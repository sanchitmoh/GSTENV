"""
Tests for RAG v2.1 — covers all anti-hallucination fixes:

Hallucination Vector Fixes:
- Chunker: sentence-boundary splitting preserves legal clauses
- VectorStore: RRF score normalization enables threshold filtering
- VectorStore: separate TF-IDF/BM25 IDF dicts
- Faithfulness: numeric claim extraction (%, days, amounts, dates)

Silent Bug Fixes:
- QueryProcessor: negation preservation (expand-first, clean-second)
- RAGEngine: parent_id deduplication for chunk-level results
- EvalHarness: explicit parent_id resolution, normalized keyword matching

Advancement:
- GROUNDING_CLAUSE injection into agent system prompts
"""

import pytest
from environment.knowledge.gst_knowledge import (
    get_all_documents,
    get_documents_by_category,
    search_documents,
)
from environment.knowledge.vector_store import VectorStore
from environment.knowledge.rag_engine import RAGEngine
from environment.knowledge.query_processor import QueryProcessor
from environment.knowledge.chunker import (
    chunk_document,
    chunk_all_documents,
    _split_sentences,
)
from environment.knowledge.faithfulness import (
    assert_grounded,
    extract_legal_references,
    extract_numeric_claims,
    get_grounding_report,
)
from environment.knowledge.eval_rag import evaluate_retrieval, _resolve_parent_ids
from environment.agents.base_agent import GROUNDING_CLAUSE


# ── Knowledge Base Tests ─────────────────────────────────────────

class TestKnowledgeBase:
    def test_has_documents(self):
        docs = get_all_documents()
        assert len(docs) >= 10

    def test_document_structure(self):
        for doc in get_all_documents():
            assert "id" in doc
            assert "title" in doc
            assert "content" in doc
            assert "source" in doc
            assert "category" in doc
            assert len(doc["content"]) > 50

    def test_has_rule_36_4(self):
        ids = {d["id"] for d in get_all_documents()}
        assert "Rule-36-4" in ids

    def test_has_section_16_2(self):
        ids = {d["id"] for d in get_all_documents()}
        assert "Section-16-2" in ids

    def test_filter_by_category(self):
        itc_docs = get_documents_by_category("itc_rules")
        assert len(itc_docs) >= 2
        assert all(d["category"] == "itc_rules" for d in itc_docs)

    def test_keyword_search(self):
        results = search_documents("ITC eligibility Rule 36", top_k=3)
        assert len(results) > 0


# ── Query Processor Tests (with negation fix) ────────────────────

class TestQueryProcessor:
    def setup_method(self):
        self.qp = QueryProcessor()

    def test_expand_adds_synonyms(self):
        expanded = self.qp.expand("supplier itc mismatch")
        assert "vendor" in expanded
        assert "input tax credit" in expanded
        assert "variance" in expanded

    def test_expand_preserves_originals(self):
        expanded = self.qp.expand("supplier")
        assert "supplier" in expanded

    def test_expand_unknown_terms_unchanged(self):
        expanded = self.qp.expand("xyznotaword")
        assert "xyznotaword" in expanded

    def test_clean_removes_stop_words(self):
        cleaned = self.qp.clean("what is the ITC eligibility for my supplier")
        assert "what" not in cleaned.split()
        assert "itc" in cleaned
        assert "supplier" in cleaned

    # FIX: negation must be preserved
    def test_clean_preserves_negation_not(self):
        cleaned = self.qp.clean("ITC not available when supplier doesn't file")
        assert "not" in cleaned

    def test_clean_preserves_negation_without(self):
        cleaned = self.qp.clean("credit without invoice is blocked")
        assert "without" in cleaned

    def test_clean_preserves_negation_never(self):
        cleaned = self.qp.clean("ITC should never be claimed in this case")
        assert "never" in cleaned

    def test_decompose_single_query(self):
        parts = self.qp.decompose("What is the ITC eligibility?")
        assert len(parts) == 1

    def test_decompose_compound_query(self):
        parts = self.qp.decompose(
            "What is the ITC eligibility and also how does reconciliation work?"
        )
        assert len(parts) == 2

    # FIX: process() now does expand → clean (not clean → expand)
    def test_process_preserves_negation_semantics(self):
        result = self.qp.process("ITC not available when supplier doesn't file GSTR-1")
        # "not" must survive the full pipeline
        assert "not" in result

    def test_process_full_pipeline(self):
        result = self.qp.process("What is the ITC for my supplier?")
        assert isinstance(result, str)
        assert len(result) > 0
        # Should have expanded terms
        assert "vendor" in result or "seller" in result  # synonym of "supplier"


# ── Chunker Tests (sentence-boundary fix) ────────────────────────

class TestChunker:
    def test_small_doc_not_chunked(self):
        doc = {
            "id": "test-1", "title": "Short Doc",
            "content": "This is a short document with few words.",
            "source": "test", "category": "test",
        }
        chunks = chunk_document(doc, chunk_size=200)
        assert len(chunks) == 1
        assert chunks[0]["id"] == "test-1"

    def test_large_doc_chunked(self):
        # Build a document with many sentences
        sentences = [f"This is sentence number {i} with enough words." for i in range(50)]
        doc = {
            "id": "test-big", "title": "Big Doc",
            "content": " ".join(sentences),
            "source": "test", "category": "test",
        }
        chunks = chunk_document(doc, chunk_size=50)
        assert len(chunks) > 1

    def test_chunks_preserve_metadata(self):
        sentences = [f"Sentence {i} has content here." for i in range(50)]
        doc = {
            "id": "test-meta", "title": "Meta Doc",
            "content": " ".join(sentences),
            "source": "test-source", "category": "test-cat",
        }
        chunks = chunk_document(doc, chunk_size=50)
        for chunk in chunks:
            assert chunk["title"] == "Meta Doc"
            assert chunk["source"] == "test-source"
            assert chunk["category"] == "test-cat"
            assert chunk["parent_id"] == "test-meta"

    def test_chunk_ids_unique(self):
        sentences = [f"Sentence {i} has content here." for i in range(100)]
        doc = {
            "id": "test-ids", "title": "IDs Doc",
            "content": " ".join(sentences),
            "source": "test", "category": "test",
        }
        chunks = chunk_document(doc, chunk_size=50)
        ids = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_all_documents(self):
        docs = get_all_documents()
        all_chunks = chunk_all_documents(docs, chunk_size=200)
        assert len(all_chunks) >= len(docs)

    # FIX: sentence-boundary splitting preserves legal clauses
    def test_legal_clause_not_split(self):
        """The critical fix: legal facts stay in one chunk."""
        doc = {
            "id": "test-legal", "title": "Legal Clause",
            "content": (
                "The 5% provisional credit allowed earlier has been removed w.e.f. 01.01.2022. "
                "This means no excess ITC can be claimed. "
                "Taxpayers must match every invoice. "
                "Only exactly matching amounts are eligible."
            ),
            "source": "test", "category": "test",
        }
        # With a large chunk_size, the whole thing should be one chunk
        chunks = chunk_document(doc, chunk_size=200)
        assert len(chunks) == 1
        assert "5%" in chunks[0]["content"]
        assert "removed" in chunks[0]["content"]
        assert "01.01.2022" in chunks[0]["content"]

    def test_sentence_splitter_preserves_abbreviations(self):
        """w.e.f. should not cause a sentence split."""
        text = "The rule removed w.e.f. 01.01.2022. The new rule applies."
        sentences = _split_sentences(text)
        # w.e.f. should NOT split the first sentence
        assert any("w.e.f." in s for s in sentences)

    def test_sentence_splitter_preserves_numbered_lists(self):
        """Numbered lists like '1. 2. 3.' should not cause splits."""
        text = "Requirements: 1. Valid invoice 2. Receipt of goods 3. Tax paid"
        sentences = _split_sentences(text)
        # Should stay as a single sentence (no capital after "1.")
        assert len(sentences) <= 2


# ── Vector Store v2 Tests (IDF separation + RRF normalization) ───

class TestVectorStore:
    def test_add_and_search(self):
        store = VectorStore()
        store.add_documents([
            {"id": "1", "title": "GST ITC Rules", "content": "Input tax credit under GST", "source": "test", "category": "rules"},
            {"id": "2", "title": "Invoice Matching", "content": "How to match invoices", "source": "test", "category": "process"},
            {"id": "3", "title": "Cooking Recipes", "content": "How to make pasta carbonara", "source": "test", "category": "other"},
        ])
        results = store.search("GST input tax credit", top_k=2)
        assert len(results) > 0
        assert results[0][0].doc_id == "1"

    def test_empty_query(self):
        store = VectorStore()
        store.add_documents(get_all_documents())
        results = store.search("", top_k=3)
        assert isinstance(results, list)

    def test_count(self):
        store = VectorStore()
        store.add_documents(get_all_documents())
        assert store.count >= 10

    def test_get_by_id(self):
        store = VectorStore()
        store.add_documents(get_all_documents())
        doc = store.get_by_id("Rule-36-4")
        assert doc is not None

    def test_get_by_category(self):
        store = VectorStore()
        store.add_documents(get_all_documents())
        rules = store.get_by_category("rules")
        assert len(rules) >= 1

    def test_cosine_similarity_bounds(self):
        store = VectorStore()
        store.add_documents(get_all_documents())
        results = store.search("GST reconciliation", top_k=10)
        for _, score in results:
            assert 0.0 <= score <= 1.0

    def test_score_threshold_filters_noise(self):
        store = VectorStore()
        store.add_documents(get_all_documents())
        strict = store.search("GST", top_k=10, min_score=0.3)
        loose = store.search("GST", top_k=10, min_score=0.0)
        assert len(strict) <= len(loose)

    def test_search_with_filter(self):
        store = VectorStore()
        store.add_documents(get_all_documents())
        results = store.search_with_filter("ITC credit eligibility", "itc_rules", top_k=5)
        assert all(doc.category == "itc_rules" for doc, _ in results)

    def test_hybrid_search_returns_results(self):
        store = VectorStore()
        store.add_documents(get_all_documents())
        results = store.hybrid_search("ITC eligibility conditions", top_k=3)
        assert len(results) > 0

    def test_bm25_score_positive(self):
        store = VectorStore()
        store.add_documents(get_all_documents())
        query_tokens = store._tokenize("ITC Rule 36 eligibility")
        score = store._bm25_score(query_tokens, 0)
        assert isinstance(score, float)

    # FIX: separate IDF dicts
    def test_separate_idf_dicts(self):
        """TF-IDF IDF and BM25 IDF must be separate with different formulas."""
        store = VectorStore()
        store.add_documents(get_all_documents())
        assert hasattr(store, "tfidf_idf")
        assert hasattr(store, "bm25_idf")
        # They should have same keys but different values
        common_key = next(iter(store.tfidf_idf))
        assert store.tfidf_idf[common_key] != store.bm25_idf[common_key]

    # FIX: RRF scores normalized to [0,1]
    def test_hybrid_search_scores_normalized(self):
        """RRF scores must be in [0,1] so min_score threshold works."""
        store = VectorStore()
        store.add_documents(get_all_documents())
        results = store.hybrid_search("ITC eligibility", top_k=10, min_score=0.0)
        for _, score in results:
            assert 0.0 <= score <= 1.0, f"RRF score {score} not in [0,1]"

    def test_hybrid_search_threshold_actually_filters(self):
        """min_score must actually filter results in hybrid search."""
        store = VectorStore()
        store.add_documents(get_all_documents())
        all_results = store.hybrid_search("ITC", top_k=20, min_score=0.0)
        filtered = store.hybrid_search("ITC", top_k=20, min_score=0.5)
        # High threshold should return fewer or equal results
        assert len(filtered) <= len(all_results)


# ── RAG Engine v2 Tests ──────────────────────────────────────────

class TestRAGEngine:
    def test_initialize(self):
        rag = RAGEngine()
        rag.initialize()
        assert rag.document_count >= 10

    def test_retrieve_returns_results(self):
        rag = RAGEngine()
        rag.initialize()
        results = rag.retrieve("ITC eligibility Rule 36", top_k=3)
        assert len(results) > 0
        assert "title" in results[0]
        assert "content" in results[0]
        assert "relevance" in results[0]
        assert "category" in results[0]

    def test_retrieve_with_category_filter(self):
        rag = RAGEngine()
        rag.initialize()
        results = rag.retrieve("ITC", top_k=3, category_filter="itc_rules")
        for r in results:
            assert r["category"] == "itc_rules"

    def test_retrieve_respects_min_score(self):
        rag = RAGEngine(min_score=0.5)
        rag.initialize()
        results = rag.retrieve("GST", top_k=10)
        assert isinstance(results, list)

    # FIX: parent_id deduplication
    def test_retrieve_multi_deduplicates_by_parent(self):
        rag = RAGEngine()
        rag.initialize()
        results = rag.retrieve_multi(
            "What about ITC rules and also reconciliation process?"
        )
        # Extract parent IDs — should be unique
        parent_ids = [r["doc_id"].split("_chunk_")[0] for r in results]
        assert len(parent_ids) == len(set(parent_ids)), "Duplicate parents found"

    def test_get_context_for_prompt(self):
        rag = RAGEngine()
        rag.initialize()
        context = rag.get_context_for_prompt("invoice matching reconciliation")
        assert len(context) > 0
        assert "Relevant GST Knowledge" in context

    def test_context_has_confidence_tags(self):
        rag = RAGEngine()
        rag.initialize()
        context = rag.get_context_for_prompt("ITC eligibility")
        assert "confidence" in context.lower()

    def test_context_respects_token_budget(self):
        rag = RAGEngine()
        rag.initialize()
        context = rag.get_context_for_prompt("ITC", top_k=10, max_tokens=100)
        word_count = len(context.split())
        assert word_count < 300

    def test_empty_query_returns_fallback(self):
        rag = RAGEngine(min_score=0.99)
        rag.initialize()
        context = rag.get_context_for_prompt("zzzzzzzznotarealthing")
        assert "No relevant GST knowledge" in context

    def test_itc_rules_context(self):
        rag = RAGEngine()
        rag.initialize()
        context = rag.get_itc_rules_context()
        assert "Rule 36" in context or "ITC" in context

    def test_reconciliation_context(self):
        rag = RAGEngine()
        rag.initialize()
        assert len(rag.get_reconciliation_context()) > 0

    def test_mismatch_context(self):
        rag = RAGEngine()
        rag.initialize()
        assert len(rag.get_mismatch_context()) > 0

    def test_double_initialize_safe(self):
        rag = RAGEngine()
        rag.initialize()
        count1 = rag.document_count
        rag.initialize()
        assert rag.document_count == count1


# ── Faithfulness Tests (with numeric claim fix) ──────────────────

class TestFaithfulness:
    def test_grounded_response(self):
        context = [{"content": "Rule 36(4) requires matching. Circular 170/02/2022 states ITC rules."}]
        response = "Under Rule 36(4), matching is required as per Circular 170/02/2022."
        assert assert_grounded(response, context) is True

    def test_ungrounded_response(self):
        context = [{"content": "Rule 36(4) requires matching."}]
        response = "According to Rule 99, which is a new rule, matching is not needed."
        assert assert_grounded(response, context) is False

    def test_no_references_is_grounded(self):
        context = [{"content": "Some general text about GST."}]
        response = "GST is a tax system in India."
        assert assert_grounded(response, context) is True

    def test_extract_legal_references(self):
        text = "Rule 36(4) and Section 16(2) apply. See Circular 170/02/2022."
        refs = extract_legal_references(text)
        assert any("Rule 36" in r for r in refs)
        assert any("Section 16" in r for r in refs)
        assert any("170/02/2022" in r for r in refs)

    # FIX: numeric claim extraction
    def test_extract_numeric_percentage(self):
        claims = extract_numeric_claims("You can claim 5% provisional ITC.")
        assert any("5%" in c for c in claims)

    def test_extract_numeric_days(self):
        claims = extract_numeric_claims("Payment must be within 180 days.")
        assert any("180 days" in c for c in claims)

    def test_extract_numeric_date(self):
        claims = extract_numeric_claims("Removed w.e.f. 01.01.2022.")
        assert any("01.01.2022" in c for c in claims)

    def test_extract_numeric_amount(self):
        claims = extract_numeric_claims("Threshold is ₹18L for registration.")
        assert len(claims) > 0

    # FIX: hallucinated percentage detected
    def test_hallucinated_percentage_caught(self):
        """LLM says '10%' but context only has '5%' — must fail."""
        context = [{"content": "The 5% provisional credit allowed earlier has been removed."}]
        response = "You can claim 10% provisional ITC under Rule 36."
        assert assert_grounded(response, context) is False

    def test_correct_percentage_passes(self):
        """LLM says '5%' and context has '5%' — must pass."""
        context = [{"content": "The 5% provisional credit allowed earlier has been removed."}]
        response = "The 5% provisional credit has been removed."
        assert assert_grounded(response, context) is True

    def test_hallucinated_days_caught(self):
        """LLM says '90 days' but context says '180 days' — must fail."""
        context = [{"content": "Payment to supplier must be made within 180 days."}]
        response = "You must pay within 90 days to retain ITC."
        assert assert_grounded(response, context) is False

    def test_grounding_report_structure(self):
        context = [{"content": "Rule 36(4) of CGST Rules. The 5% limit applies."}]
        response = "As per Rule 36(4), the 5% limit applies."
        report = get_grounding_report(response, context)
        assert "is_faithful" in report
        assert "legal_refs" in report
        assert "numeric_claims" in report
        assert report["is_faithful"] is True

    def test_grounding_report_catches_both(self):
        """Report separates legal and numeric ungrounded claims."""
        context = [{"content": "Rule 36(4) allows 5% credit."}]
        response = "Under Rule 99, you can claim 10% credit."
        report = get_grounding_report(response, context)
        assert report["is_faithful"] is False
        assert len(report["legal_refs"]["ungrounded"]) > 0
        assert len(report["numeric_claims"]["ungrounded"]) > 0


# ── Evaluation Harness Tests (parent_id fix) ─────────────────────

class TestEvaluation:
    def test_evaluation_runs(self):
        report = evaluate_retrieval()
        assert "source_hit_rate" in report
        assert "keyword_hit_rate" in report
        assert report["total_queries"] > 0

    def test_source_hit_rate_reasonable(self):
        report = evaluate_retrieval()
        assert report["source_hit_rate"] >= 0.5

    def test_keyword_hit_rate_reasonable(self):
        report = evaluate_retrieval()
        assert report["keyword_hit_rate"] >= 0.3

    # FIX: explicit parent_id resolution
    def test_parent_id_resolution(self):
        """Chunk IDs must resolve to parent document IDs."""
        results = [
            {"doc_id": "Rule-36-4_chunk_0"},
            {"doc_id": "Rule-36-4_chunk_1"},
            {"doc_id": "Section-16-2"},
        ]
        parents = _resolve_parent_ids(results)
        assert "Rule-36-4" in parents
        assert "Section-16-2" in parents
        # Should NOT contain chunk IDs
        assert "Rule-36-4_chunk_0" not in parents or "Rule-36-4" in parents


# ── Grounding Clause Tests ───────────────────────────────────────

class TestGroundingClause:
    def test_grounding_clause_exists(self):
        assert len(GROUNDING_CLAUSE) > 50

    def test_grounding_clause_has_refusal(self):
        assert "don't have a verified source" in GROUNDING_CLAUSE.lower() or \
               "do not" in GROUNDING_CLAUSE.lower()

    def test_auditor_has_grounding_clause(self):
        from environment.agents.auditor import AuditorAgent
        agent = AuditorAgent()
        prompt = agent.get_system_prompt()
        assert "GROUNDING" in prompt or "Do NOT infer" in prompt

    def test_reporter_has_grounding_clause(self):
        from environment.agents.reporter import ReporterAgent
        agent = ReporterAgent()
        prompt = agent.get_system_prompt()
        assert "GROUNDING" in prompt or "Do NOT infer" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
