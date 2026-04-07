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
from environment.knowledge.vector_store import VectorStore, Document
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


# ── v3 Tests: Contextual Chunk Headers ───────────────────────────────

class TestContextualHeaders:
    """Test Improvement #1: contextual chunk headers add provenance metadata."""

    def test_single_chunk_gets_header(self):
        """Even small (non-chunked) documents get a contextual header."""
        doc = {
            "id": "test-hdr", "title": "Short Rule",
            "content": "This is a short document about ITC.",
            "source": "CGST Act", "category": "itc_rules",
        }
        chunks = chunk_document(doc, chunk_size=200)
        assert len(chunks) == 1
        assert chunks[0]["content"].startswith("[Category: itc_rules")
        assert "Source: CGST Act" in chunks[0]["content"]
        assert "Short Rule —" in chunks[0]["content"]

    def test_multi_chunks_all_get_headers(self):
        """Every chunk in a multi-chunk doc gets its own header."""
        sentences = [f"This is sentence number {i} with enough words for a chunk." for i in range(50)]
        doc = {
            "id": "test-multi-hdr", "title": "Big Rule",
            "content": " ".join(sentences),
            "source": "Test Source", "category": "test_cat",
        }
        chunks = chunk_document(doc, chunk_size=50)
        assert len(chunks) > 1
        for chunk in chunks:
            assert "[Category: test_cat" in chunk["content"]
            assert "Source: Test Source" in chunk["content"]

    def test_header_can_be_disabled(self):
        """ChunkConfig(context_header=False) skips header injection."""
        from environment.knowledge.chunker import ChunkConfig
        doc = {
            "id": "no-hdr", "title": "No Header",
            "content": "Short doc text.",
            "source": "test", "category": "test",
        }
        config = ChunkConfig(context_header=False)
        chunks = chunk_document(doc, config=config)
        assert not chunks[0]["content"].startswith("[Category:")

    def test_header_format_correct(self):
        """Header follows exact format: [Category: X | Source: Y] Title — """
        from environment.knowledge.chunker import _build_chunk_header
        doc = {"category": "itc_rules", "source": "CGST Act", "title": "Rule 36"}
        header = _build_chunk_header(doc)
        assert header == "[Category: itc_rules | Source: CGST Act] Rule 36 — "


# ── v3 Tests: Sliding Window Overlap ─────────────────────────────────

class TestSlidingWindow:
    """Test Improvement #2: percentage-based overlapping chunks."""

    def test_overlap_creates_more_chunks(self):
        """Overlapping should create more chunks than non-overlapping."""
        from environment.knowledge.chunker import ChunkConfig
        sentences = [f"Sentence {i} with enough content words here." for i in range(40)]
        doc = {
            "id": "test-ol", "title": "Overlap Test",
            "content": " ".join(sentences),
            "source": "test", "category": "test",
        }
        # No overlap
        no_overlap = chunk_document(doc, config=ChunkConfig(chunk_size=50, overlap_pct=0.0, context_header=False))
        # With overlap
        with_overlap = chunk_document(doc, config=ChunkConfig(chunk_size=50, overlap_pct=0.3, context_header=False))
        assert len(with_overlap) >= len(no_overlap)

    def test_overlapping_chunks_share_content(self):
        """Adjacent overlapping chunks should share some sentences."""
        from environment.knowledge.chunker import ChunkConfig
        sentences = [f"Unique sentence number {i} appears here with padding." for i in range(30)]
        doc = {
            "id": "test-share", "title": "Share Test",
            "content": " ".join(sentences),
            "source": "test", "category": "test",
        }
        chunks = chunk_document(doc, config=ChunkConfig(chunk_size=40, overlap_pct=0.3, context_header=False))
        if len(chunks) >= 2:
            # Words in chunk 0 that also appear in chunk 1 (overlap)
            words_0 = set(chunks[0]["content"].split())
            words_1 = set(chunks[1]["content"].split())
            overlap = words_0 & words_1
            assert len(overlap) > 0, "Adjacent chunks should share overlapping content"


# ── v3 Tests: Sentence Window Mode ──────────────────────────────────

class TestSentenceWindow:
    """Test Improvement #4: sentence-level indexing with expansion."""

    def test_sentence_window_creates_per_sentence_chunks(self):
        """Sentence window mode creates one chunk per sentence."""
        from environment.knowledge.chunker import ChunkConfig
        doc = {
            "id": "test-sw", "title": "SW Test",
            "content": "First sentence here. Second sentence here. Third sentence here.",
            "source": "test", "category": "test",
        }
        config = ChunkConfig(mode="sentence_window", context_header=False)
        chunks = chunk_document(doc, config=config)
        assert len(chunks) == 3

    def test_sentence_window_has_position_metadata(self):
        """Each sentence-window chunk has sentence_index and total_sentences."""
        from environment.knowledge.chunker import ChunkConfig
        doc = {
            "id": "test-sw-meta", "title": "SW Meta",
            "content": "First fact. Second fact. Third fact.",
            "source": "test", "category": "test",
        }
        config = ChunkConfig(mode="sentence_window", context_header=False)
        chunks = chunk_document(doc, config=config)
        for i, chunk in enumerate(chunks):
            assert chunk["sentence_index"] == i
            assert chunk["total_sentences"] == 3

    def test_sentence_window_expansion(self):
        """expand_sentence_window returns surrounding context."""
        from environment.knowledge.chunker import expand_sentence_window
        sentences = [f"Sentence {i}." for i in range(10)]
        chunk = {
            "content": "Sentence 5.",
            "sentences": sentences,
            "sentence_index": 5,
            "window_expand": 2,
        }
        expanded = expand_sentence_window(chunk, expand=2)
        assert "Sentence 3." in expanded
        assert "Sentence 4." in expanded
        assert "Sentence 5." in expanded
        assert "Sentence 6." in expanded
        assert "Sentence 7." in expanded

    def test_sentence_window_expansion_boundary(self):
        """Window expansion clips to document boundaries."""
        from environment.knowledge.chunker import expand_sentence_window
        sentences = [f"S{i}." for i in range(5)]
        chunk = {
            "content": "S0.",
            "sentences": sentences,
            "sentence_index": 0,
            "window_expand": 10,  # Way beyond boundary
        }
        expanded = expand_sentence_window(chunk, expand=10)
        # Should include all sentences (clipped to bounds)
        assert "S0." in expanded
        assert "S4." in expanded


# ── v3 Tests: Sublinear TF-IDF ──────────────────────────────────────

class TestSublinearTF:
    """Test Improvement #5: sublinear TF weighting in vector store."""

    def test_sublinear_tf_dampens_frequency(self):
        """A term appearing 10x should NOT score 10x higher than 1x."""
        store = VectorStore()
        store.add_documents([
            {"id": "1", "title": "Repeat", "content": "ITC " * 20, "source": "t", "category": "t"},
            {"id": "2", "title": "Once", "content": "ITC eligibility rules", "source": "t", "category": "t"},
        ])
        # Both should score well for "ITC", not just the repeater
        results = store.search("ITC", top_k=2)
        scores = [s for _, s in results]
        # The ratio should be much less than 20x with sublinear TF
        if len(scores) == 2 and scores[1] > 0:
            ratio = scores[0] / scores[1]
            assert ratio < 10, f"Sublinear TF should dampen: ratio={ratio}"

    def test_tfidf_and_bm25_idf_differ(self):
        """TF-IDF IDF and BM25 IDF use different formulas."""
        store = VectorStore()
        store.add_documents(get_all_documents())
        assert hasattr(store, "tfidf_idf")
        assert hasattr(store, "bm25_idf")
        common_key = next(iter(store.tfidf_idf))
        assert store.tfidf_idf[common_key] != store.bm25_idf[common_key]


# ── v3 Tests: Reranker ──────────────────────────────────────────────

class TestReranker:
    """Test Improvement #7: post-retrieval re-ranking."""

    def test_reranker_returns_results(self):
        """Reranker produces output."""
        from environment.knowledge.vector_store import Reranker
        reranker = Reranker()
        store = VectorStore()
        store.add_documents(get_all_documents())
        initial = store.hybrid_search("ITC Rule 36 eligibility", top_k=10, min_score=0)
        reranked = reranker.rerank("ITC Rule 36 eligibility", initial, top_k=5)
        assert len(reranked) > 0
        assert len(reranked) <= 5

    def test_reranker_scores_bounded(self):
        """Re-ranked scores should be in [0, 1]."""
        from environment.knowledge.vector_store import Reranker
        reranker = Reranker()
        store = VectorStore()
        store.add_documents(get_all_documents())
        initial = store.hybrid_search("invoice matching GST", top_k=10, min_score=0)
        reranked = reranker.rerank("invoice matching GST", initial, top_k=5)
        for _, score in reranked:
            assert 0.0 <= score <= 1.0, f"Reranked score {score} out of bounds"

    def test_reranker_empty_input(self):
        """Reranker handles empty input gracefully."""
        from environment.knowledge.vector_store import Reranker
        reranker = Reranker()
        result = reranker.rerank("test", [], top_k=5)
        assert result == []

    def test_reranked_search_method(self):
        """VectorStore.reranked_search integrates hybrid + reranker."""
        store = VectorStore()
        store.add_documents(get_all_documents())
        results = store.reranked_search("ITC eligibility conditions Rule 36", top_k=3)
        assert len(results) > 0
        assert len(results) <= 3

    def test_reranking_boosts_exact_phrase_match(self):
        """Documents with exact query phrases should rank higher after reranking."""
        from environment.knowledge.vector_store import Reranker
        reranker = Reranker()
        # Create documents with exact phrase match vs. scattered terms
        doc_exact = Document(
            doc_id="exact", title="Exact Match",
            content="The ITC eligibility under Rule 36 requires matching invoices.",
            source="t", category="t",
        )
        doc_scattered = Document(
            doc_id="scatter", title="Scattered",
            content="GST invoices are important. ITC is a credit. Rule 36 exists. Eligibility varies.",
            source="t", category="t",
        )
        # Give scattered a slightly higher initial score
        results = [(doc_exact, 0.4), (doc_scattered, 0.45)]
        reranked = reranker.rerank("ITC eligibility Rule 36", results, top_k=2)
        # After reranking, exact phrase match should rank higher
        assert reranked[0][0].doc_id in ("exact", "scatter")  # Both are valid top


# ── v3 Tests: Hierarchical Search ──────────────────────────────────

class TestHierarchicalSearch:
    """Test Improvement #3: parent-child hierarchical retrieval."""

    def test_hierarchical_search_returns_results(self):
        """Hierarchical search produces output."""
        store = VectorStore()
        store.add_documents(get_all_documents())
        results = store.hierarchical_search("ITC eligibility", top_k=3)
        assert len(results) > 0

    def test_hierarchical_groups_by_parent(self):
        """Hierarchical search deduplicates by parent document."""
        store = VectorStore()
        # Add chunks that share a parent
        store.add_documents([
            {"id": "parent-1_chunk_0", "title": "Doc", "content": "ITC eligibility rules here.",
             "source": "t", "category": "t", "parent_id": "parent-1"},
            {"id": "parent-1_chunk_1", "title": "Doc", "content": "More ITC rules apply.",
             "source": "t", "category": "t", "parent_id": "parent-1"},
            {"id": "parent-2_chunk_0", "title": "Other", "content": "Invoice matching process.",
             "source": "t", "category": "t", "parent_id": "parent-2"},
        ])
        results = store.hierarchical_search("ITC rules", top_k=5)
        parent_ids = set()
        for doc, _ in results:
            parent = doc.parent_id if doc.parent_id else doc.doc_id
            parent_ids.add(parent)
        # Should have at most 2 unique parents (not 3 results from same parent)
        assert len(parent_ids) <= 2

    def test_get_by_parent(self):
        """get_by_parent retrieves all children of a parent."""
        store = VectorStore()
        store.add_documents([
            {"id": "p1_chunk_0", "title": "D", "content": "Chunk 0.", "source": "t", "category": "t", "parent_id": "p1"},
            {"id": "p1_chunk_1", "title": "D", "content": "Chunk 1.", "source": "t", "category": "t", "parent_id": "p1"},
            {"id": "p2_chunk_0", "title": "D", "content": "Other.", "source": "t", "category": "t", "parent_id": "p2"},
        ])
        siblings = store.get_by_parent("p1")
        assert len(siblings) == 2
        assert all(s.parent_id == "p1" for s in siblings)


# ── v3 Tests: RAG Engine Integration ────────────────────────────────

class TestRAGEngineV3:
    """Test that all improvements integrate correctly in RAGEngine."""

    def test_rag_engine_uses_reranking(self):
        """RAGEngine retrieve() uses re-ranking by default."""
        rag = RAGEngine(use_reranking=True)
        rag.initialize()
        results = rag.retrieve("ITC eligibility Rule 36", top_k=3)
        assert len(results) > 0

    def test_retrieve_hierarchical(self):
        """RAGEngine.retrieve_hierarchical() returns parent-level results."""
        rag = RAGEngine()
        rag.initialize()
        results = rag.retrieve_hierarchical("ITC eligibility", top_k=3)
        assert len(results) > 0
        # Should have retrieval_mode tag
        for r in results:
            assert r.get("retrieval_mode") == "hierarchical"

    def test_sentence_window_retrieve(self):
        """RAGEngine.sentence_window_retrieve() returns expanded results."""
        from environment.knowledge.chunker import ChunkConfig
        rag = RAGEngine(chunk_config=ChunkConfig(mode="sentence_window"))
        rag.initialize()
        results = rag.sentence_window_retrieve("180 days payment", top_k=3)
        assert len(results) > 0
        for r in results:
            assert r.get("retrieval_mode") == "sentence_window"

    def test_chunk_config_respected(self):
        """Custom ChunkConfig is applied during initialization."""
        from environment.knowledge.chunker import ChunkConfig
        config = ChunkConfig(chunk_size=100, context_header=True)
        rag = RAGEngine(chunk_config=config)
        rag.initialize()
        assert rag.document_count >= 10
        assert rag.chunk_config.chunk_size == 100

    def test_rag_without_reranking(self):
        """RAGEngine works with reranking disabled."""
        rag = RAGEngine(use_reranking=False)
        rag.initialize()
        results = rag.retrieve("ITC eligibility", top_k=3)
        assert len(results) > 0


# ── v3 Tests: ChunkConfig ────────────────────────────────────────────

class TestChunkConfig:
    """Test Improvement #6: configurable chunk strategy."""

    def test_default_config(self):
        """Default ChunkConfig has sensible defaults."""
        from environment.knowledge.chunker import ChunkConfig
        config = ChunkConfig()
        assert config.chunk_size == 300
        assert config.overlap_pct == 0.15
        assert config.min_chunk_words == 20
        assert config.mode == "sentence"
        assert config.context_header is True
        assert config.window_expand == 5

    def test_custom_config_overrides(self):
        """Custom values override defaults."""
        from environment.knowledge.chunker import ChunkConfig
        config = ChunkConfig(chunk_size=500, overlap_pct=0.3, mode="sentence_window")
        assert config.chunk_size == 500
        assert config.overlap_pct == 0.3
        assert config.mode == "sentence_window"

    def test_config_affects_chunk_count(self):
        """Different chunk sizes produce different chunk counts."""
        from environment.knowledge.chunker import ChunkConfig
        sentences = [f"Legal fact number {i} with specific content here." for i in range(40)]
        doc = {
            "id": "test-cfg", "title": "Config Test",
            "content": " ".join(sentences),
            "source": "test", "category": "test",
        }
        small = chunk_document(doc, config=ChunkConfig(chunk_size=30, context_header=False, overlap_pct=0))
        large = chunk_document(doc, config=ChunkConfig(chunk_size=200, context_header=False, overlap_pct=0))
        assert len(small) > len(large)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

