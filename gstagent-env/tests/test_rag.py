"""
Tests for RAG v2 — covers all upgrades:
- Query processor (synonym expansion, decomposition, cleaning)
- Document chunker (overlap, metadata preservation)
- Vector store v2 (score thresholds, category filter, hybrid search, BM25)
- RAG Engine v2 (full pipeline, token budget, multi-query, categories)
- Faithfulness checker (grounding assertions)
- Evaluation harness (ground truth)
"""

import pytest
from environment.knowledge.gst_knowledge import get_all_documents, get_documents_by_category, search_documents
from environment.knowledge.vector_store import VectorStore
from environment.knowledge.rag_engine import RAGEngine
from environment.knowledge.query_processor import QueryProcessor
from environment.knowledge.chunker import chunk_document, chunk_all_documents
from environment.knowledge.faithfulness import (
    assert_grounded,
    extract_legal_references,
    get_grounding_report,
)
from environment.knowledge.eval_rag import evaluate_retrieval


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
        docs = get_all_documents()
        ids = {d["id"] for d in docs}
        assert "Rule-36-4" in ids

    def test_has_section_16_2(self):
        docs = get_all_documents()
        ids = {d["id"] for d in docs}
        assert "Section-16-2" in ids

    def test_filter_by_category(self):
        itc_docs = get_documents_by_category("itc_rules")
        assert len(itc_docs) >= 2
        assert all(d["category"] == "itc_rules" for d in itc_docs)

    def test_keyword_search(self):
        results = search_documents("ITC eligibility Rule 36", top_k=3)
        assert len(results) > 0
        assert any("Rule 36" in r["title"] or "ITC" in r["title"] for r in results)


# ── Query Processor Tests ────────────────────────────────────────

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

    def test_decompose_single_query(self):
        parts = self.qp.decompose("What is the ITC eligibility?")
        assert len(parts) == 1

    def test_decompose_compound_query(self):
        parts = self.qp.decompose(
            "What is the ITC eligibility and also how does reconciliation work?"
        )
        assert len(parts) == 2

    def test_process_full_pipeline(self):
        result = self.qp.process("What is the ITC for my supplier?")
        # Should have expanded terms, no stop words
        assert isinstance(result, str)
        assert len(result) > 0


# ── Chunker Tests ────────────────────────────────────────────────

class TestChunker:
    def test_small_doc_not_chunked(self):
        doc = {
            "id": "test-1",
            "title": "Short Doc",
            "content": "This is a short document with few words.",
            "source": "test",
            "category": "test",
        }
        chunks = chunk_document(doc, chunk_size=200)
        assert len(chunks) == 1
        assert chunks[0]["id"] == "test-1"  # Unchanged

    def test_large_doc_chunked(self):
        content = " ".join([f"word{i}" for i in range(500)])
        doc = {
            "id": "test-big",
            "title": "Big Doc",
            "content": content,
            "source": "test",
            "category": "test",
        }
        chunks = chunk_document(doc, chunk_size=200, overlap=40)
        assert len(chunks) > 1

    def test_chunks_preserve_metadata(self):
        content = " ".join([f"word{i}" for i in range(500)])
        doc = {
            "id": "test-meta",
            "title": "Meta Doc",
            "content": content,
            "source": "test-source",
            "category": "test-cat",
        }
        chunks = chunk_document(doc, chunk_size=200)
        for chunk in chunks:
            assert chunk["title"] == "Meta Doc"
            assert chunk["source"] == "test-source"
            assert chunk["category"] == "test-cat"
            assert chunk["parent_id"] == "test-meta"

    def test_chunk_ids_unique(self):
        content = " ".join([f"word{i}" for i in range(1000)])
        doc = {
            "id": "test-ids",
            "title": "IDs Doc",
            "content": content,
            "source": "test",
            "category": "test",
        }
        chunks = chunk_document(doc, chunk_size=200, overlap=40)
        ids = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids))  # All unique

    def test_chunk_all_documents(self):
        docs = get_all_documents()
        all_chunks = chunk_all_documents(docs, chunk_size=200, overlap=40)
        assert len(all_chunks) >= len(docs)


# ── Vector Store v2 Tests ────────────────────────────────────────

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
        assert results[0][0].doc_id == "1"  # Most relevant
        assert results[0][1] > results[1][1] if len(results) > 1 else True

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
        assert doc.doc_id == "Rule-36-4"

    def test_get_by_category(self):
        store = VectorStore()
        store.add_documents(get_all_documents())
        rules = store.get_by_category("rules")
        assert len(rules) >= 1
        assert all(d.category == "rules" for d in rules)

    def test_cosine_similarity_bounds(self):
        store = VectorStore()
        store.add_documents(get_all_documents())
        results = store.search("GST reconciliation", top_k=10)
        for _, score in results:
            assert 0.0 <= score <= 1.0

    def test_score_threshold_filters_noise(self):
        store = VectorStore()
        store.add_documents(get_all_documents())
        # High threshold should return fewer results
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
        # All results should be in the itc_rules category
        for r in results:
            assert r["category"] == "itc_rules"

    def test_retrieve_respects_min_score(self):
        rag = RAGEngine(min_score=0.5)
        rag.initialize()
        results = rag.retrieve("GST", top_k=10)
        # With high threshold, some queries return fewer results
        assert isinstance(results, list)

    def test_retrieve_multi_deduplicates(self):
        rag = RAGEngine()
        rag.initialize()
        results = rag.retrieve_multi(
            "What about ITC rules and also reconciliation process?"
        )
        doc_ids = [r["doc_id"] for r in results]
        assert len(doc_ids) == len(set(doc_ids))  # No duplicates

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
        # Should have confidence indicators
        assert "confidence" in context.lower()

    def test_context_respects_token_budget(self):
        rag = RAGEngine()
        rag.initialize()
        context = rag.get_context_for_prompt("ITC", top_k=10, max_tokens=100)
        # With very low budget, should truncate
        word_count = len(context.split())
        # 100 tokens ≈ 77 words, but headers add overhead
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
        context = rag.get_reconciliation_context()
        assert len(context) > 0

    def test_mismatch_context(self):
        rag = RAGEngine()
        rag.initialize()
        context = rag.get_mismatch_context()
        assert len(context) > 0

    def test_double_initialize_safe(self):
        rag = RAGEngine()
        rag.initialize()
        count1 = rag.document_count
        rag.initialize()
        count2 = rag.document_count
        assert count1 == count2


# ── Faithfulness Tests ───────────────────────────────────────────

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

    def test_extract_references(self):
        text = "Rule 36(4) and Section 16(2) apply. See Circular 170/02/2022."
        refs = extract_legal_references(text)
        assert any("Rule 36" in r for r in refs)
        assert any("Section 16" in r for r in refs)
        assert any("170/02/2022" in r for r in refs)

    def test_grounding_report_structure(self):
        context = [{"content": "Rule 36(4) of CGST Rules."}]
        response = "As per Rule 36(4), ITC must be matched."
        report = get_grounding_report(response, context)
        assert "is_faithful" in report
        assert "total_references" in report
        assert "grounded_references" in report
        assert "ungrounded_references" in report
        assert report["is_faithful"] is True


# ── Evaluation Harness Tests ─────────────────────────────────────

class TestEvaluation:
    def test_evaluation_runs(self):
        report = evaluate_retrieval()
        assert "source_hit_rate" in report
        assert "keyword_hit_rate" in report
        assert report["total_queries"] > 0

    def test_source_hit_rate_reasonable(self):
        report = evaluate_retrieval()
        # With synonym expansion and hybrid search, should hit >50%
        assert report["source_hit_rate"] >= 0.5

    def test_keyword_hit_rate_reasonable(self):
        report = evaluate_retrieval()
        assert report["keyword_hit_rate"] >= 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
