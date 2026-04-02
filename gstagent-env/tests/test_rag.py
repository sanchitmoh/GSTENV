"""
Tests for the RAG knowledge base (Phase 10).

Covers: knowledge content, vector store, RAG engine retrieval.
"""

import pytest
from environment.knowledge.gst_knowledge import (
    get_all_documents,
    get_documents_by_category,
    search_documents,
)
from environment.knowledge.vector_store import VectorStore
from environment.knowledge.rag_engine import RAGEngine


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


# ── Vector Store Tests ───────────────────────────────────────────

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


# ── RAG Engine Tests ─────────────────────────────────────────────

class TestRAGEngine:
    def test_initialize(self):
        rag = RAGEngine()
        rag.initialize()
        assert rag.document_count >= 10

    def test_retrieve(self):
        rag = RAGEngine()
        rag.initialize()
        results = rag.retrieve("ITC eligibility Rule 36", top_k=3)
        assert len(results) > 0
        assert "title" in results[0]
        assert "content" in results[0]
        assert "relevance" in results[0]

    def test_get_context_for_prompt(self):
        rag = RAGEngine()
        rag.initialize()
        context = rag.get_context_for_prompt("invoice matching reconciliation")
        assert len(context) > 0
        assert "Relevant GST Knowledge" in context

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
