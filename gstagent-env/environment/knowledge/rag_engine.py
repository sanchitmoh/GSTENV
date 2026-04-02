"""
RAG Engine — Retrieval-Augmented Generation for GST agent reasoning.

Integrates the vector store with agent prompts to provide domain-specific
context during inference. Agents can query the knowledge base for relevant
CBIC circulars, rules, and practical guidance.
"""

from __future__ import annotations

from environment.knowledge.gst_knowledge import get_all_documents
from environment.knowledge.vector_store import VectorStore

# Global singleton — lazily initialized
_rag_engine: RAGEngine | None = None


class RAGEngine:
    """
    Retrieval-Augmented Generation engine for GST domain knowledge.

    Usage:
        engine = RAGEngine()
        engine.initialize()
        context = engine.retrieve("What is Rule 36(4)?", top_k=3)
        # Inject context into LLM prompt
    """

    def __init__(self):
        self.store = VectorStore()
        self._initialized = False

    def initialize(self) -> None:
        """Load and index all GST knowledge documents."""
        if self._initialized:
            return

        docs = get_all_documents()
        self.store.add_documents(docs)
        self._initialized = True

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Retrieve relevant knowledge for a query.

        Returns list of {title, content, source, relevance} dicts.
        """
        if not self._initialized:
            self.initialize()

        results = self.store.search(query, top_k=top_k)
        return [
            {
                "title": doc.title,
                "content": doc.content,
                "source": doc.source,
                "relevance": round(score, 4),
            }
            for doc, score in results
        ]

    def get_context_for_prompt(self, query: str, top_k: int = 3) -> str:
        """
        Get formatted context string ready for injection into an LLM prompt.

        Returns a markdown-formatted knowledge section.
        """
        results = self.retrieve(query, top_k=top_k)

        if not results:
            return ""

        sections = ["## Relevant GST Knowledge\n"]
        for i, doc in enumerate(results, 1):
            sections.append(
                f"### [{i}] {doc['title']}\n"
                f"**Source**: {doc['source']} | **Relevance**: {doc['relevance']:.0%}\n\n"
                f"{doc['content']}\n"
            )

        return "\n".join(sections)

    def get_itc_rules_context(self) -> str:
        """Get all ITC-related rules as context."""
        return self.get_context_for_prompt(
            "ITC input tax credit eligibility Rule 36 Section 16 GSTR-2B", top_k=5
        )

    def get_reconciliation_context(self) -> str:
        """Get reconciliation process context."""
        return self.get_context_for_prompt(
            "reconciliation GSTR-2B purchase register matching mismatch", top_k=5
        )

    def get_mismatch_context(self) -> str:
        """Get context for handling mismatches."""
        return self.get_context_for_prompt(
            "mismatch tolerance variance follow up supplier credit note", top_k=4
        )

    @property
    def document_count(self) -> int:
        return self.store.count


def get_rag_engine() -> RAGEngine:
    """Get or create the global RAG engine singleton."""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
        _rag_engine.initialize()
    return _rag_engine
