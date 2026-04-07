"""
RAG Engine v2 — production-grade Retrieval-Augmented Generation for GST agents.

Upgrades from v1:
- Query expansion via GST synonym processor (biggest retrieval improvement)
- Score threshold filtering (min_score=0.08 — stops noise)
- Metadata category filtering for precise retrieval
- Token-budget context assembly (no context overflow)
- Hybrid search (TF-IDF + BM25 via RRF)
- Document chunking support
- Faithfulness grounding check for LLM responses
- Confidence-tagged context sections (High/Medium/Low)

Pipeline: Query → Expand → Decompose → Hybrid Search → Filter → Budget → Format
"""

from __future__ import annotations

import structlog

from environment.knowledge.chunker import chunk_all_documents
from environment.knowledge.faithfulness import assert_grounded, get_grounding_report
from environment.knowledge.gst_knowledge import get_all_documents
from environment.knowledge.query_processor import QueryProcessor
from environment.knowledge.vector_store import VectorStore

logger = structlog.get_logger()

# Global singleton — lazily initialized
_rag_engine: RAGEngine | None = None

# Default minimum relevance score — filters out noise
DEFAULT_MIN_SCORE = 0.08


class RAGEngine:
    """
    Production-grade RAG engine for GST domain knowledge.

    Usage:
        engine = RAGEngine()
        engine.initialize()

        # Simple retrieval
        results = engine.retrieve("What is Rule 36(4)?", top_k=3)

        # Context for LLM prompt (with token budget)
        context = engine.get_context_for_prompt("invoice matching", max_tokens=2000)

        # Category-filtered retrieval
        results = engine.retrieve("ITC eligibility", category_filter="itc_rules")

        # Faithfulness check on LLM response
        is_faithful = engine.check_faithfulness(llm_response, results)
    """

    def __init__(
        self,
        min_score: float = DEFAULT_MIN_SCORE,
        use_hybrid: bool = True,
        chunk_size: int = 200,
        chunk_overlap: int = 40,
    ):
        self.store = VectorStore()
        self.query_processor = QueryProcessor()
        self.min_score = min_score
        self.use_hybrid = use_hybrid
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._initialized = False

    def initialize(self) -> None:
        """Load, chunk, and index all GST knowledge documents."""
        if self._initialized:
            return

        raw_docs = get_all_documents()

        # Chunk documents (small ones pass through unchanged)
        chunked_docs = chunk_all_documents(
            raw_docs,
            chunk_size=self.chunk_size,
            overlap_sentences=self.chunk_overlap,
        )

        self.store.add_documents(chunked_docs)
        self._initialized = True

        logger.info(
            "rag_initialized",
            raw_documents=len(raw_docs),
            indexed_chunks=self.store.count,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        category_filter: str | None = None,
        min_score: float | None = None,
    ) -> list[dict]:
        """
        Retrieve relevant knowledge for a query.

        Pipeline:
        1. Expand query with GST synonyms
        2. Search (hybrid or TF-IDF, optionally filtered by category)
        3. Filter by minimum score threshold
        4. Return structured results with relevance scores

        Args:
            query: User's question or search text
            top_k: Maximum results to return
            category_filter: Optional category to restrict search
            min_score: Override default minimum score threshold

        Returns:
            List of {title, content, source, relevance, category} dicts
        """
        if not self._initialized:
            self.initialize()

        threshold = min_score if min_score is not None else self.min_score

        # Step 1: Query expansion
        expanded_query = self.query_processor.process(query)

        # Step 2: Search (over-fetch then filter)
        fetch_k = top_k * 2

        if category_filter:
            results = self.store.search_with_filter(
                expanded_query, category_filter, top_k=fetch_k, min_score=threshold,
            )
        elif self.use_hybrid:
            results = self.store.hybrid_search(
                expanded_query, top_k=fetch_k, min_score=0,
            )
        else:
            results = self.store.search(
                expanded_query, top_k=fetch_k, min_score=threshold,
            )

        # Step 3: Build result dicts
        output = []
        for doc, score in results:
            output.append({
                "title": doc.title,
                "content": doc.content,
                "source": doc.source,
                "category": doc.category,
                "relevance": round(score, 4),
                "doc_id": doc.doc_id,
            })

        return output[:top_k]

    def retrieve_multi(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Decompose compound queries and retrieve for each sub-query.

        "What about ITC rules and also reconciliation?" → retrieves for both,
        then deduplicates by PARENT document (not chunk ID) to ensure
        diverse source coverage instead of 3 chunks from one document.
        """
        if not self._initialized:
            self.initialize()

        sub_queries = self.query_processor.decompose(query)

        if len(sub_queries) <= 1:
            return self.retrieve(query, top_k=top_k)

        seen_parents: set[str] = set()
        combined: list[dict] = []

        for sub_query in sub_queries:
            results = self.retrieve(sub_query, top_k=top_k)
            for r in results:
                parent = r["doc_id"].split("_chunk_")[0]
                if parent not in seen_parents:
                    seen_parents.add(parent)
                    combined.append(r)

        # Sort by relevance, return top_k
        combined.sort(key=lambda x: x["relevance"], reverse=True)
        return combined[:top_k]

    def get_context_for_prompt(
        self,
        query: str,
        top_k: int = 3,
        max_tokens: int = 2000,
    ) -> str:
        """
        Get formatted context with token-budget enforcement.

        Returns markdown-formatted knowledge sections with:
        - Relevance-ordered results (best first)
        - Confidence tags (High/Medium/Low)
        - Hard token limit to prevent context overflow
        - Empty message when no relevant knowledge found
        """
        results = self.retrieve(query, top_k=top_k)

        if not results:
            return "No relevant GST knowledge found for this query."

        sections = ["## Relevant GST Knowledge\n"]
        token_count = 0

        for i, doc in enumerate(results, 1):
            # Rough token estimate: words * 1.3
            chunk_tokens = int(len(doc["content"].split()) * 1.3)
            if token_count + chunk_tokens > max_tokens:
                sections.append(f"\n*({len(results) - i + 1} additional results omitted — token budget)*")
                break

            # Confidence tag based on relevance score
            relevance = doc["relevance"]
            if relevance > 0.5:
                confidence = "🟢 High"
            elif relevance > 0.2:
                confidence = "🟡 Medium"
            else:
                confidence = "🔴 Low"

            sections.append(
                f"### [{i}] {doc['title']} ({confidence} confidence)\n"
                f"**Source**: {doc['source']} | "
                f"**Relevance**: {relevance:.0%}\n\n"
                f"{doc['content']}\n"
            )
            token_count += chunk_tokens

        return "\n".join(sections)

    # ── Convenience methods for specific contexts ────────────────────

    def get_itc_rules_context(self) -> str:
        """Get ITC-related context using category filter."""
        return self.get_context_for_prompt(
            "ITC input tax credit eligibility conditions",
            top_k=5,
        )

    def get_reconciliation_context(self) -> str:
        """Get reconciliation process context."""
        return self.get_context_for_prompt(
            "reconciliation GSTR-2B purchase register matching",
            top_k=5,
        )

    def get_mismatch_context(self) -> str:
        """Get context for handling mismatches."""
        return self.get_context_for_prompt(
            "mismatch tolerance variance follow up supplier",
            top_k=4,
        )

    # ── Faithfulness Checking ────────────────────────────────────────

    def check_faithfulness(self, response: str, context_chunks: list[dict]) -> bool:
        """
        Verify that an LLM response is grounded in the retrieved context.

        Returns True if all cited legal references exist in the context.
        """
        return assert_grounded(response, context_chunks)

    def get_faithfulness_report(self, response: str, context_chunks: list[dict]) -> dict:
        """Get detailed grounding analysis of an LLM response."""
        return get_grounding_report(response, context_chunks)

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
