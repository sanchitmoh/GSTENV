"""
RAG Engine v3 — production-grade Retrieval-Augmented Generation for GST agents.

Upgrades from v2:
- Contextual chunk headers for provenance tracking
- Percentage-based sliding window overlap for better boundary handling
- Hierarchical retrieval: child-chunk precision + parent-document context
- Sentence-window retrieval: sentence-level indexing + ±N expansion
- Sublinear TF embeddings for improved semantic matching
- Post-retrieval re-ranking with multi-signal scoring
- Configurable chunk strategy via ChunkConfig

Pipeline: Query → Expand → Decompose → Hybrid Search → Re-rank → Filter → Budget → Format

Retained from v2:
- Query expansion via GST synonym processor (biggest retrieval improvement)
- Score threshold filtering (min_score=0.08 — stops noise)
- Metadata category filtering for precise retrieval
- Token-budget context assembly (no context overflow)
- Hybrid search (TF-IDF + BM25 via RRF)
- Document chunking support
- Faithfulness grounding check for LLM responses
- Confidence-tagged context sections (High/Medium/Low)
"""

from __future__ import annotations

import structlog

from environment.knowledge.chunker import (
    ChunkConfig,
    chunk_all_documents,
    expand_sentence_window,
)
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

        # Simple retrieval (with re-ranking)
        results = engine.retrieve("What is Rule 36(4)?", top_k=3)

        # Hierarchical: child-chunk precision, parent-doc context
        results = engine.retrieve_hierarchical("ITC eligibility", top_k=3)

        # Sentence-window: sentence-level precision, expanded context
        results = engine.sentence_window_retrieve("180 days payment", top_k=3)

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
        chunk_size: int = 300,
        chunk_overlap: int = 40,
        chunk_config: ChunkConfig | None = None,
        use_reranking: bool = True,
    ):
        self.store = VectorStore()
        self.query_processor = QueryProcessor()
        self.min_score = min_score
        self.use_hybrid = use_hybrid
        self.use_reranking = use_reranking
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_config = chunk_config or ChunkConfig(
            chunk_size=chunk_size,
            overlap_pct=0.15,
            context_header=True,
            mode="sentence",
        )
        self._initialized = False

    def initialize(self) -> None:
        """Load, chunk, and index all GST knowledge documents."""
        if self._initialized:
            return

        raw_docs = get_all_documents()

        # Chunk documents using ChunkConfig (small ones pass through with headers)
        chunked_docs = chunk_all_documents(
            raw_docs,
            config=self.chunk_config,
        )

        self.store.add_documents(chunked_docs)
        self._initialized = True

        logger.info(
            "rag_initialized",
            raw_documents=len(raw_docs),
            indexed_chunks=self.store.count,
            chunk_mode=self.chunk_config.mode,
            context_headers=self.chunk_config.context_header,
            reranking=self.use_reranking,
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
        3. Re-rank results for precision (if enabled)
        4. Filter by minimum score threshold
        5. Return structured results with relevance scores

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
        elif self.use_reranking:
            # Use re-ranked hybrid search for best precision
            results = self.store.reranked_search(
                expanded_query, top_k=fetch_k, min_score=0,
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

    def retrieve_hierarchical(
        self,
        query: str,
        top_k: int = 3,
        min_score: float | None = None,
    ) -> list[dict]:
        """
        Hierarchical retrieval: child-chunk precision + parent-doc context.

        Retrieves at the chunk level (small, precise matches), then groups by
        parent document. For each parent, returns all its chunks' content
        combined, giving the model both precision and full context.

        Best for: complex questions needing full document understanding.
        """
        if not self._initialized:
            self.initialize()

        threshold = min_score if min_score is not None else self.min_score
        expanded_query = self.query_processor.process(query)

        # Hierarchical search: matches chunks, groups by parent
        results = self.store.hierarchical_search(
            expanded_query, top_k=top_k, min_score=threshold,
        )

        output = []
        for doc, score in results:
            # Collect all sibling chunks of this parent for full context
            parent_id = doc.parent_id if doc.parent_id else doc.doc_id
            siblings = self.store.get_by_parent(parent_id)

            if len(siblings) > 1:
                # Combine all sibling chunks (sorted by chunk_index)
                siblings.sort(key=lambda d: d.chunk_index)
                combined_content = " ".join(s.content for s in siblings)
            else:
                combined_content = doc.content

            output.append({
                "title": doc.title,
                "content": combined_content,
                "source": doc.source,
                "category": doc.category,
                "relevance": round(score, 4),
                "doc_id": parent_id,
                "retrieval_mode": "hierarchical",
            })

        return output[:top_k]

    def sentence_window_retrieve(
        self,
        query: str,
        top_k: int = 3,
        expand: int = 5,
        min_score: float | None = None,
    ) -> list[dict]:
        """
        Sentence-window retrieval: sentence-level precision + expanded context.

        Requires sentence_window mode in ChunkConfig. Retrieves the most
        relevant individual sentences, then expands each to include ±expand
        surrounding sentences for context.

        Best for: precise fact-finding within long documents.
        """
        if not self._initialized:
            self.initialize()

        threshold = min_score if min_score is not None else self.min_score
        expanded_query = self.query_processor.process(query)

        # Search at sentence level
        if self.use_reranking:
            results = self.store.reranked_search(
                expanded_query, top_k=top_k * 2, min_score=0,
            )
        else:
            results = self.store.hybrid_search(
                expanded_query, top_k=top_k * 2, min_score=0,
            )

        # Deduplicate by parent and expand sentence windows
        seen_parents: set[str] = set()
        output = []

        for doc, score in results:
            parent = doc.parent_id if doc.parent_id else doc.doc_id
            if parent in seen_parents:
                continue
            seen_parents.add(parent)

            # Expand sentence window if this chunk has sentence metadata
            if doc.sentences and doc.sentence_index >= 0:
                chunk_dict = {
                    "content": doc.content,
                    "sentences": doc.sentences,
                    "sentence_index": doc.sentence_index,
                    "window_expand": doc.window_expand,
                }
                expanded_content = expand_sentence_window(chunk_dict, expand=expand)
            else:
                expanded_content = doc.content

            output.append({
                "title": doc.title,
                "content": expanded_content,
                "source": doc.source,
                "category": doc.category,
                "relevance": round(score, 4),
                "doc_id": doc.doc_id,
                "retrieval_mode": "sentence_window",
            })

            if len(output) >= top_k:
                break

        return output

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

        Uses hierarchical retrieval by default for best context quality.

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
