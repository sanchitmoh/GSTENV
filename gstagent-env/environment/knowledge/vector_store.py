"""
Vector Store v2 — enhanced retrieval with score thresholds and hybrid search.

Upgrades from v1:
- Score threshold filtering (min_score) — stops garbage context
- Metadata category filtering — precision improvement
- BM25 keyword scoring — hybrid TF-IDF + BM25 via Reciprocal Rank Fusion
- Over-fetch-then-filter pattern for better result quality
- Semantic embedding support ready (swap _compute_embedding method)

The TF-IDF core is retained as a zero-dependency baseline. For production,
swap to sentence-transformers (bge-small-en-v1.5) by overriding embed().
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Document:
    """A document (or chunk) in the vector store."""
    doc_id: str
    title: str
    content: str
    source: str
    category: str
    parent_id: str = ""      # For chunks: reference to parent document
    chunk_index: int = -1    # For chunks: position in parent
    vector: list[float] = field(default_factory=list)


class VectorStore:
    """
    Enhanced TF-IDF vector store with score thresholds and hybrid search.

    Supports:
    - add_documents: batch index documents (or chunks)
    - search: find top-k with score threshold
    - search_with_filter: filter by category before scoring
    - hybrid_search: combine TF-IDF + BM25 via Reciprocal Rank Fusion
    """

    # BM25 parameters
    BM25_K1 = 1.5
    BM25_B = 0.75

    def __init__(self):
        self.documents: list[Document] = []
        self.vocabulary: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self._doc_tokens: list[list[str]] = []  # Cached for BM25
        self._avg_dl: float = 0.0               # Average document length
        self._built = False

    def add_documents(self, docs: list[dict]) -> None:
        """Add documents to the store and build index."""
        for doc in docs:
            self.documents.append(Document(
                doc_id=doc["id"],
                title=doc["title"],
                content=doc["content"],
                source=doc.get("source", ""),
                category=doc.get("category", ""),
                parent_id=doc.get("parent_id", ""),
                chunk_index=doc.get("chunk_index", -1),
            ))
        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize: lowercase, split on non-alphanumeric."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def _build_index(self) -> None:
        """Build TF-IDF vectors and BM25 statistics for all documents."""
        self._doc_tokens = []
        for doc in self.documents:
            tokens = self._tokenize(doc.title + " " + doc.content)
            self._doc_tokens.append(tokens)

        # Build vocabulary
        vocab_set: set[str] = set()
        for tokens in self._doc_tokens:
            vocab_set.update(tokens)
        self.vocabulary = {token: idx for idx, token in enumerate(sorted(vocab_set))}

        # Compute IDF (shared by TF-IDF and BM25)
        n_docs = len(self.documents)
        doc_freq: Counter = Counter()
        for tokens in self._doc_tokens:
            for token in set(tokens):
                doc_freq[token] += 1

        self.idf = {
            token: math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            for token, df in doc_freq.items()
        }

        # BM25 average document length
        total_tokens = sum(len(t) for t in self._doc_tokens)
        self._avg_dl = total_tokens / max(n_docs, 1)

        # Compute TF-IDF vectors
        for doc, tokens in zip(self.documents, self._doc_tokens):
            doc.vector = self._compute_tfidf(tokens)

        self._built = True

    def _compute_tfidf(self, tokens: list[str]) -> list[float]:
        """Compute TF-IDF vector for a token list."""
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        vector = [0.0] * len(self.vocabulary)

        for token, count in tf.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                vector[idx] = (count / total) * self.idf.get(token, 0)

        return vector

    def _cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a * a for a in v1))
        mag2 = math.sqrt(sum(b * b for b in v2))
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot / (mag1 * mag2)

    def _bm25_score(self, query_tokens: list[str], doc_idx: int) -> float:
        """Compute BM25 score for a document against a query."""
        doc_tokens = self._doc_tokens[doc_idx]
        tf = Counter(doc_tokens)
        dl = len(doc_tokens)

        score = 0.0
        for qt in set(query_tokens):
            if qt not in self.idf:
                continue
            term_freq = tf.get(qt, 0)
            numerator = term_freq * (self.BM25_K1 + 1)
            denominator = term_freq + self.BM25_K1 * (
                1 - self.BM25_B + self.BM25_B * dl / max(self._avg_dl, 1)
            )
            score += self.idf[qt] * (numerator / max(denominator, 0.001))

        return score

    # ── Search Methods ───────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[Document, float]]:
        """
        TF-IDF cosine search with score threshold filtering.

        Args:
            query: Search query text
            top_k: Max results to return
            min_score: Minimum similarity threshold (0.08 recommended)
        """
        if not self._built:
            return []

        query_tokens = self._tokenize(query)
        query_vector = self._compute_tfidf(query_tokens)

        scored = []
        for doc in self.documents:
            sim = self._cosine_similarity(query_vector, doc.vector)
            if sim >= min_score:
                scored.append((doc, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search_with_filter(
        self,
        query: str,
        category: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[Document, float]]:
        """Search within a specific category only."""
        if not self._built:
            return []

        query_tokens = self._tokenize(query)
        query_vector = self._compute_tfidf(query_tokens)

        scored = []
        for doc in self.documents:
            if doc.category != category:
                continue
            sim = self._cosine_similarity(query_vector, doc.vector)
            if sim >= min_score:
                scored.append((doc, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        rrf_k: int = 60,
    ) -> list[tuple[Document, float]]:
        """
        Hybrid search: TF-IDF + BM25 combined via Reciprocal Rank Fusion.

        RRF is the industry standard for combining multiple retrieval signals.
        Score = Σ 1/(k + rank_i) across all rankers.
        """
        if not self._built:
            return []

        query_tokens = self._tokenize(query)
        query_vector = self._compute_tfidf(query_tokens)
        fetch_k = top_k * 3  # Over-fetch for better fusion

        # TF-IDF ranking
        tfidf_scored = []
        for i, doc in enumerate(self.documents):
            sim = self._cosine_similarity(query_vector, doc.vector)
            if sim > 0:
                tfidf_scored.append((i, sim))
        tfidf_scored.sort(key=lambda x: x[1], reverse=True)
        tfidf_ranked = tfidf_scored[:fetch_k]

        # BM25 ranking
        bm25_scored = []
        for i in range(len(self.documents)):
            score = self._bm25_score(query_tokens, i)
            if score > 0:
                bm25_scored.append((i, score))
        bm25_scored.sort(key=lambda x: x[1], reverse=True)
        bm25_ranked = bm25_scored[:fetch_k]

        # Reciprocal Rank Fusion
        rrf_scores: dict[int, float] = {}
        for rank, (doc_idx, _) in enumerate(tfidf_ranked):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1.0 / (rrf_k + rank)
        for rank, (doc_idx, _) in enumerate(bm25_ranked):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1.0 / (rrf_k + rank)

        # Sort by fused score
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_idx, score in fused[:top_k]:
            if score >= min_score or min_score == 0:
                results.append((self.documents[doc_idx], round(score, 6)))

        return results

    # ── Lookup Methods ───────────────────────────────────────────────

    def get_by_id(self, doc_id: str) -> Document | None:
        """Get a specific document by ID."""
        for doc in self.documents:
            if doc.doc_id == doc_id:
                return doc
        return None

    def get_by_category(self, category: str) -> list[Document]:
        """Get all documents in a category."""
        return [doc for doc in self.documents if doc.category == category]

    @property
    def count(self) -> int:
        return len(self.documents)
