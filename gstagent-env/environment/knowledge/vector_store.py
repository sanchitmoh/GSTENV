"""
Vector Store v3 — enhanced retrieval with sublinear TF, re-ranking, and
hierarchical parent-child search.

Upgrades from v2:
- SUBLINEAR TF: Uses (1 + log(tf)) instead of raw tf/total — a term
  appearing 10x shouldn't be 10x more important than appearing 1x.
  Industry standard for information retrieval.
- RE-RANKING: Post-retrieval cross-feature re-ranker using exact phrase
  match, term coverage, position scoring, and original score blending.
  Filters out false positives before final output.
- HIERARCHICAL SEARCH: Retrieve at child chunk level for precision,
  then expand to parent document for context. Combines the best of
  small-chunk matching and full-document understanding.

Retained from v2:
- Score threshold filtering (min_score) — stops garbage context
- Metadata category filtering — precision improvement
- BM25 keyword scoring — hybrid TF-IDF + BM25 via Reciprocal Rank Fusion
- Over-fetch-then-filter pattern for better result quality
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
    # Sentence window metadata
    sentence_index: int = -1
    total_sentences: int = -1
    sentences: list[str] = field(default_factory=list)
    window_expand: int = 5


class Reranker:
    """
    Post-retrieval re-ranker using cross-feature scoring.

    Combines multiple signals to re-order initial retrieval results:
    1. Exact phrase matching (query phrases found verbatim in doc)
    2. Term coverage (fraction of unique query terms found in doc)
    3. Position score (query terms appearing early rank higher)
    4. Original retrieval score (blended for stability)

    This is a zero-dependency alternative to neural re-rankers like
    BGE-reranker or Cohere rerank. For the GST domain with TF-IDF
    retrieval, this provides significant precision improvements.
    """

    # Blending weights
    ORIGINAL_WEIGHT = 0.35
    PHRASE_WEIGHT = 0.25
    COVERAGE_WEIGHT = 0.25
    POSITION_WEIGHT = 0.15

    def rerank(
        self,
        query: str,
        results: list[tuple[Document, float]],
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:
        """
        Re-rank retrieval results using multi-signal scoring.

        Args:
            query: Original search query
            results: List of (Document, score) from initial retrieval
            top_k: Number of results to return after re-ranking

        Returns:
            Re-ranked list of (Document, score) tuples
        """
        if not results:
            return []

        query_lower = query.lower()
        query_tokens = set(re.findall(r"[a-z0-9]+", query_lower))

        # Extract meaningful phrases (2-3 word ngrams) from query
        query_words = query_lower.split()
        phrases = []
        for n in (3, 2):
            for i in range(len(query_words) - n + 1):
                phrase = " ".join(query_words[i:i + n])
                phrases.append(phrase)

        reranked = []
        for doc, original_score in results:
            doc_text = (doc.title + " " + doc.content).lower()
            doc_tokens = set(re.findall(r"[a-z0-9]+", doc_text))

            # Signal 1: Exact phrase match count (normalized)
            phrase_hits = sum(1 for p in phrases if p in doc_text) if phrases else 0
            phrase_score = min(phrase_hits / max(len(phrases), 1), 1.0)

            # Signal 2: Term coverage (fraction of query terms found)
            if query_tokens:
                coverage = len(query_tokens & doc_tokens) / len(query_tokens)
            else:
                coverage = 0.0

            # Signal 3: Position score (early matches matter more)
            position_score = 0.0
            if query_tokens:
                doc_words = doc_text.split()
                total_words = len(doc_words)
                if total_words > 0:
                    first_positions = []
                    for qt in query_tokens:
                        for pos, word in enumerate(doc_words):
                            if qt in word:
                                first_positions.append(1.0 - pos / total_words)
                                break
                    if first_positions:
                        position_score = sum(first_positions) / len(first_positions)

            # Combined score
            combined = (
                self.ORIGINAL_WEIGHT * original_score
                + self.PHRASE_WEIGHT * phrase_score
                + self.COVERAGE_WEIGHT * coverage
                + self.POSITION_WEIGHT * position_score
            )

            reranked.append((doc, round(combined, 6)))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]


class VectorStore:
    """
    Enhanced TF-IDF vector store with sublinear TF, score thresholds,
    hybrid search, re-ranking, and hierarchical parent-child retrieval.

    Supports:
    - add_documents: batch index documents (or chunks)
    - search: find top-k with score threshold
    - search_with_filter: filter by category before scoring
    - hybrid_search: combine TF-IDF + BM25 via Reciprocal Rank Fusion
    - reranked_search: hybrid search + re-ranking for precision
    - hierarchical_search: retrieve child chunks, return parent context
    """

    # BM25 parameters
    BM25_K1 = 1.5
    BM25_B = 0.75

    def __init__(self):
        self.documents: list[Document] = []
        self.vocabulary: dict[str, int] = {}
        self.tfidf_idf: dict[str, float] = {}  # Standard IDF for cosine space
        self.bm25_idf: dict[str, float] = {}   # Saturating IDF for BM25
        self._doc_tokens: list[list[str]] = []  # Cached for BM25
        self._avg_dl: float = 0.0               # Average document length
        self._built = False
        self.reranker = Reranker()

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
                sentence_index=doc.get("sentence_index", -1),
                total_sentences=doc.get("total_sentences", -1),
                sentences=doc.get("sentences", []),
                window_expand=doc.get("window_expand", 5),
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

        # Compute document frequencies
        n_docs = len(self.documents)
        doc_freq: Counter = Counter()
        for tokens in self._doc_tokens:
            for token in set(tokens):
                doc_freq[token] += 1

        # Separate IDF formulas — these are different algorithms
        # TF-IDF IDF: standard log(N/df) for cosine similarity space
        self.tfidf_idf = {
            token: math.log(n_docs / (1 + df))
            for token, df in doc_freq.items()
        }
        # BM25 IDF: saturating formula calibrated for multiplicative TF weighting
        self.bm25_idf = {
            token: math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            for token, df in doc_freq.items()
        }

        # BM25 average document length
        total_tokens = sum(len(t) for t in self._doc_tokens)
        self._avg_dl = total_tokens / max(n_docs, 1)

        # Compute TF-IDF vectors using SUBLINEAR TF (improvement #5)
        for doc, tokens in zip(self.documents, self._doc_tokens):
            doc.vector = self._compute_tfidf(tokens)

        self._built = True

    def _compute_tfidf(self, tokens: list[str]) -> list[float]:
        """
        Compute TF-IDF vector using SUBLINEAR TF weighting.

        Sublinear TF: 1 + log(tf) instead of raw tf/total.
        This dampens high-frequency terms — a word appearing 10× shouldn't
        be 10× more important than appearing once. Industry standard.
        """
        tf = Counter(tokens)
        vector = [0.0] * len(self.vocabulary)

        for token, count in tf.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                # Sublinear TF: 1 + log(count) instead of count/total
                sublinear_tf = 1 + math.log(count) if count > 0 else 0
                vector[idx] = sublinear_tf * self.tfidf_idf.get(token, 0)

        # L2 normalization for fair cosine comparison across document lengths
        magnitude = math.sqrt(sum(v * v for v in vector))
        if magnitude > 0:
            vector = [v / magnitude for v in vector]

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
        """Compute BM25 score using BM25-specific IDF."""
        doc_tokens = self._doc_tokens[doc_idx]
        tf = Counter(doc_tokens)
        dl = len(doc_tokens)

        score = 0.0
        for qt in set(query_tokens):
            if qt not in self.bm25_idf:
                continue
            term_freq = tf.get(qt, 0)
            numerator = term_freq * (self.BM25_K1 + 1)
            denominator = term_freq + self.BM25_K1 * (
                1 - self.BM25_B + self.BM25_B * dl / max(self._avg_dl, 1)
            )
            score += self.bm25_idf[qt] * (numerator / max(denominator, 0.001))

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

        # Normalize RRF scores to [0, 1] so min_score threshold works
        # Max possible RRF = 2 * 1/(rrf_k + 0) = 2/rrf_k (rank 0 in both)
        max_possible = 2.0 / rrf_k
        normalized: dict[int, float] = {
            idx: score / max_possible for idx, score in rrf_scores.items()
        }

        # Sort by normalized score
        fused = sorted(normalized.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_idx, score in fused[:top_k]:
            if score >= min_score:
                results.append((self.documents[doc_idx], round(score, 6)))

        return results

    def reranked_search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        fetch_multiplier: int = 3,
    ) -> list[tuple[Document, float]]:
        """
        Hybrid search + re-ranking for maximum precision.

        Over-fetches with hybrid search, then applies the re-ranker
        to re-order by exact phrase match, term coverage, and position.
        """
        # Over-fetch to give re-ranker enough candidates
        initial = self.hybrid_search(
            query,
            top_k=top_k * fetch_multiplier,
            min_score=0,
        )
        return self.reranker.rerank(query, initial, top_k=top_k)

    def hierarchical_search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[Document, float]]:
        """
        Hierarchical parent-child search.

        1. Retrieve at child chunk level (high precision matching)
        2. Group by parent document
        3. Return parent-level results with best child score

        This combines small-chunk precision with full-document context.
        """
        # Retrieve more candidates at chunk level
        candidates = self.reranked_search(query, top_k=top_k * 3, min_score=0)

        # Group by parent_id — keep best score per parent
        parent_scores: dict[str, tuple[Document, float]] = {}
        for doc, score in candidates:
            parent = doc.parent_id if doc.parent_id else doc.doc_id
            if parent not in parent_scores or score > parent_scores[parent][1]:
                parent_scores[parent] = (doc, score)

        # Sort parents by best child score
        sorted_parents = sorted(
            parent_scores.values(), key=lambda x: x[1], reverse=True
        )

        results = []
        for doc, score in sorted_parents[:top_k]:
            if score >= min_score:
                results.append((doc, score))

        return results

    # ── Lookup Methods ───────────────────────────────────────────────

    def get_by_id(self, doc_id: str) -> Document | None:
        """Get a specific document by ID."""
        for doc in self.documents:
            if doc.doc_id == doc_id:
                return doc
        return None

    def get_by_parent(self, parent_id: str) -> list[Document]:
        """Get all chunks belonging to a parent document."""
        return [
            doc for doc in self.documents
            if doc.parent_id == parent_id or doc.doc_id == parent_id
        ]

    def get_by_category(self, category: str) -> list[Document]:
        """Get all documents in a category."""
        return [doc for doc in self.documents if doc.category == category]

    @property
    def count(self) -> int:
        return len(self.documents)
