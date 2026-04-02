"""
Vector Store — lightweight embedding-based retrieval for GST knowledge.

Uses TF-IDF as a fast, zero-dependency vector store. Can be swapped for
ChromaDB/Pinecone in production (Phase 3 roadmap).

Why TF-IDF instead of external embeddings:
- Zero API calls needed (works offline, no cost)
- Sub-millisecond retrieval
- Good enough for structured domain knowledge
- Upgradeable: swap the embed() and search() methods for real embeddings
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Document:
    """A document in the vector store."""
    doc_id: str
    title: str
    content: str
    source: str
    category: str
    vector: list[float] = field(default_factory=list)


class VectorStore:
    """
    TF-IDF based vector store for GST knowledge retrieval.

    Supports:
    - add_documents: batch index documents
    - search: find top-k most relevant documents for a query
    - get_by_id: retrieve specific document
    """

    def __init__(self):
        self.documents: list[Document] = []
        self.vocabulary: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self._built = False

    def add_documents(self, docs: list[dict]) -> None:
        """Add documents to the store."""
        for doc in docs:
            self.documents.append(Document(
                doc_id=doc["id"],
                title=doc["title"],
                content=doc["content"],
                source=doc.get("source", ""),
                category=doc.get("category", ""),
            ))
        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def _build_index(self) -> None:
        """Build TF-IDF vectors for all documents."""
        # Build vocabulary
        all_tokens: list[list[str]] = []
        for doc in self.documents:
            tokens = self._tokenize(doc.title + " " + doc.content)
            all_tokens.append(tokens)

        # Build vocabulary from all unique tokens
        vocab_set: set[str] = set()
        for tokens in all_tokens:
            vocab_set.update(tokens)
        self.vocabulary = {token: idx for idx, token in enumerate(sorted(vocab_set))}

        # Compute IDF
        n_docs = len(self.documents)
        doc_freq: Counter = Counter()
        for tokens in all_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] += 1

        self.idf = {
            token: math.log(n_docs / (1 + df))
            for token, df in doc_freq.items()
        }

        # Compute TF-IDF vectors
        for doc, tokens in zip(self.documents, all_tokens):
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

    def search(self, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
        """
        Search for documents most relevant to the query.

        Returns list of (Document, similarity_score) tuples.
        """
        if not self._built:
            return []

        query_tokens = self._tokenize(query)
        query_vector = self._compute_tfidf(query_tokens)

        scored = []
        for doc in self.documents:
            sim = self._cosine_similarity(query_vector, doc.vector)
            if sim > 0:
                scored.append((doc, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

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
