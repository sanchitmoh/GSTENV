"""
Query Router — automatic retrieval strategy selection.

Classifies queries by type and routes to the optimal retrieval method:
- Fact lookup → sentence_window (pinpoint precision)
- Process/workflow → hierarchical (full document context)
- Multi-topic → multi-query decomposition + RRF fusion
- Compliance rule → category-filtered search
- General → standard reranked search

Also implements:
- RAG-Fusion: generate query variants and fuse results via RRF
- Semantic Cache: LRU cache with normalized fingerprinting
- HyDE: Hypothetical Document Embedding for question→document matching
"""

from __future__ import annotations

import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field


# ── Query Classification ─────────────────────────────────────────────

@dataclass
class QueryRoute:
    """Result of query classification."""
    strategy: str          # "sentence_window" | "hierarchical" | "multi" | "filtered" | "standard"
    category_filter: str | None = None
    confidence: float = 0.0
    reason: str = ""


class QueryRouter:
    """
    Classify queries and route to optimal retrieval strategy.

    Uses pattern matching on query structure and keywords to determine
    which retrieval method will produce the best results.
    """

    # Patterns for fact-lookup queries (→ sentence_window)
    FACT_PATTERNS: list[str] = [
        r"what is the (?:rate|percentage|amount|limit|threshold)",
        r"how (?:much|many|long)",
        r"when is .+ (?:due|required|mandatory)",
        r"what (?:percentage|rate|amount)",
        r"\d+%",
        r"(?:₹|rs|inr)\s*[\d,]+",
        r"(?:how many|what number of) days",
    ]

    # Patterns for process/workflow queries (→ hierarchical)
    PROCESS_PATTERNS: list[str] = [
        r"how to",
        r"step[- ]by[- ]step",
        r"process (?:of|for)",
        r"procedure for",
        r"what are the steps",
        r"explain the process",
        r"workflow",
    ]

    # Patterns for multi-topic queries (→ multi-query decomposition)
    MULTI_PATTERNS: list[str] = [
        r"\band\b.*\balso\b",
        r"\band\b.*\bwhat\b",
        r"\bplus\b",
        r"\badditionally\b",
        r"\bas well as\b",
    ]

    # Category keywords for filtered retrieval
    CATEGORY_KEYWORDS: dict[str, list[str]] = {
        "itc_rules": ["itc", "input tax credit", "credit", "section 16", "rule 36"],
        "reconciliation": ["reconciliation", "matching", "gstr-2b", "mismatch", "variance"],
        "compliance": ["penalty", "interest", "late", "demand", "notice", "section 73"],
        "registration": ["registration", "threshold", "gstin", "cancel"],
        "returns": ["gstr-1", "gstr-3b", "return", "filing", "due date", "qrmp"],
        "e_invoice": ["e-invoice", "einvoice", "irn", "irp"],
        "exports": ["export", "zero-rated", "lut", "refund"],
        "eway_bill": ["e-way bill", "eway", "transport", "ewb"],
    }

    def classify(self, query: str) -> QueryRoute:
        """Classify a query and return the optimal retrieval route."""
        query_lower = query.lower().strip()

        # Check for multi-topic queries first (highest priority split)
        for pattern in self.MULTI_PATTERNS:
            if re.search(pattern, query_lower):
                return QueryRoute(
                    strategy="multi",
                    confidence=0.8,
                    reason=f"Multi-topic pattern: {pattern}",
                )

        # Check for fact-lookup queries (→ sentence window for precision)
        for pattern in self.FACT_PATTERNS:
            if re.search(pattern, query_lower):
                cat = self._detect_category(query_lower)
                return QueryRoute(
                    strategy="sentence_window",
                    category_filter=cat,
                    confidence=0.85,
                    reason=f"Fact lookup pattern: {pattern}",
                )

        # Check for process/workflow queries (→ hierarchical for context)
        for pattern in self.PROCESS_PATTERNS:
            if re.search(pattern, query_lower):
                return QueryRoute(
                    strategy="hierarchical",
                    confidence=0.8,
                    reason=f"Process pattern: {pattern}",
                )

        # Check for category-specific queries
        cat = self._detect_category(query_lower)
        if cat:
            return QueryRoute(
                strategy="filtered",
                category_filter=cat,
                confidence=0.7,
                reason=f"Category detected: {cat}",
            )

        # Default to standard reranked search
        return QueryRoute(
            strategy="standard",
            confidence=0.5,
            reason="No specific pattern matched — using standard search",
        )

    def _detect_category(self, query_lower: str) -> str | None:
        """Detect the GST knowledge category from query content."""
        best_cat = None
        best_count = 0

        for cat, keywords in self.CATEGORY_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in query_lower)
            if count > best_count:
                best_count = count
                best_cat = cat

        return best_cat if best_count > 0 else None


# ── Semantic Cache ───────────────────────────────────────────────────

# Stop words for fingerprint normalization
_CACHE_STOPS = {
    "what", "how", "when", "where", "why", "who", "which",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "the", "a", "an", "or", "but", "in", "on", "at",
    "to", "for", "of", "with", "by", "from", "as", "into",
    "this", "that", "these", "those", "it", "its",
    "can", "will", "would", "should", "could", "may", "might",
    "my", "your", "our", "their", "if", "i", "me", "we",
    "about", "just", "very", "so", "than",
}


class SemanticCache:
    """
    LRU cache for retrieval results, keyed by normalized query fingerprint.

    Avoids redundant retrieval for queries that differ only in stopwords
    or word order. During a reconciliation run, the agent often asks
    variants of the same ITC question — this eliminates ~60-80% of
    redundant retrieval calls.
    """

    def __init__(self, max_size: int = 64):
        self._cache: OrderedDict[str, list[dict]] = OrderedDict()
        self.max_size = max_size
        self.hits: int = 0
        self.misses: int = 0

    def _fingerprint(self, query: str, strategy: str = "standard") -> str:
        """Normalize query to a cache key: lowercase, remove stops, sort tokens.

        v4 fix: include retrieval strategy in key to prevent cross-strategy
        cache collisions (e.g., RAG-Fusion result returned for sentence-window request).
        """
        tokens = re.findall(r"[a-z0-9]+", query.lower())
        # Keep negation tokens (critical for meaning), remove other stops
        meaningful = [t for t in tokens if t not in _CACHE_STOPS or t in {"not", "no", "without", "never"}]
        return strategy + ":" + " ".join(sorted(set(meaningful)))

    def get(self, query: str, strategy: str = "standard") -> list[dict] | None:
        """Look up cached results. Returns None on miss."""
        key = self._fingerprint(query, strategy)
        if key in self._cache:
            self.hits += 1
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, query: str, results: list[dict], strategy: str = "standard") -> None:
        """Store results in cache."""
        key = self._fingerprint(query, strategy)
        self._cache[key] = results
        self._cache.move_to_end(key)
        # Evict oldest if over capacity
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_stats(self) -> dict:
        """Return cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 3),
        }


# ── RAG-Fusion — Multi-Query Variant Generation ─────────────────────

class RAGFusion:
    """
    Generate multiple reformulations of a query, retrieve for each,
    then fuse results via Reciprocal Rank Fusion.

    This addresses vocabulary mismatch beyond simple synonym expansion —
    different phrasings of the same question retrieve different document sets.
    Fusing these gives better recall than any single reformulation.
    """

    # Question-type reformulation templates for semantic diversity
    # (v4 fix: old templates produced near-identical keyword soup)
    VARIANT_TEMPLATES: list[str] = [
        "{topic} GST rules provisions",
        "rule governing {topic}",
        "{topic} eligibility conditions requirements",
        "CBIC clarification {topic}",
        "{topic} compliance procedure process",
    ]

    def generate_variants(self, query: str) -> list[str]:
        """
        Generate 3-4 reformulations of the query for multi-retrieval.

        Uses template-based rewriting — extracts the core topic from the
        query and inserts it into domain-specific templates.
        """
        # Extract core topic (remove question words and stops)
        tokens = re.findall(r"[a-z0-9]+", query.lower())
        topic_tokens = [t for t in tokens if t not in _CACHE_STOPS and len(t) > 2]

        if not topic_tokens:
            return [query]

        topic = " ".join(topic_tokens[:6])  # First 6 meaningful words

        variants = [query]  # Always include original
        for template in self.VARIANT_TEMPLATES:
            variant = template.format(topic=topic)
            if variant != query:
                variants.append(variant)

        return variants[:5]  # Cap at 5 variants

    def fuse_results(
        self,
        all_results: list[list[tuple]],
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> list[tuple]:
        """
        Fuse results from multiple retrievals using Reciprocal Rank Fusion.

        Each result set votes for documents based on their rank position.
        RRF = Σ 1/(k + rank_i) across all result sets.
        """
        rrf_scores: dict[str, tuple[object, float]] = {}

        for result_set in all_results:
            for rank, (doc, score) in enumerate(result_set):
                doc_key = doc.doc_id if hasattr(doc, "doc_id") else str(doc)
                rrf_contribution = 1.0 / (rrf_k + rank)
                if doc_key in rrf_scores:
                    existing_doc, existing_score = rrf_scores[doc_key]
                    rrf_scores[doc_key] = (existing_doc, existing_score + rrf_contribution)
                else:
                    rrf_scores[doc_key] = (doc, rrf_contribution)

        # Normalize scores
        max_score = max((s for _, s in rrf_scores.values()), default=1.0)
        if max_score > 0:
            normalized = [
                (doc, round(score / max_score, 6))
                for doc, score in rrf_scores.values()
            ]
        else:
            normalized = [(doc, score) for doc, score in rrf_scores.values()]

        normalized.sort(key=lambda x: x[1], reverse=True)
        return normalized[:top_k]


# ── HyDE — Hypothetical Document Embeddings ─────────────────────────

class HyDE:
    """
    Hypothetical Document Embeddings — generate a hypothetical answer
    to the query, then search for documents similar to that answer.

    This bridges the question→document gap: questions and answers have
    different vocabulary distributions. By generating a hypothetical
    answer first, the search becomes answer→document (same distribution).

    This is a lightweight template-based implementation — no LLM needed.
    For production, the hypothetical answer would be generated by an LLM.
    """

    # Domain-specific answer templates for common query types
    ANSWER_TEMPLATES: dict[str, str] = {
        "itc": (
            "Input Tax Credit (ITC) under GST is governed by Section 16 and Rule 36(4). "
            "ITC can only be claimed for invoices reflected in GSTR-2B. The conditions "
            "include possession of tax invoice, receipt of goods/services, tax paid to "
            "government, and filing of return. {query_terms}"
        ),
        "reconciliation": (
            "GST reconciliation involves matching the purchase register with GSTR-2B data. "
            "Download GSTR-2B, compare invoice numbers, GSTINs, and amounts. Mismatches "
            "within 5% are acceptable, 5-20% require partial claim, over 20% are ineligible. "
            "{query_terms}"
        ),
        "penalty": (
            "GST penalties and interest are governed by Sections 73, 74, and 50. Interest "
            "on ITC wrongly availed is 24% if utilized and 18% if availed but not utilized. "
            "Late filing attracts penalty under Section 122. {query_terms}"
        ),
        "registration": (
            "GST registration is mandatory for businesses with aggregate turnover exceeding "
            "₹40 lakh for goods (₹20 lakh for services and special category states). "
            "Registration requires PAN, Aadhaar, business proof. {query_terms}"
        ),
        "default": (
            "Under GST law, the provisions apply as per the CGST Act 2017, IGST Act, and "
            "related rules and CBIC circulars. {query_terms}"
        ),
    }

    def generate_hypothetical_doc(self, query: str) -> str:
        """
        Generate a hypothetical document that would answer the query.

        Uses template-based generation with query term injection.
        """
        query_lower = query.lower()
        query_terms = " ".join(re.findall(r"[a-z0-9]+", query_lower))

        # Select best template based on query content
        if any(kw in query_lower for kw in ["itc", "input tax credit", "credit", "section 16", "rule 36"]):
            template_key = "itc"
        elif any(kw in query_lower for kw in ["reconciliation", "matching", "mismatch", "gstr-2b"]):
            template_key = "reconciliation"
        elif any(kw in query_lower for kw in ["penalty", "interest", "late", "section 50", "section 73"]):
            template_key = "penalty"
        elif any(kw in query_lower for kw in ["registration", "threshold", "gstin"]):
            template_key = "registration"
        else:
            template_key = "default"

        return self.ANSWER_TEMPLATES[template_key].format(query_terms=query_terms)


# ── GraphRAG — Citation Graph Traversal ──────────────────────────────

class KnowledgeGraph:
    """
    Lightweight knowledge graph for GST regulations.

    Models citation relationships between GST provisions:
    - Section 16(2) → Rule 36(4) → Circular 170
    - Rule 37 → Section 16(2)

    On retrieval, if a document is retrieved, its graph neighbors
    are also pulled (1-hop expansion) for richer context.
    """

    def __init__(self):
        self._edges: dict[str, set[str]] = defaultdict(set)
        self._built = False

    def add_edge(self, from_id: str, to_id: str) -> None:
        """Add a bidirectional citation link."""
        self._edges[from_id].add(to_id)
        self._edges[to_id].add(from_id)

    def get_neighbors(self, doc_id: str, max_hops: int = 1) -> set[str]:
        """Get all documents reachable within max_hops."""
        visited: set[str] = set()
        frontier = {doc_id}

        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for node in frontier:
                if node not in visited:
                    visited.add(node)
                    next_frontier.update(self._edges.get(node, set()))
            frontier = next_frontier - visited

        visited.update(frontier)
        visited.discard(doc_id)  # Don't include the query doc itself
        return visited

    def build_from_documents(self, documents: list[dict]) -> None:
        """
        Automatically build citation graph from document content.

        Scans each document for references to other documents'
        IDs, titles, or rule/section numbers.
        """
        # Build a lookup of doc IDs and their key identifiers
        doc_keys: dict[str, list[str]] = {}
        for doc in documents:
            doc_id = doc.get("id", "")
            keys = [doc_id.lower()]
            title = doc.get("title", "").lower()
            # Extract key identifiers from title
            for match in re.findall(r"(?:section|rule|circular)\s+[\d()/.]+", title):
                keys.append(match.lower())
            doc_keys[doc_id] = keys

        # Scan each document for references to others
        for doc in documents:
            doc_id = doc.get("id", "")
            content = doc.get("content", "").lower()

            for other_id, keys in doc_keys.items():
                if other_id == doc_id:
                    continue
                for key in keys:
                    if key in content:
                        self.add_edge(doc_id, other_id)
                        break

        self._built = True

    @property
    def edge_count(self) -> int:
        return sum(len(v) for v in self._edges.values()) // 2

    @property
    def node_count(self) -> int:
        return len(self._edges)


# ── Self-RAG — Retrieval-Aware Generation Control ────────────────────

class SelfRAGController:
    """
    Lightweight Self-RAG controller that decides:
    1. Whether retrieval is needed (some queries are self-contained)
    2. Whether retrieved context is relevant (filters noise)
    3. How to present context to the LLM (with relevance signals)

    This avoids the "always retrieve" anti-pattern that dilutes context
    with irrelevant documents for simple or well-known queries.
    """

    # Queries that don't need retrieval (self-contained)
    NO_RETRIEVAL_PATTERNS: list[str] = [
        r"^hello",
        r"^hi\b",
        r"^thank",
        r"^ok\b",
        r"^yes\b",
        r"^no\b",
    ]

    # Minimum relevance score to include in context
    RELEVANCE_THRESHOLD = 0.15

    # Domain terms that indicate a substantive query regardless of length
    DOMAIN_TERMS: set[str] = {
        "itc", "gst", "gstr", "gstin", "cgst", "igst", "sgst", "rcm",
        "invoice", "credit", "refund", "penalty", "interest", "eway",
        "reconciliation", "compliance", "registration", "section", "rule",
        "circular", "notification", "eligibility", "reversal", "input",
    }

    def needs_retrieval(self, query: str) -> bool:
        """Decide whether this query needs retrieval."""
        query_lower = query.lower().strip()

        # Skip retrieval for greetings and acknowledgments
        for pattern in self.NO_RETRIEVAL_PATTERNS:
            if re.search(pattern, query_lower):
                return False

        # Very short queries (< 3 words) unless they contain domain terms
        words = query_lower.split()
        if len(words) < 3:
            # Check if any word is a domain-specific term
            if not any(w in self.DOMAIN_TERMS for w in words):
                return False

        return True

    def filter_relevant(
        self, results: list[dict], min_relevance: float | None = None,
    ) -> list[dict]:
        """
        Filter retrieval results to only include relevant documents.

        Removes noise documents that scored below the relevance threshold.
        """
        threshold = min_relevance or self.RELEVANCE_THRESHOLD

        filtered = [r for r in results if r.get("relevance", 0) >= threshold]

        # Always return at least 1 result if any were found
        if not filtered and results:
            return [max(results, key=lambda r: r.get("relevance", 0))]

        return filtered

    def should_cite(self, result: dict) -> str:
        """
        Return citation confidence level for a result.

        Used to add authority signals to the context.
        Thresholds calibrated for TF-IDF cosine similarity (0.05–0.5 range).
        """
        score = result.get("relevance", 0)
        if score >= 0.25:
            return "AUTHORITATIVE"
        elif score >= 0.08:
            return "SUPPORTING"
        else:
            return "LOW_CONFIDENCE"
