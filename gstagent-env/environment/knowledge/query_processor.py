"""
Query Processor — pre-retrieval query rewriting for GST domain.

Handles the #1 retrieval killer: vocabulary mismatch between user queries
and indexed documents. A user asking "what happens if my supplier doesn't file?"
needs to match against "GSTR-1 late filing buyer ITC consequence."

Features:
- GST domain synonym expansion (50+ mappings)
- Compound query decomposition
- Stop word removal for cleaner TF-IDF/embedding matching
"""

from __future__ import annotations

import re


class QueryProcessor:
    """
    Rewrite and expand queries before they hit the vector store.

    Bridges the gap between natural language and GST jargon.
    """

    # Domain-specific synonym mappings
    GST_SYNONYMS: dict[str, list[str]] = {
        # Parties
        "supplier": ["vendor", "seller", "gstin holder"],
        "buyer": ["recipient", "purchaser", "registered person"],
        "vendor": ["supplier", "seller"],
        # Credits
        "itc": ["input tax credit", "credit", "gst credit"],
        "credit": ["itc", "input tax credit"],
        "claim": ["avail", "take", "utilize"],
        "avail": ["claim", "take"],
        # Discrepancy terms
        "mismatch": ["variance", "difference", "discrepancy", "gap"],
        "variance": ["mismatch", "difference", "discrepancy"],
        "missing": ["not appearing", "absent", "not reflected", "not found"],
        "excess": ["more", "over-claimed", "surplus"],
        # Actions
        "block": ["restrict", "disallow", "reverse", "deny"],
        "reverse": ["block", "return", "undo", "reversal"],
        "file": ["submit", "upload", "declare"],
        # Documents
        "gstr1": ["gstr-1", "outward supply", "supplier filing"],
        "gstr2b": ["gstr-2b", "auto-generated", "inward supply"],
        "gstr3b": ["gstr-3b", "monthly return", "summary return"],
        "invoice": ["bill", "tax invoice", "document"],
        # Amounts
        "tax": ["gst", "cgst", "sgst", "igst", "tax amount"],
        "amount": ["value", "sum", "total", "figure"],
        # Time
        "late": ["delayed", "overdue", "past due", "after deadline"],
        "monthly": ["every month", "per month", "each month"],
        # Rules & Compliance
        "rule": ["provision", "regulation", "requirement"],
        "section": ["clause", "provision", "act"],
        "penalty": ["fine", "interest", "charge", "consequence"],
        "audit": ["inspection", "review", "verification", "scrutiny"],
        # Business
        "msme": ["small business", "micro enterprise", "sme", "startup"],
        "cash flow": ["working capital", "liquidity", "funds"],
        "reconciliation": ["matching", "comparison", "verification", "recon"],
    }

    # Stop words to remove for cleaner matching
    STOP_WORDS: set[str] = {
        "what", "how", "when", "where", "why", "who", "which",
        "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did",
        "the", "a", "an", "and", "or", "but", "in", "on", "at",
        "to", "for", "of", "with", "by", "from", "as", "into",
        "this", "that", "these", "those", "it", "its",
        "can", "will", "would", "should", "could", "may", "might",
        "my", "your", "our", "their", "if", "i", "me", "we",
        "not", "no", "don", "doesn", "didn", "won", "isn", "aren",
        "about", "also", "just", "very", "so", "than",
    }

    def expand(self, query: str) -> str:
        """
        Expand query with domain synonyms before embedding/search.

        Adds GST-specific synonyms so "supplier doesn't file" also searches
        for "vendor filing" and "seller submit GSTR-1".
        """
        tokens = query.lower().split()
        expanded = set(tokens)

        for token in tokens:
            clean_token = re.sub(r"[^a-z0-9]", "", token)
            if clean_token in self.GST_SYNONYMS:
                expanded.update(self.GST_SYNONYMS[clean_token])

        return " ".join(sorted(expanded))

    def clean(self, query: str) -> str:
        """Remove stop words for cleaner keyword matching."""
        tokens = query.lower().split()
        return " ".join(t for t in tokens if t not in self.STOP_WORDS and len(t) > 1)

    def decompose(self, query: str) -> list[str]:
        """
        Break compound questions into sub-queries.

        "What about ITC rules and also reconciliation process?"
        → ["ITC rules", "reconciliation process"]
        """
        parts = re.split(
            r"\band\b|\balso\b|\bwhat about\b|\badditionally\b|\bplus\b",
            query,
            flags=re.IGNORECASE,
        )
        cleaned = [p.strip() for p in parts if len(p.strip()) > 10]
        return cleaned if len(cleaned) > 1 else [query]

    def process(self, query: str) -> str:
        """
        Full pipeline: clean → expand → return optimized query.

        This is the main entry point for the retrieval pipeline.
        """
        cleaned = self.clean(query)
        expanded = self.expand(cleaned)
        return expanded
