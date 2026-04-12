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

        # ── NEW: E-Invoicing ────────────────────────────────────────
        "einvoice": ["e-invoice", "electronic invoice", "irn", "irp"],
        "irn": ["invoice reference number", "e-invoice", "einvoice"],
        "irp": ["invoice registration portal", "e-invoice portal"],

        # ── NEW: Registration ───────────────────────────────────────
        "gstin": ["registration", "gstin number", "gst identification number", "registered"],
        "registration": ["gstin", "registered", "enrolment", "reg"],
        "threshold": ["limit", "turnover limit", "eligibility"],
        "cancel": ["cancellation", "revoke", "revocation", "deregister"],

        # ── NEW: Returns ────────────────────────────────────────────
        "return": ["filing", "gstr", "form", "declaration"],
        "qrmp": ["quarterly return", "quarterly filing", "small taxpayer"],
        "annual": ["gstr-9", "yearly", "gstr9"],

        # ── NEW: Penalties & Interest ───────────────────────────────
        "interest": ["penalty", "late fee", "charge", "section 50"],
        "demand": ["notice", "show cause", "scn", "section 73", "section 74"],

        # ── NEW: RCM ────────────────────────────────────────────────
        "rcm": ["reverse charge", "reverse charge mechanism", "self-invoice"],
        "unregistered": ["non-registered", "urd", "unregistered dealer"],

        # ── NEW: Composition ────────────────────────────────────────
        "composition": ["composition scheme", "section 10", "fixed rate", "cmp-08"],

        # ── NEW: Blocked Credits ────────────────────────────────────
        "blocked": ["ineligible", "restricted", "section 17(5)", "not allowed"],

        # ── NEW: Exports ────────────────────────────────────────────
        "export": ["zero-rated", "lut", "shipping bill", "refund"],
        "refund": ["reimbursement", "claim back", "rfd-01"],
        "lut": ["letter of undertaking", "bond", "export without igst"],

        # ── NEW: TDS/TCS ────────────────────────────────────────────
        "tds": ["tax deducted at source", "gstr-7", "deductor"],
        "tcs": ["tax collected at source", "gstr-8", "e-commerce"],

        # ── NEW: Place of Supply ────────────────────────────────────
        "place": ["location", "state", "jurisdiction"],
        "interstate": ["inter-state", "cross-state", "igst"],
        "intrastate": ["intra-state", "within state", "cgst sgst"],

        # ── NEW: E-Way Bill ─────────────────────────────────────────
        "eway": ["e-way bill", "ewb", "transport", "goods movement"],
        "transport": ["movement", "shipping", "freight", "gta"],

        # ── NEW: ISD ────────────────────────────────────────────────
        "isd": ["input service distributor", "gstr-6", "branch credit"],

        # ── NEW: Audit ──────────────────────────────────────────────
        "assessment": ["scrutiny", "demand", "notice", "order"],
    }

    # Negation tokens — NEVER strip these, they invert query meaning
    NEGATION_TOKENS: set[str] = {
        "not", "no", "don", "doesn", "didn", "won", "isn", "aren",
        "cant", "cannot", "without", "never", "neither", "nor",
    }

    # Stop words to remove — negation tokens deliberately excluded
    STOP_WORDS: set[str] = {
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
        """Remove stop words while preserving negation tokens."""
        tokens = query.lower().split()
        return " ".join(
            t for t in tokens
            if (t in self.NEGATION_TOKENS or t not in self.STOP_WORDS) and len(t) > 1
        )

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
        Full pipeline: expand → clean → return optimized query.

        IMPORTANT: expand FIRST so synonyms are added while negation
        context is intact, THEN clean to remove neutral stop words.
        Reverse order would strip "not" before expansion, inverting meaning.
        """
        expanded = self.expand(query)
        cleaned = self.clean(expanded)
        return cleaned
