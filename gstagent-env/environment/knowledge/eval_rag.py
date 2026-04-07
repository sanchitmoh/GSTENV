"""
RAG Evaluation Harness v2 — ground truth test cases for retrieval quality.

Measures:
- Source Hit Rate: Are the right parent documents being retrieved?
- Keyword Hit Rate: Does the retrieved context contain expected terms?

Fixes from v1:
- Uses explicit parent_id resolution instead of fragile substring matching
- Normalizes whitespace and currency symbols for keyword matching

Run before and after every change to the retrieval pipeline.
This is what keeps the system honest — no eval = flying blind.
"""

from __future__ import annotations

import re

from environment.knowledge.rag_engine import RAGEngine

# Ground truth evaluation set — curated by domain experts
GST_EVAL_SET: list[dict] = [
    # ── Original 8 queries ───────────────────────────────────────────
    {
        "query": "Can I claim ITC if my supplier hasn't filed GSTR-1?",
        "expected_sources": ["Section-16-2", "CBIC-Circular-170"],
        "expected_keywords": ["gstr-2b", "condition", "paid to government", "supplier"],
    },
    {
        "query": "What is the provisional ITC percentage allowed?",
        "expected_sources": ["Rule-36-4", "CBIC-Circular-183"],
        "expected_keywords": ["5%", "removed", "01.01.2022"],
    },
    {
        "query": "How many days to pay supplier before ITC reversal?",
        "expected_sources": ["Rule-37"],
        "expected_keywords": ["180 days", "reverse", "interest"],
    },
    {
        "query": "What happens when there is a mismatch between purchase register and GSTR-2B?",
        "expected_sources": ["Mismatch-Tolerance", "Reconciliation-Best-Practices"],
        "expected_keywords": ["5%", "20%", "variance", "supplier"],
    },
    {
        "query": "How does GSTR-2B get generated?",
        "expected_sources": ["GSTR-2B-Auto-Generation"],
        "expected_keywords": ["14th", "gstr-1", "auto-generated"],
    },
    {
        "query": "What is the impact of blocked ITC on small businesses?",
        "expected_sources": ["MSME-Cash-Flow-Impact"],
        "expected_keywords": ["working capital", "msme", "blocked"],
    },
    {
        "query": "How to handle credit notes in reconciliation?",
        "expected_sources": ["Credit-Note-Handling"],
        "expected_keywords": ["credit note", "reduce", "itc"],
    },
    {
        "query": "What is the step by step reconciliation process?",
        "expected_sources": ["Reconciliation-Best-Practices"],
        "expected_keywords": ["download", "match", "invoice number", "gstin"],
    },

    # ── NEW: E-Invoicing queries ─────────────────────────────────────
    {
        "query": "Is e-invoicing mandatory and what is IRN?",
        "expected_sources": ["E-Invoice-Mandate"],
        "expected_keywords": ["irn", "irp", "b2b", "₹5 cr"],
    },
    {
        "query": "Can I cancel an e-invoice after 24 hours?",
        "expected_sources": ["E-Invoice-Cancellation"],
        "expected_keywords": ["24 hours", "credit note", "cancelled"],
    },

    # ── NEW: Registration queries ────────────────────────────────────
    {
        "query": "What is the turnover limit for GST registration?",
        "expected_sources": ["GST-Registration-Threshold"],
        "expected_keywords": ["₹40 lakh", "₹20 lakh", "mandatory"],
    },

    # ── NEW: Returns queries ─────────────────────────────────────────
    {
        "query": "What is the due date for filing GSTR-3B?",
        "expected_sources": ["GSTR-3B-Filing"],
        "expected_keywords": ["20th", "18%", "nil return"],
    },
    {
        "query": "What is QRMP scheme and who is eligible?",
        "expected_sources": ["QRMP-Scheme"],
        "expected_keywords": ["quarterly", "₹5 cr", "pmt-06"],
    },

    # ── NEW: Penalties queries ───────────────────────────────────────
    {
        "query": "What is the interest rate on wrong ITC claims?",
        "expected_sources": ["GST-Interest-on-ITC", "GST-Penalties-Interest"],
        "expected_keywords": ["24%", "18%", "availed", "utilized"],
    },

    # ── NEW: RCM queries ────────────────────────────────────────────
    {
        "query": "What is reverse charge mechanism and when does it apply?",
        "expected_sources": ["Reverse-Charge-Mechanism"],
        "expected_keywords": ["recipient", "rcm", "self-invoice"],
    },

    # ── NEW: Blocked credits query ──────────────────────────────────
    {
        "query": "Which items are blocked from ITC under Section 17(5)?",
        "expected_sources": ["Blocked-Credits-Section-17-5"],
        "expected_keywords": ["motor vehicle", "food", "construction"],
    },

    # ── NEW: Place of supply query ──────────────────────────────────
    {
        "query": "How is place of supply determined for inter-state goods?",
        "expected_sources": ["Place-of-Supply-Goods"],
        "expected_keywords": ["igst", "movement", "terminates"],
    },

    # ── NEW: Exports query ──────────────────────────────────────────
    {
        "query": "How do I claim GST refund on exports?",
        "expected_sources": ["GST-Exports-Refund"],
        "expected_keywords": ["zero-rated", "lut", "refund"],
    },

    # ── NEW: Time limit query ───────────────────────────────────────
    {
        "query": "What is the time limit for claiming ITC under Section 16(4)?",
        "expected_sources": ["Section-16-4-Time-Limit"],
        "expected_keywords": ["30th november", "annual return", "lapses"],
    },

    # ── NEW: E-Way Bill query ───────────────────────────────────────
    {
        "query": "When is e-way bill required for goods transport?",
        "expected_sources": ["E-Way-Bill"],
        "expected_keywords": ["₹50,000", "200 km", "part-b"],
    },
]


def _normalize(text: str) -> str:
    """Normalize text for flexible matching: lowercase, collapse whitespace, handle currency."""
    text = text.lower()
    text = text.replace("₹", "rs").replace("rs.", "rs")
    text = re.sub(r"\s+", " ", text)
    return text


def _resolve_parent_ids(results: list[dict]) -> set[str]:
    """Extract parent document IDs from results (handles chunk IDs)."""
    parents = set()
    for r in results:
        doc_id = r.get("doc_id", "")
        parent = doc_id.split("_chunk_")[0]
        parents.add(parent)
    return parents


def evaluate_retrieval(engine: RAGEngine | None = None, verbose: bool = False) -> dict:
    """
    Run the evaluation harness against the RAG engine.

    Returns:
        Dict with source_hit_rate, keyword_hit_rate, per_query details.
    """
    if engine is None:
        engine = RAGEngine()
        engine.initialize()

    total_source_hits = 0
    total_keyword_score = 0.0
    per_query: list[dict] = []

    for case in GST_EVAL_SET:
        results = engine.retrieve(case["query"], top_k=3)

        # Source hit: resolve parent IDs and check for exact match
        retrieved_parents = _resolve_parent_ids(results)
        source_hit = any(
            src in retrieved_parents for src in case["expected_sources"]
        )

        # Keyword hit: normalize context for flexible matching
        raw_context = " ".join(r["content"] for r in results)
        context = _normalize(raw_context)

        keyword_hits = sum(
            1 for kw in case["expected_keywords"]
            if _normalize(kw) in context
        )
        keyword_score = keyword_hits / max(len(case["expected_keywords"]), 1)

        total_source_hits += int(source_hit)
        total_keyword_score += keyword_score

        query_result = {
            "query": case["query"],
            "source_hit": source_hit,
            "keyword_score": round(keyword_score, 2),
            "retrieved_parents": sorted(retrieved_parents),
            "expected_sources": case["expected_sources"],
        }
        per_query.append(query_result)

        if verbose:
            status = "✅" if source_hit else "❌"
            print(f"{status} [{keyword_score:.0%}] {case['query']}")
            if not source_hit:
                print(f"   Expected: {case['expected_sources']}")
                print(f"   Got:      {sorted(retrieved_parents)}")

    n = len(GST_EVAL_SET)
    report = {
        "source_hit_rate": round(total_source_hits / n, 2),
        "keyword_hit_rate": round(total_keyword_score / n, 2),
        "total_queries": n,
        "source_hits": total_source_hits,
        "per_query": per_query,
    }

    if verbose:
        print(f"\n📊 Source Hit Rate: {report['source_hit_rate']:.0%}")
        print(f"📊 Keyword Hit Rate: {report['keyword_hit_rate']:.0%}")

    return report


if __name__ == "__main__":
    print("🔍 Running RAG Evaluation Harness v2...\n")
    evaluate_retrieval(verbose=True)
