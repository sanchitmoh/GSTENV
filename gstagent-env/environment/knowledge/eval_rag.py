"""
RAG Evaluation Harness v3 — expanded ground truth test cases + advanced metrics.

Measures:
- Source Hit Rate: Are the right parent documents being retrieved?
- Keyword Hit Rate: Does the retrieved context contain expected terms?
- MRR (Mean Reciprocal Rank): Position of first relevant document
- Recall@K: Hit rate at K=1, K=3, K=5
- Per-query latency: Time per retrieval call

Fixes from v1:
- Uses explicit parent_id resolution instead of fragile substring matching
- Normalizes whitespace and currency symbols for keyword matching

New in v3:
- 13 additional eval queries covering RCM, composition, TDS, annual returns,
  proportional reversal, audit, ISD, anti-profiteering, debit notes, Rule 86B,
  adversarial, and negative/cross-category queries
- MRR metric for ranking quality
- Recall@K at multiple K values
- Latency tracking per query

Run before and after every change to the retrieval pipeline.
This is what keeps the system honest — no eval = flying blind.
"""

from __future__ import annotations

import re
import time

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

    # ── v3 NEW: RCM specified services query ────────────────────────
    {
        "query": "Which services are covered under reverse charge mechanism?",
        "expected_sources": ["RCM-Specified-Services", "Reverse-Charge-Mechanism"],
        "expected_keywords": ["advocate", "gta", "director"],
    },

    # ── v3 NEW: Composition scheme query ────────────────────────────
    {
        "query": "Who is eligible for the composition scheme and what are the rates?",
        "expected_sources": ["Composition-Scheme"],
        "expected_keywords": ["₹1.5 cr", "composition", "cmp-08"],
    },

    # ── v3 NEW: TDS under GST query ─────────────────────────────────
    {
        "query": "When is TDS applicable under GST and what is the rate?",
        "expected_sources": ["GST-TDS"],
        "expected_keywords": ["2%", "₹2.5 lakh", "gstr-7"],
    },

    # ── v3 NEW: Annual return query ─────────────────────────────────
    {
        "query": "Who needs to file GSTR-9 annual return?",
        "expected_sources": ["GSTR-9-Annual-Return"],
        "expected_keywords": ["31st december", "₹2 cr", "annual"],
    },

    # ── v3 NEW: Proportional reversal query ─────────────────────────
    {
        "query": "How is ITC reversed for mixed-use inputs under Rule 42?",
        "expected_sources": ["ITC-Proportional-Reversal-Rule-42-43"],
        "expected_keywords": ["turnover ratio", "exempt", "proportionally"],
    },

    # ── v3 NEW: GST audit query ─────────────────────────────────────
    {
        "query": "What are the different types of GST audit?",
        "expected_sources": ["GST-Audit-Assessment"],
        "expected_keywords": ["section 65", "special audit", "scrutiny"],
    },

    # ── v3 NEW: ISD query ───────────────────────────────────────────
    {
        "query": "What is Input Service Distributor and how does it work?",
        "expected_sources": ["Input-Service-Distributor"],
        "expected_keywords": ["gstr-6", "distribute", "branches"],
    },

    # ── v3 NEW: Debit note ITC query ────────────────────────────────
    {
        "query": "Can I claim ITC on debit notes and supplementary invoices?",
        "expected_sources": ["Debit-Note-ITC"],
        "expected_keywords": ["debit note", "financial year", "section 16"],
    },

    # ── v3 NEW: Rule 86B cash restriction ───────────────────────────
    {
        "query": "What is the 1% cash payment rule under Rule 86B?",
        "expected_sources": ["Rule-86B-Cash-Restriction"],
        "expected_keywords": ["₹50 lakh", "1%", "cash"],
    },

    # ── v3 NEW: Electronic ledgers query ────────────────────────────
    {
        "query": "How do electronic ledgers work on the GST portal?",
        "expected_sources": ["Electronic-Ledgers"],
        "expected_keywords": ["cash ledger", "credit ledger", "utilization"],
    },

    # ── v3 NEW: Anti-profiteering query ─────────────────────────────
    {
        "query": "What are the anti-profiteering provisions under GST?",
        "expected_sources": ["Anti-Profiteering"],
        "expected_keywords": ["section 171", "reduction", "penalty"],
    },

    # ── v3 NEW: Adversarial — irrelevant query ──────────────────────
    {
        "query": "How to make chocolate cake recipe?",
        "expected_sources": [],  # Should NOT match any GST doc
        "expected_keywords": [],
        "type": "negative",  # Marker for negative test
    },

    # ── v3 NEW: GSTR-1 filing details ──────────────────────────────
    {
        "query": "What is the due date for filing GSTR-1?",
        "expected_sources": ["GSTR-1-Filing"],
        "expected_keywords": ["11th", "gstr-1", "b2b"],
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


def evaluate_retrieval(
    engine: RAGEngine | None = None,
    verbose: bool = False,
    k_values: list[int] | None = None,
) -> dict:
    """
    Run the evaluation harness against the RAG engine.

    Returns:
        Dict with source_hit_rate, keyword_hit_rate, mrr, recall_at_k,
        avg_latency_ms, per_query details.
    """
    if engine is None:
        engine = RAGEngine()
        engine.initialize()

    if k_values is None:
        k_values = [1, 3, 5]

    max_k = max(k_values)
    total_source_hits = 0
    total_keyword_score = 0.0
    reciprocal_ranks: list[float] = []
    recall_at_k_hits: dict[int, int] = {k: 0 for k in k_values}
    latencies: list[float] = []
    per_query: list[dict] = []

    # Separate regular and negative queries
    regular_cases = [c for c in GST_EVAL_SET if c.get("type") != "negative"]
    negative_cases = [c for c in GST_EVAL_SET if c.get("type") == "negative"]

    for case in regular_cases:
        t0 = time.time()
        results = engine.retrieve(case["query"], top_k=max_k)
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)

        # Source hit: resolve parent IDs and check for exact match
        retrieved_parents = _resolve_parent_ids(results)
        source_hit = any(
            src in retrieved_parents for src in case["expected_sources"]
        )

        # MRR: find rank of first relevant source (1-indexed)
        rr = 0.0
        for rank, r in enumerate(results, 1):
            parent = r["doc_id"].split("_chunk_")[0]
            if parent in case["expected_sources"]:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        # Recall@K: check if any expected source appears in top-K results
        for k in k_values:
            top_k_parents = _resolve_parent_ids(results[:k])
            if any(src in top_k_parents for src in case["expected_sources"]):
                recall_at_k_hits[k] += 1

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
            "reciprocal_rank": round(rr, 4),
            "latency_ms": round(latency_ms, 1),
        }
        per_query.append(query_result)

        if verbose:
            status = "[PASS]" if source_hit else "[FAIL]"
            print(f"{status} [{keyword_score:.0%}] (RR={rr:.2f}, {latency_ms:.0f}ms) {case['query']}")
            if not source_hit:
                print(f"   Expected: {case['expected_sources']}")
                print(f"   Got:      {sorted(retrieved_parents)}")

    # Evaluate negative queries — should NOT match specific docs
    negative_pass = 0
    for case in negative_cases:
        t0 = time.time()
        results = engine.retrieve(case["query"], top_k=3)
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)

        # For negative queries, success = low relevance scores
        is_negative_pass = True
        if results:
            max_relevance = max(r.get("relevance", 0) for r in results)
            # If best match is still low, it's a pass
            is_negative_pass = max_relevance < 0.3

        negative_pass += int(is_negative_pass)

        query_result = {
            "query": case["query"],
            "type": "negative",
            "negative_pass": is_negative_pass,
            "max_relevance": round(max(r.get("relevance", 0) for r in results), 4) if results else 0,
            "latency_ms": round(latency_ms, 1),
        }
        per_query.append(query_result)

        if verbose:
            status = "[PASS]" if is_negative_pass else "[WARN]"
            print(f"{status} [NEG] ({latency_ms:.0f}ms) {case['query']}")

    n_regular = len(regular_cases)
    n_negative = len(negative_cases)

    # Compute aggregate metrics
    mrr = round(sum(reciprocal_ranks) / max(n_regular, 1), 4)
    recall_at_k = {
        f"recall@{k}": round(recall_at_k_hits[k] / max(n_regular, 1), 2)
        for k in k_values
    }
    avg_latency = round(sum(latencies) / max(len(latencies), 1), 1)

    report = {
        "source_hit_rate": round(total_source_hits / max(n_regular, 1), 2),
        "keyword_hit_rate": round(total_keyword_score / max(n_regular, 1), 2),
        "mrr": mrr,
        **recall_at_k,
        "avg_latency_ms": avg_latency,
        "total_queries": n_regular + n_negative,
        "regular_queries": n_regular,
        "negative_queries": n_negative,
        "negative_pass_rate": round(negative_pass / max(n_negative, 1), 2),
        "source_hits": total_source_hits,
        "per_query": per_query,
    }

    if verbose:
        print(f"\n>> Source Hit Rate: {report['source_hit_rate']:.0%}")
        print(f">> Keyword Hit Rate: {report['keyword_hit_rate']:.0%}")
        print(f">> MRR: {mrr:.4f}")
        for k in k_values:
            print(f">> Recall@{k}: {recall_at_k[f'recall@{k}']:.0%}")
        print(f">> Avg Latency: {avg_latency:.0f}ms")
        if n_negative > 0:
            print(f">> Negative Query Pass Rate: {report['negative_pass_rate']:.0%}")

    return report


if __name__ == "__main__":
    print("Running RAG Evaluation Harness v3...\n")
    evaluate_retrieval(verbose=True)

