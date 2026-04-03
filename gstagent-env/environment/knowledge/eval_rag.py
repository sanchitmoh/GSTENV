"""
RAG Evaluation Harness — ground truth test cases for retrieval quality.

Measures:
- Source Hit Rate: Are the right documents being retrieved?
- Keyword Hit Rate: Does the retrieved context contain expected terms?

Run before and after every change to the retrieval pipeline.
This is what keeps the system honest — no eval = flying blind.
"""

from __future__ import annotations

from environment.knowledge.rag_engine import RAGEngine

# Ground truth evaluation set — curated by domain experts
GST_EVAL_SET: list[dict] = [
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
]


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

        # Source hit: did we retrieve at least one expected document?
        retrieved_ids = set()
        for r in results:
            retrieved_ids.add(r.get("doc_id", ""))
            # Also check parent_id for chunks
            parent = r.get("doc_id", "").split("_chunk_")[0]
            retrieved_ids.add(parent)

        source_hit = any(
            src in retrieved_ids or any(src in rid for rid in retrieved_ids)
            for src in case["expected_sources"]
        )

        # Keyword hit: what fraction of expected keywords appear in context?
        context = " ".join(r["content"] for r in results).lower()
        keyword_hits = sum(
            1 for kw in case["expected_keywords"]
            if kw.lower() in context
        )
        keyword_score = keyword_hits / max(len(case["expected_keywords"]), 1)

        total_source_hits += int(source_hit)
        total_keyword_score += keyword_score

        query_result = {
            "query": case["query"],
            "source_hit": source_hit,
            "keyword_score": round(keyword_score, 2),
            "retrieved_ids": list(retrieved_ids),
            "expected_sources": case["expected_sources"],
        }
        per_query.append(query_result)

        if verbose:
            status = "✅" if source_hit else "❌"
            print(f"{status} [{keyword_score:.0%}] {case['query']}")
            if not source_hit:
                print(f"   Expected: {case['expected_sources']}")
                print(f"   Got:      {list(retrieved_ids)}")

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
    print("🔍 Running RAG Evaluation Harness...\n")
    evaluate_retrieval(verbose=True)
