"""
Advanced Multi-Agent Inference with RAG v2.

This is the upgraded inference script that uses:
1. Multi-agent orchestration (Matcher → Auditor → Reporter → Validator)
2. RAG v2 — query expansion, hybrid search, score thresholds, faithfulness checks
3. Curriculum learning for difficulty progression
4. Full observability metrics
5. Retrieval evaluation harness for quality monitoring

Usage:
    python inference_advanced.py

Environment variables:
    API_BASE_URL (default: http://localhost:7860)
"""

from __future__ import annotations

import json
import os
import sys
import time

from environment.agents.orchestrator import Orchestrator
from environment.config import API_BASE_URL, MODEL_NAME, OPENAI_API_KEY
from environment.curriculum import CurriculumConfig, CurriculumManager
from environment.knowledge.rag_engine import get_rag_engine
from environment.knowledge.eval_rag import evaluate_retrieval
from environment.observability import PerformanceTracker, get_metrics_collector


def main():
    api_base = API_BASE_URL
    model = MODEL_NAME

    print("=" * 70)
    print("  GST AGENT ENVIRONMENT — Advanced Multi-Agent Inference")
    print("=" * 70)
    print(f"  API:   {api_base}")
    print(f"  Model: {model}")

    if not OPENAI_API_KEY:
        print("\n  ⚠️  Warning: OPENAI_API_KEY not set. Agent LLM calls will fail.")
        print("     Set it in .env or as an environment variable.")

    # ── Initialize RAG v2 ────────────────────────────────────────
    print("\n📚 Initializing RAG v2 Knowledge Base...")
    rag = get_rag_engine()
    print(f"   Indexed {rag.document_count} GST knowledge documents/chunks")
    print(f"   Features: query expansion, hybrid search, score threshold, faithfulness")

    # Run eval harness to baseline retrieval quality
    eval_report = evaluate_retrieval(rag)
    print(f"   📊 Eval: source_hit={eval_report['source_hit_rate']:.0%}, "
          f"keyword_hit={eval_report['keyword_hit_rate']:.0%}")

    # ── Initialize Curriculum ────────────────────────────────────
    config = CurriculumConfig(
        easy_threshold=0.75,
        medium_threshold=0.70,
        hard_threshold=0.65,
        min_episodes=1,
        auto_advance=True,
    )
    curriculum = CurriculumManager(config)
    print(f"   Curriculum: starting at {curriculum.get_current_difficulty()}")

    # ── Initialize Orchestrator ──────────────────────────────────
    orchestrator = Orchestrator(api_base_url=api_base)

    # ── Initialize Metrics ───────────────────────────────────────
    metrics = get_metrics_collector()

    # ── Run All Tasks ────────────────────────────────────────────
    all_scores = {}
    total_start = time.time()

    for task_id in ["invoice_match", "itc_audit", "full_recon"]:
        print(f"\n{'─' * 60}")
        print(f"  Task: {task_id} | Difficulty: {curriculum.get_current_difficulty()}")

        # Show RAG v2 context for this task (category-filtered retrieval)
        if task_id == "itc_audit":
            context = rag.get_itc_rules_context()
            print(f"  RAG context injected: {len(context)} chars of ITC rules")
        elif task_id == "full_recon":
            context = rag.get_reconciliation_context()
            print(f"  RAG context injected: {len(context)} chars of reconciliation guidance")
        else:
            context = rag.get_context_for_prompt(
                f"invoice matching {task_id}", max_tokens=1500
            )
            print(f"  RAG context injected: {len(context)} chars")

        # Run task
        tracker = PerformanceTracker(f"adv-{task_id}", task_id)
        try:
            score = orchestrator.run_task(task_id)
            all_scores[task_id] = score
            tracker.set_final_score(score)

            # Record in curriculum
            advancement = curriculum.record_score(task_id, score)
            if advancement.get("advanced"):
                print(f"  🎓 Curriculum Advanced: {advancement['from']} → {advancement['to']}")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            all_scores[task_id] = 0.0
            tracker.set_final_score(0.0)

        # Record episode metrics
        episode = tracker.get_episode_metrics()
        metrics.record_episode(episode)

    total_time = time.time() - total_start

    # ── Final Report ─────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  FINAL RESULTS — Multi-Agent + RAG + Curriculum")
    print(f"{'=' * 70}")
    for task, score in all_scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {task:20s} : {score:6.4f} [{bar}]")

    avg = sum(all_scores.values()) / max(len(all_scores), 1)
    print(f"  {'Average':20s} : {avg:6.4f}")
    print(f"  Total time: {total_time:.1f}s")

    # Curriculum summary
    cs = curriculum.get_summary()
    print(f"\n  Curriculum: {cs['current_level']} | Mastered: {cs['mastered']}")

    # Metrics summary
    ms = metrics.get_summary()
    print(f"  Steps: {ms['total_steps']} | Errors: {ms['total_errors']}")
    print(f"  Avg scores: {ms['avg_scores_by_task']}")

    return all_scores


if __name__ == "__main__":
    scores = main()
    # Exit with non-zero if all scores are 0
    if all(s == 0 for s in scores.values()):
        sys.exit(1)
