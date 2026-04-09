"""Pre-submission validation script — run all critical checks."""
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def check_rag_eval():
    """Run RAG evaluation harness and print metrics."""
    print("\n" + "=" * 60)
    print("  RAG EVALUATION HARNESS")
    print("=" * 60)
    from environment.knowledge.eval_rag import evaluate_retrieval
    results = evaluate_retrieval()
    print(f"  MRR:             {results['mrr']:.4f}")
    print(f"  NDCG@3:          {results['ndcg@3']:.4f}")
    print(f"  Source Hit Rate: {results['source_hit_rate']:.4f}")
    print(f"  Keyword Hit Rate:{results['keyword_hit_rate']:.4f}")
    print(f"  Queries:         {results['total_queries']}")
    assert results["mrr"] > 0.3, f"MRR too low: {results['mrr']}"
    assert results["source_hit_rate"] > 0.4, f"Source hit rate too low: {results['source_hit_rate']}"
    print("  PASS")

def check_environment():
    """Run a full environment lifecycle: reset -> step -> submit."""
    print("\n" + "=" * 60)
    print("  ENVIRONMENT LIFECYCLE")
    print("=" * 60)
    from environment.env import GSTAgentEnv
    from environment.models import GSTAction

    for task_id in ["invoice_match", "itc_audit", "full_recon"]:
        env = GSTAgentEnv()
        obs = env.reset(task_id)
        inv_id = obs.purchase_register[0].invoice_id
        print(f"  [{task_id}] reset OK: {len(obs.purchase_register)} invoices, session={obs.session_id[:8]}")

        # Step: match first invoice
        action = GSTAction(action_type="match_invoice", invoice_id=inv_id)
        obs2, reward, done, info = env.step(action)
        print(f"  [{task_id}] step OK: reward={reward}, step={obs2.step_number}")

        # Submit
        action2 = GSTAction(action_type="submit_report", payload={"matches": {inv_id: "present"}})
        obs3, reward2, done2, info2 = env.step(action2)
        print(f"  [{task_id}] submit OK: score={reward2:.4f}, grader={info2.get('grader')}")
        assert done2, "Episode should be done after submit"
    print("  PASS")

def check_agents():
    """Verify all agents can be instantiated and have required methods."""
    print("\n" + "=" * 60)
    print("  AGENT PIPELINE")
    print("=" * 60)
    from environment.agents.base_agent import BaseAgent, AgentMessage
    from environment.agents.matcher import MatcherAgent
    from environment.agents.orchestrator import Orchestrator

    matcher = MatcherAgent()
    print(f"  MatcherAgent: name={matcher.name}, model={matcher.model}")
    assert hasattr(matcher, "inject_context"), "Missing inject_context"
    assert hasattr(matcher, "get_full_system_prompt"), "Missing get_full_system_prompt"

    # Test RAG context injection
    matcher.inject_context("Test RAG context")
    prompt = matcher.get_full_system_prompt()
    assert "Test RAG context" in prompt, "RAG context not injected into prompt"
    print("  RAG context injection: OK")

    orch = Orchestrator.__init__
    print("  Orchestrator importable: OK")
    print("  PASS")

def check_imports():
    """Verify all top-level imports work."""
    print("\n" + "=" * 60)
    print("  IMPORT VALIDATION")
    print("=" * 60)
    modules = [
        "environment.server",
        "environment.env",
        "environment.models",
        "environment.config",
        "environment.auth",
        "environment.graders.grader_invoice_match",
        "environment.graders.grader_itc_audit",
        "environment.graders.grader_full_recon",
        "environment.rules.gst_rules",
        "environment.tasks.task1_invoice_match",
        "environment.tasks.task2_itc_audit",
        "environment.tasks.task3_full_recon",
        "environment.agents.base_agent",
        "environment.agents.matcher",
        "environment.agents.orchestrator",
        "environment.curriculum",
        "environment.observability",
        "environment.knowledge.rag_engine",
        "environment.knowledge.query_router",
        "environment.knowledge.query_processor",
        "environment.knowledge.vector_store",
        "environment.knowledge.chunker",
        "environment.knowledge.gst_knowledge",
        "environment.knowledge.eval_rag",
        "environment.knowledge.faithfulness",
        "environment.leaderboard_db",
        "environment.data.generator",
    ]
    for mod in modules:
        __import__(mod)
        print(f"  {mod}: OK")

    # Top-level scripts (not packages)
    import importlib.util
    for script in ["inference.py", "inference_advanced.py"]:
        spec = importlib.util.spec_from_file_location(script.replace(".py", ""), script)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        print(f"  {script}: OK")
    print(f"  All {len(modules)} modules imported successfully")
    print("  PASS")

def check_rag_retrieval_modes():
    """Test all RAG retrieval strategies."""
    print("\n" + "=" * 60)
    print("  RAG RETRIEVAL MODES")
    print("=" * 60)
    from environment.knowledge.rag_engine import get_rag_engine
    rag = get_rag_engine()

    # Standard retrieval
    ctx = rag.get_context_for_prompt("What is ITC under GST?")
    assert len(ctx) > 0, "Standard retrieval returned empty"
    print(f"  Standard retrieval: {len(ctx)} chars")

    # ITC rules context
    ctx2 = rag.get_itc_rules_context()
    assert len(ctx2) > 0, "ITC rules context returned empty"
    print(f"  ITC rules context:  {len(ctx2)} chars")

    # Reconciliation context
    ctx3 = rag.get_reconciliation_context()
    assert len(ctx3) > 0, "Reconciliation context returned empty"
    print(f"  Recon context:      {len(ctx3)} chars")

    # Cache test
    _ = rag.get_context_for_prompt("What is ITC under GST?")  # Should be cached
    stats = rag.cache_stats
    print(f"  Cache stats:        hits={stats.get('hits', 0)}, misses={stats.get('misses', 0)}")
    print("  PASS")

def check_query_router():
    """Test query routing strategies."""
    print("\n" + "=" * 60)
    print("  QUERY ROUTER")
    print("=" * 60)
    from environment.knowledge.query_router import QueryRouter, SemanticCache, RAGFusion, SelfRAGController

    router = QueryRouter()
    tests = [
        ("What is the rate of IGST on exports?", "sentence_window"),
        ("How to file GSTR-3B step by step?", "hierarchical"),
        ("ITC eligibility", "filtered"),
    ]
    for query, expected in tests:
        route = router.classify(query)
        status = "OK" if route.strategy == expected else f"UNEXPECTED (got {route.strategy})"
        print(f"  '{query[:40]}...' -> {route.strategy} [{status}]")

    # Semantic cache
    cache = SemanticCache(max_size=10)
    cache.put("test query", [{"doc": "test"}])
    result = cache.get("test query")
    assert result is not None, "Cache miss on identical query"
    print(f"  SemanticCache: OK (hit_rate={cache.hit_rate:.1%})")

    # RAG Fusion
    fusion = RAGFusion()
    variants = fusion.generate_variants("What is ITC eligibility?")
    assert len(variants) >= 2, "Too few variants"
    print(f"  RAGFusion: {len(variants)} variants generated")

    # Self-RAG
    self_rag = SelfRAGController()
    assert not self_rag.needs_retrieval("hello"), "Should skip retrieval for greeting"
    assert self_rag.needs_retrieval("What is ITC eligibility under GST?"), "Should retrieve for domain query"
    print("  SelfRAG: OK")
    print("  PASS")

def check_curriculum():
    """Test curriculum learning."""
    print("\n" + "=" * 60)
    print("  CURRICULUM LEARNING")
    print("=" * 60)
    from environment.curriculum import CurriculumManager, CurriculumConfig
    cm = CurriculumManager(CurriculumConfig())
    print(f"  Current difficulty: {cm.get_current_difficulty()}")
    cm.record_score("invoice_match", 0.9)
    cm.record_score("invoice_match", 0.85)
    cm.record_score("invoice_match", 0.82)
    new_diff = cm.get_current_difficulty()
    print(f"  After high scores:  {new_diff}")
    print("  PASS")

if __name__ == "__main__":
    try:
        check_imports()
        check_environment()
        check_rag_retrieval_modes()
        check_rag_eval()
        check_query_router()
        check_agents()
        check_curriculum()

        print("\n" + "=" * 60)
        print("  ALL PRE-SUBMISSION CHECKS PASSED")
        print("=" * 60)
    except Exception as e:
        print(f"\n  FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
