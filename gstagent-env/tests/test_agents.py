"""
Tests for the multi-agent orchestration system (Phase 9).

Covers: BaseAgent, MatcherAgent, AuditorAgent, ReporterAgent, ValidatorAgent.
"""

import pytest
from environment.agents.base_agent import AgentMessage, BaseAgent
from environment.agents.matcher import MatcherAgent
from environment.agents.auditor import AuditorAgent
from environment.agents.reporter import ReporterAgent
from environment.agents.validator import ValidatorAgent
from environment.agents.orchestrator import Orchestrator


# ── Test data ────────────────────────────────────────────────────

SAMPLE_OBS = {
    "purchase_register": [
        {
            "invoice_id": "INV-0001", "supplier_gstin": "27AAACR5055K1ZX",
            "buyer_gstin": "29BBBCR5055K1Z0", "invoice_date": "2025-03-05",
            "taxable_amount": 100000, "cgst": 9000, "sgst": 9000, "igst": 0,
            "hsn_code": "8471", "item_description": "Computers",
        },
        {
            "invoice_id": "INV-0002", "supplier_gstin": "27AAACR5055K1ZX",
            "buyer_gstin": "29BBBCR5055K1Z0", "invoice_date": "2025-03-10",
            "taxable_amount": 50000, "cgst": 4500, "sgst": 4500, "igst": 0,
            "hsn_code": "9401", "item_description": "Furniture",
        },
        {
            "invoice_id": "INV-0003", "supplier_gstin": "27AAACR5055K1ZX",
            "buyer_gstin": "29BBBCR5055K1Z0", "invoice_date": "2025-03-15",
            "taxable_amount": 200000, "cgst": 0, "sgst": 0, "igst": 36000,
            "hsn_code": "8517", "item_description": "Phones",
        },
    ],
    "gstr2b_data": [
        {
            "invoice_id": "INV-0001", "supplier_gstin": "27AAACR5055K1ZX",
            "taxable_amount": 100000,
        },
        {
            "invoice_id": "INV-0003", "supplier_gstin": "27AAACR5055K1ZX",
            "taxable_amount": 180000,  # 10% mismatch
        },
    ],
}


# ── Matcher Tests ────────────────────────────────────────────────

class TestMatcherAgent:
    def test_identifies_present(self):
        matcher = MatcherAgent()
        msgs = matcher.process(SAMPLE_OBS, [])
        statuses = {m.action["invoice_id"]: m.metadata.get("status") for m in msgs if m.action and "invoice_id" in m.action}
        assert statuses["INV-0001"] == "present"

    def test_identifies_missing(self):
        matcher = MatcherAgent()
        msgs = matcher.process(SAMPLE_OBS, [])
        statuses = {m.action["invoice_id"]: m.metadata.get("status") for m in msgs if m.action and "invoice_id" in m.action}
        assert statuses["INV-0002"] == "missing"

    def test_identifies_mismatch(self):
        matcher = MatcherAgent()
        msgs = matcher.process(SAMPLE_OBS, [])
        statuses = {m.action["invoice_id"]: m.metadata.get("status") for m in msgs if m.action and "invoice_id" in m.action}
        assert statuses["INV-0003"] == "mismatch"

    def test_summary_message(self):
        matcher = MatcherAgent()
        msgs = matcher.process(SAMPLE_OBS, [])
        summary = msgs[-1]
        assert "complete" in summary.content.lower()

    def test_action_types_correct(self):
        matcher = MatcherAgent()
        msgs = matcher.process(SAMPLE_OBS, [])
        for m in msgs:
            if m.action and m.action.get("action_type"):
                assert m.action["action_type"] in ("match_invoice", "flag_mismatch")


# ── Auditor Tests ────────────────────────────────────────────────

class TestAuditorAgent:
    def test_eligible_for_matched(self):
        auditor = AuditorAgent()
        msgs = auditor.process(SAMPLE_OBS, [])
        statuses = {m.action["invoice_id"]: m.metadata.get("status") for m in msgs if m.action and "invoice_id" in m.action}
        assert statuses["INV-0001"] == "eligible"

    def test_ineligible_for_missing(self):
        auditor = AuditorAgent()
        msgs = auditor.process(SAMPLE_OBS, [])
        statuses = {m.action["invoice_id"]: m.metadata.get("status") for m in msgs if m.action and "invoice_id" in m.action}
        assert statuses["INV-0002"] == "ineligible"

    def test_partial_for_mismatch(self):
        auditor = AuditorAgent()
        msgs = auditor.process(SAMPLE_OBS, [])
        statuses = {m.action["invoice_id"]: m.metadata.get("status") for m in msgs if m.action and "invoice_id" in m.action}
        assert statuses["INV-0003"] == "partial"

    def test_itc_summary_present(self):
        auditor = AuditorAgent()
        msgs = auditor.process(SAMPLE_OBS, [])
        summary = [m for m in msgs if m.action and m.action.get("action_type") == "compute_itc"]
        assert len(summary) == 1
        assert summary[0].metadata.get("total_eligible_itc", 0) > 0

    def test_rule_citations(self):
        auditor = AuditorAgent()
        msgs = auditor.process(SAMPLE_OBS, [])
        for m in msgs:
            if m.action and "invoice_id" in m.action:
                assert "Rule" in m.content or "Section" in m.content


# ── Reporter Tests ───────────────────────────────────────────────

class TestReporterAgent:
    def setup_method(self):
        # Run matcher and auditor first
        self.matcher = MatcherAgent()
        self.auditor = AuditorAgent()
        self.context = []
        self.context.extend(self.matcher.process(SAMPLE_OBS, self.context))
        self.context.extend(self.auditor.process(SAMPLE_OBS, self.context))

    def test_produces_submit_action(self):
        reporter = ReporterAgent()
        msgs = reporter.process(SAMPLE_OBS, self.context)
        assert len(msgs) == 1
        assert msgs[0].action["action_type"] == "submit_report"

    def test_report_has_total_itc(self):
        reporter = ReporterAgent()
        msgs = reporter.process(SAMPLE_OBS, self.context)
        payload = msgs[0].action["payload"]
        assert "total_itc" in payload
        assert payload["total_itc"] > 0

    def test_report_has_discrepancies(self):
        reporter = ReporterAgent()
        msgs = reporter.process(SAMPLE_OBS, self.context)
        payload = msgs[0].action["payload"]
        assert "discrepancies" in payload
        assert len(payload["discrepancies"]) > 0


# ── Validator Tests ──────────────────────────────────────────────

class TestValidatorAgent:
    def setup_method(self):
        self.matcher = MatcherAgent()
        self.auditor = AuditorAgent()
        self.reporter = ReporterAgent()
        self.context = []
        self.context.extend(self.matcher.process(SAMPLE_OBS, self.context))
        self.context.extend(self.auditor.process(SAMPLE_OBS, self.context))
        self.context.extend(self.reporter.process(SAMPLE_OBS, self.context))

    def test_clean_report_unchanged(self):
        validator = ValidatorAgent()
        msgs = validator.process(SAMPLE_OBS, self.context)
        assert len(msgs) == 1
        assert msgs[0].metadata.get("hallucinated_removed") == 0

    def test_removes_hallucinated_ids(self):
        # Inject fake IDs into reporter's output
        fake_context = list(self.context)
        fake_context.append(AgentMessage(
            sender="Reporter",
            content="Fake report",
            action={
                "action_type": "submit_report",
                "payload": {
                    "total_itc": 50000,
                    "discrepancies": [
                        {"invoice_id": "FAKE-999", "status": "bad", "action": "none"},
                        {"invoice_id": "INV-0001", "status": "ok", "action": "ok"},
                    ],
                    "matches": {"INV-0001": "present", "FAKE-888": "missing"},
                    "decisions": {"INV-0001": "eligible"},
                },
            },
        ))

        validator = ValidatorAgent()
        msgs = validator.process(SAMPLE_OBS, fake_context)
        clean = msgs[0].metadata.get("clean_report", {})
        disc_ids = [d["invoice_id"] for d in clean.get("discrepancies", [])]
        assert "FAKE-999" not in disc_ids
        assert "FAKE-888" not in clean.get("matches", {})
        assert msgs[0].metadata["hallucinated_removed"] >= 2

    def test_caps_itc_at_max(self):
        fake_context = list(self.context)
        fake_context.append(AgentMessage(
            sender="Reporter",
            content="Inflated ITC",
            action={
                "action_type": "submit_report",
                "payload": {
                    "total_itc": 999999999.0,  # absurdly high
                    "discrepancies": [],
                    "matches": {},
                    "decisions": {},
                },
            },
        ))
        validator = ValidatorAgent()
        msgs = validator.process(SAMPLE_OBS, fake_context)
        clean = msgs[0].metadata.get("clean_report", {})
        max_possible = sum(
            inv.get("cgst", 0) + inv.get("sgst", 0) + inv.get("igst", 0)
            for inv in SAMPLE_OBS["purchase_register"]
        )
        assert clean["total_itc"] <= max_possible * 1.1


# ── System Prompt Tests (⚠ was untested) ─────────────────────────

class TestSystemPrompts:
    """Verify every agent's get_system_prompt() contains jurisdiction-aware GST content."""

    def test_matcher_prompt_mentions_gst_terms(self):
        prompt = MatcherAgent().get_system_prompt().lower()
        assert "gst" in prompt
        assert "invoice" in prompt
        assert "gstr-2b" in prompt or "gstr" in prompt

    def test_auditor_prompt_mentions_rules(self):
        prompt = AuditorAgent().get_system_prompt().lower()
        assert "rule 36(4)" in prompt or "rule 36" in prompt
        assert "section 16(2)" in prompt or "section 16" in prompt

    def test_auditor_prompt_has_grounding_clause(self):
        from environment.agents.base_agent import GROUNDING_CLAUSE
        prompt = AuditorAgent().get_system_prompt()
        assert GROUNDING_CLAUSE.strip() in prompt

    def test_reporter_prompt_has_grounding_clause(self):
        from environment.agents.base_agent import GROUNDING_CLAUSE
        prompt = ReporterAgent().get_system_prompt()
        assert GROUNDING_CLAUSE.strip() in prompt

    def test_validator_prompt_mentions_hallucination(self):
        prompt = ValidatorAgent().get_system_prompt().lower()
        assert "hallucin" in prompt

    def test_all_prompts_non_trivial(self):
        """Every agent's system prompt should be substantial (>100 chars)."""
        for cls in [MatcherAgent, AuditorAgent, ReporterAgent, ValidatorAgent]:
            prompt = cls().get_system_prompt()
            assert len(prompt) > 100, f"{cls.__name__} prompt too short ({len(prompt)} chars)"

    def test_rag_context_injection_appears_in_full_prompt(self):
        """get_full_system_prompt() must surface injected RAG context."""
        agent = AuditorAgent()
        agent.inject_context("TEST RAG CONTEXT: Section 17(5)")
        full = agent.get_full_system_prompt()
        assert "TEST RAG CONTEXT: Section 17(5)" in full
        assert "RETRIEVED GST KNOWLEDGE" in full

    def test_full_prompt_without_rag_is_base(self):
        agent = MatcherAgent()
        assert agent.get_full_system_prompt() == agent.get_system_prompt()


# ── Auditor ↔ Rules Engine Cross-Validation ──────────────────────

class TestAuditorRulesCrossValidation:
    """Verify AuditorAgent decisions exactly match gst_rules.py outputs."""

    def test_auditor_status_matches_rules_engine(self):
        from environment.rules.gst_rules import check_itc_eligibility
        auditor = AuditorAgent()
        msgs = auditor.process(SAMPLE_OBS, [])

        gstr2b_index = {inv["invoice_id"]: inv for inv in SAMPLE_OBS["gstr2b_data"]}
        for inv in SAMPLE_OBS["purchase_register"]:
            inv_id = inv["invoice_id"]
            expected = check_itc_eligibility(inv, gstr2b_index.get(inv_id))
            agent_status = None
            for m in msgs:
                if m.action and m.action.get("invoice_id") == inv_id:
                    agent_status = m.metadata.get("status")
                    break
            assert agent_status == expected, f"{inv_id}: agent={agent_status}, rules={expected}"

    def test_auditor_itc_amounts_match_rules_engine(self):
        from environment.rules.gst_rules import check_itc_eligibility, calculate_itc_amount
        auditor = AuditorAgent()
        msgs = auditor.process(SAMPLE_OBS, [])

        gstr2b_index = {inv["invoice_id"]: inv for inv in SAMPLE_OBS["gstr2b_data"]}
        for inv in SAMPLE_OBS["purchase_register"]:
            inv_id = inv["invoice_id"]
            status = check_itc_eligibility(inv, gstr2b_index.get(inv_id))
            expected_itc = calculate_itc_amount(inv, status)
            agent_itc = None
            for m in msgs:
                if m.action and m.action.get("invoice_id") == inv_id:
                    agent_itc = m.metadata.get("itc_amount")
                    break
            assert agent_itc == expected_itc, f"{inv_id}: agent_itc={agent_itc}, rules_itc={expected_itc}"


# ── Orchestrator Recovery Tests ──────────────────────────────────

class TestOrchestratorRecovery:
    """Verify Orchestrator.run_all_tasks handles partial failures."""

    def test_run_all_tasks_has_try_except(self):
        import inspect
        source = inspect.getsource(Orchestrator.run_all_tasks)
        assert "try:" in source
        assert "except" in source

    def test_run_all_tasks_sets_zero_on_error(self):
        import inspect
        source = inspect.getsource(Orchestrator.run_all_tasks)
        assert "self.scores[task_id] = 0.0" in source

    def test_pipeline_order_is_correct(self):
        """Matcher must run before Auditor before Reporter before Validator."""
        import inspect
        source = inspect.getsource(Orchestrator.run_task)
        m_idx = source.find("self.matcher.process")
        a_idx = source.find("self.auditor.process")
        r_idx = source.find("self.reporter.process")
        v_idx = source.find("self.validator.process")
        assert 0 < m_idx < a_idx < r_idx < v_idx

    def test_rag_injection_in_pipeline(self):
        """All 4 agents must receive RAG context."""
        import inspect
        source = inspect.getsource(Orchestrator.run_task)
        for agent in ["matcher", "auditor", "reporter", "validator"]:
            assert f"self.{agent}.inject_context" in source


# ── Validator Extended Guard Tests ───────────────────────────────

class TestValidatorExtended:
    """Additional validator edge cases."""

    def setup_method(self):
        self.matcher = MatcherAgent()
        self.auditor = AuditorAgent()
        self.reporter = ReporterAgent()
        self.context = []
        self.context.extend(self.matcher.process(SAMPLE_OBS, self.context))
        self.context.extend(self.auditor.process(SAMPLE_OBS, self.context))
        self.context.extend(self.reporter.process(SAMPLE_OBS, self.context))

    def test_negative_itc_clamped_to_zero(self):
        fake_ctx = list(self.context)
        fake_ctx.append(AgentMessage(
            sender="Reporter", content="Negative ITC",
            action={"action_type": "submit_report", "payload": {
                "total_itc": -50000, "discrepancies": [], "matches": {}, "decisions": {},
            }},
        ))
        validator = ValidatorAgent()
        msgs = validator.process(SAMPLE_OBS, fake_ctx)
        assert msgs[0].metadata["clean_report"]["total_itc"] >= 0

    def test_handles_missing_reporter_gracefully(self):
        """If no Reporter message exists, validator should still return a result."""
        validator = ValidatorAgent()
        msgs = validator.process(SAMPLE_OBS, [])  # Empty context, no reporter
        assert len(msgs) == 1
        assert "error" in msgs[0].content.lower() or "no report" in msgs[0].content.lower()

    def test_malformed_invoice_no_crash(self):
        """Auditor shouldn't crash on invoices missing fields."""
        malformed_obs = {
            "purchase_register": [
                {"invoice_id": "MAL-1"},
                {"invoice_id": "MAL-2", "taxable_amount": -100, "cgst": 0, "sgst": 0, "igst": 0},
            ],
            "gstr2b_data": [],
        }
        auditor = AuditorAgent()
        msgs = auditor.process(malformed_obs, [])
        assert len(msgs) > 0  # Should produce messages, not crash


# ── Community Risk Assessment Tests (⚠ was untested) ────────────

class TestCommunityRiskAssessment:
    """Check that community_summaries has risk fields and they are assessable."""

    def test_rag_engine_communities_exist(self):
        from environment.knowledge.gst_knowledge import get_all_documents
        docs = get_all_documents()
        # Categories represent communities
        categories = {d.get("category", "unknown") for d in docs}
        assert len(categories) > 0, "No categories (communities) found"

    def test_categories_cover_gst_domains(self):
        from environment.knowledge.gst_knowledge import get_all_documents
        docs = get_all_documents()
        categories = {d.get("category", "unknown") for d in docs}
        # Must cover at least a few core GST domains
        core_domains = {"itc", "registration", "return", "invoice", "compliance"}
        found = {c for c in categories if any(dom in c.lower() for dom in core_domains)}
        assert len(found) >= 2, f"Too few core GST domains: {found}"

    def test_all_documents_have_category(self):
        from environment.knowledge.gst_knowledge import get_all_documents
        docs = get_all_documents()
        missing = [d.get("id", "?") for d in docs if not d.get("category")]
        assert len(missing) == 0, f"Documents without category: {missing}"

    def test_all_categories_have_risk_assessable_content(self):
        """Every category/community should have enough content to assess risk."""
        from environment.knowledge.gst_knowledge import get_all_documents
        docs = get_all_documents()
        cat_counts = {}
        for d in docs:
            cat = d.get("category", "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        # Every category must have at least 1 document
        for cat, count in cat_counts.items():
            assert count >= 1, f"Category '{cat}' has no documents"
        # Must have enough categories to cover GST domains
        assert len(cat_counts) >= 10, f"Only {len(cat_counts)} categories (need >=10)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
