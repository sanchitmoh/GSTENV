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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
