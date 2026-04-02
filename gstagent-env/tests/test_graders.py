"""
Unit tests for all 3 graders.

Covers: perfect input, wrong input, empty input, determinism, bounds.
"""

import pytest
from environment.graders import grader_full_recon, grader_invoice_match, grader_itc_audit


# ── Task 1: Invoice Match Grader ─────────────────────────────────────


class TestGraderInvoiceMatch:
    def setup_method(self):
        self.ground_truth = {
            "INV-0001": "present",
            "INV-0002": "missing",
            "INV-0003": "present",
            "INV-0004": "missing",
        }

    def test_perfect_score(self):
        agent = dict(self.ground_truth)
        score = grader_invoice_match.grade(agent, self.ground_truth)
        assert score == 1.0

    def test_all_wrong(self):
        agent = {k: ("present" if v == "missing" else "missing") for k, v in self.ground_truth.items()}
        score = grader_invoice_match.grade(agent, self.ground_truth)
        assert score < 1.0

    def test_empty_input(self):
        assert grader_invoice_match.grade({}, self.ground_truth) == 0.0
        assert grader_invoice_match.grade(self.ground_truth, {}) == 0.0

    def test_deterministic(self):
        agent = {"INV-0001": "present", "INV-0002": "present"}
        s1 = grader_invoice_match.grade(agent, self.ground_truth)
        s2 = grader_invoice_match.grade(agent, self.ground_truth)
        assert s1 == s2

    def test_score_bounds(self):
        agent = {"INV-0001": "missing", "INV-0099": "present"}
        score = grader_invoice_match.grade(agent, self.ground_truth)
        assert 0.0 <= score <= 1.0


# ── Task 2: ITC Audit Grader ─────────────────────────────────────────


class TestGraderItcAudit:
    def setup_method(self):
        self.ground_truth = {
            "INV-0001": "eligible",
            "INV-0002": "ineligible",
            "INV-0003": "partial",
        }

    def test_perfect_score(self):
        agent = dict(self.ground_truth)
        score = grader_itc_audit.grade(agent, self.ground_truth)
        assert score == 1.0

    def test_all_wrong(self):
        agent = {"INV-0001": "ineligible", "INV-0002": "eligible", "INV-0003": "eligible"}
        score = grader_itc_audit.grade(agent, self.ground_truth)
        assert score < 1.0

    def test_empty_input(self):
        assert grader_itc_audit.grade({}, self.ground_truth) == 0.0

    def test_deterministic(self):
        agent = {"INV-0001": "eligible", "INV-0002": "eligible", "INV-0003": "eligible"}
        s1 = grader_itc_audit.grade(agent, self.ground_truth)
        s2 = grader_itc_audit.grade(agent, self.ground_truth)
        assert s1 == s2

    def test_with_amounts(self):
        agent_dec = dict(self.ground_truth)
        agent_amt = {"INV-0001": 900.0, "INV-0002": 0.0, "INV-0003": 400.0}
        truth_amt = {"INV-0001": 1000.0, "INV-0002": 0.0, "INV-0003": 500.0}
        score = grader_itc_audit.grade(agent_dec, self.ground_truth, agent_amt, truth_amt)
        assert 0.0 <= score <= 1.0


# ── Task 3: Full Recon Grader ─────────────────────────────────────────


class TestGraderFullRecon:
    def setup_method(self):
        self.ground_truth = {
            "total_itc": 50000.0,
            "discrepancies": {
                "INV-0002": {"status": "ineligible", "action": "Follow up with supplier"},
                "INV-0005": {"status": "partial", "action": "Claim only matched amount"},
            },
            "all_invoice_ids": {"INV-0001", "INV-0002", "INV-0003", "INV-0004", "INV-0005"},
        }

    def test_perfect_report(self):
        report = {
            "total_itc": 50000.0,
            "discrepancies": [
                {"invoice_id": "INV-0002", "status": "ineligible", "action": "Follow up with supplier"},
                {"invoice_id": "INV-0005", "status": "partial", "action": "Claim only matched amount"},
            ],
        }
        result = grader_full_recon.grade(report, self.ground_truth, 5, 20)
        assert result["total"] > 0.8

    def test_empty_input(self):
        result = grader_full_recon.grade({}, self.ground_truth)
        assert result["total"] == 0.0

    def test_hallucination_penalty_bounded(self):
        """Fix #16: even 10 fake IDs should only incur max 0.2 penalty."""
        report = {
            "total_itc": 0,
            "discrepancies": [
                {"invoice_id": f"FAKE-{i}", "status": "bad", "action": "bad"}
                for i in range(10)
            ],
        }
        result = grader_full_recon.grade(report, self.ground_truth, 20, 20)
        assert result["hallucination_penalty"] <= 0.2
        assert result["total"] >= 0.0

    def test_itc_close_not_zero(self):
        """Fix #17: 1 rupee off should NOT give 0.0."""
        report = {
            "total_itc": 49999.0,
            "discrepancies": [],
        }
        result = grader_full_recon.grade(report, self.ground_truth, 10, 20)
        assert result["itc_accuracy"] > 0.99

    def test_efficiency_bonus(self):
        """Fix #18: using fewer steps should give bonus."""
        report = {"total_itc": 50000.0, "discrepancies": []}
        fast = grader_full_recon.grade(report, self.ground_truth, 2, 20)
        slow = grader_full_recon.grade(report, self.ground_truth, 18, 20)
        assert fast["efficiency_bonus"] > slow["efficiency_bonus"]

    def test_deterministic(self):
        report = {"total_itc": 30000.0, "discrepancies": []}
        r1 = grader_full_recon.grade(report, self.ground_truth, 10, 20)
        r2 = grader_full_recon.grade(report, self.ground_truth, 10, 20)
        assert r1["total"] == r2["total"]

    def test_score_always_bounded(self):
        """Score must always be in [0.0, 1.0]."""
        # Worst case: wrong ITC + all hallucinated + max steps
        report = {
            "total_itc": 999999.0,
            "discrepancies": [
                {"invoice_id": f"GHOST-{i}", "status": "x", "action": "x"}
                for i in range(20)
            ],
        }
        result = grader_full_recon.grade(report, self.ground_truth, 20, 20)
        assert 0.0 <= result["total"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
