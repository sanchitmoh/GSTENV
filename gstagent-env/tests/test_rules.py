"""
Tests for the GST Rules Engine.

Covers: eligible, partial, ineligible paths, edge cases, recommended actions.
"""

import pytest
from environment.rules.gst_rules import (
    check_itc_eligibility,
    compute_total_eligible_itc,
    get_recommended_action,
    calculate_itc_amount,
)


class TestITCEligibility:
    def test_eligible_exact_match(self):
        inv = {"taxable_amount": 10000}
        gstr = {"taxable_amount": 10000}
        assert check_itc_eligibility(inv, gstr) == "eligible"

    def test_eligible_within_5pct(self):
        inv = {"taxable_amount": 10000}
        gstr = {"taxable_amount": 10400}  # 4% diff
        assert check_itc_eligibility(inv, gstr) == "eligible"

    def test_partial_between_5_and_20pct(self):
        inv = {"taxable_amount": 10000}
        gstr = {"taxable_amount": 8500}  # 15% diff
        assert check_itc_eligibility(inv, gstr) == "partial"

    def test_ineligible_over_20pct(self):
        inv = {"taxable_amount": 10000}
        gstr = {"taxable_amount": 7000}  # 30% diff
        assert check_itc_eligibility(inv, gstr) == "ineligible"

    def test_ineligible_missing_gstr2b(self):
        inv = {"taxable_amount": 10000}
        assert check_itc_eligibility(inv, None) == "ineligible"

    def test_ineligible_zero_amount(self):
        inv = {"taxable_amount": 0}
        gstr = {"taxable_amount": 1000}
        assert check_itc_eligibility(inv, gstr) == "ineligible"


class TestRecommendedActions:
    def test_eligible_action(self):
        action = get_recommended_action("eligible")
        assert "full ITC" in action.lower() or "claim full" in action.lower()

    def test_ineligible_action(self):
        action = get_recommended_action("ineligible")
        assert "follow up" in action.lower() or "supplier" in action.lower()

    def test_partial_action(self):
        action = get_recommended_action("partial")
        assert "matched amount" in action.lower() or "reverse" in action.lower()

    def test_unknown_status(self):
        action = get_recommended_action("unknown")
        assert "unknown" in action.lower()


class TestITCAmount:
    def test_eligible_full_tax(self):
        inv = {"cgst": 900, "sgst": 900, "igst": 0}
        assert calculate_itc_amount(inv, "eligible") == 1800

    def test_ineligible_zero(self):
        inv = {"cgst": 900, "sgst": 900, "igst": 0}
        assert calculate_itc_amount(inv, "ineligible") == 0.0

    def test_partial_80pct(self):
        inv = {"cgst": 1000, "sgst": 1000, "igst": 0}
        amount = calculate_itc_amount(inv, "partial")
        assert amount == 1600.0  # 80% of 2000


class TestTotalITC:
    def test_compute_total(self):
        invoices = [
            {"invoice_id": "INV-0001", "taxable_amount": 10000, "cgst": 900, "sgst": 900, "igst": 0, "supplier_gstin": "27AAA"},
            {"invoice_id": "INV-0002", "taxable_amount": 20000, "cgst": 0, "sgst": 0, "igst": 3600, "supplier_gstin": "29BBB"},
        ]
        gstr2b_index = {
            "INV-0001": {"invoice_id": "INV-0001", "taxable_amount": 10000},
        }
        total, details = compute_total_eligible_itc(invoices, gstr2b_index)
        assert total > 0
        assert len(details) == 2
        assert details[0]["status"] == "eligible"
        assert details[1]["status"] == "ineligible"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
