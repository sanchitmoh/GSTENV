"""
Tests for the generator module.

Covers: seed determinism, GSTIN format, HSN validity, date range, GSTR-2B noise.
"""

import json
import random
from datetime import date

import pytest
from environment.data.generator import (
    SEED,
    generate_gstin,
    generate_gstr2b,
    generate_invoice,
    generate_invoice_batch,
)


class TestGSTINGeneration:
    def test_format_15_chars(self):
        gstin = generate_gstin("27")
        assert len(gstin) == 15

    def test_starts_with_state_code(self):
        gstin = generate_gstin("29")
        assert gstin[:2] == "29"

    def test_seed_determinism(self):
        random.seed(SEED)
        g1 = generate_gstin()
        random.seed(SEED)
        g2 = generate_gstin()
        assert g1 == g2

    def test_different_each_call(self):
        random.seed(99)
        g1 = generate_gstin()
        g2 = generate_gstin()
        assert g1 != g2


class TestInvoiceGeneration:
    def test_invoice_has_required_fields(self):
        random.seed(SEED)
        inv = generate_invoice(invoice_num=1)
        required = [
            "invoice_id", "supplier_gstin", "buyer_gstin",
            "invoice_date", "taxable_amount", "cgst", "sgst",
            "igst", "hsn_code", "item_description",
        ]
        for field in required:
            assert field in inv, f"Missing field: {field}"

    def test_invoice_id_format(self):
        random.seed(SEED)
        inv = generate_invoice(invoice_num=42)
        assert inv["invoice_id"] == "INV-0042"

    def test_same_month_dates(self):
        random.seed(SEED)
        base = date(2025, 3, 1)
        invoices = generate_invoice_batch(10, base_date=base)
        for inv in invoices:
            d = date.fromisoformat(inv["invoice_date"])
            assert d.year == 2025
            assert d.month == 3

    def test_realistic_amounts(self):
        random.seed(SEED)
        inv = generate_invoice()
        assert 5000 <= inv["taxable_amount"] <= 500000

    def test_real_hsn_codes(self):
        with open("environment/data/hsn_codes.json") as f:
            valid_codes = {h["code"] for h in json.load(f)}
        random.seed(SEED)
        inv = generate_invoice()
        assert inv["hsn_code"] in valid_codes


class TestGSTR2BGeneration:
    def test_missing_invoices(self):
        random.seed(SEED)
        invoices = generate_invoice_batch(100)
        random.seed(SEED + 10)
        gstr2b = generate_gstr2b(invoices, missing_rate=0.3, mismatch_rate=0.0)
        assert len(gstr2b) < len(invoices)

    def test_zero_missing_rate(self):
        random.seed(SEED)
        invoices = generate_invoice_batch(10)
        gstr2b = generate_gstr2b(invoices, missing_rate=0.0, mismatch_rate=0.0)
        assert len(gstr2b) == len(invoices)

    def test_amount_mismatches_present(self):
        random.seed(SEED)
        invoices = generate_invoice_batch(100)
        random.seed(SEED + 20)
        gstr2b = generate_gstr2b(invoices, missing_rate=0.0, mismatch_rate=1.0)
        mismatch_count = sum(
            1 for g, i in zip(gstr2b, invoices)
            if abs(g["taxable_amount"] - i["taxable_amount"]) > 0.01
        )
        assert mismatch_count > 0


# ── GSTIN Checksum Validation (was ✓ found but not deeply tested) ─

class TestGSTINChecksum:
    """Verify _luhn_checksum produces valid check characters."""

    def test_checksum_consistency_100_gstins(self):
        """Every generated GSTIN must pass its own checksum."""
        from environment.data.generator import _luhn_checksum
        random.seed(42)
        for _ in range(100):
            gstin = generate_gstin()
            base = gstin[:14]
            expected = _luhn_checksum(base)
            assert gstin[14] == expected, f"Checksum fail: {gstin}"

    def test_gstin_format_structure(self):
        """GSTIN = 2 digits + 5 letters + 4 digits + 1 letter + 1 digit + Z + check."""
        random.seed(SEED)
        for _ in range(20):
            g = generate_gstin()
            assert len(g) == 15
            assert g[:2].isdigit()        # state code
            assert g[2:7].isalpha()       # PAN letters
            assert g[7:11].isdigit()      # PAN digits
            assert g[11].isalpha()         # PAN letter
            assert g[12].isdigit()         # entity number
            assert g[13] == "Z"            # fixed Z

    def test_all_state_codes_valid(self):
        from environment.data.generator import STATE_CODES
        random.seed(SEED)
        for sc in STATE_CODES:
            g = generate_gstin(state_code=sc)
            assert g[:2] == sc


# ── Invoice Batch Mismatch Coverage ──────────────────────────────

class TestInvoiceMismatchTypes:
    """Verify generate_invoice_batch covers amount, missing, and tax type mismatches."""

    def test_amount_mismatches_generated(self):
        random.seed(SEED)
        invoices = generate_invoice_batch(50)
        random.seed(SEED + 20)
        gstr2b = generate_gstr2b(invoices, missing_rate=0.0, mismatch_rate=1.0)
        mismatched = sum(
            1 for g, i in zip(gstr2b, invoices)
            if abs(g["taxable_amount"] - i["taxable_amount"]) > 0.01
        )
        assert mismatched > 0

    def test_missing_invoices_generated(self):
        random.seed(SEED)
        invoices = generate_invoice_batch(50)
        random.seed(SEED + 30)
        gstr2b = generate_gstr2b(invoices, missing_rate=0.5, mismatch_rate=0.0)
        assert len(gstr2b) < len(invoices)

    def test_both_intra_and_inter_state(self):
        """Batch must contain both CGST+SGST (intra) and IGST (inter) invoices."""
        random.seed(SEED)
        batch = generate_invoice_batch(50)
        has_intra = any(inv["cgst"] > 0 for inv in batch)
        has_inter = any(inv["igst"] > 0 for inv in batch)
        assert has_intra, "No intra-state invoices"
        assert has_inter, "No inter-state invoices"

    def test_tax_recalculated_on_mismatch(self):
        """When amount is altered, taxes should change proportionally."""
        random.seed(SEED)
        invoices = generate_invoice_batch(10)
        random.seed(SEED + 40)
        gstr2b = generate_gstr2b(invoices, missing_rate=0.0, mismatch_rate=1.0)
        for g, i in zip(gstr2b, invoices):
            if abs(g["taxable_amount"] - i["taxable_amount"]) > 0.01:
                # At least one tax component should differ
                tax_changed = (
                    g["cgst"] != i["cgst"] or
                    g["sgst"] != i["sgst"] or
                    g["igst"] != i["igst"]
                )
                assert tax_changed, f"Tax not recalculated for {g['invoice_id']}"
                break  # One example is sufficient

    def test_gstr2b_seed_determinism(self):
        """Same seed → identical GSTR-2B output."""
        random.seed(SEED)
        invoices = generate_invoice_batch(20)
        random.seed(SEED + 100)
        g1 = generate_gstr2b(invoices, missing_rate=0.3, mismatch_rate=0.1)
        random.seed(SEED + 100)
        g2 = generate_gstr2b(invoices, missing_rate=0.3, mismatch_rate=0.1)
        assert len(g1) == len(g2)
        for a, b in zip(g1, g2):
            assert a["invoice_id"] == b["invoice_id"]
            assert a["taxable_amount"] == b["taxable_amount"]


# ── Knowledge Base Coverage ──────────────────────────────────────

class TestKnowledgeBaseCoverage:
    """Confirm gst_knowledge.py covers all required GST topics."""

    def _all_text(self):
        from environment.knowledge.gst_knowledge import get_all_documents
        docs = get_all_documents()
        return docs, " ".join(
            (d.get("title", "") + " " + d.get("content", "")).lower()
            for d in docs
        )

    def test_covers_cgst(self):
        _, text = self._all_text()
        assert "cgst" in text

    def test_covers_sgst(self):
        _, text = self._all_text()
        assert "sgst" in text

    def test_covers_igst(self):
        _, text = self._all_text()
        assert "igst" in text

    def test_covers_rcm(self):
        _, text = self._all_text()
        assert "reverse charge" in text or "rcm" in text

    def test_covers_composition_scheme(self):
        _, text = self._all_text()
        assert "composition" in text

    def test_has_rcm_category(self):
        from environment.knowledge.gst_knowledge import get_all_documents
        cats = {d.get("category") for d in get_all_documents()}
        assert "rcm" in cats

    def test_has_composition_category(self):
        from environment.knowledge.gst_knowledge import get_all_documents
        cats = {d.get("category") for d in get_all_documents()}
        assert "composition" in cats

    def test_minimum_document_count(self):
        from environment.knowledge.gst_knowledge import get_all_documents
        docs = get_all_documents()
        assert len(docs) >= 30, f"Only {len(docs)} docs (need >=30)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
