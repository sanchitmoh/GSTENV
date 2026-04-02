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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
