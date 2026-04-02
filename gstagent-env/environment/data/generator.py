"""
Synthetic GST data generator with deterministic seeded output.

Generates realistic fake invoices and GSTR-2B data for testing AI agents.
Uses seed control so every run produces identical output (Fix #6).
GSTINs use Luhn checksum but are designed to FAIL real GSTN validation (Fix #5).
HSN codes sourced from actual tariff list (Fix #7).
All dates within same fiscal month (Fix #8).
"""

from __future__ import annotations

import json
import random
import string
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

from environment.config import SEED

# Deterministic seeds — same output every run
random.seed(SEED)
fake = Faker("en_IN")
Faker.seed(SEED)

# Load real HSN codes
HSN_FILE = Path(__file__).parent / "hsn_codes.json"
with open(HSN_FILE) as f:
    HSN_CODES = json.load(f)

# Indian state codes used in GSTINs
STATE_CODES = ["27", "29", "07", "33", "06", "24", "19", "36", "32", "09"]


def _luhn_checksum(digits: str) -> str:
    """Compute a Luhn check character for GSTIN-like strings."""
    charset = string.digits + string.ascii_uppercase
    total = 0
    for i, ch in enumerate(digits):
        val = charset.index(ch.upper()) if ch.upper() in charset else 0
        if i % 2 == 1:
            val *= 2
        total += val // len(charset) + val % len(charset)
    check = (len(charset) - (total % len(charset))) % len(charset)
    return charset[check]


def generate_gstin(state_code: str | None = None) -> str:
    """
    Generate a fake but validly formatted GSTIN.

    Format: 2-digit state + 10-char PAN-like + 1 digit + Z + check char
    These will FAIL real GSTN validation because the PAN portion is random.
    """
    sc = state_code or random.choice(STATE_CODES)
    # PAN-like: 5 uppercase + 4 digits + 1 uppercase
    pan = (
        "".join(random.choices(string.ascii_uppercase, k=5))
        + "".join(random.choices(string.digits, k=4))
        + random.choice(string.ascii_uppercase)
    )
    entity_num = str(random.randint(1, 9))
    base = f"{sc}{pan}{entity_num}Z"
    check = _luhn_checksum(base)
    return f"{base}{check}"


def generate_invoice(
    supplier_gstin: str | None = None,
    buyer_gstin: str | None = None,
    invoice_num: int = 1,
    base_date: date | None = None,
) -> dict:
    """
    Generate a single realistic GST invoice.

    All invoices within same fiscal month for realism (Fix #8).
    """
    if base_date is None:
        base_date = date(2025, 3, 1)  # March 2025 — a typical MSME filing month

    supplier = supplier_gstin or generate_gstin()
    buyer = buyer_gstin or generate_gstin()

    # Random date within the same month
    day_offset = random.randint(0, 27)
    inv_date = base_date + timedelta(days=day_offset)

    # Realistic MSME amounts: ₹5,000 to ₹5,00,000
    taxable = round(random.uniform(5000, 500000), 2)

    # Determine if intra-state or inter-state
    is_intra = supplier[:2] == buyer[:2]
    hsn_entry = random.choice(HSN_CODES)

    if is_intra:
        gst_rate = random.choice([0.025, 0.06, 0.09, 0.14])  # half of 5%, 12%, 18%, 28%
        cgst = round(taxable * gst_rate, 2)
        sgst = cgst
        igst = 0.0
    else:
        gst_rate = random.choice([0.05, 0.12, 0.18, 0.28])
        cgst = 0.0
        sgst = 0.0
        igst = round(taxable * gst_rate, 2)

    return {
        "invoice_id": f"INV-{invoice_num:04d}",
        "supplier_gstin": supplier,
        "buyer_gstin": buyer,
        "invoice_date": inv_date.isoformat(),
        "taxable_amount": taxable,
        "cgst": cgst,
        "sgst": sgst,
        "igst": igst,
        "hsn_code": hsn_entry["code"],
        "item_description": hsn_entry["description"],
    }


def generate_invoice_batch(
    count: int,
    buyer_gstin: str | None = None,
    base_date: date | None = None,
) -> list[dict]:
    """Generate a batch of invoices from various suppliers to one buyer."""
    buyer = buyer_gstin or generate_gstin()
    suppliers = [generate_gstin() for _ in range(max(3, count // 3))]

    invoices = []
    for i in range(count):
        supplier = random.choice(suppliers)
        inv = generate_invoice(
            supplier_gstin=supplier,
            buyer_gstin=buyer,
            invoice_num=i + 1,
            base_date=base_date,
        )
        invoices.append(inv)

    return invoices


def generate_gstr2b(
    invoice_list: list[dict],
    missing_rate: float = 0.3,
    mismatch_rate: float = 0.1,
) -> list[dict]:
    """
    Simulate GSTR-2B by introducing controlled noise.

    - missing_rate: fraction of invoices randomly removed (supplier didn't file)
    - mismatch_rate: fraction of remaining invoices with altered amounts
    """
    import copy

    gstr2b = []
    for inv in invoice_list:
        # Randomly skip some invoices (simulate non-filing)
        if random.random() < missing_rate:
            continue

        record = copy.deepcopy(inv)

        # Randomly alter amounts (simulate keying errors)
        if random.random() < mismatch_rate:
            # Alter by 5% to 30%
            factor = random.uniform(0.7, 1.3)
            record["taxable_amount"] = round(record["taxable_amount"] * factor, 2)
            # Recalculate taxes
            if record["igst"] > 0:
                rate = record["igst"] / inv["taxable_amount"] if inv["taxable_amount"] else 0
                record["igst"] = round(record["taxable_amount"] * rate, 2)
            else:
                rate = record["cgst"] / inv["taxable_amount"] if inv["taxable_amount"] else 0
                record["cgst"] = round(record["taxable_amount"] * rate, 2)
                record["sgst"] = record["cgst"]

        gstr2b.append(record)

    return gstr2b


def save_dataset(invoices: list[dict], gstr2b: list[dict], filepath: str) -> None:
    """Save a paired dataset as JSON."""
    data = {
        "purchase_register": invoices,
        "gstr2b": gstr2b,
        "metadata": {
            "total_invoices": len(invoices),
            "gstr2b_count": len(gstr2b),
            "missing_count": len(invoices) - len(gstr2b),
            "seed": SEED,
        },
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    data_dir = Path(__file__).parent

    # Reset seed for reproducibility
    random.seed(SEED)
    Faker.seed(SEED)

    buyer = generate_gstin("27")  # Maharashtra buyer

    # Easy: 10 invoices, ~2 missing, no amount mismatches
    random.seed(SEED)
    easy_invoices = generate_invoice_batch(10, buyer_gstin=buyer)
    easy_gstr2b = generate_gstr2b(easy_invoices, missing_rate=0.2, mismatch_rate=0.0)
    save_dataset(easy_invoices, easy_gstr2b, str(data_dir / "invoices_easy.json"))
    print(f"Easy: {len(easy_invoices)} invoices, {len(easy_gstr2b)} in GSTR-2B")

    # Medium: 20 invoices, ~6 missing, some amount mismatches
    random.seed(SEED + 1)
    med_invoices = generate_invoice_batch(20, buyer_gstin=buyer)
    med_gstr2b = generate_gstr2b(med_invoices, missing_rate=0.3, mismatch_rate=0.1)
    save_dataset(med_invoices, med_gstr2b, str(data_dir / "invoices_medium.json"))
    print(f"Medium: {len(med_invoices)} invoices, {len(med_gstr2b)} in GSTR-2B")

    # Hard: 50 invoices, ~15 missing, mixed mismatch types
    random.seed(SEED + 2)
    hard_invoices = generate_invoice_batch(50, buyer_gstin=buyer)
    hard_gstr2b = generate_gstr2b(hard_invoices, missing_rate=0.3, mismatch_rate=0.15)
    save_dataset(hard_invoices, hard_gstr2b, str(data_dir / "invoices_hard.json"))
    print(f"Hard: {len(hard_invoices)} invoices, {len(hard_gstr2b)} in GSTR-2B")

    print("\nAll datasets generated successfully.")
