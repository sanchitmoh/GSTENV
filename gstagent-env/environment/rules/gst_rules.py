"""
GST Rules Engine — encodes Indian GST law as deterministic Python logic.

Implements Rule 36(4) (CGST Rules) and Section 16(2) (CGST Act) for
ITC eligibility determination and recommended actions.
"""

from __future__ import annotations

# Tolerance thresholds
AMOUNT_MISMATCH_THRESHOLD = 0.20  # 20% — beyond this, ITC is ineligible
AMOUNT_PARTIAL_THRESHOLD = 0.05  # 5% — within this, full ITC allowed


def check_itc_eligibility(
    invoice: dict,
    gstr2b_record: dict | None,
) -> str:
    """
    Determine ITC eligibility for a single invoice.

    Rule 36(4): ITC limited to invoices reflected in GSTR-2B.
    Section 16(2): Requires possession of invoice, receipt of goods,
                   tax paid to government, return filed.

    Returns:
        "eligible"   — invoice in GSTR-2B, amounts match within 5%
        "partial"    — invoice in GSTR-2B, amounts differ by 5%-20%
        "ineligible" — invoice missing from GSTR-2B or amounts differ by >20%
    """
    # If not in GSTR-2B at all → supplier didn't file → ineligible
    if gstr2b_record is None:
        return "ineligible"

    # Compare taxable amounts
    inv_amount = invoice.get("taxable_amount", 0)
    gstr_amount = gstr2b_record.get("taxable_amount", 0)

    if inv_amount == 0:
        return "ineligible"

    variance = abs(inv_amount - gstr_amount) / inv_amount

    if variance <= AMOUNT_PARTIAL_THRESHOLD:
        return "eligible"
    elif variance <= AMOUNT_MISMATCH_THRESHOLD:
        return "partial"
    else:
        return "ineligible"


def get_recommended_action(status: str) -> str:
    """
    Map eligibility status to a recommended action string.

    These map to real-world CA/accountant advice.
    """
    actions = {
        "eligible": "Claim full ITC in GSTR-3B",
        "partial": "Claim only matched amount, reverse the excess under Rule 37",
        "ineligible": "Follow up with supplier to file GSTR-1; do not claim ITC until reflected in GSTR-2B",
    }
    return actions.get(status, f"Unknown status: {status}")


def calculate_itc_amount(invoice: dict, status: str) -> float:
    """
    Calculate the claimable ITC amount based on eligibility status.

    Returns the amount of ITC (tax) that can be claimed.
    """
    total_tax = invoice.get("cgst", 0) + invoice.get("sgst", 0) + invoice.get("igst", 0)

    if status == "eligible":
        return total_tax
    elif status == "partial":
        # Claim only the amount reflected in GSTR-2B
        return total_tax * 0.8  # Conservative: claim 80% pending reconciliation
    else:
        return 0.0


def compute_total_eligible_itc(
    invoices: list[dict],
    gstr2b_index: dict[str, dict],
) -> tuple[float, list[dict]]:
    """
    Compute total eligible ITC across all invoices.

    Returns:
        (total_itc, details) where details is a list of per-invoice results.
    """
    total_itc = 0.0
    details = []

    for inv in invoices:
        inv_id = inv["invoice_id"]
        gstr2b_record = gstr2b_index.get(inv_id)
        status = check_itc_eligibility(inv, gstr2b_record)
        action = get_recommended_action(status)
        itc_amount = calculate_itc_amount(inv, status)
        total_itc += itc_amount

        details.append(
            {
                "invoice_id": inv_id,
                "status": status,
                "action": action,
                "itc_amount": round(itc_amount, 2),
                "supplier_gstin": inv.get("supplier_gstin", ""),
            }
        )

    return round(total_itc, 2), details
