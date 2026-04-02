"""
GST Knowledge Base — curated domain knowledge for RAG.

Contains CBIC circulars, Rule 36(4) amendments, Section 16(2) details,
and practical CA/accountant guidance. This is the "knowledge corpus"
that gets embedded and retrieved during agent reasoning.
"""

from __future__ import annotations

# Each document has: id, title, content, source, category
GST_KNOWLEDGE_BASE: list[dict] = [
    {
        "id": "CBIC-Circular-170",
        "title": "Circular 170/02/2022 — ITC reconciliation with GSTR-2B",
        "content": (
            "CBIC Circular No. 170/02/2022-GST dated 06.07.2022 clarifies that "
            "ITC can only be availed on the basis of GSTR-2B. Taxpayers must "
            "reconcile their purchase register with GSTR-2B before filing GSTR-3B. "
            "Any ITC claimed in excess of GSTR-2B must be reversed. The circular "
            "emphasizes that GSTR-2B is the sole basis for ITC availment from "
            "1st January 2022 onwards."
        ),
        "source": "CBIC Circular 170/02/2022-GST",
        "category": "itc_rules",
    },
    {
        "id": "CBIC-Circular-183",
        "title": "Circular 183/15/2022 — Provisional ITC under Rule 36(4)",
        "content": (
            "Rule 36(4) was amended to restrict provisional ITC. Earlier, "
            "taxpayers could claim 5% provisional ITC on invoices not reflected "
            "in GSTR-2B. From January 2022, this provision was removed. Now, "
            "ITC is available only for invoices appearing in GSTR-2B. This "
            "change makes GSTR-2B reconciliation critical for every taxpayer. "
            "Suppliers must file GSTR-1 on time for buyers to claim ITC."
        ),
        "source": "CBIC Circular 183/15/2022-GST",
        "category": "itc_rules",
    },
    {
        "id": "Section-16-2",
        "title": "Section 16(2) — Conditions for ITC availment",
        "content": (
            "Input Tax Credit under Section 16(2) requires four conditions: "
            "(a) possession of tax invoice or debit note, "
            "(b) receipt of goods or services, "
            "(c) tax has actually been paid to the government by the supplier, "
            "(d) the taxpayer has filed the required return. "
            "All four conditions must be satisfied simultaneously. If the "
            "supplier has not filed GSTR-1, condition (c) is not met and ITC "
            "cannot be claimed even if the buyer has the invoice."
        ),
        "source": "CGST Act, Section 16(2)",
        "category": "legislation",
    },
    {
        "id": "Rule-36-4",
        "title": "Rule 36(4) — Restriction on ITC not reflected in GSTR-2B",
        "content": (
            "Rule 36(4) of CGST Rules states that ITC to be availed by a "
            "registered person in respect of invoices or debit notes not "
            "reflected in GSTR-2B shall not exceed the ITC available in "
            "respect of invoices that ARE reflected in GSTR-2B. The 5% "
            "provisional credit allowed earlier has been removed w.e.f. "
            "01.01.2022 via Notification No. 40/2021. Taxpayers must now "
            "ensure 100% matching with GSTR-2B before claiming ITC."
        ),
        "source": "CGST Rules, Rule 36(4), Notification 40/2021",
        "category": "rules",
    },
    {
        "id": "Rule-37",
        "title": "Rule 37 — Reversal of ITC for non-payment within 180 days",
        "content": (
            "If a buyer fails to pay the supplier within 180 days from the "
            "invoice date, the ITC claimed must be reversed and added to "
            "output tax liability along with interest under Section 50. "
            "The ITC can be reclaimed when payment is eventually made. "
            "This rule applies to all types of supplies — goods and services."
        ),
        "source": "CGST Rules, Rule 37",
        "category": "rules",
    },
    {
        "id": "Mismatch-Tolerance",
        "title": "Practical mismatch tolerance in GST reconciliation",
        "content": (
            "In practice, CAs and tax professionals follow these thresholds: "
            "Variance ≤5% between purchase register and GSTR-2B → claim full ITC. "
            "Variance between 5-20% → claim only the amount reflected in GSTR-2B, "
            "reverse the excess, and follow up with supplier. "
            "Variance >20% → do NOT claim ITC, immediately contact supplier "
            "to rectify their GSTR-1 filing. These thresholds are based on "
            "common audit practices and departmental guidelines."
        ),
        "source": "CA Practice Guidelines",
        "category": "practical",
    },
    {
        "id": "GSTR-2B-Auto-Generation",
        "title": "How GSTR-2B is auto-generated from suppliers' GSTR-1",
        "content": (
            "GSTR-2B is auto-generated on the 14th of every month based on "
            "suppliers' GSTR-1 filings by the 11th. It shows: "
            "1. Invoices filed by registered suppliers in their GSTR-1, "
            "2. ITC available and ITC not available (e.g., blocked credits), "
            "3. ITC reversed due to credit notes. "
            "If a supplier files late or amends their GSTR-1, the buyer's "
            "GSTR-2B will be updated in the next month's cycle. "
            "Common reasons for missing invoices in GSTR-2B: "
            "supplier not registered, late filing, wrong GSTIN entered, "
            "B2B invoice filed as B2C."
        ),
        "source": "GST Portal Documentation",
        "category": "process",
    },
    {
        "id": "MSME-Cash-Flow-Impact",
        "title": "ITC blocking impact on MSME cash flow",
        "content": (
            "Blocked ITC directly impacts MSME working capital. A typical "
            "MSME with ₹1 Cr annual turnover and 18% GST rate has ₹18L "
            "annual ITC. If 20-30% is blocked due to reconciliation issues, "
            "₹3.6-5.4L gets locked up. For small businesses operating on "
            "5-10% margins, this can mean the difference between survival "
            "and closure. Automated reconciliation reduces blockage from "
            "30% to under 5%, freeing up critical working capital."
        ),
        "source": "MSME Industry Report 2024",
        "category": "business_impact",
    },
    {
        "id": "HSN-Classification",
        "title": "HSN code classification for GST",
        "content": (
            "HSN (Harmonized System of Nomenclature) codes classify goods "
            "for taxation. Under GST: "
            "4-digit HSN required for turnover >₹5 Cr, "
            "6-digit HSN required for exports, "
            "8-digit HSN used for customs. "
            "Services use SAC (Services Accounting Code) under Chapter 99. "
            "Common HSN codes for MSMEs: 8471 (computers), 8517 (phones), "
            "9401 (chairs), 3004 (medicines), 9988 (IT services). "
            "Misclassification leads to wrong tax rate and ITC issues."
        ),
        "source": "CBIC HSN Classification Guide",
        "category": "classification",
    },
    {
        "id": "GSTIN-Format",
        "title": "GSTIN structure and validation",
        "content": (
            "GSTIN is a 15-character code: "
            "Positions 1-2: State code (e.g., 27=Maharashtra, 29=Karnataka) "
            "Positions 3-12: PAN of the entity "
            "Position 13: Entity number (1-9 for multiple registrations) "
            "Position 14: 'Z' by default "
            "Position 15: Check digit (Luhn algorithm) "
            "Common errors: wrong state code, transposed PAN digits. "
            "Always validate GSTIN format before processing invoices."
        ),
        "source": "GSTN Technical Documentation",
        "category": "technical",
    },
    {
        "id": "Reconciliation-Best-Practices",
        "title": "Monthly GST reconciliation best practices",
        "content": (
            "Step-by-step reconciliation process: "
            "1. Download GSTR-2B from GST portal (available 14th of each month) "
            "2. Export purchase register from accounting software "
            "3. Match by invoice number and supplier GSTIN "
            "4. Flag missing invoices — contact supplier to file GSTR-1 "
            "5. Flag amount mismatches — verify with purchase order "
            "6. Calculate eligible ITC after removing ineligible items "
            "7. File GSTR-3B with reconciled ITC amount "
            "8. Document mismatches for audit trail "
            "Automate steps 1-6 using AI for 10x speed improvement."
        ),
        "source": "CA Practice Framework 2024",
        "category": "process",
    },
    {
        "id": "Credit-Note-Handling",
        "title": "Credit notes and their impact on ITC",
        "content": (
            "When a supplier issues a credit note (for returns, discounts, "
            "or corrections), the buyer must reduce their ITC accordingly. "
            "Credit notes appear in GSTR-2B as negative entries. "
            "If a credit note reduces the invoice value, the ITC claimed "
            "on the original invoice must be proportionally reduced. "
            "Failure to account for credit notes is a common audit finding."
        ),
        "source": "CGST Act, Section 34",
        "category": "legislation",
    },
]


def get_all_documents() -> list[dict]:
    """Return all knowledge base documents."""
    return GST_KNOWLEDGE_BASE


def get_documents_by_category(category: str) -> list[dict]:
    """Return documents filtered by category."""
    return [doc for doc in GST_KNOWLEDGE_BASE if doc["category"] == category]


def search_documents(query: str, top_k: int = 5) -> list[dict]:
    """Simple keyword search across all documents."""
    query_lower = query.lower()
    scored = []

    for doc in GST_KNOWLEDGE_BASE:
        # Score based on keyword overlap
        text = (doc["title"] + " " + doc["content"]).lower()
        words = query_lower.split()
        score = sum(1 for w in words if w in text)
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]
