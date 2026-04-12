"""
GST Knowledge Base — curated domain knowledge for RAG.

Contains CBIC circulars, Rule 36(4) amendments, Section 16(2) details,
and practical CA/accountant guidance. This is the "knowledge corpus"
that gets embedded and retrieved during agent reasoning.
"""

from __future__ import annotations

# Each document has: id, title, content, source, category
GST_KNOWLEDGE_BASE: list[dict] = [
    # ── Original 12 documents (ITC, reconciliation, basics) ──────────
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
            "Rule 37 of CGST Rules requires reversal of ITC if a buyer fails "
            "to pay the supplier within 180 days from the invoice date. The ITC "
            "claimed must be reversed and added to output tax liability along "
            "with interest under Section 50. The ITC can be reclaimed when "
            "payment is eventually made. Rule 37 applies to all types of "
            "supplies — goods and services."
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

    # ── NEW: E-Invoicing ────────────────────────────────────────────
    {
        "id": "E-Invoice-Mandate",
        "title": "E-invoicing mandate and IRN generation under GST",
        "content": (
            "E-invoicing is mandatory for businesses with aggregate turnover "
            "exceeding ₹5 Cr (from 01.08.2023 via Notification 10/2023). "
            "All B2B invoices must be registered on the Invoice Registration "
            "Portal (IRP) to obtain an Invoice Reference Number (IRN). "
            "The IRP generates a signed QR code and returns a signed JSON. "
            "E-invoices are auto-populated into GSTR-1 and GSTR-2B, reducing "
            "manual entry errors. Non-compliance results in the invoice being "
            "treated as invalid — buyer cannot claim ITC on such invoices. "
            "Common errors: duplicate IRN requests, incorrect GSTIN in JSON "
            "schema, HSN code mismatch between ERP and e-invoice portal."
        ),
        "source": "CBIC Notification 10/2023, E-Invoice Schema v1.1",
        "category": "e_invoicing",
    },
    {
        "id": "E-Invoice-Cancellation",
        "title": "E-invoice cancellation and amendment rules",
        "content": (
            "An e-invoice can be cancelled on the IRP within 24 hours of "
            "generation. After 24 hours, it cannot be cancelled on IRP — the "
            "supplier must issue a credit note and report it in GSTR-1. "
            "Amendments to e-invoices are not allowed on IRP. To correct "
            "errors, cancel the original (if within 24 hours) and re-generate, "
            "or issue a debit/credit note. Partial cancellation is not "
            "supported. The cancelled IRN cannot be reused. "
            "For reconciliation, agents must check if credit notes correspond "
            "to cancelled e-invoices to avoid double ITC reversal."
        ),
        "source": "GSTN E-Invoice API Documentation",
        "category": "e_invoicing",
    },

    # ── NEW: GST Registration ───────────────────────────────────────
    {
        "id": "GST-Registration-Threshold",
        "title": "GST registration thresholds and requirements",
        "content": (
            "GST registration is mandatory when aggregate turnover exceeds: "
            "₹40 Lakh for goods suppliers (₹20 Lakh for special category states), "
            "₹20 Lakh for service providers (₹10 Lakh for special category states). "
            "Special category states include: Arunachal Pradesh, Manipur, Meghalaya, "
            "Mizoram, Nagaland, Sikkim, Tripura, and Uttarakhand. "
            "Compulsory registration regardless of turnover for: inter-state suppliers, "
            "e-commerce operators, TDS/TCS deductors, casual taxable persons, "
            "and non-resident taxable persons. "
            "Registration must be obtained within 30 days of crossing the threshold."
        ),
        "source": "CGST Act, Section 22 & Section 24",
        "category": "registration",
    },
    {
        "id": "GST-Registration-Cancellation",
        "title": "Cancellation and revocation of GST registration",
        "content": (
            "GST registration can be cancelled by the officer if: returns not filed "
            "for 6 consecutive months (regular) or 3 consecutive quarters (composition), "
            "registration obtained by fraud, or business discontinued. "
            "Voluntary cancellation requires filing all pending returns and paying "
            "any dues. ITC must be reversed on closing stock. "
            "Revocation of cancellation can be applied within 30 days of cancellation "
            "order (extendable to 90 days by Additional Commissioner). "
            "Suppliers whose registration is cancelled are treated as unregistered — "
            "buyers cannot claim ITC on their invoices."
        ),
        "source": "CGST Act, Section 29 & Section 30",
        "category": "registration",
    },

    # ── NEW: Returns Filing ─────────────────────────────────────────
    {
        "id": "GSTR-1-Filing",
        "title": "GSTR-1 filing — outward supply details",
        "content": (
            "GSTR-1 is the return for outward supplies filed by every registered "
            "taxpayer. Due dates: 11th of next month for monthly filers (turnover "
            "> ₹5 Cr), 13th of month after quarter for QRMP scheme filers. "
            "GSTR-1 contains: B2B invoices (with GSTIN), B2C large (>₹2.5L), "
            "B2C others (consolidated), credit/debit notes, exports, nil-rated "
            "and exempt supplies. Invoice Furnishing Facility (IFF) allows QRMP "
            "filers to upload B2B invoices in first 2 months of each quarter. "
            "Late filing attracts ₹50/day (₹20/day for nil return) up to ₹10,000. "
            "Late GSTR-1 directly impacts buyer's GSTR-2B and their ITC claims."
        ),
        "source": "CGST Rules, Rule 59",
        "category": "returns",
    },
    {
        "id": "GSTR-3B-Filing",
        "title": "GSTR-3B filing — monthly summary return",
        "content": (
            "GSTR-3B is the monthly summary return for payment of GST. Due date: "
            "20th of the next month (staggered for different states: 20th, 22nd, "
            "or 24th). GSTR-3B requires reporting: outward supplies (taxable, "
            "zero-rated, nil-rated, exempt), inward supplies liable to reverse "
            "charge, eligible ITC (from GSTR-2B), ITC reversed, and net tax "
            "payable. ITC claimed in GSTR-3B must not exceed ITC available in "
            "GSTR-2B. Interest at 18% per annum applies on late payment of tax. "
            "Late fee: ₹50/day (₹20/day for nil return) capped at ₹5,000 per "
            "return. Filing GSTR-3B is mandatory even if there is no business "
            "activity — nil returns must still be filed."
        ),
        "source": "CGST Act, Section 39",
        "category": "returns",
    },
    {
        "id": "QRMP-Scheme",
        "title": "QRMP scheme for small taxpayers",
        "content": (
            "The Quarterly Return Monthly Payment (QRMP) scheme is available for "
            "taxpayers with aggregate turnover up to ₹5 Cr. Under QRMP: returns "
            "(GSTR-1 and GSTR-3B) are filed quarterly, but tax payment is monthly "
            "using PMT-06 challan by 25th of the next month. "
            "Two payment methods: Fixed Sum Method (35% of cash liability of "
            "previous quarter) or Self-Assessment Method (actual tax for the month). "
            "Invoice Furnishing Facility (IFF) allows uploading B2B invoices in "
            "first two months of the quarter, enabling buyer ITC without waiting "
            "for quarterly GSTR-1. Switching between monthly and QRMP is allowed "
            "at the beginning of each quarter on the GST portal."
        ),
        "source": "CBIC Notification 84/2020, CGST Rules",
        "category": "returns",
    },

    # ── NEW: Penalties & Interest ───────────────────────────────────
    {
        "id": "GST-Penalties-Interest",
        "title": "Penalties and interest provisions under GST",
        "content": (
            "Key penalty provisions under GST: "
            "Section 122: General penalty for contraventions — minimum ₹10,000 "
            "or tax amount, whichever is higher. Fraud attracts 100% penalty. "
            "Section 73: Demand for tax not paid — no penalty if tax plus interest "
            "paid within 30 days of show cause notice. "
            "Section 74: Demand for tax evaded through fraud — penalty equal to "
            "100% of tax. Can be reduced to 15% if paid within 30 days. "
            "Interest rates: 18% per annum on late payment (Section 50(1)), "
            "24% per annum on undue ITC claimed and utilized (Section 50(3)). "
            "Late fees: ₹50/day for GSTR-1, ₹50/day for GSTR-3B, capped at "
            "₹5,000 per return. Nil returns attract reduced late fee of ₹20/day."
        ),
        "source": "CGST Act, Sections 50, 73, 74, 122",
        "category": "penalties",
    },
    {
        "id": "GST-Interest-on-ITC",
        "title": "Interest implications of wrong ITC claims",
        "content": (
            "Wrongly availed and utilized ITC attracts interest at 24% per annum "
            "under Section 50(3) of CGST Act. This is higher than the standard "
            "18% for late tax payment. Wrongly availed but NOT utilized ITC "
            "attracts no interest — only reversal is needed. "
            "The distinction between 'availed' and 'utilized' is critical: "
            "ITC is 'availed' when claimed in GSTR-3B, and 'utilized' when "
            "set off against output tax liability. "
            "Common scenarios triggering 24% interest: ITC claimed on blocked "
            "credits (Section 17(5)), ITC on invoices from cancelled GSTINs, "
            "ITC claimed beyond GSTR-2B without legitimate basis. "
            "Voluntary reversal before notice reduces risk of interest."
        ),
        "source": "CGST Act, Section 50(3), CBIC Circular 183/15/2022",
        "category": "penalties",
    },

    # ── NEW: Reverse Charge Mechanism ───────────────────────────────
    {
        "id": "Reverse-Charge-Mechanism",
        "title": "Reverse charge mechanism (RCM) under GST",
        "content": (
            "Under Reverse Charge Mechanism (RCM), the recipient pays GST "
            "instead of the supplier. RCM applies in three scenarios: "
            "1. Supplies from unregistered persons to registered persons "
            "(Section 9(4) — currently applicable only to specified categories), "
            "2. Specified goods/services listed in Section 9(3) notification "
            "(e.g., legal services from advocates, transport by GTA, sponsorship), "
            "3. E-commerce operators for specified services (Section 9(5)). "
            "ITC on RCM: Recipient can claim ITC on tax paid under RCM, "
            "provided all conditions of Section 16(2) are met. RCM supplies "
            "must be reported separately in GSTR-3B Table 3.1(d). "
            "Self-invoices must be issued for RCM from unregistered persons. "
            "RCM liability cannot be discharged using ITC balance."
        ),
        "source": "CGST Act, Section 9(3), 9(4), 9(5)",
        "category": "rcm",
    },
    {
        "id": "RCM-Specified-Services",
        "title": "List of services under reverse charge",
        "content": (
            "Key services under reverse charge (recipient pays GST): "
            "1. Legal services by an advocate or firm of advocates — 18% GST. "
            "2. Goods Transport Agency (GTA) services — 5% (no ITC) or 12%. "
            "3. Sponsorship services — 18% GST. "
            "4. Services by a director to a company — 18% GST. "
            "5. Insurance agent services to insurance company — 18% GST. "
            "6. Security services by individual/firm to a registered person — 18%. "
            "7. Renting of motor vehicles (if supplier individual/HUF) — 5%. "
            "For reconciliation: RCM invoices won't appear in GSTR-2B since "
            "they are self-assessed. Agents must identify RCM invoices separately "
            "and ensure the recipient has paid tax and filed self-invoice."
        ),
        "source": "CGST Notification 13/2017 (as amended)",
        "category": "rcm",
    },

    # ── NEW: Composition Scheme ─────────────────────────────────────
    {
        "id": "Composition-Scheme",
        "title": "Composition scheme under GST",
        "content": (
            "The composition scheme (Section 10) allows small taxpayers to pay "
            "GST at a fixed rate on turnover instead of normal rates. "
            "Eligibility: aggregate turnover up to ₹1.5 Cr (₹75 Lakh for special "
            "category states). Not available for: service providers (except "
            "restaurants), inter-state suppliers, e-commerce operators, and "
            "manufacturers of notified goods (ice cream, pan masala, tobacco). "
            "Tax rates: 1% for manufacturers, 5% for restaurants, 1% for traders. "
            "Composition dealers CANNOT collect tax from buyers and CANNOT claim "
            "ITC. They file CMP-08 quarterly and GSTR-4 annually. "
            "For reconciliation: invoices from composition dealers should NOT "
            "have GST charged — if they do, the buyer must NOT claim ITC on them."
        ),
        "source": "CGST Act, Section 10",
        "category": "composition",
    },

    # ── NEW: Blocked Credits ────────────────────────────────────────
    {
        "id": "Blocked-Credits-Section-17-5",
        "title": "Blocked ITC under Section 17(5)",
        "content": (
            "Section 17(5) lists supplies on which ITC is blocked (cannot be claimed): "
            "(a) Motor vehicles and conveyances (exceptions: passenger transport, "
            "driving training, vehicles for further supply). "
            "(b) Food and beverages, outdoor catering, health and beauty services "
            "(exception: if used for making taxable outward supply of same category). "
            "(c) Membership of clubs, health and fitness centres. "
            "(d) Rent-a-cab, life insurance, health insurance (exception: if "
            "mandatory under law, or used for making taxable outward supply). "
            "(e) Travel benefits extended to employees on vacation. "
            "(f) Works contract services for construction of immovable property "
            "(except plant and machinery). "
            "(g) Goods/services for construction of immovable property on own account. "
            "(h) Tax paid under composition scheme. "
            "(i) Goods lost, stolen, destroyed, written off, or given as free samples."
        ),
        "source": "CGST Act, Section 17(5)",
        "category": "blocked_credits",
    },

    # ── NEW: Annual Return ──────────────────────────────────────────
    {
        "id": "GSTR-9-Annual-Return",
        "title": "GSTR-9 annual return filing",
        "content": (
            "GSTR-9 is the annual return filed by every registered taxpayer. "
            "Due date: 31st December of the following financial year. "
            "Mandatory for taxpayers with turnover exceeding ₹2 Cr. "
            "Optional for taxpayers with turnover up to ₹2 Cr (from FY 2022-23). "
            "GSTR-9 consolidates: all monthly/quarterly returns (GSTR-1, GSTR-3B), "
            "outward and inward supplies, ITC availed and reversed, tax paid, and "
            "any adjustments. For turnover exceeding ₹5 Cr, GSTR-9C "
            "(reconciliation statement) may also be required — certifying that "
            "figures in GSTR-9 match audited financial statements. "
            "Common reconciliation issues in GSTR-9: mismatch between GSTR-1 "
            "and GSTR-3B, ITC differences between books and GSTR-2B, "
            "unmatched credit notes, and classification errors in HSN summary."
        ),
        "source": "CGST Act, Section 44; CGST Rules, Rule 80",
        "category": "returns",
    },

    # ── NEW: TDS/TCS under GST ──────────────────────────────────────
    {
        "id": "GST-TDS",
        "title": "Tax Deducted at Source (TDS) under GST",
        "content": (
            "TDS under GST (Section 51) applies when: government departments, "
            "local authorities, government agencies, or persons notified by "
            "government make payments exceeding ₹2.5 Lakh for a single contract. "
            "TDS rate: 2% of taxable value (1% CGST + 1% SGST for intra-state, "
            "2% IGST for inter-state). The deductor must file GSTR-7 monthly "
            "by the 10th of the following month. The supplier (deductee) claims "
            "TDS credit in their electronic cash ledger. "
            "Common issues: TDS not reflected in supplier's ledger due to "
            "incorrect GSTIN entry, mismatch in contract value, or late GSTR-7 "
            "filing by the deductor."
        ),
        "source": "CGST Act, Section 51",
        "category": "tds_tcs",
    },
    {
        "id": "GST-TCS",
        "title": "Tax Collected at Source (TCS) under GST",
        "content": (
            "TCS under GST (Section 52) applies to e-commerce operators "
            "who collect tax at source on behalf of suppliers selling through "
            "their platform. TCS rate: 1% of net taxable supplies (0.5% CGST + "
            "0.5% SGST for intra-state, 1% IGST for inter-state). "
            "E-commerce operators must file GSTR-8 monthly by the 10th of "
            "following month. Suppliers can claim TCS credit in their "
            "electronic cash ledger while filing GSTR-3B. "
            "Net taxable supplies = aggregate value of taxable supplies minus "
            "returns. TCS is applicable on actual supplies, not on orders placed."
        ),
        "source": "CGST Act, Section 52",
        "category": "tds_tcs",
    },

    # ── NEW: Place of Supply ────────────────────────────────────────
    {
        "id": "Place-of-Supply-Goods",
        "title": "Place of supply rules for goods",
        "content": (
            "Determining place of supply is critical for correct tax levy "
            "(CGST+SGST for intra-state, IGST for inter-state). "
            "For goods: place of supply is the location where movement of goods "
            "terminates for delivery to the recipient (Section 10(1)(a)). "
            "If no movement: place of supply is the location of goods at the "
            "time of delivery (Section 10(1)(b)). "
            "Bill-to-ship-to: If goods are delivered to a third person on "
            "direction of the buyer, place of supply is the location of such "
            "third person (Section 10(1)(b)). "
            "For imports: place of supply is the location of the importer. "
            "For exports: place of supply is the location outside India. "
            "Wrong place of supply leads to: wrong GSTIN on invoice, IGST paid "
            "instead of CGST+SGST or vice versa, ITC credit issues for buyer."
        ),
        "source": "IGST Act, Section 10",
        "category": "place_of_supply",
    },
    {
        "id": "Place-of-Supply-Services",
        "title": "Place of supply rules for services",
        "content": (
            "Default rule (Section 12(2)): Place of supply of services is the "
            "location of the recipient if registered, else the location of the "
            "supplier. Special rules apply for: "
            "1. Immovable property services — place where property is located. "
            "2. Restaurant/catering — place where services are performed. "
            "3. Training/performance events — place where event is held. "
            "4. Transportation of goods — place where goods are handed over. "
            "5. Telecom/internet services — billing address of recipient. "
            "6. Banking/financial services — location of recipient. "
            "For B2B services: always the registered place of recipient. "
            "For B2C services: various rules apply based on service type. "
            "Incorrect place of supply is a major cause of ITC mismatch — "
            "supplier charges IGST but buyer's state expects CGST/SGST."
        ),
        "source": "IGST Act, Section 12, Section 13",
        "category": "place_of_supply",
    },

    # ── NEW: ITC Proportional Reversal ──────────────────────────────
    {
        "id": "ITC-Proportional-Reversal-Rule-42-43",
        "title": "ITC reversal for mixed-use inputs under Rule 42 and 43",
        "content": (
            "When inputs or input services are used partly for taxable and partly "
            "for exempt/non-business purposes, ITC must be proportionally reversed. "
            "Rule 42 (inputs/input services): ITC on common inputs is apportioned "
            "based on turnover ratio — exempt turnover / total turnover. "
            "Rule 43 (capital goods): ITC on common capital goods is also reversed "
            "proportionally, calculated each month and adjusted at year-end. "
            "Example: If exempt turnover is 30% of total, 30% of common ITC "
            "must be reversed. Turnover includes: taxable, exempt, nil-rated, "
            "non-GST supply. Banking/financial services use a 50/50 formula "
            "instead of turnover-based reversal. "
            "Reversal must be done monthly and finalized in September return."
        ),
        "source": "CGST Rules, Rule 42 and Rule 43",
        "category": "itc_rules",
    },

    # ── NEW: GST on Exports ─────────────────────────────────────────
    {
        "id": "GST-Exports-Refund",
        "title": "GST on exports and refund mechanism",
        "content": (
            "Exports are zero-rated under GST (IGST Act Section 16). "
            "Two options for exporters: "
            "1. Export under bond/LUT (Letter of Undertaking) without payment "
            "of IGST — claim refund of accumulated ITC. "
            "2. Export with payment of IGST — claim refund of IGST paid. "
            "Refund timeline: Must be filed within 2 years from relevant date. "
            "Refund of IGST on exports: auto-processed by matching shipping bill "
            "data with GSTR-1 export invoices (ICEGATE-GST integration). "
            "Refund of ITC on exports under LUT: filed via RFD-01 on GST portal. "
            "Common issues: shipping bill mismatch with GSTR-1 invoice data, "
            "IGST not reported in Table 6A of GSTR-1, bank account not validated "
            "on GST portal, incomplete export documentation (BRC/FIRC for services)."
        ),
        "source": "IGST Act, Section 16; CGST Rules, Rule 89, Rule 96",
        "category": "exports",
    },

    # ── NEW: Anti-Profiteering ──────────────────────────────────────
    {
        "id": "Anti-Profiteering",
        "title": "Anti-profiteering provisions under GST",
        "content": (
            "Section 171 of CGST Act mandates that any reduction in GST rate "
            "or benefit of ITC must be passed on to the recipient by way of "
            "commensurate reduction in prices. The National Anti-Profiteering "
            "Authority (NAA) — now replaced by the Competition Commission of "
            "India (CCI) from December 2022 — investigates complaints. "
            "Penalty: If profiteering is established, the supplier must reduce "
            "prices and refund the profiteered amount with 18% interest. "
            "If the recipient cannot be identified, the amount is deposited in "
            "the Consumer Welfare Fund. Repeated offences can lead to "
            "cancellation of GST registration."
        ),
        "source": "CGST Act, Section 171",
        "category": "anti_profiteering",
    },

    # ── NEW: Input Service Distributor ──────────────────────────────
    {
        "id": "Input-Service-Distributor",
        "title": "Input Service Distributor (ISD) mechanism",
        "content": (
            "An Input Service Distributor (ISD) is a registered office that "
            "receives invoices for input services and distributes ITC to its "
            "branches or units. ISD is mandatory from April 2025 for distributing "
            "ITC of services procured from third parties (cross-charge alternative). "
            "ISD must file GSTR-6 monthly by 13th of the following month. "
            "ITC distribution must be proportional to turnover of each branch "
            "in the previous financial year. Credit notes from ISD reduce the "
            "distributed ITC. ISD can only distribute ITC on input services, "
            "not on inputs or capital goods. "
            "For reconciliation: GSTR-6 data flows into recipient's GSTR-2B "
            "as ISD credit — must be reconciled separately from normal ITC."
        ),
        "source": "CGST Act, Section 20; CGST Rules, Rule 39",
        "category": "isd",
    },

    # ── NEW: E-Way Bill ─────────────────────────────────────────────
    {
        "id": "E-Way-Bill",
        "title": "E-Way Bill requirements for goods movement",
        "content": (
            "E-Way Bill (EWB) is mandatory when goods valued above ₹50,000 "
            "are transported (inter-state or intra-state). Generated on the "
            "E-Way Bill portal (ewaybillgst.gov.in). "
            "Validity: up to 200 km = 1 day, each additional 200 km = 1 extra day. "
            "For over-dimensional cargo: 20 km per day. "
            "Part-A of EWB: invoice details, supplier/recipient GSTIN, HSN, value. "
            "Part-B of EWB: vehicle number (required when goods move by road). "
            "E-Way Bill is auto-generated for e-invoices with goods value ≥₹50,000. "
            "Common issues: expired EWB during transit, Part-B not updated for "
            "vehicle change, goods arriving without valid EWB — attracts penalty "
            "of ₹10,000 or tax amount, whichever is higher (Section 129). "
            "No EWB required for: documents, exempt goods, goods moved within 50 km."
        ),
        "source": "CGST Rules, Rule 138; CGST Act, Section 68 & 129",
        "category": "logistics",
    },

    # ── NEW: Section 16(4) Time Limit ───────────────────────────────
    {
        "id": "Section-16-4-Time-Limit",
        "title": "Time limit for claiming ITC under Section 16(4)",
        "content": (
            "Section 16(4) imposes a strict time limit on ITC claims. "
            "ITC for any invoice or debit note can only be claimed up to: "
            "30th November of the year following the financial year to which the "
            "invoice pertains, OR the date of filing the annual return (GSTR-9), "
            "whichever is earlier. "
            "Example: For an invoice dated 15 March 2025 (FY 2024-25), ITC must "
            "be claimed by 30 November 2025 or date of GSTR-9 filing for FY 24-25. "
            "After this deadline, ITC lapses permanently — it cannot be claimed "
            "even if all other conditions (Section 16(2)) are met. "
            "This makes timely reconciliation critical — delayed supplier filing "
            "can cause permanent ITC loss if GSTR-2B is not updated before the "
            "Section 16(4) deadline."
        ),
        "source": "CGST Act, Section 16(4)",
        "category": "itc_rules",
    },

    # ── NEW: Debit Note ITC Rules ───────────────────────────────────
    {
        "id": "Debit-Note-ITC",
        "title": "ITC on debit notes and supplementary invoices",
        "content": (
            "Debit notes are issued by the supplier when the taxable value or "
            "tax charged in the original invoice is less than the actual amount. "
            "Section 16(2) allows ITC on debit notes just like invoices. "
            "The time limit for ITC on debit notes (per amendment effective "
            "01.01.2021): calculated from the financial year of the debit note, "
            "NOT the original invoice date. This gives taxpayers additional time "
            "to claim ITC on supplementary invoices. "
            "Debit notes appear in GSTR-2B as positive entries (additional ITC). "
            "For reconciliation: debit notes must be matched separately from "
            "the original invoice — both amount and GSTIN must match GSTR-2B."
        ),
        "source": "CGST Act, Section 16(2), Section 34; Finance Act 2020 Amendment",
        "category": "itc_rules",
    },

    # ── NEW: GST Audit & Assessment ─────────────────────────────────
    {
        "id": "GST-Audit-Assessment",
        "title": "GST audit and assessment procedures",
        "content": (
            "GST audit can be triggered in multiple ways: "
            "1. Departmental audit (Section 65) — conducted by tax authorities "
            "for any registered taxpayer, with 15 days notice. "
            "2. Special audit (Section 66) — ordered during scrutiny if affairs "
            "are complex, conducted by a chartered accountant nominated by "
            "commissioner. Must be completed within 90 days (extendable to 180). "
            "3. Scrutiny of returns (Section 61) — preliminary review of returns, "
            "may lead to demand if discrepancies found. "
            "Assessment types: self-assessment (Section 59, normal filing), "
            "provisional assessment (Section 60), scrutiny assessment (Section 61), "
            "best judgement assessment (Section 62 — for non-filers), and "
            "assessment of unregistered persons (Section 63). "
            "Key audit focus areas: ITC claimed vs GSTR-2B, output tax vs GSTR-1, "
            "reverse charge compliance, and blocked credit claims."
        ),
        "source": "CGST Act, Sections 59-66",
        "category": "audit",
    },

    # ── NEW: Electronic Ledgers ─────────────────────────────────────
    {
        "id": "Electronic-Ledgers",
        "title": "Electronic ledgers on the GST portal",
        "content": (
            "The GST portal maintains three electronic ledgers for each taxpayer: "
            "1. Electronic Cash Ledger — all tax payments via challans, TDS/TCS "
            "credited, and refunds. Balance can be used for any tax payment. "
            "2. Electronic Credit Ledger — ITC claimed in GSTR-3B. Can be used "
            "to pay output tax (not interest, late fees, or penalties). "
            "ITC cannot be used to pay RCM liability — must be paid in cash. "
            "3. Electronic Liability Ledger — all tax dues, interest, penalties, "
            "fees, and other amounts payable. "
            "Order of utilization: IGST credit first (against IGST, then CGST, "
            "then SGST), then CGST credit (against CGST, then IGST), then SGST "
            "credit (against SGST, then IGST). Cross-utilization of CGST and SGST "
            "is not allowed."
        ),
        "source": "CGST Act, Sections 49, 49A, 49B; CGST Rules, Rule 86, 86A, 87",
        "category": "technical",
    },

    # ── NEW: Rule 86B — Cash Payment Restriction ────────────────────
    {
        "id": "Rule-86B-Cash-Restriction",
        "title": "Rule 86B cash payment restriction for high ITC users",
        "content": (
            "Rule 86B restricts use of ITC in electronic credit ledger: "
            "if taxable supply value exceeds ₹50 Lakh in a month, at least "
            "1% of output tax must be paid in cash (from Electronic Cash Ledger). "
            "This prevents businesses from discharging 100% tax liability via ITC. "
            "Exemptions from Rule 86B: if income tax paid in last two years "
            "exceeds ₹1 Lakh, if refund of unutilized ITC exceeds ₹1 Lakh in "
            "previous FY, or if cumulative tax paid in cash up to the month "
            "exceeds 1% of total output tax. "
            "Rule 86B impacts cash flow planning for MSMEs with high ITC — "
            "they must maintain sufficient cash balance for the 1% minimum."
        ),
        "source": "CGST Rules, Rule 86B (Notification 94/2020)",
        "category": "rules",
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
