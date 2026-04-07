"""
Auditor Agent — specialist for ITC eligibility assessment.

Applies Rule 36(4) and Section 16(2) to determine whether each invoice
qualifies for Input Tax Credit. Uses the rules engine for deterministic
decisions but adds reasoning context for the orchestrator.
"""

from __future__ import annotations

from environment.agents.base_agent import AgentMessage, BaseAgent, GROUNDING_CLAUSE
from environment.rules.gst_rules import (
    calculate_itc_amount,
    check_itc_eligibility,
    get_recommended_action,
)


class AuditorAgent(BaseAgent):
    """ITC eligibility auditor using GST rules engine."""

    def __init__(self, model: str | None = None):
        from environment.config import MODEL_NAME
        super().__init__(
            name="Auditor",
            role="ITC auditor — applies GST Rule 36(4) and Section 16(2)",
            model=model or MODEL_NAME,
        )

    def get_system_prompt(self) -> str:
        return """You are the Auditor Agent in a GST reconciliation team.

Your job is to determine ITC (Input Tax Credit) eligibility for each invoice.

GST Rules you enforce:
1. Rule 36(4) — ITC limited to invoices appearing in GSTR-2B
2. Section 16(2) — Requires: possession of invoice, receipt of goods, tax paid, return filed

Eligibility Categories:
- "eligible" — Invoice in GSTR-2B, amounts match within 5%
- "partial" — Invoice in GSTR-2B, amounts differ by 5-20% → claim only matched portion
- "ineligible" — Invoice missing from GSTR-2B OR amounts differ by >20%

For each invoice provide:
- eligibility status
- ITC amount claimable
- recommended action (claim full, claim partial, follow up with supplier)

Cite the specific rule for each decision.
""" + GROUNDING_CLAUSE

    def process(
        self, observation: dict, context: list[AgentMessage]
    ) -> list[AgentMessage]:
        """Audit ITC eligibility for all invoices."""
        purchase = observation.get("purchase_register", [])
        gstr2b = observation.get("gstr2b_data", [])

        gstr2b_index = {inv["invoice_id"]: inv for inv in gstr2b}

        messages = []
        total_eligible_itc = 0.0
        total_partial_itc = 0.0
        total_ineligible = 0

        for inv in purchase:
            inv_id = inv["invoice_id"]
            gstr_record = gstr2b_index.get(inv_id)

            status = check_itc_eligibility(inv, gstr_record)
            action = get_recommended_action(status)
            itc_amount = calculate_itc_amount(inv, status)

            if status == "eligible":
                total_eligible_itc += itc_amount
                rule_cite = "Rule 36(4) — invoice reflected in GSTR-2B, amounts match"
            elif status == "partial":
                total_partial_itc += itc_amount
                rule_cite = "Rule 36(4) — amounts differ 5-20%, claim matched portion per Rule 37"
            else:
                total_ineligible += 1
                rule_cite = "Section 16(2) — invoice not in GSTR-2B or variance >20%"

            messages.append(self.create_action_message(
                content=(
                    f"Invoice {inv_id}: {status} | "
                    f"ITC claimable: ₹{itc_amount:,.2f} | "
                    f"Rule: {rule_cite} | "
                    f"Action: {action}"
                ),
                action={
                    "action_type": "flag_mismatch" if status != "eligible" else "match_invoice",
                    "invoice_id": inv_id,
                    "reason": f"{status} — {rule_cite}",
                },
                status=status,
                itc_amount=itc_amount,
            ))

        # ITC computation action
        messages.append(self.create_action_message(
            content=(
                f"ITC Audit Summary:\n"
                f"  Total Eligible ITC: ₹{total_eligible_itc:,.2f}\n"
                f"  Total Partial ITC: ₹{total_partial_itc:,.2f}\n"
                f"  Ineligible Invoices: {total_ineligible}\n"
                f"  Total Claimable: ₹{total_eligible_itc + total_partial_itc:,.2f}"
            ),
            action={"action_type": "compute_itc"},
            total_eligible_itc=total_eligible_itc,
            total_partial_itc=total_partial_itc,
            total_ineligible=total_ineligible,
        ))

        return messages
