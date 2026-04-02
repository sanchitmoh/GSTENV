"""
Matcher Agent — specialist for invoice matching.

Compares purchase register entries against GSTR-2B records.
Identifies present, missing, and amount-mismatched invoices.
Optimized for speed — uses simpler model for high throughput.
"""

from __future__ import annotations

from environment.agents.base_agent import AgentMessage, BaseAgent


class MatcherAgent(BaseAgent):
    """Invoice matching specialist."""

    def __init__(self, model: str | None = None):
        from environment.config import FAST_MODEL_NAME
        super().__init__(
            name="Matcher",
            role="Invoice matcher — compares purchase register to GSTR-2B",
            model=model or FAST_MODEL_NAME,
        )

    def get_system_prompt(self) -> str:
        return """You are the Matcher Agent in a GST reconciliation team.

Your ONLY job is to match invoices from the purchase register against GSTR-2B data.

For each invoice, determine:
- "present" — invoice exists in GSTR-2B with matching details
- "missing" — invoice NOT found in GSTR-2B (supplier may not have filed)
- "mismatch" — invoice found but amounts differ significantly

Output your analysis as a JSON list:
[{"invoice_id": "INV-0001", "status": "present|missing|mismatch", "variance": 0.0}]

Be thorough. Check EVERY invoice. Do NOT skip any.
Do NOT invent invoice IDs that don't exist in the purchase register."""

    def process(
        self, observation: dict, context: list[AgentMessage]
    ) -> list[AgentMessage]:
        """Analyze invoices and produce match results."""
        purchase = observation.get("purchase_register", [])
        gstr2b = observation.get("gstr2b_data", [])

        # Build GSTR-2B index for O(1) lookup
        gstr2b_index = {inv["invoice_id"]: inv for inv in gstr2b}

        results = []
        actions = []

        for inv in purchase:
            inv_id = inv["invoice_id"]
            gstr_record = gstr2b_index.get(inv_id)

            if gstr_record is None:
                status = "missing"
                variance = 1.0
            else:
                inv_amt = inv.get("taxable_amount", 0)
                gstr_amt = gstr_record.get("taxable_amount", 0)
                variance = abs(inv_amt - gstr_amt) / max(inv_amt, 1) if inv_amt else 0
                status = "present" if variance <= 0.05 else "mismatch"

            results.append({
                "invoice_id": inv_id,
                "status": status,
                "variance": round(variance, 4),
            })

            # Generate actions
            actions.append(self.create_action_message(
                content=f"Invoice {inv_id}: {status} (variance: {variance:.2%})",
                action={
                    "action_type": "match_invoice" if status == "present" else "flag_mismatch",
                    "invoice_id": inv_id,
                    "reason": f"{status} — variance {variance:.2%}" if status != "present" else None,
                },
                status=status,
                variance=variance,
            ))

        # Summary message
        missing = sum(1 for r in results if r["status"] == "missing")
        mismatched = sum(1 for r in results if r["status"] == "mismatch")
        matched = sum(1 for r in results if r["status"] == "present")

        summary = self.create_action_message(
            content=(
                f"Matching complete: {matched} matched, {missing} missing, "
                f"{mismatched} mismatched out of {len(purchase)} invoices."
            ),
            matched=matched,
            missing=missing,
            mismatched=mismatched,
        )
        actions.append(summary)

        return actions
