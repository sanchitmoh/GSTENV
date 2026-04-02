"""
Validator Agent — hallucination guard.

Checks agent-generated invoice IDs against actual data before submission.
Removes any fabricated IDs that don't exist in the purchase register.
Prevents hallucination penalty from the grader.
"""

from __future__ import annotations

from environment.agents.base_agent import AgentMessage, BaseAgent


class ValidatorAgent(BaseAgent):
    """Hallucination detection and report validation specialist."""

    def __init__(self, model: str | None = None):
        from environment.config import FAST_MODEL_NAME
        super().__init__(
            name="Validator",
            role="Validator — catches hallucinated invoice IDs and validates report integrity",
            model=model or FAST_MODEL_NAME,
        )

    def get_system_prompt(self) -> str:
        return """You are the Validator Agent in a GST reconciliation team.

Your CRITICAL job is to prevent hallucinations:
1. Check every invoice_id in the report against the actual purchase register
2. Remove any invoice_id that does NOT exist in the data
3. Verify ITC amounts are reasonable (not negative, not absurdly high)
4. Ensure the report structure is complete

You are the LAST agent before submission. Nothing gets through with fake data."""

    def process(
        self, observation: dict, context: list[AgentMessage]
    ) -> list[AgentMessage]:
        """Validate the report before submission."""
        # Get valid invoice IDs from observation
        valid_ids = {
            inv["invoice_id"]
            for inv in observation.get("purchase_register", [])
        }

        # Find the reporter's submission
        report_msg = None
        for msg in reversed(context):
            if msg.sender == "Reporter" and msg.action:
                if msg.action.get("action_type") == "submit_report":
                    report_msg = msg
                    break

        if not report_msg or not report_msg.action:
            return [self.create_action_message(
                content="ERROR: No report found to validate.",
                action=None,
            )]

        payload = report_msg.action.get("payload", {})

        # Validate discrepancies
        clean_discrepancies = []
        hallucinated_count = 0
        for disc in payload.get("discrepancies", []):
            disc_id = disc.get("invoice_id", "")
            if disc_id in valid_ids:
                clean_discrepancies.append(disc)
            else:
                hallucinated_count += 1

        # Validate matches
        clean_matches = {}
        for inv_id, status in payload.get("matches", {}).items():
            if inv_id in valid_ids:
                clean_matches[inv_id] = status
            else:
                hallucinated_count += 1

        # Validate decisions
        clean_decisions = {}
        for inv_id, status in payload.get("decisions", {}).items():
            if inv_id in valid_ids:
                clean_decisions[inv_id] = status
            else:
                hallucinated_count += 1

        # Validate ITC amount
        total_itc = payload.get("total_itc", 0.0)
        if total_itc < 0:
            total_itc = 0.0
        # Cap at reasonable max (sum of all invoice taxes)
        max_possible = sum(
            inv.get("cgst", 0) + inv.get("sgst", 0) + inv.get("igst", 0)
            for inv in observation.get("purchase_register", [])
        )
        if total_itc > max_possible * 1.1:
            total_itc = max_possible

        # Build clean report
        clean_payload = {
            "total_itc": round(total_itc, 2),
            "discrepancies": clean_discrepancies,
            "matches": clean_matches,
            "decisions": clean_decisions,
        }

        return [self.create_action_message(
            content=(
                f"Validation Complete:\n"
                f"  Hallucinated IDs removed: {hallucinated_count}\n"
                f"  Clean discrepancies: {len(clean_discrepancies)}\n"
                f"  Clean matches: {len(clean_matches)}\n"
                f"  Validated ITC: ₹{total_itc:,.2f}\n"
                f"  Report is ready for submission."
            ),
            action={
                "action_type": "submit_report",
                "payload": clean_payload,
            },
            hallucinated_removed=hallucinated_count,
            clean_report=clean_payload,
        )]
