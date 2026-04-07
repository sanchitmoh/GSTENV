"""
Reporter Agent — specialist for generating structured reconciliation reports.

Collects results from Matcher and Auditor agents, then produces a
comprehensive report suitable for submission via submit_report action.
"""

from __future__ import annotations

from environment.agents.base_agent import AgentMessage, BaseAgent, GROUNDING_CLAUSE


class ReporterAgent(BaseAgent):
    """Report generation specialist."""

    def __init__(self, model: str | None = None):
        from environment.config import MODEL_NAME
        super().__init__(
            name="Reporter",
            role="Report generator — synthesizes findings into structured reconciliation report",
            model=model or MODEL_NAME,
        )

    def get_system_prompt(self) -> str:
        return """You are the Reporter Agent in a GST reconciliation team.

Your job is to take the findings from the Matcher and Auditor agents
and compile them into a structured reconciliation report.

The report must include:
1. Total eligible ITC amount
2. List of discrepancies (missing/mismatched invoices)
3. Recommended action for each discrepancy
4. Match status for every invoice

Output format for submit_report action:
{
  "total_itc": <float>,
  "discrepancies": [{"invoice_id": "...", "status": "...", "action": "..."}],
  "matches": {"INV-0001": "present", "INV-0002": "missing", ...},
  "decisions": {"INV-0001": "eligible", "INV-0002": "ineligible", ...}
}

Be precise. Use exact invoice IDs from the data. Never invent IDs.
""" + GROUNDING_CLAUSE

    def process(
        self, observation: dict, context: list[AgentMessage]
    ) -> list[AgentMessage]:
        """Compile final report from agent context."""
        # Extract match results from Matcher messages
        matches = {}
        decisions = {}
        discrepancies = []
        total_itc = 0.0

        for msg in context:
            if msg.sender == "Matcher" and msg.action:
                inv_id = msg.action.get("invoice_id", "")
                if inv_id:
                    status = msg.metadata.get("status", "unknown")
                    matches[inv_id] = "present" if status == "present" else "missing"

            elif msg.sender == "Auditor" and msg.action:
                inv_id = msg.action.get("invoice_id", "")
                if inv_id:
                    status = msg.metadata.get("status", "unknown")
                    itc = msg.metadata.get("itc_amount", 0.0)
                    decisions[inv_id] = status
                    total_itc += itc

                    if status in ("partial", "ineligible"):
                        reason = msg.action.get("reason", "")
                        action_str = (
                            "Follow up with supplier to file GSTR-1; do not claim ITC until reflected in GSTR-2B"
                            if status == "ineligible"
                            else "Claim only matched amount, reverse the excess under Rule 37"
                        )
                        discrepancies.append({
                            "invoice_id": inv_id,
                            "status": status,
                            "action": action_str,
                        })

                # Check for ITC summary from Auditor
                if msg.action and msg.action.get("action_type") == "compute_itc":
                    auditor_total = msg.metadata.get("total_eligible_itc", 0)
                    auditor_partial = msg.metadata.get("total_partial_itc", 0)
                    if auditor_total + auditor_partial > 0:
                        total_itc = auditor_total + auditor_partial

        # Build the final report
        report_payload = {
            "total_itc": round(total_itc, 2),
            "discrepancies": discrepancies,
            "matches": matches,
            "decisions": decisions,
        }

        return [self.create_action_message(
            content=(
                f"Reconciliation Report Compiled:\n"
                f"  Total Eligible ITC: ₹{total_itc:,.2f}\n"
                f"  Discrepancies Found: {len(discrepancies)}\n"
                f"  Invoices Matched: {len(matches)}\n"
                f"  Decisions Made: {len(decisions)}\n"
                f"\nSubmitting report..."
            ),
            action={
                "action_type": "submit_report",
                "payload": report_payload,
            },
            report=report_payload,
        )]
