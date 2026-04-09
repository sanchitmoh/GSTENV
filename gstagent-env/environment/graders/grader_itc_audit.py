"""
Grader for Task 2: ITC Audit.

Evaluates eligibility classification accuracy plus amount accuracy — Fix #15.
Handles empty input gracefully — Fix #14.
Deterministic: same inputs always produce same output.
"""

from __future__ import annotations

# Score must be strictly inside (0, 1) — 0.0 and 1.0 are rejected by validator
_SCORE_MIN = 1e-4
_SCORE_MAX = 1.0 - 1e-4


def _clamp(score: float) -> float:
    return max(_SCORE_MIN, min(_SCORE_MAX, score))


def grade(
    agent_decisions: dict,
    ground_truth: dict,
    agent_amounts: dict | None = None,
    truth_amounts: dict | None = None,
) -> float:
    """
    Grade ITC audit results.

    Args:
        agent_decisions: {invoice_id: "eligible"|"partial"|"ineligible"} from agent
        ground_truth: {invoice_id: "eligible"|"partial"|"ineligible"} actual
        agent_amounts: {invoice_id: float} ITC amounts claimed (optional)
        truth_amounts: {invoice_id: float} correct ITC amounts (optional)

    Returns:
        Score strictly in (0, 1) — never exactly 0.0 or 1.0
    """
    # Fix #14: guard against empty input
    if not agent_decisions or not ground_truth:
        return _SCORE_MIN

    # Decision accuracy: correct eligibility classifications
    all_ids = set(ground_truth.keys())
    correct = 0
    total = len(all_ids)

    if total == 0:
        return _SCORE_MIN

    for inv_id in all_ids:
        agent_val = agent_decisions.get(inv_id, "unknown")
        truth_val = ground_truth.get(inv_id, "unknown")
        if agent_val == truth_val:
            correct += 1

    decision_accuracy = correct / total

    # Fix #15: Amount accuracy sub-score
    amount_accuracy = 1.0
    if agent_amounts and truth_amounts:
        amount_errors = []
        for inv_id in all_ids:
            agent_amt = agent_amounts.get(inv_id, 0.0)
            truth_amt = truth_amounts.get(inv_id, 0.0)
            if truth_amt > 0:
                error = abs(agent_amt - truth_amt) / truth_amt
                amount_errors.append(max(0.0, 1.0 - error))
            elif agent_amt == 0:
                amount_errors.append(1.0)
            else:
                amount_errors.append(0.0)

        if amount_errors:
            amount_accuracy = sum(amount_errors) / len(amount_errors)

    # Weighted: 70% decision accuracy + 30% amount accuracy
    score = 0.7 * decision_accuracy + 0.3 * amount_accuracy

    return _clamp(round(score, 4))
