"""
Grader for Task 1: Invoice Matching.

Computes proper F1 score (precision + recall) — Fix #13.
Handles empty input gracefully — Fix #14.
Deterministic: same inputs always produce same output.
"""

from __future__ import annotations

# Score must be strictly inside (0, 1) — 0.0 and 1.0 are rejected by validator
# CRITICAL: log_end uses :.3f — 1e-4 rounds to '0.000', (1-1e-4) rounds to '1.000'
_SCORE_MIN = 0.001   # 0.001 -> '0.001' with :.3f
_SCORE_MAX = 0.999   # 0.999 -> '0.999' with :.3f


def _clamp(score: float) -> float:
    return max(_SCORE_MIN, min(_SCORE_MAX, score))


def grade(agent_matches: dict, ground_truth: dict) -> float:
    """
    Grade invoice matching results using F1 score.

    Args:
        agent_matches: {invoice_id: "present" | "missing"} from agent
        ground_truth: {invoice_id: "present" | "missing"} actual answers

    Returns:
        Score strictly in (0, 1) — never exactly 0.0 or 1.0
    """
    # Fix #14: guard against empty input
    if not agent_matches or not ground_truth:
        return _SCORE_MIN

    # Upgrade #9: deduplicate agent matches — last call per invoice_id wins.
    # Prevents duplicate match_invoice calls from inflating false positives.
    if isinstance(agent_matches, list):
        # Accept list-of-dicts format too: [{invoice_id: .., status: ..}, ...]
        deduped: dict[str, str] = {}
        for entry in agent_matches:
            if isinstance(entry, dict):
                inv_id = entry.get("invoice_id", "")
                status = entry.get("status", "unknown")
                if inv_id:
                    deduped[inv_id] = status
        agent_matches = deduped

    # For F1 we treat "missing" as the positive class (what we want to detect)
    # True positives: agent correctly identified as missing
    # False positives: agent said missing but actually present
    # False negatives: agent said present but actually missing

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    all_ids = set(ground_truth.keys()) | set(agent_matches.keys())

    for inv_id in all_ids:
        agent_val = agent_matches.get(inv_id, "unknown")
        truth_val = ground_truth.get(inv_id, "unknown")

        if truth_val == "missing" and agent_val == "missing":
            true_positives += 1
        elif truth_val == "present" and agent_val == "missing":
            false_positives += 1
        elif truth_val == "missing" and agent_val != "missing":
            false_negatives += 1
        elif truth_val == "present" and agent_val == "present":
            true_negatives += 1

    # Precision: of what agent called "missing", how many actually were?
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )

    # Recall: of actual "missing", how many did agent find?
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    # F1 score
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    # Also factor in correct "present" classifications
    total_correct = true_positives + true_negatives
    total = len(all_ids) if all_ids else 1
    accuracy = total_correct / total

    # Blend: 60% F1 (mismatch detection) + 40% accuracy (overall)
    score = 0.6 * f1 + 0.4 * accuracy

    return _clamp(round(score, 4))
