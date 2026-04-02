"""
Grader for Task 1: Invoice Matching.

Computes proper F1 score (precision + recall) — Fix #13.
Handles empty input gracefully — Fix #14.
Deterministic: same inputs always produce same output.
"""

from __future__ import annotations


def grade(agent_matches: dict, ground_truth: dict) -> float:
    """
    Grade invoice matching results using F1 score.

    Args:
        agent_matches: {invoice_id: "present" | "missing"} from agent
        ground_truth: {invoice_id: "present" | "missing"} actual answers

    Returns:
        F1 score between 0.0 and 1.0
    """
    # Fix #14: guard against empty input
    if not agent_matches or not ground_truth:
        return 0.0

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

    return max(0.0, min(1.0, round(score, 4)))
