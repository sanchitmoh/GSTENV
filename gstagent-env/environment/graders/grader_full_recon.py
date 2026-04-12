"""
Grader for Task 3: Full Reconciliation.

Four-signal composite reward with:
- Proportional ITC accuracy decay (Fix #17)
- Proportional hallucination penalty capped at 0.2 (Fix #16)
- Efficiency bonus for fewer steps (Fix #18)
- Empty input guard (Fix #14)
- Fuzzy action matching via difflib (Upgrade #8)
- Efficiency bonus gated on recall > 0.5 (Upgrade #4-fix)

Deterministic: same inputs always produce same output.
"""

from __future__ import annotations

import difflib

# Score must be strictly inside (0, 1) — 0.0 and 1.0 are rejected by validator
# CRITICAL: log_end uses :.3f — 1e-4 rounds to '0.000', (1-1e-4) rounds to '1.000'
_SCORE_MIN = 0.001   # 0.001 -> '0.001' with :.3f
_SCORE_MAX = 0.999   # 0.999 -> '0.999' with :.3f


def _clamp(score: float) -> float:
    return max(_SCORE_MIN, min(_SCORE_MAX, score))


def grade(
    agent_report: dict,
    ground_truth: dict,
    steps_used: int = 0,
    max_steps: int = 20,
) -> dict:
    """
    Grade full reconciliation report with multi-signal composite reward.

    Args:
        agent_report: {
            "total_itc": float,
            "discrepancies": [{"invoice_id": str, "status": str, "action": str}],
            "matches": {invoice_id: status}
        }
        ground_truth: {
            "total_itc": float,
            "discrepancies": {invoice_id: {"status": str, "action": str}},
            "all_invoice_ids": set of valid IDs
        }
        steps_used: how many steps the agent took
        max_steps: maximum allowed steps

    Returns:
        dict with total score (strictly in (0,1)) and component breakdown
    """
    # Fix #14: guard against empty input — still return min score, not 0.0
    if not agent_report or not ground_truth:
        return {
            "total": _SCORE_MIN,
            "itc_accuracy": 0.0,
            "recall_score": 0.0,
            "action_correctness": 0.0,
            "efficiency_bonus": 0.0,
            "hallucination_penalty": 0.0,
        }

    # === 1. ITC Accuracy (Fix #17: proportional decay, not binary threshold) ===
    agent_itc = agent_report.get("total_itc", 0.0)
    correct_itc = ground_truth.get("total_itc", 0.0)

    if correct_itc > 0:
        itc_accuracy = max(0.0, 1.0 - abs(agent_itc - correct_itc) / correct_itc)
    elif agent_itc == 0:
        itc_accuracy = 1.0
    else:
        itc_accuracy = 0.0

    # === 2. Discrepancy Recall ===
    truth_discrepancies = ground_truth.get("discrepancies", {})
    agent_discrepancies = agent_report.get("discrepancies", [])

    total_mismatches = len(truth_discrepancies)
    found_mismatches = 0

    # Upgrade #9: Deduplicate agent discrepancies by invoice_id (last one wins)
    seen_disc_ids: dict[str, dict] = {}
    for disc in agent_discrepancies:
        disc_id = disc.get("invoice_id", "")
        if disc_id:
            seen_disc_ids[disc_id] = disc  # dedup — last declaration of same ID wins

    agent_disc_ids = set(seen_disc_ids.keys())
    for disc_id in agent_disc_ids:
        if disc_id in truth_discrepancies:
            found_mismatches += 1

    recall_score = (
        found_mismatches / total_mismatches if total_mismatches > 0 else 1.0
    )

    # === 3. Action Correctness (Upgrade #8: fuzzy matching via difflib) ===
    correct_actions = 0
    total_flagged = len(seen_disc_ids)

    for disc_id, disc in seen_disc_ids.items():
        agent_action = disc.get("action", "")
        if disc_id in truth_discrepancies:
            expected_action = truth_discrepancies[disc_id].get("action", "")
            if (
                agent_action == expected_action
                or _action_matches(agent_action, expected_action)
            ):
                correct_actions += 1

    action_correctness = (
        correct_actions / total_flagged if total_flagged > 0 else 0.0
    )

    # === 4. Hallucination Penalty (Fix #16: proportional, max 0.2) ===
    valid_ids = ground_truth.get("all_invoice_ids", set())
    total_agent_claims = len(agent_disc_ids)
    fake_ids = sum(1 for d_id in agent_disc_ids if d_id not in valid_ids)

    hallucination_rate = fake_ids / max(total_agent_claims, 1)
    hallucination_penalty = 0.2 * hallucination_rate  # Always 0.0 to 0.2

    # === 5. Efficiency Bonus (Upgrade: gated on recall > 0.5 to prevent lazy agents) ===
    if recall_score > 0.5:
        efficiency_bonus = 0.1 * (1.0 - steps_used / max(max_steps, 1))
        efficiency_bonus = max(0.0, efficiency_bonus)
    else:
        efficiency_bonus = 0.0  # Lazy agent (low recall) gets no efficiency reward

    # === Final Score — clamped to strictly open (0, 1) ===
    raw_score = (
        0.4 * itc_accuracy
        + 0.3 * recall_score
        + 0.2 * action_correctness
        - hallucination_penalty
        + efficiency_bonus
    )
    total = _clamp(round(raw_score, 4))

    return {
        "total": total,
        "itc_accuracy": round(itc_accuracy, 4),
        "recall_score": round(recall_score, 4),
        "action_correctness": round(action_correctness, 4),
        "efficiency_bonus": round(efficiency_bonus, 4),
        "hallucination_penalty": round(hallucination_penalty, 4),
    }


def _action_matches(agent_action: str, expected_action: str) -> bool:
    """
    Fuzzy action matching — allows semantic variations.

    Upgrade #8: Uses difflib.SequenceMatcher for sequence similarity (≥0.45)
    plus key-phrase matching for known patterns.
    """
    if not agent_action or not expected_action:
        return False

    agent_lower = agent_action.lower().strip()
    expected_lower = expected_action.lower().strip()

    # Exact match (fast path)
    if agent_lower == expected_lower:
        return True

    # Fuzzy similarity via difflib (threshold: 0.45 = loose but not too loose)
    similarity = difflib.SequenceMatcher(None, agent_lower, expected_lower).ratio()
    if similarity >= 0.45:
        return True

    # Key-phrase matching for known action patterns (backward-compatible)
    key_phrases = [
        "claim full",
        "follow up",
        "follow-up",
        "reverse",
        "supplier",
        "gstr-1",
        "matched amount",
        "contact supplier",
        "rectify",
        "ineligible",
        "partial",
    ]

    for phrase in key_phrases:
        if phrase in expected_lower and phrase in agent_lower:
            return True

    return False
