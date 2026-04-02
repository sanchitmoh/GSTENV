"""
Property-based tests for graders using Hypothesis (Fix #32).

Ensures grader output is ALWAYS in [0.0, 1.0] for ANY random input.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from environment.graders import grader_full_recon, grader_invoice_match, grader_itc_audit


# Strategies
invoice_id = st.text(
    alphabet=st.sampled_from("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"),
    min_size=1,
    max_size=10,
)
status_match = st.sampled_from(["present", "missing", "unknown", ""])
status_elig = st.sampled_from(["eligible", "partial", "ineligible", "unknown", ""])

match_dict = st.dictionaries(invoice_id, status_match, min_size=0, max_size=20)
elig_dict = st.dictionaries(invoice_id, status_elig, min_size=0, max_size=20)


@given(agent=match_dict, truth=match_dict)
@settings(max_examples=200)
def test_invoice_match_always_bounded(agent, truth):
    score = grader_invoice_match.grade(agent, truth)
    assert 0.0 <= score <= 1.0


@given(agent=elig_dict, truth=elig_dict)
@settings(max_examples=200)
def test_itc_audit_always_bounded(agent, truth):
    score = grader_itc_audit.grade(agent, truth)
    assert 0.0 <= score <= 1.0


@given(
    total_itc=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
    correct_itc=st.floats(min_value=0, max_value=1e6, allow_nan=False),
    num_disc=st.integers(min_value=0, max_value=15),
    steps=st.integers(min_value=0, max_value=50),
    max_steps=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=200)
def test_full_recon_always_bounded(total_itc, correct_itc, num_disc, steps, max_steps):
    agent_report = {
        "total_itc": total_itc,
        "discrepancies": [
            {"invoice_id": f"INV-{i}", "status": "unknown", "action": "none"}
            for i in range(num_disc)
        ],
    }
    ground_truth = {
        "total_itc": correct_itc,
        "discrepancies": {f"INV-{i}": {"status": "ineligible", "action": "Follow up"} for i in range(3)},
        "all_invoice_ids": {f"INV-{i}" for i in range(10)},
    }
    result = grader_full_recon.grade(agent_report, ground_truth, steps, max_steps)
    assert 0.0 <= result["total"] <= 1.0
    assert 0.0 <= result["hallucination_penalty"] <= 0.2
