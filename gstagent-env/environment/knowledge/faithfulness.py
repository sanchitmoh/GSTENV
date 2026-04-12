"""
Faithfulness Checker v2 — grounding assertions for legal AND numeric claims.

v1 only caught legal references (Rule 36, Circular 170/02/2022).
v2 also catches numeric claims that matter in GST compliance:
- Percentages: "5%", "18%", "20%"
- Day counts: "180 days", "30 days"
- Amounts: "₹18L", "₹1 Cr"
- Dates: "01.01.2022", "14th of every month"

An LLM can hallucinate "claim 10% provisional ITC" and v1 returns True
because no Section or Circular was cited. v2 catches the "10%" as
ungrounded in the context where only "5%" appears.
"""

from __future__ import annotations

import re


def assert_grounded(response: str, context_chunks: list[dict]) -> bool:
    """
    Check if key claims in the response are traceable to retrieved context.

    Verifies:
    1. Legal references (circular numbers, rules, sections)
    2. Numeric claims (percentages, amounts, day counts, dates)

    Returns True if all verifiable claims are grounded.
    Returns True if nothing verifiable is found (nothing to check).
    Returns False if any hallucinated claim is detected.
    """
    combined_context = " ".join(
        c.get("content", "").lower() for c in context_chunks
    )
    # Strip contextual chunk headers: [Category: X | Source: Y] Title —
    # These headers contain legal references that would falsely ground
    # hallucinated citations (e.g., "Rule 36(4)" in header matches even
    # when the actual chunk content doesn't support the claim).
    combined_context = re.sub(r'\[category:[^\]]+\][^\n]*?\u2014\s*', '', combined_context)
    combined_context = re.sub(r"\s+", " ", combined_context)

    # Check legal references
    legal_refs = extract_legal_references(response)
    for ref in legal_refs:
        normalized_ref = ref.replace(" ", "").lower()
        normalized_context = combined_context.replace(" ", "")
        if normalized_ref not in normalized_context:
            return False

    # Check numeric claims
    numeric_claims = extract_numeric_claims(response)
    for claim in numeric_claims:
        normalized_claim = claim.replace(" ", "").lower()
        normalized_context = combined_context.replace(" ", "")
        if normalized_claim not in normalized_context:
            return False

    return True


def extract_legal_references(text: str) -> list[str]:
    """
    Extract legal/regulatory references from text.

    Captures: circulars, rules, sections, notifications.
    """
    patterns = [
        r"\d{2,3}/\d{2,4}/\d{4}",          # Circular: 170/02/2022
        r"\d{2,3}/\d{4}",                   # Notification: 40/2021
        r"Rule\s+\d+(?:\(\d+\))?",          # Rule 36, Rule 36(4)
        r"Section\s+\d+(?:\(\d+\))?",       # Section 16, Section 16(2)
    ]

    refs = []
    for pattern in patterns:
        refs.extend(re.findall(pattern, text, re.IGNORECASE))

    return list(set(refs))


def extract_numeric_claims(text: str) -> list[str]:
    """
    Extract GST-critical numeric claims from text.

    Captures percentages, day counts, currency amounts, and dates
    that are commonly hallucinated in GST compliance responses.
    """
    patterns = [
        r"\d+(?:\.\d+)?%",                      # Percentages: 5%, 18%, 0.5%
        r"\d+\s*days?",                           # Day counts: 180 days, 30 day
        r"(?:₹|Rs\.?|INR)\s*[\d,.]+\s*(?:L|Cr|lakh|crore)?",  # Amounts: ₹18L, Rs.1Cr
        r"\d{1,2}\.\d{1,2}\.\d{4}",             # Dates: 01.01.2022
        r"\d{1,2}(?:st|nd|rd|th)\s+of\s+(?:every|each)\s+month",  # "14th of every month"
    ]

    claims = []
    for pattern in patterns:
        claims.extend(re.findall(pattern, text, re.IGNORECASE))

    return list(set(claims))


def get_grounding_report(response: str, context_chunks: list[dict]) -> dict:
    """
    Generate a detailed grounding report for a response.

    Returns dict with separate tracking for legal refs and numeric claims.
    """
    combined_context = " ".join(
        c.get("content", "").lower() for c in context_chunks
    )
    # Strip contextual chunk headers before checking grounding
    combined_context = re.sub(r'\[category:[^\]]+\][^\n]*?\u2014\s*', '', combined_context)
    combined_normalized = re.sub(r"\s+", " ", combined_context).replace(" ", "")

    legal_refs = extract_legal_references(response)
    numeric_claims = extract_numeric_claims(response)

    grounded_legal = []
    ungrounded_legal = []
    for ref in legal_refs:
        if ref.replace(" ", "").lower() in combined_normalized:
            grounded_legal.append(ref)
        else:
            ungrounded_legal.append(ref)

    grounded_numeric = []
    ungrounded_numeric = []
    for claim in numeric_claims:
        if claim.replace(" ", "").lower() in combined_normalized:
            grounded_numeric.append(claim)
        else:
            ungrounded_numeric.append(claim)

    all_ungrounded = ungrounded_legal + ungrounded_numeric

    return {
        "is_faithful": len(all_ungrounded) == 0,
        "total_references": len(legal_refs) + len(numeric_claims),
        "grounded_references": grounded_legal + grounded_numeric,
        "ungrounded_references": all_ungrounded,
        "legal_refs": {"grounded": grounded_legal, "ungrounded": ungrounded_legal},
        "numeric_claims": {"grounded": grounded_numeric, "ungrounded": ungrounded_numeric},
    }
