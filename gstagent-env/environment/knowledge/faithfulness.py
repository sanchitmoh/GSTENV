"""
Faithfulness Checker — post-generation grounding assertions.

Ensures LLM responses stay faithful to retrieved context by verifying
that cited legal references (circular numbers, rule numbers, section
references) actually exist in the retrieved chunks.

Critical for compliance domains like GST where a hallucinated circular
number can lead to incorrect tax treatment.
"""

from __future__ import annotations

import re


def assert_grounded(response: str, context_chunks: list[dict]) -> bool:
    """
    Check if key legal references in the response are traceable to retrieved chunks.

    Extracts circular numbers, rule numbers, and section references from the
    LLM response and verifies each one exists in the provided context.

    Args:
        response: The LLM-generated response text
        context_chunks: List of retrieved knowledge chunks (dicts with 'content')

    Returns:
        True if all cited references are grounded in the context.
        False if any hallucinated reference is detected.
        True if no references are cited (nothing to verify).
    """
    # Combine all context content for checking
    combined_context = " ".join(
        c.get("content", "").lower() for c in context_chunks
    )
    # Normalize whitespace for matching
    combined_context = re.sub(r"\s+", " ", combined_context)

    # Extract legal references from the response
    cited_refs = extract_legal_references(response)

    if not cited_refs:
        return True  # No references to verify

    for ref in cited_refs:
        normalized_ref = ref.replace(" ", "").lower()
        normalized_context = combined_context.replace(" ", "")
        if normalized_ref not in normalized_context:
            return False  # Hallucinated reference!

    return True


def extract_legal_references(text: str) -> list[str]:
    """
    Extract legal/regulatory references from text.

    Captures:
    - Circular numbers: "170/02/2022", "183/15/2022"
    - Rule references: "Rule 36", "Rule 37"
    - Section references: "Section 16", "Section 34"
    - Notification numbers: "Notification 40/2021"
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


def get_grounding_report(response: str, context_chunks: list[dict]) -> dict:
    """
    Generate a detailed grounding report for a response.

    Returns:
        Dict with grounded status, cited references, and any ungrounded refs.
    """
    combined_context = " ".join(
        c.get("content", "").lower() for c in context_chunks
    )
    combined_context_normalized = re.sub(r"\s+", " ", combined_context).replace(" ", "")

    cited_refs = extract_legal_references(response)
    grounded = []
    ungrounded = []

    for ref in cited_refs:
        normalized_ref = ref.replace(" ", "").lower()
        if normalized_ref in combined_context_normalized:
            grounded.append(ref)
        else:
            ungrounded.append(ref)

    return {
        "is_faithful": len(ungrounded) == 0,
        "total_references": len(cited_refs),
        "grounded_references": grounded,
        "ungrounded_references": ungrounded,
    }
