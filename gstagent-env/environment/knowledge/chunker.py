"""
Document Chunker — sentence-aware splitting for legal document integrity.

CRITICAL: Legal clauses like "The 5% provisional credit has been removed
w.e.f. 01.01.2022" are single facts. Word-boundary splitting can split
mid-sentence, causing a chunk with "provisional credit allowed" sans the
removal date — which gives the LLM partial, authoritative-sounding info
that leads to confident wrong answers.

This chunker splits on sentence boundaries, then groups sentences into
chunks that respect the size budget while keeping legal facts intact.
"""

from __future__ import annotations

import re


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences, preserving legal abbreviations.

    Handles GST-specific abbreviations (w.e.f., e.g., i.e., etc.)
    and numbered lists (1. 2. 3.) without false splits.
    """
    # Protect common abbreviations from sentence splitting
    protected = text
    abbreviations = [
        (r"w\.e\.f\.", "WEF_PLACEHOLDER"),
        (r"e\.g\.", "EG_PLACEHOLDER"),
        (r"i\.e\.", "IE_PLACEHOLDER"),
        (r"etc\.", "ETC_PLACEHOLDER"),
        (r"No\.", "NO_PLACEHOLDER"),
        (r"Sr\.", "SR_PLACEHOLDER"),
        (r"Cr\.", "CR_PLACEHOLDER"),
        (r"Rs\.", "RS_PLACEHOLDER"),
        (r"pct\.", "PCT_PLACEHOLDER"),
    ]

    for pattern, placeholder in abbreviations:
        protected = re.sub(pattern, placeholder, protected)

    # Protect numbered lists (1. 2. 3.) — don't split on these
    protected = re.sub(r"(\d+)\.\s", r"\1LISTDOT_PLACEHOLDER ", protected)

    # Split on sentence-ending punctuation followed by space + capital or end
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", protected)

    # Restore placeholders
    result = []
    for s in sentences:
        restored = s
        for pattern, placeholder in abbreviations:
            original = pattern.replace("\\", "")
            restored = restored.replace(placeholder, original)
        restored = re.sub(r"(\d+)LISTDOT_PLACEHOLDER ", r"\1. ", restored)
        restored = restored.strip()
        if restored:
            result.append(restored)

    return result


def chunk_document(
    doc: dict,
    chunk_size: int = 200,
    overlap_sentences: int = 1,
) -> list[dict]:
    """
    Split a document into sentence-aligned overlapping chunks.

    Args:
        doc: Document dict with id, title, content, source, category
        chunk_size: Target max words per chunk (soft limit — won't split sentences)
        overlap_sentences: Number of sentences to repeat across chunk boundaries

    Returns:
        List of chunk dicts. Single-chunk documents returned as-is.
    """
    sentences = _split_sentences(doc["content"])

    # Don't chunk small documents
    total_words = sum(len(s.split()) for s in sentences)
    if total_words <= chunk_size:
        return [doc]

    chunks = []
    chunk_idx = 0
    i = 0

    while i < len(sentences):
        current_chunk_sentences = []
        word_count = 0

        # Fill chunk up to word budget, never splitting mid-sentence
        while i < len(sentences):
            sentence_words = len(sentences[i].split())
            if word_count + sentence_words > chunk_size and current_chunk_sentences:
                break  # Chunk is full (but always include at least 1 sentence)
            current_chunk_sentences.append(sentences[i])
            word_count += sentence_words
            i += 1

        chunk_text = " ".join(current_chunk_sentences)
        chunks.append({
            "id": f"{doc['id']}_chunk_{chunk_idx}",
            "parent_id": doc["id"],
            "title": doc["title"],
            "content": chunk_text,
            "source": doc["source"],
            "category": doc["category"],
            "chunk_index": chunk_idx,
        })
        chunk_idx += 1

        # Overlap: back up by overlap_sentences for context continuity
        if i < len(sentences) and overlap_sentences > 0:
            i = max(i - overlap_sentences, len(current_chunk_sentences))

    return chunks


def chunk_all_documents(
    docs: list[dict],
    chunk_size: int = 200,
    overlap_sentences: int = 1,
) -> list[dict]:
    """
    Chunk all documents in a corpus using sentence-aware splitting.

    Small documents (under chunk_size words) pass through unchanged.
    """
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size, overlap_sentences))
    return all_chunks
