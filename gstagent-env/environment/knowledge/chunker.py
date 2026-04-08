"""
Document Chunker v3 — production-grade chunking with 4 RAG best practices.

Improvements over v2:
1. CONTEXTUAL CHUNK HEADERS — every chunk is prefixed with metadata so the
   model knows where it came from: [Category: itc_rules | Source: CGST Act]
2. SLIDING WINDOWS — percentage-based overlap (default 15%) instead of fixed
   1-sentence overlap. Prevents context loss at chunk boundaries.
3. SENTENCE-WINDOW MODE — embed individual sentences, store position metadata.
   On retrieval the engine expands to ±N neighbors for full context.
4. CONFIGURABLE STRATEGY — ChunkConfig dataclass replaces ad-hoc parameters.
   Tune chunk_size, overlap, min size, and mode per use-case.

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
from dataclasses import dataclass


# ── Configuration ────────────────────────────────────────────────────

@dataclass
class ChunkConfig:
    """
    Tunable chunking strategy.

    Attributes:
        chunk_size:       Target max words per chunk (soft limit — won't split sentences)
        overlap_pct:      Fraction of chunk sentences to repeat across boundaries (0.15 = 15%)
        min_chunk_words:  Don't create chunks smaller than this (merge into previous)
        mode:             "sentence" (default v2 behavior), "sentence_window" (per-sentence),
                          "fixed" (pure word-count, no sentence awareness)
        context_header:   If True, prepend [Category | Source] Title — to each chunk
        window_expand:    For sentence_window mode: expand ±N sentences on retrieval
    """
    chunk_size: int = 300
    overlap_pct: float = 0.15
    min_chunk_words: int = 20
    mode: str = "sentence"        # "sentence" | "sentence_window" | "fixed"
    context_header: bool = True
    window_expand: int = 5


# Default config — used when callers don't provide one
DEFAULT_CONFIG = ChunkConfig()

# ── Category-Aware Chunk Sizes ───────────────────────────────────
# Every GST doc in the knowledge base is under 180 words.
# chunk_size=300 means the guard `if total_words <= config.chunk_size`
# fires for every single document, making the chunker dead code.
# These category-aware sizes ensure 30+ of 40 docs split into 2-3 chunks,
# activating hierarchical search, sentence-window, and parent_id tracking.

CHUNK_SIZES: dict[str, int] = {
    "legislation":      50,   # Dense single-clause legal text
    "rules":            55,   # Rule clauses — self-contained facts
    "itc_rules":        65,   # Circular paragraphs
    "process":          90,   # Numbered steps — keep step groups
    "practical":        70,   # Threshold tables
    "business_impact":  100,  # Narrative
    "e_invoicing":      70,   # Technical spec, dense
    "returns":          80,   # Multi-part filing rules
    "penalties":        60,   # Short legal clauses
    "rcm":              70,   # RCM rules + service lists
    "composition":      65,
    "blocked_credits":  60,   # Enumerated list items
    "exports":          70,
    "registration":     65,
    "tds_tcs":          60,
    "place_of_supply":  70,
    "audit":            80,
    "technical":        60,
    "logistics":        75,
    "isd":              65,
    "anti_profiteering": 60,
    "classification":   70,
}


# ── Sentence Splitting ───────────────────────────────────────────────

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


# ── Contextual Headers ───────────────────────────────────────────────

def _build_chunk_header(doc: dict) -> str:
    """
    Build a contextual header prefix for a chunk.

    Adds provenance metadata so the model knows where the chunk came from.
    Example: "[Category: itc_rules | Source: CGST Act, Section 16(2)] Section 16(2) — "
    """
    category = doc.get("category", "unknown")
    source = doc.get("source", "unknown")
    title = doc.get("title", "")
    return f"[Category: {category} | Source: {source}] {title} — "


# ── Core Chunking Functions ──────────────────────────────────────────

def chunk_document(
    doc: dict,
    chunk_size: int = 300,
    overlap_sentences: int = 1,
    config: ChunkConfig | None = None,
) -> list[dict]:
    """
    Split a document into sentence-aligned overlapping chunks.

    Args:
        doc: Document dict with id, title, content, source, category
        chunk_size: Target max words per chunk (soft limit — won't split sentences)
        overlap_sentences: Legacy param — if config is provided, overlap_pct is used instead
        config: ChunkConfig for full control. If None, uses legacy parameters.

    Returns:
        List of chunk dicts. Single-chunk documents returned as-is.
    """
    if config is None:
        # Use category-aware chunk size if caller didn't specify one
        effective_size = chunk_size
        if chunk_size == 300:  # Default — use category-aware size instead
            category = doc.get("category", "")
            effective_size = CHUNK_SIZES.get(category, 70)
        config = ChunkConfig(
            chunk_size=effective_size,
            overlap_pct=0.0 if overlap_sentences == 0 else 0.15,
            context_header=True,
        )

    # Route to mode-specific chunker
    if config.mode == "sentence_window":
        return _sentence_window_chunks(doc, config)

    sentences = _split_sentences(doc["content"])

    # Don't chunk small documents
    total_words = sum(len(s.split()) for s in sentences)
    if total_words <= config.chunk_size:
        # Still add contextual header even for single-chunk docs
        content = doc["content"]
        if config.context_header:
            content = _build_chunk_header(doc) + content
        result = dict(doc)
        result["content"] = content
        return [result]

    chunks = []
    chunk_idx = 0
    i = 0

    while i < len(sentences):
        current_chunk_sentences: list[str] = []
        word_count = 0

        # Fill chunk up to word budget, never splitting mid-sentence
        while i < len(sentences):
            sentence_words = len(sentences[i].split())
            if word_count + sentence_words > config.chunk_size and current_chunk_sentences:
                break  # Chunk is full (but always include at least 1 sentence)
            current_chunk_sentences.append(sentences[i])
            word_count += sentence_words
            i += 1

        # Skip tiny trailing chunks — merge into previous
        if word_count < config.min_chunk_words and chunks:
            prev_content = chunks[-1]["content"]
            # Remove header from previous if present, re-add after merge
            chunks[-1]["content"] = prev_content + " " + " ".join(current_chunk_sentences)
            break

        chunk_text = " ".join(current_chunk_sentences)

        # Add contextual header
        if config.context_header:
            chunk_text = _build_chunk_header(doc) + chunk_text

        chunks.append({
            "id": f"{doc['id']}_chunk_{chunk_idx}",
            "parent_id": doc["id"],
            "title": doc["title"],
            "content": chunk_text,
            "source": doc["source"],
            "category": doc["category"],
            "chunk_index": chunk_idx,
            "total_chunks": -1,  # Filled after loop
            "sentences": current_chunk_sentences,  # Keep raw sentences for window expansion
        })
        chunk_idx += 1

        # Sliding window: overlap by percentage of sentences in this chunk
        if i < len(sentences) and config.overlap_pct > 0:
            overlap_count = max(2, int(len(current_chunk_sentences) * config.overlap_pct))
            i = max(i - overlap_count, i - len(current_chunk_sentences) + 1)

    # Fill total_chunks count
    for chunk in chunks:
        chunk["total_chunks"] = len(chunks)

    return chunks


def _sentence_window_chunks(doc: dict, config: ChunkConfig) -> list[dict]:
    """
    Create one chunk per sentence with position metadata.

    Enables fine-grained retrieval — find the most relevant sentence,
    then expand to ±window_expand surrounding sentences for context.
    """
    sentences = _split_sentences(doc["content"])

    if len(sentences) <= 1:
        content = doc["content"]
        if config.context_header:
            content = _build_chunk_header(doc) + content
        result = dict(doc)
        result["content"] = content
        return [result]

    chunks = []
    for idx, sentence in enumerate(sentences):
        # Content for indexing: just the sentence (+ header for context)
        content = sentence
        if config.context_header:
            content = _build_chunk_header(doc) + content

        chunks.append({
            "id": f"{doc['id']}_sw_{idx}",
            "parent_id": doc["id"],
            "title": doc["title"],
            "content": content,
            "source": doc["source"],
            "category": doc["category"],
            "chunk_index": idx,
            "total_chunks": len(sentences),
            "sentence_index": idx,
            "total_sentences": len(sentences),
            "sentences": sentences,  # Full sentence list for window expansion
            "window_expand": config.window_expand,
        })

    return chunks


def expand_sentence_window(chunk: dict, expand: int | None = None) -> str:
    """
    Expand a sentence-window chunk to include surrounding sentences.

    Given a chunk that was created in sentence_window mode, returns the
    expanded text including ±expand neighboring sentences.

    Args:
        chunk: A chunk dict created by _sentence_window_chunks
        expand: Number of sentences to include on each side. Defaults to
                the chunk's stored window_expand value, or 5.

    Returns:
        Expanded text string with surrounding context.
    """
    sentences = chunk.get("sentences", [])
    if not sentences:
        return chunk.get("content", "")

    idx = chunk.get("sentence_index", 0)
    n = expand if expand is not None else chunk.get("window_expand", 5)

    start = max(0, idx - n)
    end = min(len(sentences), idx + n + 1)

    expanded_text = " ".join(sentences[start:end])

    # Add contextual header if the original chunk had one
    header_marker = "[Category:"
    content = chunk.get("content", "")
    if content.startswith(header_marker):
        header_end = content.find(" — ")
        if header_end > 0:
            header = content[:header_end + 3]
            return header + expanded_text

    return expanded_text


# ── Batch Functions ──────────────────────────────────────────────────

def chunk_all_documents(
    docs: list[dict],
    chunk_size: int = 300,
    overlap_sentences: int = 1,
    config: ChunkConfig | None = None,
) -> list[dict]:
    """
    Chunk all documents in a corpus using sentence-aware splitting.

    Small documents (under chunk_size words) pass through unchanged
    (but still get contextual headers if enabled).
    """
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size, overlap_sentences, config))
    return all_chunks

