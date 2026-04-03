"""
Document Chunker — split large documents into overlapping chunks.

When the knowledge base grows (100+ circulars), full-document indexing
pollutes unrelated queries. This module splits documents into semantic
chunks while preserving parent metadata for traceability.

Features:
- Configurable chunk size and overlap
- Parent-child relationship tracking
- Metadata preservation across chunks
"""

from __future__ import annotations


def chunk_document(
    doc: dict,
    chunk_size: int = 200,
    overlap: int = 40,
) -> list[dict]:
    """
    Split a document into overlapping word-level chunks.

    Args:
        doc: Document dict with id, title, content, source, category
        chunk_size: Max words per chunk
        overlap: Number of overlapping words between consecutive chunks

    Returns:
        List of chunk dicts, each inheriting parent metadata.
        Single-chunk documents returned as-is (no unnecessary splitting).
    """
    words = doc["content"].split()

    # Don't chunk small documents — just return as-is
    if len(words) <= chunk_size:
        return [doc]

    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])

        chunks.append({
            "id": f"{doc['id']}_chunk_{chunk_idx}",
            "parent_id": doc["id"],
            "title": doc["title"],
            "content": chunk_text,
            "source": doc["source"],
            "category": doc["category"],
            "chunk_index": chunk_idx,
        })

        start += chunk_size - overlap
        chunk_idx += 1

    return chunks


def chunk_all_documents(
    docs: list[dict],
    chunk_size: int = 200,
    overlap: int = 40,
) -> list[dict]:
    """
    Chunk all documents in a corpus.

    Small documents (under chunk_size) are returned unchanged.
    """
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size, overlap))
    return all_chunks
