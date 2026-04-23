"""
utils/chunker.py
----------------
Splits a long string into overlapping fixed-size character chunks.

Why overlap?
  A sentence or concept may straddle a chunk boundary. Overlapping ensures
  the relevant context is fully present in at least one chunk, improving
  retrieval quality.

Example (CHUNK_SIZE=20, OVERLAP=5):
  "The quick brown fox jumps over the lazy dog"
  → ["The quick brown fox ", "x jumps over the la", "e lazy dog"]
"""

from utils.config import settings


def split_into_chunks(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    """
    Split `text` into chunks of `chunk_size` characters with `overlap` chars
    of context carried forward from the previous chunk.

    Args:
        text:       Full document text.
        chunk_size: Max characters per chunk (default from settings).
        overlap:    Chars to repeat at the start of the next chunk (default from settings).

    Returns:
        List of non-empty string chunks.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    if not text.strip():
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Move forward by (chunk_size - overlap) so next chunk shares `overlap` chars
        start += chunk_size - overlap

    return chunks
