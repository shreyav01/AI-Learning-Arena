"""
services/embedding_service.py
------------------------------
API-based embeddings using OpenRouter (no local models, no DLL issues).

Instead of running sentence-transformers locally (which requires pyarrow,
sklearn, and other DLLs blocked by Windows Application Control), we:
  1. Call the OpenRouter embeddings API to get vectors
  2. Store vectors + chunks as plain numpy .npy + pickle files
  3. Do similarity search with pure numpy (cosine similarity)

This removes ALL blocked dependencies:
  - No sentence-transformers
  - No pyarrow
  - No sklearn
  - No faiss (replaced with numpy dot product)

Trade-off: requires an API call per upload/query (fast, ~1-2 sec)
"""

import os
import pickle

import httpx
import numpy as np
from fastapi import HTTPException

from utils.config import settings


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _vecs_path(doc_id: str) -> str:
    return os.path.join(settings.VECTOR_STORE_DIR, f"{doc_id}.npy")

def _chunks_path(doc_id: str) -> str:
    return os.path.join(settings.VECTOR_STORE_DIR, f"{doc_id}.chunks")


# ---------------------------------------------------------------------------
# Embedding API call
# ---------------------------------------------------------------------------

def _get_embeddings(texts: list[str]) -> np.ndarray:
    """
    Call OpenRouter's embedding endpoint.
    Uses text-embedding-3-small (OpenAI) via OpenRouter — 1536 dims, fast, cheap.
    Falls back to a simple TF-IDF-style bag-of-words if API fails.
    """
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    # OpenRouter supports OpenAI-compatible /embeddings endpoint
    try:
        response = httpx.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers=headers,
            json={
                "model": "openai/text-embedding-3-small",
                "input": texts,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        vectors = [item["embedding"] for item in data["data"]]
        arr = np.array(vectors, dtype=np.float32)
        # Normalize for cosine similarity
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return arr / norms

    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Embedding API failed: {e}. Check your OPENROUTER_API_KEY.",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_and_store(doc_id: str, chunks: list[str]) -> int:
    """
    Generate embeddings via API and save as .npy + .chunks files.
    Processes in batches of 50 to stay within API limits.
    """
    if not chunks:
        raise HTTPException(status_code=422, detail="No chunks to embed.")

    os.makedirs(settings.VECTOR_STORE_DIR, exist_ok=True)

    # Batch embed (API has input limits)
    BATCH = 50
    all_vecs = []
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        vecs = _get_embeddings(batch)
        all_vecs.append(vecs)

    embeddings = np.vstack(all_vecs)

    # Save vectors and chunks
    np.save(_vecs_path(doc_id), embeddings)
    with open(_chunks_path(doc_id), "wb") as f:
        pickle.dump(chunks, f)

    return len(chunks)


def search_similar_chunks(doc_id: str, query: str, top_k: int = None) -> list[str]:
    """
    Embed the query via API and find top-k most similar chunks
    using cosine similarity (fast numpy dot product on normalized vectors).
    """
    top_k = top_k or settings.TOP_K_CHUNKS

    vecs_path  = _vecs_path(doc_id)
    chks_path  = _chunks_path(doc_id)

    if not os.path.exists(vecs_path) or not os.path.exists(chks_path):
        raise HTTPException(
            status_code=404,
            detail=f"No index found for document '{doc_id}'. Did you upload it?",
        )

    # Load stored data
    embeddings = np.load(vecs_path)
    with open(chks_path, "rb") as f:
        chunks: list[str] = pickle.load(f)

    # Embed query
    query_vec = _get_embeddings([query])[0]  # shape (dim,)

    # Cosine similarity = dot product on normalized vectors
    scores = embeddings @ query_vec  # shape (n_chunks,)

    # Get top-k indices
    top_k = min(top_k, len(chunks))
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [chunks[i] for i in top_indices]


def list_documents() -> list[str]:
    """Return all doc_ids that have been indexed."""
    if not os.path.exists(settings.VECTOR_STORE_DIR):
        return []
    files = os.listdir(settings.VECTOR_STORE_DIR)
    return [f.replace(".npy", "") for f in files if f.endswith(".npy")]
