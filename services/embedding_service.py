"""
services/embedding_service.py
------------------------------
Manages the sentence-transformer embedding model and per-document FAISS indexes.

Architecture
------------
Each uploaded document gets its OWN FAISS index stored on disk:
  vector_store/<doc_id>.index   — FAISS flat L2 index
  vector_store/<doc_id>.chunks  — newline-joined raw chunk strings (parallel array)

Why a flat index per document (not one shared index)?
  • Simpler: no need to store doc_id metadata alongside each vector.
  • Supports isolated queries: "ask only about document X".
  • Easy deletion: remove two files and the doc is gone.
  • For a production system you'd use a proper vector DB (Qdrant/Weaviate/Pinecone).

Embedding model
---------------
`all-MiniLM-L6-v2` produces 384-dimensional vectors, fast CPU inference, good quality.
It is downloaded once and cached by sentence-transformers.
"""

import os
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import HTTPException

from utils.config import settings

# ---------------------------------------------------------------------------
# Singleton model — loaded once at import time, reused for every request.
# ---------------------------------------------------------------------------
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _index_path(doc_id: str) -> str:
    return os.path.join(settings.VECTOR_STORE_DIR, f"{doc_id}.index")


def _chunks_path(doc_id: str) -> str:
    return os.path.join(settings.VECTOR_STORE_DIR, f"{doc_id}.chunks")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_and_store(doc_id: str, chunks: list[str]) -> int:
    """
    Generate embeddings for `chunks` and persist a FAISS index to disk.

    Args:
        doc_id: Unique document identifier (used as filename stem).
        chunks: List of text chunks from the document.

    Returns:
        Number of chunks indexed.
    """
    if not chunks:
        raise HTTPException(status_code=422, detail="No chunks to embed — document may be empty.")

    os.makedirs(settings.VECTOR_STORE_DIR, exist_ok=True)

    model = get_model()

    # Encode all chunks → numpy float32 matrix of shape (n_chunks, dim)
    # batch_size=64 keeps memory tight and maximises CPU throughput
    # normalize_embeddings=True lets us use dot-product instead of L2 (faster search)
    embeddings: np.ndarray = model.encode(
        chunks,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = embeddings.astype(np.float32)

    dimension = embeddings.shape[1]

    # FAISS IndexFlatL2: exact nearest-neighbour search via L2 (Euclidean) distance.
    # For production with millions of chunks, switch to IndexIVFFlat or HNSW.
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Persist index
    faiss.write_index(index, _index_path(doc_id))

    # Persist raw chunks (parallel to index vectors)
    with open(_chunks_path(doc_id), "wb") as f:
        pickle.dump(chunks, f)

    return len(chunks)


def search_similar_chunks(doc_id: str, query: str, top_k: int = None) -> list[str]:
    """
    Embed `query` and retrieve the top-k most similar chunks from the document index.

    Args:
        doc_id: Document to search within.
        query:  User question or prompt.
        top_k:  Number of chunks to return (default from settings).

    Returns:
        List of chunk strings ordered by relevance (closest first).

    Raises:
        HTTPException(404) if the document index doesn't exist.
    """
    top_k = top_k or settings.TOP_K_CHUNKS

    idx_path = _index_path(doc_id)
    chk_path = _chunks_path(doc_id)

    if not os.path.exists(idx_path) or not os.path.exists(chk_path):
        raise HTTPException(
            status_code=404,
            detail=f"No index found for document '{doc_id}'. Did you upload it?",
        )

    # Load index and chunks from disk
    index = faiss.read_index(idx_path)
    with open(chk_path, "rb") as f:
        chunks: list[str] = pickle.load(f)

    # Embed the query
    model = get_model()
    query_vec: np.ndarray = model.encode([query], convert_to_numpy=True).astype(np.float32)

    # Clamp top_k to available chunks
    top_k = min(top_k, len(chunks))

    # Search — returns distances and indices arrays of shape (1, top_k)
    _distances, indices = index.search(query_vec, top_k)

    # Map FAISS indices back to chunk strings
    relevant_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    return relevant_chunks


def list_documents() -> list[str]:
    """Return all doc_ids that have been indexed."""
    if not os.path.exists(settings.VECTOR_STORE_DIR):
        return []
    files = os.listdir(settings.VECTOR_STORE_DIR)
    return [f.replace(".index", "") for f in files if f.endswith(".index")]
