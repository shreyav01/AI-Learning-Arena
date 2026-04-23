"""
main.py
-------
FastAPI application entry point for the NotebookLM-like backend.

Startup responsibilities:
  - Validate required environment variables.
  - Create storage directories.
  - Pre-load the embedding model so the first request isn't slow.
  - Register all routers.
  - Configure CORS.

Run with:
  uvicorn main:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from utils.config import settings
from services.embedding_service import get_model
from routes import upload, ask, generate_questions, arena


# ---------------------------------------------------------------------------
# Lifespan: runs on startup and shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup:
      1. Validate API key is present.
      2. Ensure storage directories exist.
      3. Warm up the embedding model (downloads on first run).
    """
    # 1. Config validation
    settings.validate()

    # 2. Storage dirs
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.VECTOR_STORE_DIR, exist_ok=True)

    # 3. Pre-load embedding model into memory
    print("⏳  Loading embedding model — this may take a moment on first run...")
    get_model()
    print(f"✅  Embedding model '{settings.EMBEDDING_MODEL}' ready.")
    print(f"✅  Using LLM: {settings.OPENROUTER_MODEL} via OpenRouter")

    yield  # application runs here

    # Shutdown (add cleanup here if needed)
    print("👋  Shutting down NotebookLM backend.")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NotebookLM Backend",
    description=(
        "Upload PDFs, ask questions via RAG, and generate study materials "
        "(MCQs, short & long answers) grounded in your documents."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS — allow all origins for local development.
# In production, restrict `allow_origins` to your frontend domain.
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Replace with ["https://yourfrontend.com"] in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(upload.router, tags=["Documents"])
app.include_router(ask.router, tags=["RAG Q&A"])
app.include_router(generate_questions.router, tags=["Study Material"])
app.include_router(arena.router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"])
def health_check():
    """Simple liveness probe."""
    return {
        "status": "ok",
        "model": settings.OPENROUTER_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
    }


@app.get("/documents", tags=["Documents"])
def list_documents():
    """List all doc_ids that have been indexed."""
    from services.embedding_service import list_documents as _list
    return {"documents": _list()}
