"""
routes/upload.py
----------------
POST /upload
  Accepts a PDF file, extracts text, chunks it, generates embeddings,
  and stores a FAISS index. Returns a doc_id the client uses for /ask
  and /generate-questions.

Multipart form field: `file` (PDF only)
Optional form field:  `doc_id` (string, auto-generated UUID if omitted)
"""

import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from services.pdf_service import save_upload, extract_text
from services.embedding_service import embed_and_store
from utils.chunker import split_into_chunks

router = APIRouter()


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    pages_extracted: bool
    total_chunks: int
    message: str


@router.post("/upload", response_model=UploadResponse, summary="Upload and index a PDF")
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to upload"),
    doc_id: str = Form(None, description="Optional custom document ID (UUID generated if omitted)"),
):
    """
    Full pipeline for a new document:
    1. Validate it's a PDF.
    2. Save to disk.
    3. Extract text with pdfplumber.
    4. Split text into overlapping chunks.
    5. Generate sentence-transformer embeddings.
    6. Store FAISS index on disk.
    """
    # --- Validation ---
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are accepted.")

    # Generate a stable doc_id if not provided
    if not doc_id:
        doc_id = str(uuid.uuid4())

    # --- Step 1: Save ---
    pdf_path = save_upload(file, doc_id)

    # --- Step 2: Extract text ---
    full_text = extract_text(pdf_path)

    # --- Step 3: Chunk ---
    chunks = split_into_chunks(full_text)
    if not chunks:
        raise HTTPException(status_code=422, detail="Document produced no text chunks.")

    # --- Step 4 & 5: Embed + store FAISS ---
    total_chunks = embed_and_store(doc_id, chunks)

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        pages_extracted=True,
        total_chunks=total_chunks,
        message=(
            f"Document indexed successfully. "
            f"Use doc_id='{doc_id}' for /ask and /generate-questions."
        ),
    )
