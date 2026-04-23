"""
services/pdf_service.py
-----------------------
Handles PDF upload persistence and text extraction using pdfplumber.

pdfplumber is chosen over PyPDF2 / pypdf because it handles:
  - Multi-column layouts
  - Tables (via page.extract_table())
  - Better whitespace normalisation

Flow:
  save_upload()  →  extract_text()  →  caller hands text to chunker
"""

import os
import shutil
import pdfplumber
from fastapi import UploadFile, HTTPException

from utils.config import settings


def save_upload(file: UploadFile, doc_id: str) -> str:
    """
    Persist the uploaded file to disk under UPLOAD_DIR/<doc_id>.pdf.

    Returns the absolute path to the saved file.
    """
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    dest_path = os.path.join(settings.UPLOAD_DIR, f"{doc_id}.pdf")

    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save PDF: {e}")

    return dest_path


def extract_text(pdf_path: str) -> str:
    """
    Extract all text from a PDF file.

    Strategy:
      1. Open with pdfplumber.
      2. For each page, extract plain text (preserves reading order).
      3. Join pages with double newline so chunk boundaries don't merge pages.

    Returns the full document text as a single string.
    Raises HTTPException(422) if no text could be extracted (e.g. scanned PDF).
    """
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"PDF not found at {pdf_path}")

    pages_text: list[str] = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    pages_text.append(text.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {e}")

    if not pages_text:
        raise HTTPException(
            status_code=422,
            detail=(
                "No extractable text found. "
                "The PDF may be scanned/image-based. "
                "Please use a text-based PDF."
            ),
        )

    full_text = "\n\n".join(pages_text)
    return full_text
