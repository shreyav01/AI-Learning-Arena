"""
routes/generate_questions.py
-----------------------------
POST /generate-questions
  Generates structured study material from an uploaded document:
    • 5 Multiple-Choice Questions (MCQs) with options A-D and correct answer
    • 5 Short-Answer Questions with concise answers
    • 3 Long-Answer Questions with detailed answers

Strategy:
  Instead of sending the entire document (which may exceed context limits),
  we retrieve the top-k chunks most relevant to the topic hint (or use a
  broad "summarize all key concepts" query if no topic is provided).
  This keeps prompt size manageable while covering the document's substance.
"""

import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.embedding_service import search_similar_chunks
from services.llm_service import generate_study_material
from utils.config import settings

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    doc_id: str = Field(..., description="Document ID returned by /upload")
    topic_hint: str = Field(
        default="key concepts, definitions, and important details",
        description=(
            "Optional topic to focus questions on. "
            "Leave blank to cover the document broadly."
        ),
    )
    top_k: int = Field(
        default=15,
        ge=5,
        le=30,
        description="Chunks to retrieve for context (more = broader coverage)",
    )


class MCQ(BaseModel):
    question: str
    options: dict[str, str]          # {"A": "...", "B": "...", ...}
    correct_answer: str              # "A" / "B" / "C" / "D"
    explanation: str


class QA(BaseModel):
    question: str
    answer: str


class GenerateResponse(BaseModel):
    doc_id: str
    mcqs: list[MCQ]
    short_answer_questions: list[QA]
    long_answer_questions: list[QA]
    chunks_used: int


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/generate-questions",
    response_model=GenerateResponse,
    summary="Generate MCQs and study questions from a document",
)
async def generate_questions(body: GenerateRequest):
    """
    Study material pipeline:
    1. Retrieve top-k chunks relevant to the topic hint via FAISS.
    2. Send chunks to LLM with a structured JSON prompt.
    3. Parse and validate the JSON response.
    4. Return typed study material.
    """
    # --- Retrieve broad context ---
    relevant_chunks = search_similar_chunks(
        body.doc_id,
        query=body.topic_hint,
        top_k=body.top_k,
    )

    # --- Generate study material ---
    raw_json_str = await generate_study_material(relevant_chunks)

    # --- Parse LLM JSON output ---
    try:
        # Strip potential markdown fences the model might add despite instructions
        clean = raw_json_str.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=502,
            detail=f"LLM returned invalid JSON: {e}. Raw output: {raw_json_str[:300]}",
        )

    # --- Validate structure ---
    try:
        mcqs = [MCQ(**item) for item in data.get("mcqs", [])]
        short_qs = [QA(**item) for item in data.get("short_answer_questions", [])]
        long_qs = [QA(**item) for item in data.get("long_answer_questions", [])]
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to parse study material structure: {e}",
        )

    # Warn (but don't fail) if counts are off
    if len(mcqs) < 5 or len(short_qs) < 5 or len(long_qs) < 3:
        # Still return whatever was generated
        pass

    return GenerateResponse(
        doc_id=body.doc_id,
        mcqs=mcqs,
        short_answer_questions=short_qs,
        long_answer_questions=long_qs,
        chunks_used=len(relevant_chunks),
    )
