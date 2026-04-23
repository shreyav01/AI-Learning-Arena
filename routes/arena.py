"""
routes/arena.py
---------------
POST /arena/evaluate
  The AI Learning Arena endpoint.

  Accepts:
    - doc_id         : which document to ground evaluation in
    - question       : the question being answered
    - student_answer : the student's submitted answer
    - top_k          : how many chunks to retrieve (default 6)

  Returns:
    - analyst  : factual correctness check
    - critic   : gaps and missing depth
    - teacher  : ideal answer explanation
    - verdict  : score, grade, summary (structured JSON)

POST /arena/question
  Pull a single random question from the document's generated question bank.
  Useful for "give me a question to answer" flow.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from services.embedding_service import search_similar_chunks
from services.arena_service import evaluate_answer
from services.llm_service import call_llm
from utils.config import settings

router = APIRouter(prefix="/arena", tags=["Arena"])


# ── Evaluate endpoint ────────────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    doc_id: str         = Field(..., description="Document ID from /upload")
    question: str       = Field(..., min_length=5)
    student_answer: str = Field(..., min_length=3)
    top_k: int          = Field(default=6, ge=1, le=20)


class VerdictModel(BaseModel):
    score:       int
    out_of:      int
    grade:       str
    summary:     str
    strengths:   str
    improvement: str


class EvaluateResponse(BaseModel):
    question:       str
    student_answer: str
    analyst:        str
    critic:         str
    teacher:        str
    verdict:        VerdictModel
    chunks_used:    int


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(body: EvaluateRequest):
    """
    Run the 4-agent jury evaluation on a student's answer.
    Analyst + Critic + Teacher run in parallel, then Verdict synthesises.
    """
    # Retrieve document context relevant to the question
    chunks = search_similar_chunks(body.doc_id, body.question, top_k=body.top_k)

    # Run the multi-agent pipeline
    result = await evaluate_answer(body.question, body.student_answer, chunks)

    return EvaluateResponse(
        question=body.question,
        student_answer=body.student_answer,
        analyst=result["analyst"],
        critic=result["critic"],
        teacher=result["teacher"],
        verdict=VerdictModel(**result["verdict"]),
        chunks_used=len(chunks),
    )


# ── Quick question generator ─────────────────────────────────────────────────

class QuickQuestionRequest(BaseModel):
    doc_id:     str = Field(..., description="Document ID from /upload")
    topic_hint: str = Field(default="key concepts", description="Optional topic focus")
    top_k:      int = Field(default=8, ge=3, le=20)


class QuickQuestionResponse(BaseModel):
    question:    str
    question_type: str   # "conceptual" | "applied" | "analytical"
    difficulty:  str     # "easy" | "medium" | "hard"


QUESTION_GEN_PROMPT = """\
You are a university professor creating exam questions.
Generate exactly ONE high-quality question based on the document excerpts.
The question should require genuine understanding, not just memorisation.

Respond in strict JSON only:
{
  "question": "Your question here?",
  "question_type": "conceptual",
  "difficulty": "medium"
}

question_type must be one of: conceptual, applied, analytical
difficulty must be one of: easy, medium, hard
No markdown, no extra text — pure JSON only.
"""


@router.post("/question", response_model=QuickQuestionResponse)
async def get_quick_question(body: QuickQuestionRequest):
    """
    Generate a single exam-style question from the document.
    Used by the Arena UI's 'Give me a question' button.
    """
    import json

    chunks = search_similar_chunks(body.doc_id, body.topic_hint, top_k=body.top_k)
    excerpts = "\n\n---\n\n".join(
        f"[Excerpt {i+1}]\n{c}" for i, c in enumerate(chunks)
    )
    user_prompt = f"DOCUMENT EXCERPTS:\n{excerpts}\n\nGenerate one question."

    raw = await call_llm(QUESTION_GEN_PROMPT, user_prompt)
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        data = json.loads(clean)
        return QuickQuestionResponse(**data)
    except Exception:
        return QuickQuestionResponse(
            question=raw[:300],
            question_type="conceptual",
            difficulty="medium",
        )
