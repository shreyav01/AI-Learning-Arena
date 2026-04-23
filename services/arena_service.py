"""
services/arena_service.py
--------------------------
The AI Learning Arena — multi-agent evaluation loop.

When a student submits an answer, it is evaluated by 4 sequential agents,
each with a distinct role. Each agent reads the document context + question
+ student answer, then contributes its unique perspective.

Agent pipeline:
  1. Analyst  — checks factual correctness against the source document
  2. Critic   — finds gaps, missing nuance, or weak reasoning
  3. Teacher  — explains what the ideal answer looks like (constructive)
  4. Verdict  — synthesises all three and assigns a score 0–10

Why sequential agents vs one big prompt?
  Separate agents produce cleaner, more focused outputs. Each "persona"
  is tuned to do one job well, preventing the model from trying to be
  correct + critical + explanatory all at once (which muddies all three).
"""

import asyncio
import json
from fastapi import HTTPException

from services.llm_service import call_llm


# ── Agent system prompts ─────────────────────────────────────────────────────

ANALYST_PROMPT = """\
You are the Analyst — a rigorous fact-checker.
Your ONLY job: compare the student's answer against the provided document excerpts.
Identify what is CORRECT, what is INCORRECT, and what is PARTIALLY correct.
Be specific. Quote or reference the document when pointing out errors.
Keep your response under 120 words. Use plain text, no bullet points.
"""

CRITIC_PROMPT = """\
You are the Critic — a sharp academic reviewer.
Your ONLY job: identify GAPS and MISSING DEPTH in the student's answer.
What important concepts from the document did they leave out?
What reasoning is shallow or underdeveloped?
Do NOT re-explain the topic — only point out what is missing or weak.
Keep your response under 120 words. Be direct and specific.
"""

TEACHER_PROMPT = """\
You are the Teacher — a clear, patient educator.
Your ONLY job: explain what an ideal answer would look like.
Use the document excerpts as your source. Make your explanation
easy to understand. Help the student see the full picture they missed.
Keep your response under 150 words. Write in a warm, encouraging tone.
"""

VERDICT_PROMPT = """\
You are the Verdict — the final judge.
You have access to: the question, the student's answer, the document context,
and feedback from the Analyst, Critic, and Teacher.
Your job: give a final score from 0 to 10 and a one-sentence summary.

Respond in strict JSON only (no markdown, no extra text):
{
  "score": 7,
  "out_of": 10,
  "grade": "B",
  "summary": "One sentence verdict on the student's performance.",
  "strengths": "What they did well in one sentence.",
  "improvement": "The single most important thing to improve."
}

Grade scale: 0-3=F, 4-5=D, 6=C, 7=B, 8=A-, 9=A, 10=A+
"""


# ── Helper to build the shared context block ────────────────────────────────

def _build_context(question: str, student_answer: str, chunks: list[str]) -> str:
    excerpts = "\n\n---\n\n".join(
        f"[Excerpt {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )
    return (
        f"DOCUMENT EXCERPTS:\n{excerpts}\n\n"
        f"QUESTION: {question}\n\n"
        f"STUDENT'S ANSWER: {student_answer}"
    )


# ── Individual agent callers ─────────────────────────────────────────────────

async def _run_analyst(context: str) -> str:
    return await call_llm(ANALYST_PROMPT, context)


async def _run_critic(context: str) -> str:
    return await call_llm(CRITIC_PROMPT, context)


async def _run_teacher(context: str) -> str:
    return await call_llm(TEACHER_PROMPT, context)


async def _run_verdict(
    context: str,
    analyst: str,
    critic: str,
    teacher: str,
) -> dict:
    verdict_context = (
        f"{context}\n\n"
        f"--- ANALYST FEEDBACK ---\n{analyst}\n\n"
        f"--- CRITIC FEEDBACK ---\n{critic}\n\n"
        f"--- TEACHER FEEDBACK ---\n{teacher}"
    )
    raw = await call_llm(VERDICT_PROMPT, verdict_context)

    # Strip any accidental markdown fences
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Fallback if model doesn't comply with JSON format
        return {
            "score": 0,
            "out_of": 10,
            "grade": "?",
            "summary": raw[:200],
            "strengths": "Unable to parse structured verdict.",
            "improvement": "Please try again.",
        }


# ── Main orchestrator ────────────────────────────────────────────────────────

async def evaluate_answer(
    question: str,
    student_answer: str,
    chunks: list[str],
) -> dict:
    """
    Run the full 4-agent evaluation pipeline.

    Steps:
      1. Analyst, Critic, Teacher run in PARALLEL (saves ~2/3 of wait time)
      2. Verdict runs after all three complete (needs their outputs)

    Returns a dict with all four agent responses.
    """
    context = _build_context(question, student_answer, chunks)

    # Step 1 — run first three agents concurrently
    analyst, critic, teacher = await asyncio.gather(
        _run_analyst(context),
        _run_critic(context),
        _run_teacher(context),
    )

    # Step 2 — verdict synthesises all three
    verdict = await _run_verdict(context, analyst, critic, teacher)

    return {
        "analyst":  analyst,
        "critic":   critic,
        "teacher":  teacher,
        "verdict":  verdict,
    }
