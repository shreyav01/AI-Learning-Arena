"""
services/llm_service.py
-----------------------
Thin wrapper around the OpenRouter Chat Completions API.

OpenRouter proxies dozens of LLMs (GPT-4, Claude, Mistral, Llama, etc.)
through a single OpenAI-compatible endpoint. We use httpx (async-capable)
rather than the openai SDK to keep dependencies lean.

All prompts enforce grounding: the system message explicitly instructs the
model to answer ONLY from provided context, preventing hallucination.
"""

import httpx
from fastapi import HTTPException

from utils.config import settings

# ---------------------------------------------------------------------------
# HTTP client — shared across requests (connection pooling)
# ---------------------------------------------------------------------------
_client = httpx.AsyncClient(timeout=60.0)


async def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Send a chat completion request to OpenRouter.

    Args:
        system_prompt: Instructions / role definition for the model.
        user_prompt:   The actual user message (context + question).

    Returns:
        The model's response as a plain string.

    Raises:
        HTTPException(502) on upstream API errors.
    """
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # OpenRouter recommends these headers for better routing
        "HTTP-Referer": "https://notebooklm-backend.local",
        "X-Title": "NotebookLM Backend",
    }

    payload = {
        "model": settings.OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,   # Low temperature = more factual, less creative
        "max_tokens": 2048,
    }

    try:
        response = await _client.post(
            settings.OPENROUTER_BASE_URL,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter API error {e.response.status_code}: {e.response.text}",
        )
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Network error calling LLM: {e}")

    data = response.json()

    # Standard OpenAI-compatible response structure
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise HTTPException(status_code=502, detail=f"Unexpected LLM response shape: {data}")


# ---------------------------------------------------------------------------
# Grounded RAG — builds context window from retrieved chunks
# ---------------------------------------------------------------------------

GROUNDING_SYSTEM_PROMPT = """\
You are a precise study assistant. You MUST answer ONLY using the document \
excerpts provided in the context below. Do NOT use any outside knowledge. \
If the answer is not present in the provided context, say: \
"I could not find an answer in the uploaded documents."\
"""


async def answer_from_context(question: str, chunks: list[str]) -> str:
    """
    Answer a user question grounded strictly in the retrieved chunks.
    """
    context_block = "\n\n---\n\n".join(
        [f"[Excerpt {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)]
    )
    user_prompt = (
        f"CONTEXT FROM DOCUMENT:\n{context_block}\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer based solely on the context above."
    )
    return await call_llm(GROUNDING_SYSTEM_PROMPT, user_prompt)


# ---------------------------------------------------------------------------
# Study material generation
# ---------------------------------------------------------------------------

STUDY_SYSTEM_PROMPT = """\
You are an expert educator creating study materials STRICTLY from the provided \
document excerpts. Every question and answer must be derivable from the given \
context. Do NOT invent facts not present in the excerpts.\
"""


async def generate_study_material(chunks: list[str]) -> str:
    """
    Generate MCQs, short-answer, and long-answer questions from chunks.

    Returns raw LLM text — the route handler parses the structured output.
    """
    context_block = "\n\n---\n\n".join(
        [f"[Excerpt {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)]
    )

    user_prompt = f"""\
DOCUMENT EXCERPTS:
{context_block}

Using ONLY the above excerpts, generate the following study materials in \
strict JSON format (no markdown fences, pure JSON):

{{
  "mcqs": [
    {{
      "question": "...",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "correct_answer": "A",
      "explanation": "Brief explanation citing the text."
    }}
    // 5 items total
  ],
  "short_answer_questions": [
    {{
      "question": "...",
      "answer": "..."
    }}
    // 5 items total
  ],
  "long_answer_questions": [
    {{
      "question": "...",
      "answer": "..."
    }}
    // 3 items total
  ]
}}

Return ONLY valid JSON. No preamble, no explanation outside the JSON.
"""
    return await call_llm(STUDY_SYSTEM_PROMPT, user_prompt)
