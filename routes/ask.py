"""
routes/ask.py
-------------
POST /ask
  Performs RAG (Retrieval-Augmented Generation):
    1. Embed the user's question.
    2. Retrieve top-k similar chunks from the document's FAISS index.
    3. Send [context + question] to the LLM with a strict grounding prompt.
    4. Return the grounded answer.

The LLM is explicitly instructed NOT to answer from general knowledge,
ensuring responses are grounded in the uploaded document.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from services.embedding_service import search_similar_chunks
from services.llm_service import answer_from_context
from utils.config import settings

router = APIRouter()


class AskRequest(BaseModel):
    doc_id: str = Field(..., description="Document ID returned by /upload")
    question: str = Field(..., min_length=3, description="Question to ask about the document")
    top_k: int = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of chunks to retrieve (default from settings)",
    )


class AskResponse(BaseModel):
    doc_id: str
    question: str
    answer: str
    chunks_used: int
    retrieved_context: list[str]  # returned so client can inspect sources


@router.post("/ask", response_model=AskResponse, summary="Ask a question about a document (RAG)")
async def ask_question(body: AskRequest):
    """
    RAG pipeline:
    - Embeds the question with the same model used for document chunks.
    - FAISS nearest-neighbour search finds the most relevant passages.
    - LLM answers grounded strictly on those passages.
    """
    top_k = body.top_k or settings.TOP_K_CHUNKS

    # --- Retrieve relevant chunks ---
    relevant_chunks = search_similar_chunks(body.doc_id, body.question, top_k=top_k)

    # --- Generate grounded answer ---
    answer = await answer_from_context(body.question, relevant_chunks)

    return AskResponse(
        doc_id=body.doc_id,
        question=body.question,
        answer=answer,
        chunks_used=len(relevant_chunks),
        retrieved_context=relevant_chunks,
    )
