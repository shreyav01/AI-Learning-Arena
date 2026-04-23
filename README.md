# AI-Learning-Arena
AI system that evaluates how you think, not just what you answer.Multi-agent AI system for evaluating reasoning and improving learning.

AI Learning Arena is a context-aware study assistant that evaluates user responses using a multi-agent AI framework. Instead of simply checking whether an answer is right or wrong, the system analyzes responses to identify gaps in understanding, provide clear explanations, and offer structured feedback.

The application allows users to upload their own study material (PDF or text), automatically generate relevant quiz questions, and interact with an AI-powered evaluation pipeline. It also supports context-aware question answering using a Retrieval-Augmented Generation (RAG) approach, ensuring responses are grounded in the user’s content.

By combining multiple AI roles — Analyst, Critic, Teacher, and Verdict — the system simulates a more realistic learning process focused on reasoning, improvement, and deeper understanding rather than just correctness.
A simplified NotebookLM-style backend built with **FastAPI**, **FAISS**, and **OpenRouter**.

Upload PDF documents, ask grounded questions via RAG, and generate structured study materials — all answers tied strictly to your uploaded content.

---

## Architecture

```
notebooklm-backend/
├── main.py                        # FastAPI app, CORS, startup
├── requirements.txt
├── .env.example                   # Copy → .env
│
├── routes/
│   ├── upload.py                  # POST /upload
│   ├── ask.py                     # POST /ask
│   └── generate_questions.py      # POST /generate-questions
│
├── services/
│   ├── pdf_service.py             # pdfplumber extraction
│   ├── embedding_service.py       # SentenceTransformer + FAISS
│   └── llm_service.py             # OpenRouter API calls
│
├── utils/
│   ├── config.py                  # Settings from .env
│   └── chunker.py                 # Overlapping text chunker
│
├── uploads/                       # Saved PDFs (auto-created)
└── vector_store/                  # FAISS indexes (auto-created)
```

---

## Quick Start

### 1. Install dependencies

```bash
cd notebooklm-backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

Get a free API key at [openrouter.ai](https://openrouter.ai).

### 3. Run the server

```bash
uvicorn main:app --reload --port 8000
```

Visit **http://localhost:8000/docs** for the interactive Swagger UI.

---

## API Reference

### `POST /upload`
Upload a PDF and index it.

**Form data:**
| Field    | Type   | Required | Description                          |
|----------|--------|----------|--------------------------------------|
| `file`   | File   | ✅       | PDF file                             |
| `doc_id` | string | ❌       | Custom ID (UUID auto-generated)      |

**Response:**
```json
{
  "doc_id": "abc-123",
  "filename": "lecture.pdf",
  "pages_extracted": true,
  "total_chunks": 47,
  "message": "Document indexed successfully..."
}
```

---

### `POST /ask`
Ask a question grounded in the document.

```json
{
  "doc_id": "abc-123",
  "question": "What are the main causes of inflation?",
  "top_k": 5
}
```

**Response:**
```json
{
  "doc_id": "abc-123",
  "question": "...",
  "answer": "According to the document...",
  "chunks_used": 5,
  "retrieved_context": ["chunk1...", "chunk2..."]
}
```

---

### `POST /generate-questions`
Generate MCQs, short-answer, and long-answer study questions.

```json
{
  "doc_id": "abc-123",
  "topic_hint": "supply and demand",
  "top_k": 15
}
```

**Response:**
```json
{
  "mcqs": [
    {
      "question": "...",
      "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
      "correct_answer": "B",
      "explanation": "..."
    }
  ],
  "short_answer_questions": [{"question": "...", "answer": "..."}],
  "long_answer_questions": [{"question": "...", "answer": "..."}]
}
```

---

### `GET /documents`
List all indexed document IDs.

### `GET /health`
Liveness probe.

---

## Design Decisions

| Decision | Rationale |
|---|---|
| Per-document FAISS index | Simpler isolation; no metadata filtering needed |
| `all-MiniLM-L6-v2` embeddings | Fast, 384-dim, good quality, runs on CPU |
| Overlapping chunks | Prevents context loss at chunk boundaries |
| Low LLM temperature (0.2) | Factual, consistent answers |
| Strict grounding prompt | Model told to refuse if answer not in context |
| OpenRouter | Single key for 50+ LLM providers; easy model swap |

---

## Changing the LLM

Edit `.env`:
```
OPENROUTER_MODEL=openai/gpt-4o
# or
OPENROUTER_MODEL=anthropic/claude-3-haiku
# or
OPENROUTER_MODEL=google/gemma-3-27b-it:free
```

See all models at [openrouter.ai/models](https://openrouter.ai/models).
