"""
app.py — AI Learning Arena
===========================
NotebookLM Mini upgraded with multi-agent answer evaluation.

Tabs:
  01 Upload  — index a PDF
  02 Ask     — RAG question answering
  03 Study   — generate MCQs + short/long questions
  04 Arena   — submit your answer, get evaluated by 4 AI agents

Run:
    pip install streamlit requests
    streamlit run app.py
"""

import streamlit as st
import requests

# ══════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════
BACKEND_URL    = "http://127.0.0.1:8000"
UPLOAD_TIMEOUT = 600
ASK_TIMEOUT    = 120
STUDY_TIMEOUT  = 180
ARENA_TIMEOUT  = 180

# ══════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Learning Arena",
    page_icon="⚔️",
    layout="centered",
)

# ══════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;0,8..60,600;1,8..60,300;1,8..60,400&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Serif 4', Georgia, serif !important;
    background-color: #faf8f4 !important;
    color: #1c1a17 !important;
}
.stApp { background: #faf8f4 !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 860px !important; }

/* ── Masthead ── */
.masthead {
    background: #0f0e0c;
    padding: 2.2rem 3rem 1.8rem;
    position: relative;
    overflow: hidden;
}
.masthead::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        90deg,
        transparent,
        transparent 60px,
        rgba(200,168,112,0.03) 60px,
        rgba(200,168,112,0.03) 61px
    );
}
.masthead-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #c8a870;
    margin-bottom: 0.5rem;
}
.masthead-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1.05;
    color: #faf8f4;
    letter-spacing: -0.02em;
    margin-bottom: 0.45rem;
}
.masthead-title em { font-weight: 400; font-style: italic; color: #c8a870; }
.masthead-desc {
    font-family: 'Source Serif 4', serif;
    font-style: italic;
    font-size: 0.93rem;
    color: #6a6050;
    font-weight: 300;
}

/* ── Doc status bar ── */
.doc-bar {
    background: #f0ece2;
    border-bottom: 1px solid #d8d0be;
    padding: 0.62rem 3rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #6a6050;
    display: flex; align-items: center; gap: 0.55rem;
}
.doc-bar .dot { width:7px;height:7px;border-radius:50%;background:#5a8a5a;flex-shrink:0; }
.doc-bar .dot.off { background:#c0b8a0; }
.doc-bar strong { color:#1c1a17; font-weight:500; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #f0ece2 !important;
    border-bottom: 2px solid #0f0e0c !important;
    padding: 0 3rem !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.67rem !important;
    letter-spacing: 0.13em !important;
    text-transform: uppercase !important;
    color: #7a7060 !important;
    padding: 0.75rem 1.4rem !important;
    border-bottom: 3px solid transparent !important;
    margin-bottom: -2px !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #0f0e0c !important;
    border-bottom: 3px solid #c8a870 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding: 2.5rem 3rem !important;
    background: #faf8f4 !important;
}

/* ── Tab headings ── */
.tab-h { font-family:'Playfair Display',serif;font-size:1.55rem;font-weight:600;color:#1c1a17;margin-bottom:0.3rem;line-height:1.2; }
.tab-sub { font-family:'Source Serif 4',serif;font-style:italic;font-size:0.9rem;color:#8a8070;font-weight:300;margin-bottom:1.8rem;padding-bottom:1.2rem;border-bottom:1px solid #d8d0be; }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #fff !important;
    border: 1.5px solid #c8c0ae !important;
    border-radius: 3px !important;
    color: #1c1a17 !important;
    font-family: 'Source Serif 4', serif !important;
    font-size: 0.98rem !important;
    padding: 0.65rem 0.9rem !important;
    line-height: 1.6 !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #0f0e0c !important;
    box-shadow: 0 0 0 2px rgba(15,14,12,0.07) !important;
}
.stTextInput label, .stTextArea label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    color: #6a6050 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #f5f1e6 !important;
    border: 1.5px dashed #bfb090 !important;
    border-radius: 4px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #0f0e0c !important;
    color: #faf8f4 !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.58rem 1.8rem !important;
    transition: background 0.15s !important;
}
.stButton > button:hover { background: #3a3530 !important; }
.stButton > button:disabled { background:#d8d0be !important; color:#a09880 !important; }

/* ── Answer / source blocks ── */
.ans-label { font-family:'JetBrains Mono',monospace;font-size:0.6rem;letter-spacing:0.2em;text-transform:uppercase;color:#a09070;margin-bottom:0.55rem; }
.ans-block { background:#fff;border-left:3px solid #0f0e0c;border-radius:0 4px 4px 0;padding:1.4rem 1.8rem;font-size:1rem;line-height:1.85;color:#1c1a17;font-weight:300; }
.src-chunk { background:#f5f1e6;border-left:2px solid #c0b090;border-radius:0 3px 3px 0;padding:0.75rem 1rem;margin-bottom:0.5rem;font-family:'Source Serif 4',serif;font-style:italic;font-size:0.82rem;line-height:1.65;color:#5a5448; }
.src-num { font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:#a09070;font-style:normal;margin-bottom:0.25rem; }

/* ── MCQ / QA cards ── */
.mcq-card { background:#fff;border:1px solid #e0d8c6;border-radius:4px;padding:1.3rem 1.5rem;margin-bottom:0.9rem; }
.mcq-num { font-family:'JetBrains Mono',monospace;font-size:0.6rem;letter-spacing:0.18em;text-transform:uppercase;color:#a09070;margin-bottom:0.45rem; }
.mcq-q { font-family:'Source Serif 4',serif;font-size:0.98rem;font-weight:600;color:#1c1a17;margin-bottom:0.8rem;line-height:1.5; }
.mcq-opt { font-family:'Source Serif 4',serif;font-size:0.87rem;color:#5a5448;padding:0.2rem 0.35rem;border-radius:2px;line-height:1.5; }
.mcq-opt.ok { background:#ecf4ec;color:#2a5a2a;font-weight:600; }
.mcq-expl { margin-top:0.75rem;padding-top:0.65rem;border-top:1px solid #e8e0ce;font-family:'Source Serif 4',serif;font-style:italic;font-size:0.82rem;color:#8a8070;line-height:1.6; }
.qa-wrap { background:#fff;border:1px solid #e0d8c6;border-radius:4px;padding:1.2rem 1.6rem; }
.qa-item { margin-bottom:1.1rem;padding-bottom:1.1rem;border-bottom:1px solid #ece8de; }
.qa-item:last-child { margin-bottom:0;padding-bottom:0;border-bottom:none; }
.qa-q { font-family:'Playfair Display',serif;font-size:0.97rem;font-weight:600;color:#1c1a17;margin-bottom:0.4rem;line-height:1.45; }
.qa-a { font-family:'Source Serif 4',serif;font-size:0.9rem;color:#4a4438;line-height:1.78;font-weight:300; }

/* ── Section heads ── */
.sec-head { display:flex;align-items:center;gap:1rem;margin:2rem 0 1rem; }
.sec-tag { font-family:'JetBrains Mono',monospace;font-size:0.62rem;letter-spacing:0.18em;text-transform:uppercase;color:#faf8f4;background:#0f0e0c;padding:0.22rem 0.65rem;border-radius:2px;white-space:nowrap; }
.sec-line { flex:1;height:1px;background:#d8d0be; }

/* ══════════════════════════════════
   ARENA-SPECIFIC STYLES
══════════════════════════════════ */

/* Question card */
.arena-question {
    background: #0f0e0c;
    border-radius: 6px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.5rem;
}
.arena-question-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #c8a870;
    margin-bottom: 0.5rem;
}
.arena-question-text {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    font-style: italic;
    color: #faf8f4;
    line-height: 1.5;
}
.arena-question-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: #6a6050;
    margin-top: 0.5rem;
}

/* Score ring */
.verdict-panel {
    background: #0f0e0c;
    border-radius: 8px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 2rem;
}
.score-ring {
    flex-shrink: 0;
    width: 90px; height: 90px;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    border: 3px solid #c8a870;
}
.score-number {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #faf8f4;
    line-height: 1;
}
.score-denom {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: #6a6050;
}
.verdict-right {}
.verdict-grade {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    color: #c8a870;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.verdict-summary {
    font-family: 'Source Serif 4', serif;
    font-size: 1rem;
    color: #faf8f4;
    font-style: italic;
    line-height: 1.5;
    margin-bottom: 0.6rem;
}
.verdict-meta {
    font-family: 'Source Serif 4', serif;
    font-size: 0.82rem;
    color: #6a6050;
    line-height: 1.5;
}
.verdict-meta span { color: #a09070; }

/* Agent cards */
.agent-card {
    background: #fff;
    border: 1px solid #e0d8c6;
    border-radius: 6px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 0.9rem;
}
.agent-header {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 0.8rem;
}
.agent-icon {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem;
    flex-shrink: 0;
}
.agent-icon.analyst  { background: #e8f0f8; }
.agent-icon.critic   { background: #f8ece8; }
.agent-icon.teacher  { background: #eaf4ea; }
.agent-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #1c1a17;
    font-weight: 500;
}
.agent-role {
    font-family: 'Source Serif 4', serif;
    font-size: 0.75rem;
    font-style: italic;
    color: #a09070;
}
.agent-body {
    font-family: 'Source Serif 4', serif;
    font-size: 0.92rem;
    color: #2a2820;
    line-height: 1.78;
    font-weight: 300;
}

/* ── Misc ── */
.rule { border:none;border-top:1px solid #d8d0be;margin:1.8rem 0; }
[data-testid="stExpander"] { background:#f5f1e6 !important;border:1px solid #d8d0be !important;border-radius:4px !important; }
[data-testid="stAlert"] { font-family:'Source Serif 4',serif !important;font-size:0.9rem !important;border-radius:3px !important; }
[data-testid="stSpinner"] p { font-family:'JetBrains Mono',monospace !important;font-size:0.7rem !important;color:#8a8070 !important;letter-spacing:0.06em !important; }
.stCaption p { font-family:'Source Serif 4',serif !important;font-style:italic !important;font-size:0.82rem !important;color:#a09070 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════
for key, val in {
    "doc_id": None,
    "doc_name": None,
    "answer": None,
    "answer_q": None,
    "answer_chunks": [],
    "study_data": None,
    "arena_question": None,
    "arena_question_meta": None,
    "arena_eval": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ══════════════════════════════════════════════════════
#  MASTHEAD
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="masthead">
    <div class="masthead-eyebrow">⚔ &nbsp;Multi-Agent Learning System</div>
    <div class="masthead-title">AI Learning <em>Arena</em></div>
    <div class="masthead-desc">Upload · Ask · Study · Get evaluated by an AI jury</div>
</div>
""", unsafe_allow_html=True)

if st.session_state.doc_id:
    short_id = st.session_state.doc_id[:10] + "…"
    name = st.session_state.doc_name or "document"
    st.markdown(f"""
    <div class="doc-bar">
        <div class="dot"></div>
        <span>Active:&nbsp;<strong>{name}</strong></span>
        <span style="color:#c0b8a0">·</span>
        <span style="color:#a09070">{short_id}</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="doc-bar">
        <div class="dot off"></div>
        <span>No document loaded — upload a PDF to begin</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["01  Upload", "02  Ask", "03  Study", "04  Arena ⚔"])


# ──────────────────────────────────────────────────────
#  TAB 1 — UPLOAD
# ──────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="tab-h">Document Upload</div>', unsafe_allow_html=True)
    st.markdown('<div class="tab-sub">Select a PDF to index. Text is extracted, split into overlapping chunks, and a FAISS semantic index is built for retrieval.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Select a PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded_file:
        st.markdown(f"""
        <div style="background:#fff;border:1px solid #e0d8c6;border-radius:4px;
                    padding:0.9rem 1.2rem;margin:0.8rem 0 1rem;display:inline-block;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                        letter-spacing:0.18em;text-transform:uppercase;color:#a09070;
                        margin-bottom:0.3rem;">Selected file</div>
            <div style="font-family:'Source Serif 4',serif;font-size:0.95rem;color:#1c1a17;">
                📄 &nbsp;{uploaded_file.name}
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                        color:#a09070;margin-top:0.25rem;">{uploaded_file.size/1024:.1f} KB</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Index Document →", key="upload_btn"):
            with st.spinner("Extracting text · Chunking · Building FAISS index…"):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                        timeout=UPLOAD_TIMEOUT,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.doc_id    = data["doc_id"]
                        st.session_state.doc_name  = uploaded_file.name
                        st.session_state.answer    = None
                        st.session_state.study_data = None
                        st.session_state.arena_eval = None
                        st.success(f"✓  Indexed {data['total_chunks']} chunks from **{uploaded_file.name}**")
                        st.markdown(f"""
                        <div style="background:#f5f1e6;border:1px solid #d0c8b6;border-radius:4px;
                                    padding:1rem 1.3rem;margin-top:0.8rem;">
                            <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                                        letter-spacing:0.18em;text-transform:uppercase;
                                        color:#a09070;margin-bottom:0.4rem;">Document ID</div>
                            <div style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;
                                        color:#1c1a17;word-break:break-all;">{data['doc_id']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption("Switch to any tab to continue.")
                    else:
                        st.error(f"Upload failed ({resp.status_code}): {resp.json().get('detail','Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach backend. Is it running at `http://127.0.0.1:8000`?")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
    else:
        st.caption("Supported format: PDF · Text-based PDFs only")


# ──────────────────────────────────────────────────────
#  TAB 2 — ASK
# ──────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="tab-h">Ask a Question</div>', unsafe_allow_html=True)
    st.markdown('<div class="tab-sub">Your question is matched against the most relevant passages in the document. Answers are grounded strictly in your content.</div>', unsafe_allow_html=True)

    if not st.session_state.doc_id:
        st.info("Upload a document in the **Upload** tab first.")
    else:
        question = st.text_area("Your question", placeholder="e.g. What is the central argument?", height=105, key="q_input")

        if st.button("Submit Question →", disabled=not question.strip(), key="ask_btn"):
            with st.spinner("Retrieving passages · Generating grounded answer…"):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/ask",
                        json={"doc_id": st.session_state.doc_id, "question": question.strip(), "top_k": 5},
                        timeout=ASK_TIMEOUT,
                    )
                    if resp.status_code == 200:
                        d = resp.json()
                        st.session_state.answer        = d["answer"]
                        st.session_state.answer_q      = question.strip()
                        st.session_state.answer_chunks = d.get("retrieved_context", [])
                    else:
                        st.error(f"Error ({resp.status_code}): {resp.json().get('detail','Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach backend.")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

        if st.session_state.answer:
            st.markdown('<hr class="rule">', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                        letter-spacing:0.18em;text-transform:uppercase;color:#a09070;
                        margin-bottom:0.35rem;">Question</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.05rem;
                        font-style:italic;color:#1c1a17;margin-bottom:1.2rem;
                        line-height:1.5;">"{st.session_state.answer_q}"</div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="ans-label">Answer</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ans-block">{st.session_state.answer}</div>', unsafe_allow_html=True)
            chunks = st.session_state.answer_chunks
            if chunks:
                with st.expander(f"View {len(chunks)} source passages"):
                    for i, chunk in enumerate(chunks, 1):
                        st.markdown(f'<div class="src-chunk"><div class="src-num">Passage {i}</div>{chunk}</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────
#  TAB 3 — STUDY
# ──────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="tab-h">Generate Study Material</div>', unsafe_allow_html=True)
    st.markdown('<div class="tab-sub">Generates 5 MCQs, 5 short-answer, and 3 long-answer questions — grounded in your document.</div>', unsafe_allow_html=True)

    if not st.session_state.doc_id:
        st.info("Upload a document in the **Upload** tab first.")
    else:
        topic = st.text_input("Topic focus (optional)", placeholder="e.g. neural networks, supply and demand…", key="topic_input")

        if st.button("Generate Questions →", key="gen_btn"):
            with st.spinner("Analysing document · Composing questions…"):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/generate-questions",
                        json={"doc_id": st.session_state.doc_id, "topic_hint": topic.strip() or "key concepts, definitions, and important details", "top_k": 15},
                        timeout=STUDY_TIMEOUT,
                    )
                    if resp.status_code == 200:
                        st.session_state.study_data = resp.json()
                        # Clear any previous MCQ selections
                        for j in range(1, 10):
                            st.session_state.pop(f"mcq_{j}_answer", None)
                    else:
                        st.error(f"Error ({resp.status_code}): {resp.json().get('detail','Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach backend.")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

        if st.session_state.study_data:
            data = st.session_state.study_data
            st.markdown('<hr class="rule">', unsafe_allow_html=True)

            mcqs = data.get("mcqs", [])
            if mcqs:
                st.markdown('<div class="sec-head"><span class="sec-tag">Multiple Choice</span><div class="sec-line"></div></div>', unsafe_allow_html=True)

                # Track score across all MCQs
                answered   = 0
                correct_ct = 0

                for i, q in enumerate(mcqs, 1):
                    options     = q.get("options", {})
                    correct_key = q.get("correct_answer", "")
                    explanation = q.get("explanation", "")
                    session_key = f"mcq_{i}_answer"   # unique key per question

                    # Question heading
                    st.markdown(f"""
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                                letter-spacing:0.18em;text-transform:uppercase;color:#a09070;
                                margin: 1.2rem 0 0.3rem;">Question {i} / {len(mcqs)}</div>
                    <div style="font-family:'Source Serif 4',serif;font-size:1rem;
                                font-weight:600;color:#1c1a17;line-height:1.5;
                                margin-bottom:0.6rem;">{q.get('question','')}</div>
                    """, unsafe_allow_html=True)

                    # Build radio choices as "A. Option text"
                    choices = [
                        f"{k}.  {v}"
                        for k, v in options.items() if v
                    ]
                    # Add a blank default so nothing is pre-selected
                    radio_options = ["— Select an answer —"] + choices

                    selected = st.radio(
                        label=f"q{i}",
                        options=radio_options,
                        index=0,
                        key=session_key,
                        label_visibility="collapsed",
                    )

                    # Only show feedback once user picks a real option
                    if selected and selected != "— Select an answer —":
                        chosen_key = selected.split(".")[0].strip()   # extract "A" / "B" etc.
                        answered += 1
                        is_correct = chosen_key == correct_key

                        if is_correct:
                            correct_ct += 1
                            st.markdown(f"""
                            <div style="background:#eaf4ea;border-left:3px solid #5a8a5a;
                                        border-radius:0 4px 4px 0;padding:0.8rem 1.1rem;
                                        margin:0.4rem 0 0.8rem;">
                                <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                                            letter-spacing:0.1em;text-transform:uppercase;
                                            color:#2a5a2a;margin-bottom:0.3rem;">✓ &nbsp;Correct!</div>
                                <div style="font-family:'Source Serif 4',serif;font-size:0.88rem;
                                            color:#2a4a2a;line-height:1.65;">{explanation}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            correct_text = options.get(correct_key, "")
                            st.markdown(f"""
                            <div style="background:#f8eeec;border-left:3px solid #a05a5a;
                                        border-radius:0 4px 4px 0;padding:0.8rem 1.1rem;
                                        margin:0.4rem 0 0.8rem;">
                                <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                                            letter-spacing:0.1em;text-transform:uppercase;
                                            color:#7a2a2a;margin-bottom:0.3rem;">✗ &nbsp;Incorrect</div>
                                <div style="font-family:'Source Serif 4',serif;font-size:0.88rem;
                                            color:#4a2020;margin-bottom:0.4rem;">
                                    The correct answer is <strong>{correct_key}. {correct_text}</strong>
                                </div>
                                <div style="font-family:'Source Serif 4',serif;font-size:0.85rem;
                                            font-style:italic;color:#6a4040;line-height:1.65;">
                                    {explanation}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        # Subtle spacer
                        st.markdown('<div style="margin-bottom:0.6rem;"></div>', unsafe_allow_html=True)

                    st.markdown('<div style="border-top:1px solid #ece8de;margin:0.5rem 0 0.8rem;"></div>', unsafe_allow_html=True)

                # ── Score summary bar ──
                if answered > 0:
                    pct = int((correct_ct / answered) * 100)
                    bar_color = "#5a8a5a" if pct >= 70 else "#c8a870" if pct >= 40 else "#a05a5a"
                    st.markdown(f"""
                    <div style="background:#0f0e0c;border-radius:6px;padding:1.1rem 1.5rem;
                                margin:1rem 0;display:flex;align-items:center;gap:1.5rem;">
                        <div style="font-family:'Playfair Display',serif;font-size:1.8rem;
                                    font-weight:700;color:{bar_color};line-height:1;">
                            {correct_ct}/{answered}
                        </div>
                        <div>
                            <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;
                                        letter-spacing:0.18em;text-transform:uppercase;color:#6a6050;">
                                Score so far
                            </div>
                            <div style="font-family:'Source Serif 4',serif;font-size:0.9rem;
                                        color:#a09070;font-style:italic;">
                                {pct}% correct · {len(mcqs) - answered} question{'s' if len(mcqs)-answered != 1 else ''} remaining
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            short_qs = data.get("short_answer_questions", [])
            if short_qs:
                st.markdown('<div class="sec-head"><span class="sec-tag">Short Answer</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
                rows = "".join(f'<div class="qa-item"><div class="qa-q">{q.get("question","")}</div><div class="qa-a">{q.get("answer","")}</div></div>' for q in short_qs)
                st.markdown(f'<div class="qa-wrap">{rows}</div>', unsafe_allow_html=True)

            long_qs = data.get("long_answer_questions", [])
            if long_qs:
                st.markdown('<div class="sec-head"><span class="sec-tag">Long Answer</span><div class="sec-line"></div></div>', unsafe_allow_html=True)
                for i, q in enumerate(long_qs, 1):
                    preview = q.get("question","")
                    with st.expander(f"Q{i} — {preview[:75]}{'…' if len(preview)>75 else ''}"):
                        st.markdown(f'<div style="font-family:\'Playfair Display\',serif;font-size:0.98rem;font-weight:600;color:#1c1a17;margin-bottom:0.85rem;line-height:1.45;">{q.get("question","")}</div><div style="font-family:\'Source Serif 4\',serif;font-size:0.91rem;color:#4a4438;line-height:1.82;font-weight:300;">{q.get("answer","")}</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────
#  TAB 4 — ARENA
# ──────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="tab-h">AI Learning Arena ⚔</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="tab-sub">Answer a question, then face the AI Jury. '
        'Four agents — Analyst, Critic, Teacher, and Verdict — '
        'evaluate your answer and tell you exactly why you\'re right or wrong.</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.doc_id:
        st.info("Upload a document in the **Upload** tab first.")
    else:
        # ── Step 1: Get a question ──────────────────────
        st.markdown("""
        <div class="sec-head">
            <span class="sec-tag">Step 1 — Choose a Question</span>
            <div class="sec-line"></div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            topic_arena = st.text_input(
                "Topic hint (optional)",
                placeholder="e.g. ensemble methods, photosynthesis…",
                key="arena_topic",
            )
            if st.button("🎲  Generate Question from PDF", key="gen_q_btn"):
                with st.spinner("Generating question from your document…"):
                    try:
                        resp = requests.post(
                            f"{BACKEND_URL}/arena/question",
                            json={
                                "doc_id": st.session_state.doc_id,
                                "topic_hint": topic_arena.strip() or "key concepts",
                                "top_k": 8,
                            },
                            timeout=60,
                        )
                        if resp.status_code == 200:
                            d = resp.json()
                            st.session_state.arena_question = d["question"]
                            st.session_state.arena_question_meta = f"{d['question_type'].capitalize()} · {d['difficulty'].capitalize()}"
                            st.session_state.arena_eval = None
                        else:
                            st.error(f"Error: {resp.json().get('detail','Unknown error')}")
                    except Exception as e:
                        st.error(f"Error: {e}")

        with col2:
            custom_q = st.text_area(
                "Or type your own question",
                placeholder="Type any question about the document…",
                height=100,
                key="arena_custom_q",
            )
            if st.button("✏️  Use This Question", key="use_custom_q", disabled=not custom_q.strip()):
                st.session_state.arena_question = custom_q.strip()
                st.session_state.arena_question_meta = "Custom · Your question"
                st.session_state.arena_eval = None

        # ── Display active question ──────────────────────
        if st.session_state.arena_question:
            st.markdown(f"""
            <div class="arena-question">
                <div class="arena-question-label">📋 &nbsp;Active Question</div>
                <div class="arena-question-text">{st.session_state.arena_question}</div>
                <div class="arena-question-meta">{st.session_state.arena_question_meta or ""}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Step 2: Submit answer ────────────────────
            st.markdown("""
            <div class="sec-head">
                <span class="sec-tag">Step 2 — Write Your Answer</span>
                <div class="sec-line"></div>
            </div>
            """, unsafe_allow_html=True)

            student_answer = st.text_area(
                "Your answer",
                placeholder="Write your answer here. Be as detailed as you can — the jury evaluates depth and accuracy.",
                height=160,
                key="student_answer",
            )

            if st.button(
                "⚔  Submit to the AI Jury",
                disabled=not student_answer.strip(),
                key="submit_arena",
            ):
                with st.spinner("Analyst · Critic · Teacher deliberating… Verdict incoming…"):
                    try:
                        resp = requests.post(
                            f"{BACKEND_URL}/arena/evaluate",
                            json={
                                "doc_id": st.session_state.doc_id,
                                "question": st.session_state.arena_question,
                                "student_answer": student_answer.strip(),
                                "top_k": 6,
                            },
                            timeout=ARENA_TIMEOUT,
                        )
                        if resp.status_code == 200:
                            st.session_state.arena_eval = resp.json()
                        else:
                            st.error(f"Error ({resp.status_code}): {resp.json().get('detail','Unknown error')}")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot reach backend.")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")

        # ── Step 3: Show evaluation ──────────────────────
        if st.session_state.arena_eval:
            ev = st.session_state.arena_eval
            v  = ev["verdict"]

            st.markdown("""
            <div class="sec-head">
                <span class="sec-tag">Step 3 — Jury Verdict</span>
                <div class="sec-line"></div>
            </div>
            """, unsafe_allow_html=True)

            # Score ring + summary
            score = v.get("score", 0)
            grade = v.get("grade", "?")

            # Pick ring color based on score
            ring_color = "#5a8a5a" if score >= 7 else "#c8a870" if score >= 5 else "#a05a5a"

            st.markdown(f"""
            <div class="verdict-panel">
                <div class="score-ring" style="border-color:{ring_color};">
                    <div class="score-number" style="color:{ring_color};">{score}</div>
                    <div class="score-denom">/ 10</div>
                </div>
                <div class="verdict-right">
                    <div class="verdict-grade">Grade: {grade}</div>
                    <div class="verdict-summary">{v.get("summary","")}</div>
                    <div class="verdict-meta">
                        <span>Strengths: </span>{v.get("strengths","")}<br>
                        <span>To improve: </span>{v.get("improvement","")}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Agent cards
            st.markdown("""
            <div class="sec-head">
                <span class="sec-tag">Full Jury Feedback</span>
                <div class="sec-line"></div>
            </div>
            """, unsafe_allow_html=True)

            agents = [
                ("analyst",  "🔍", "analyst",  "Analyst",  "Checks factual correctness against the document"),
                ("critic",   "⚡", "critic",   "Critic",   "Identifies gaps and missing depth"),
                ("teacher",  "📖", "teacher",  "Teacher",  "Explains what an ideal answer looks like"),
            ]

            for key, icon, css_class, name, role in agents:
                st.markdown(f"""
                <div class="agent-card">
                    <div class="agent-header">
                        <div class="agent-icon {css_class}">{icon}</div>
                        <div>
                            <div class="agent-name">{name}</div>
                            <div class="agent-role">{role}</div>
                        </div>
                    </div>
                    <div class="agent-body">{ev.get(key, "")}</div>
                </div>
                """, unsafe_allow_html=True)

            st.caption(f"Evaluation grounded in {ev.get('chunks_used', 0)} document passages.")

        elif not st.session_state.arena_question:
            st.caption("Generate or type a question above to begin.")


# ══════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════
st.markdown(f"""
<div style="border-top:1px solid #d8d0be;margin-top:2.5rem;padding:1.1rem 3rem;
            font-family:'JetBrains Mono',monospace;font-size:0.6rem;
            letter-spacing:0.14em;color:#c0b8a0;text-align:center;">
    AI LEARNING ARENA &nbsp;·&nbsp; {BACKEND_URL}
</div>
""", unsafe_allow_html=True)
