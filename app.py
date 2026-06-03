import streamlit as st
import streamlit.components.v1 as components
import json
import time
import re
import random
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── cache model ─────────────────────────────────────────────────────
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

_cached_model = load_embedding_model()
from src.embedder import set_model
set_model(_cached_model)

from src.retriever import retrieve, detect_query_intent
from src.generator import generate_answer_streaming, format_citations
from src.recommender import recommend_by_query, recommend_arxiv
from src.memory import ConversationMemory
from src.indexer import get_index_stats
from src.arxiv_fetcher import download_all
from src.index_manager import add_paper, add_new_papers, full_rebuild
from src.tts import text_to_speech
from src.flowchart import generate_flowchart

# ── page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuralScribe — Research RAG",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── DESIGN SYSTEM ────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

:root {
  --bg:        #050508;
  --bg1:       #0a0a10;
  --bg2:       #0f0f18;
  --bg3:       #161620;
  --border:    rgba(255,255,255,0.06);
  --border2:   rgba(255,255,255,0.12);
  --accent:    #7c6aff;
  --accent2:   #a78bfa;
  --cyan:      #22d3ee;
  --green:     #34d399;
  --amber:     #fbbf24;
  --red:       #f87171;
  --text:      #e2e8f0;
  --text2:     #94a3b8;
  --text3:     #64748b;
  --glow:      0 0 40px rgba(124,106,255,0.15);
  --glow2:     0 0 20px rgba(34,211,238,0.1);
}

* { box-sizing: border-box; }

/* ── Base ── */
.stApp {
  background: var(--bg) !important;
  font-family: 'DM Sans', sans-serif !important;
  color: var(--text) !important;
}

/* Animated mesh background */
.stApp::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse 80% 50% at 20% 10%, rgba(124,106,255,0.07) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 80%, rgba(34,211,238,0.05) 0%, transparent 60%),
    radial-gradient(ellipse 40% 60% at 50% 50%, rgba(167,139,250,0.03) 0%, transparent 70%);
  pointer-events: none;
  z-index: 0;
}

/* ── Typography ── */
h1,h2,h3,h4,h5,h6 {
  font-family: 'Syne', sans-serif !important;
  letter-spacing: -0.02em !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: var(--bg1) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div {
  padding-top: 1.5rem !important;
}

/* ── Tabs ── */
div[data-baseweb="tab-list"] {
  background: transparent !important;
  gap: 4px !important;
  border-bottom: 1px solid var(--border) !important;
  padding-bottom: 0 !important;
}
button[data-baseweb="tab"] {
  font-family: 'Syne', sans-serif !important;
  font-size: 0.85rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.04em !important;
  text-transform: uppercase !important;
  padding: 10px 20px !important;
  border-radius: 6px 6px 0 0 !important;
  color: var(--text3) !important;
  background: transparent !important;
  border: none !important;
  transition: all 0.2s ease !important;
}
button[data-baseweb="tab"]:hover {
  color: var(--text2) !important;
  background: rgba(255,255,255,0.03) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
  color: var(--accent2) !important;
  background: rgba(124,106,255,0.08) !important;
  border-bottom: 2px solid var(--accent) !important;
}

/* ── Chat messages ── */
div[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  padding: 0.25rem 0 !important;
}

/* ── Chat input ── */
div[data-testid="stChatInput"] {
  background: var(--bg2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 12px !important;
  box-shadow: var(--glow) !important;
}
div[data-testid="stChatInput"]:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 1px var(--accent), var(--glow) !important;
}

/* ── Buttons ── */
.stButton > button {
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.8rem !important;
  letter-spacing: 0.05em !important;
  border-radius: 8px !important;
  border: 1px solid var(--border2) !important;
  background: var(--bg3) !important;
  color: var(--text2) !important;
  transition: all 0.2s ease !important;
  padding: 6px 14px !important;
}
.stButton > button:hover {
  border-color: var(--accent) !important;
  color: var(--accent2) !important;
  background: rgba(124,106,255,0.08) !important;
  transform: translateY(-1px) !important;
}

/* ── Expanders ── */
div[data-testid="stExpander"] {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  overflow: hidden !important;
}
div[data-testid="stExpander"] summary {
  font-family: 'Syne', sans-serif !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.03em !important;
  color: var(--text2) !important;
  padding: 10px 14px !important;
}
div[data-testid="stExpander"] summary:hover {
  color: var(--accent2) !important;
  background: rgba(124,106,255,0.05) !important;
}

/* ── Metrics ── */
div[data-testid="stMetricValue"] {
  font-family: 'Syne', sans-serif !important;
  font-weight: 800 !important;
  font-size: 1.6rem !important;
  color: var(--accent2) !important;
}
div[data-testid="stMetricLabel"] {
  font-size: 0.72rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  color: var(--text3) !important;
}

/* ── Sliders ── */
div[data-testid="stSlider"] div[role="slider"] {
  background: var(--accent) !important;
}

/* ── Toggle ── */
label[data-testid="stToggleLabel"] {
  font-size: 0.85rem !important;
  color: var(--text2) !important;
}

/* ── Divider ── */
hr {
  border-color: var(--border) !important;
  margin: 1rem 0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--bg3); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Custom components ── */
.ns-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 0.25rem;
}
.ns-logo {
  width: 36px; height: 36px;
  background: linear-gradient(135deg, var(--accent), var(--cyan));
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.1rem;
  box-shadow: 0 0 20px rgba(124,106,255,0.4);
  flex-shrink: 0;
}
.ns-title {
  font-family: 'Syne', sans-serif;
  font-size: 1.05rem;
  font-weight: 800;
  letter-spacing: -0.01em;
  color: var(--text);
  line-height: 1.1;
}
.ns-subtitle {
  font-size: 0.7rem;
  color: var(--text3);
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.page-title {
  font-family: 'Syne', sans-serif;
  font-size: 2.8rem;
  font-weight: 800;
  letter-spacing: -0.04em;
  line-height: 1;
  background: linear-gradient(135deg, #e2e8f0 0%, var(--accent2) 50%, var(--cyan) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 0.3rem;
}
.page-sub {
  font-size: 0.95rem;
  color: var(--text3);
  margin-bottom: 2rem;
  font-weight: 300;
}

.chip {
  display: inline-block;
  background: rgba(124,106,255,0.12);
  border: 1px solid rgba(124,106,255,0.25);
  border-radius: 20px;
  padding: 2px 10px;
  font-size: 0.72rem;
  font-family: 'Syne', sans-serif;
  font-weight: 600;
  letter-spacing: 0.04em;
  color: var(--accent2);
}
.chip-green {
  background: rgba(52,211,153,0.1);
  border-color: rgba(52,211,153,0.25);
  color: var(--green);
}
.chip-cyan {
  background: rgba(34,211,238,0.1);
  border-color: rgba(34,211,238,0.25);
  color: var(--cyan);
}
.chip-amber {
  background: rgba(251,191,36,0.1);
  border-color: rgba(251,191,36,0.25);
  color: var(--amber);
}

.stat-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1rem;
  text-align: center;
  transition: border-color 0.2s;
}
.stat-card:hover { border-color: var(--border2); }
.stat-num {
  font-family: 'Syne', sans-serif;
  font-size: 1.8rem;
  font-weight: 800;
  color: var(--accent2);
  line-height: 1;
}
.stat-label {
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text3);
  margin-top: 4px;
}

.citation-row {
  padding: 10px 0;
  border-bottom: 1px solid var(--border);
}
.citation-row:last-child { border-bottom: none; }
.citation-title {
  font-family: 'Syne', sans-serif;
  font-size: 0.88rem;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 3px;
}
.citation-meta {
  font-size: 0.75rem;
  color: var(--text3);
}

.suggest-btn {
  display: inline-block;
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 14px;
  font-size: 0.8rem;
  color: var(--text2);
  cursor: pointer;
  margin: 4px;
  transition: all 0.2s;
  font-family: 'DM Sans', sans-serif;
}
.suggest-btn:hover {
  border-color: var(--accent);
  color: var(--accent2);
  background: rgba(124,106,255,0.08);
}

.section-label {
  font-family: 'Syne', sans-serif;
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text3);
  margin-bottom: 0.75rem;
}

.perf-bar-bg {
  background: var(--bg3);
  border-radius: 4px;
  height: 6px;
  overflow: hidden;
  margin-top: 4px;
}
.perf-bar-fill {
  height: 100%;
  border-radius: 4px;
  background: linear-gradient(90deg, var(--accent), var(--cyan));
  transition: width 0.8s ease;
}

.audio-hint {
  font-size: 0.75rem;
  color: var(--text3);
  font-style: italic;
  padding: 6px 0;
}

/* File uploader */
div[data-testid="stFileUploaderDropzone"] {
  background: var(--bg2) !important;
  border: 1px dashed var(--border2) !important;
  border-radius: 10px !important;
}
div[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--accent) !important;
  background: rgba(124,106,255,0.04) !important;
}

/* Success/error/info */
div[data-testid="stAlert"] {
  border-radius: 10px !important;
  border-left-width: 3px !important;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
  border-radius: 10px !important;
  overflow: hidden !important;
}

/* Spinner */
div[data-testid="stSpinner"] {
  color: var(--accent2) !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ── helpers ──────────────────────────────────────────────────────────

def strip_md(text: str) -> str:
    return re.sub(r'\*{1,2}|_{1,2}', '', str(text)).strip()

def render_mermaid(diagram_code: str, height: int = 380):
    match = re.search(r'```mermaid\s*([\s\S]*?)\s*```', diagram_code)
    raw = match.group(1).strip() if match else diagram_code.strip()
    html = f"""
    <div style="background:transparent; padding:4px;">
      <div class="mermaid">{raw}</div>
    </div>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{
        startOnLoad: true, theme: 'dark',
        themeVariables: {{
          primaryColor: '#1e1b4b', primaryTextColor: '#a78bfa',
          primaryBorderColor: '#7c6aff', lineColor: '#64748b',
          secondaryColor: '#0f172a', tertiaryColor: '#0f172a',
          background: 'transparent', mainBkg: '#16162a',
          nodeBorder: '#7c6aff', clusterBkg: '#0f172a',
          titleColor: '#e2e8f0', edgeLabelBackground: '#16162a',
          fontFamily: 'DM Sans, sans-serif'
        }}
      }});
    </script>"""
    components.html(html, height=height, scrolling=True)

def load_chunks_for_suggestions(n: int = 6) -> list[str]:
    """Sample questions from across all indexed papers."""
    try:
        p = Path("indexes/chunks_metadata.json")
        if not p.exists():
            return []
        with open(p) as f:
            chunks = json.load(f)
        # Get unique papers
        by_paper = {}
        for c in chunks:
            src = c["metadata"]["source"]
            if src not in by_paper:
                by_paper[src] = []
            by_paper[src].append(c)
        # Sample one chunk from each paper
        sampled = []
        papers = list(by_paper.keys())
        random.shuffle(papers)
        for paper in papers[:n]:
            paper_chunks = [c for c in by_paper[paper]
                           if c["metadata"]["section"] in
                           ("abstract","introduction","methodology","method","results")]
            if paper_chunks:
                sampled.append(random.choice(paper_chunks))
        # Build suggestion questions from titles
        suggestions = []
        question_templates = [
            "How does {} work?",
            "What is the key contribution of {}?",
            "Explain the methodology in {}",
            "What are the main results of {}?",
            "How does {} compare to prior work?",
            "What problem does {} solve?",
        ]
        for i, chunk in enumerate(sampled[:n]):
            title = strip_md(chunk["metadata"].get("title", "this paper"))
            short_title = title[:45] + ("…" if len(title) > 45 else "")
            tmpl = question_templates[i % len(question_templates)]
            suggestions.append(tmpl.format(short_title))
        return suggestions
    except Exception:
        return []

def perf_bar(value: float, color: str = "accent") -> str:
    pct = int(value * 100)
    return f"""<div class="perf-bar-bg"><div class="perf-bar-fill" style="width:{pct}%"></div></div>"""


# ── session state ────────────────────────────────────────────────────
defaults = {
    "memory": ConversationMemory(),
    "chat": [],
    "query_count": 0,
    "tts_audio": {},
    "flowcharts": {},
    "last_uploaded": None,
    "suggestions": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Refresh suggestions if empty
if not st.session_state.suggestions:
    st.session_state.suggestions = load_chunks_for_suggestions(6)


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="ns-header">
      <div class="ns-logo">⬡</div>
      <div>
        <div class="ns-title">NeuralScribe</div>
        <div class="ns-subtitle">Research Intelligence</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Index stats
    try:
        stats = get_index_stats()
        papers = stats.get("unique_papers", 0)
        chunks = stats.get("total_chunks", 0)
        vecs   = stats.get("faiss_vectors", 0)
        st.markdown(f"""
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:1rem;">
          <div class="stat-card">
            <div class="stat-num">{papers}</div>
            <div class="stat-label">Papers</div>
          </div>
          <div class="stat-card">
            <div class="stat-num">{chunks}</div>
            <div class="stat-label">Chunks</div>
          </div>
        </div>
        <div style="font-size:0.7rem; color:var(--text3); text-align:center; margin-bottom:0.5rem;">
          {vecs} vectors indexed
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        st.markdown('<p style="font-size:0.75rem; color:var(--text3);">Index not built yet</p>', unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="section-label">Retrieval</div>', unsafe_allow_html=True)
    use_hyde = st.toggle("HyDE expansion", value=True,
                         help="Generates a hypothetical answer to improve vector search")
    top_k = st.slider("Chunks", 3, 10, 5, label_visibility="collapsed")
    st.markdown(f'<div style="font-size:0.72rem; color:var(--text3); margin-top:-8px;">Retrieving top {top_k} chunks</div>', unsafe_allow_html=True)

    st.divider()

    # Eval scores
    scores_path = Path("indexes/eval_scores.json")
    if scores_path.exists():
        st.markdown('<div class="section-label">Eval Scores</div>', unsafe_allow_html=True)
        try:
            with open(scores_path) as f:
                all_scores = json.load(f)
            for label, s in list(all_scores.items())[-1:]:
                if s:
                    faith = s.get('faithfulness', 0)
                    relev = s.get('relevance', s.get('answer_relevancy', 0))
                    compl = s.get('completeness', s.get('context_precision', 0))
                    st.markdown(f"""
                    <div style="margin-bottom:0.5rem;">
                      <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:var(--text3); margin-bottom:2px;">
                        <span>Faithfulness</span><span style="color:var(--green);">{faith:.2f}</span>
                      </div>
                      {perf_bar(faith)}
                      <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:var(--text3); margin:6px 0 2px;">
                        <span>Relevance</span><span style="color:var(--cyan);">{relev:.2f}</span>
                      </div>
                      {perf_bar(relev)}
                    </div>
                    """, unsafe_allow_html=True)
        except Exception:
            pass

    st.divider()

    col1, col2 = st.columns(2)
    if col1.button("Clear", use_container_width=True):
        st.session_state.memory.clear()
        st.session_state.chat = []
        st.session_state.tts_audio = {}
        st.session_state.flowcharts = {}
        st.session_state.suggestions = load_chunks_for_suggestions(6)
        st.rerun()
    if col2.button("Save", use_container_width=True):
        st.session_state.memory.save()
        st.success("Saved")

    turns = len(st.session_state.chat) // 2
    st.markdown(f'<div style="font-size:0.7rem; color:var(--text3); text-align:center; margin-top:0.5rem;">{turns} turns · {st.session_state.query_count} queries</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="page-title">Research Intelligence</div>
<div class="page-sub">Hybrid RAG · Multi-paper synthesis · Local LLM · No cloud required</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["  PAPERS  ", "  CHAT  ", "  BENCHMARK  "])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — PAPERS
# ══════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-label">Upload PDF</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop a research paper",
            type=["pdf"],
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            if st.session_state.get("last_uploaded") != uploaded_file.name:
                papers_dir = Path("data/papers")
                papers_dir.mkdir(parents=True, exist_ok=True)
                pdf_path = papers_dir / uploaded_file.name
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                with st.spinner(f"Indexing…"):
                    try:
                        add_paper(pdf_path)
                        st.session_state["last_uploaded"] = uploaded_file.name
                        st.session_state.suggestions = load_chunks_for_suggestions(6)
                        st.success(f"✓ Indexed: {uploaded_file.name[:50]}")
                        time.sleep(1.0)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.success(f"✓ Already indexed: {uploaded_file.name[:50]}")

        st.markdown('<div class="section-label" style="margin-top:1.5rem;">Index Maintenance</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button("Scan New", use_container_width=True):
            with st.spinner("Scanning…"):
                try:
                    add_new_papers("data/papers")
                    st.session_state.suggestions = load_chunks_for_suggestions(6)
                    st.success("Done")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        if c2.button("Full Rebuild", use_container_width=True):
            with st.spinner("Rebuilding…"):
                try:
                    full_rebuild("data/papers")
                    st.session_state.suggestions = load_chunks_for_suggestions(6)
                    st.success("Done")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    with col_right:
        st.markdown('<div class="section-label">Import from arXiv</div>', unsafe_allow_html=True)
        arxiv_input = st.text_input(
            "Topics or arXiv IDs",
            placeholder="e.g. Attention mechanism, GAN, 1706.03762",
            label_visibility="collapsed"
        )
        c1, c2 = st.columns(2)
        max_papers = c1.slider("Max per topic", 1, 10, 3)
        min_year   = c2.slider("Min year", 2010, 2026, 2018)

        if st.button("Download & Index", use_container_width=True):
            if not arxiv_input:
                st.warning("Enter at least one topic.")
            else:
                topics = [t.strip() for t in arxiv_input.split(",") if t.strip()]
                config = {"topics": topics, "max_per_topic": max_papers,
                          "output_dir": "data/papers", "min_year": min_year}
                with st.spinner("Downloading from arXiv…"):
                    try:
                        new_dl = download_all(config)
                        if new_dl:
                            st.success(f"Downloaded {len(new_dl)} papers")
                            with st.spinner("Indexing…"):
                                add_new_papers("data/papers")
                            st.session_state.suggestions = load_chunks_for_suggestions(6)
                            st.rerun()
                        else:
                            st.info("No new papers (already downloaded).")
                    except Exception as e:
                        st.error(str(e))

    # Paper list
    st.divider()
    st.markdown('<div class="section-label">Indexed Papers</div>', unsafe_allow_html=True)
    try:
        with open("indexes/chunks_metadata.json") as f:
            all_chunks = json.load(f)
        seen, papers_list = set(), []
        for c in all_chunks:
            m = c["metadata"]
            if m["source"] not in seen:
                seen.add(m["source"])
                papers_list.append(m)
        papers_list.sort(key=lambda x: x.get("year", ""), reverse=True)

        cols = st.columns(3)
        for i, p in enumerate(papers_list):
            with cols[i % 3]:
                title   = strip_md(p.get("title", p["source"]))
                authors = strip_md(p.get("authors", ""))
                year    = p.get("year", "")
                section = p.get("section", "")
                st.markdown(f"""
                <div class="stat-card" style="text-align:left; margin-bottom:8px;">
                  <div style="font-family:'Syne',sans-serif; font-size:0.8rem; font-weight:600;
                              color:var(--text); margin-bottom:4px; line-height:1.3;">
                    {title[:70]}{"…" if len(title)>70 else ""}
                  </div>
                  <div style="font-size:0.7rem; color:var(--text3);">{authors[:40]}</div>
                  <div style="margin-top:6px;">
                    <span class="chip chip-cyan">{year}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
    except Exception:
        st.markdown('<p style="color:var(--text3); font-size:0.85rem;">No papers indexed yet.</p>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — CHAT
# ══════════════════════════════════════════════════════════════════════
with tab2:

    # ── Render history ───────────────────────────────────────────────
    for idx, msg in enumerate(st.session_state.chat):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant":

                # Citations
                if msg.get("citations"):
                    cits = msg["citations"]
                    with st.expander(f"📄 Sources — {len(cits)} papers"):
                        for i, c in enumerate(cits, 1):
                            title   = strip_md(c["title"])
                            authors = strip_md(c["authors"])
                            doi_str = f' <a href="{c["doi"]}" style="color:var(--cyan);">DOI</a>' if c.get("doi") else ""
                            score_str = f'<span class="chip" style="font-size:0.65rem;">{c["score"]:.3f}</span>' if c.get("score") else ""
                            st.markdown(f"""
                            <div class="citation-row">
                              <div class="citation-title">{i}. {title[:80]}{doi_str}</div>
                              <div class="citation-meta">
                                {authors[:50]} · {c['year']} ·
                                <span class="chip chip-green">{c['section']}</span>
                                {score_str}
                              </div>
                            </div>
                            """, unsafe_allow_html=True)

                # Recommendations
                local_recs = msg.get("local_recs", [])
                arxiv_recs = msg.get("arxiv_recs", [])
                if local_recs or arxiv_recs:
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        if local_recs:
                            with st.expander(f"🔗 Similar — {len(local_recs)} local"):
                                for r in local_recs:
                                    sim = int(r["similarity"] * 100)
                                    title = strip_md(r["title"])
                                    st.markdown(f"""
                                    <div class="citation-row">
                                      <div class="citation-title">{title[:65]}</div>
                                      <div class="citation-meta">
                                        {strip_md(r['authors'])[:40]} · {r['year']} ·
                                        <span class="chip chip-amber">{sim}% match</span>
                                      </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                    with rc2:
                        if arxiv_recs:
                            with st.expander(f"🌐 arXiv — {len(arxiv_recs)} related"):
                                for r in arxiv_recs:
                                    title = strip_md(r["title"])
                                    st.markdown(f"""
                                    <div class="citation-row">
                                      <div class="citation-title">
                                        <a href="{r['url']}" target="_blank" style="color:var(--accent2); text-decoration:none;">{title[:65]}</a>
                                      </div>
                                      <div class="citation-meta">{r['authors'][:40]} · {r['year']}</div>
                                    </div>
                                    """, unsafe_allow_html=True)

                # TTS + Flowchart
                act1, act2 = st.columns(2)
                with act1:
                    if idx in st.session_state.tts_audio:
                        st.audio(st.session_state.tts_audio[idx], format="audio/wav")
                    else:
                        if st.button("▶ Listen", key=f"tts_{idx}", use_container_width=True):
                            with st.spinner("Synthesising speech…"):
                                audio = text_to_speech(msg["content"])
                                if audio:
                                    st.session_state.tts_audio[idx] = audio
                                    st.rerun()
                                else:
                                    st.error("TTS failed.")
                with act2:
                    if idx in st.session_state.flowcharts:
                        render_mermaid(st.session_state.flowcharts[idx])
                    else:
                        user_q = ""
                        if idx > 0 and st.session_state.chat[idx-1]["role"] == "user":
                            user_q = st.session_state.chat[idx-1]["content"]
                        if st.button("⬡ Flow Diagram", key=f"flow_{idx}", use_container_width=True):
                            with st.spinner("Generating diagram…"):
                                fc = generate_flowchart(user_q, msg["content"])
                                if fc:
                                    st.session_state.flowcharts[idx] = fc
                                    st.rerun()
                                else:
                                    st.error("Diagram generation failed.")

    # ── Suggested questions (multi-paper) ────────────────────────────
    if not st.session_state.chat and st.session_state.suggestions:
        st.markdown('<div class="section-label" style="margin-top:1rem; margin-bottom:0.5rem;">Suggested — from your papers</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, sug in enumerate(st.session_state.suggestions[:6]):
            with cols[i % 3]:
                if st.button(sug, key=f"sug_{i}", use_container_width=True):
                    st.session_state["_pending_query"] = sug
                    st.rerun()

    # ── Chat input ───────────────────────────────────────────────────
    query = st.chat_input("Ask anything about your papers…")

    # Handle suggestion click
    if "_pending_query" in st.session_state:
        query = st.session_state.pop("_pending_query")

    if query:
        st.session_state.query_count += 1
        st.session_state.chat.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        # Retrieve — BUG 1 FIX: use_hyde flag, never pass tuple
        with st.spinner("Searching…"):
            t1 = time.time()
            chunks    = retrieve(query, top_k=top_k, use_hyde=use_hyde)
            citations = format_citations(chunks)

        with st.chat_message("assistant"):
            answer = st.write_stream(
                generate_answer_streaming(query, chunks, st.session_state.memory.get())
            )

            local_recs = recommend_by_query(query, top_k=3)
            arxiv_recs = recommend_arxiv(query, top_k=3)

            if citations:
                with st.expander(f"📄 Sources — {len(citations)} papers", expanded=False):
                    for i, c in enumerate(citations, 1):
                        title   = strip_md(c["title"])
                        authors = strip_md(c["authors"])
                        doi_str = f' <a href="{c["doi"]}" style="color:var(--cyan);">DOI</a>' if c.get("doi") else ""
                        score_str = f'<span class="chip" style="font-size:0.65rem;">{c["score"]:.3f}</span>' if c.get("score") else ""
                        st.markdown(f"""
                        <div class="citation-row">
                          <div class="citation-title">{i}. {title[:80]}{doi_str}</div>
                          <div class="citation-meta">
                            {authors[:50]} · {c['year']} ·
                            <span class="chip chip-green">{c['section']}</span>
                            {score_str}
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

            if local_recs or arxiv_recs:
                rc1, rc2 = st.columns(2)
                with rc1:
                    if local_recs:
                        with st.expander(f"🔗 Similar — {len(local_recs)} local"):
                            for r in local_recs:
                                sim = int(r["similarity"] * 100)
                                title = strip_md(r["title"])
                                st.markdown(f"""
                                <div class="citation-row">
                                  <div class="citation-title">{title[:65]}</div>
                                  <div class="citation-meta">
                                    {strip_md(r['authors'])[:40]} · {r['year']} ·
                                    <span class="chip chip-amber">{sim}% match</span>
                                  </div>
                                </div>
                                """, unsafe_allow_html=True)
                with rc2:
                    if arxiv_recs:
                        with st.expander(f"🌐 arXiv — {len(arxiv_recs)} related"):
                            for r in arxiv_recs:
                                title = strip_md(r["title"])
                                st.markdown(f"""
                                <div class="citation-row">
                                  <div class="citation-title">
                                    <a href="{r['url']}" target="_blank" style="color:var(--accent2); text-decoration:none;">{title[:65]}</a>
                                  </div>
                                  <div class="citation-meta">{r['authors'][:40]} · {r['year']}</div>
                                </div>
                                """, unsafe_allow_html=True)

            new_idx = len(st.session_state.chat)
            act1, act2 = st.columns(2)
            with act1:
                st.markdown('<div class="audio-hint">▶ Listen available after refresh</div>', unsafe_allow_html=True)
            with act2:
                st.markdown('<div class="audio-hint">⬡ Flow diagram available after refresh</div>', unsafe_allow_html=True)

        st.session_state.memory.add("user", query)
        st.session_state.memory.add("assistant", answer)
        st.session_state.memory.add_topic(query[:50])
        st.session_state.chat.append({
            "role": "assistant", "content": answer,
            "citations": citations, "local_recs": local_recs, "arxiv_recs": arxiv_recs,
        })
        st.rerun()


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — BENCHMARK
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-label">Retrieval Configuration Benchmarks</div>', unsafe_allow_html=True)

    summary_path = Path("indexes/benchmark_summary.json")
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary_data = json.load(f)

            records = []
            for label, details in summary_data.items():
                ret   = details.get("retrieval", {})
                judge = details.get("judge", {})
                records.append({
                    "Config":       label,
                    "Hit@1":        ret.get("hit@1", 0.0),
                    "MRR":          ret.get("mrr", 0.0),
                    "nDCG@5":       ret.get("ndcg@5", 0.0),
                    "Faithfulness": judge.get("faithfulness", 0.0) if judge else 0.0,
                    "Relevance":    judge.get("relevance", 0.0) if judge else 0.0,
                    "Mean Score":   details.get("mean_score", 0.0),
                })

            df = pd.DataFrame(records)

            # Visual metric cards for best config
            best = df.loc[df["Mean Score"].idxmax()]
            cols = st.columns(4)
            metrics = [
                ("Hit@1",        best["Hit@1"],        "var(--green)"),
                ("MRR",          best["MRR"],           "var(--cyan)"),
                ("nDCG@5",       best["nDCG@5"],        "var(--accent2)"),
                ("Mean Score",   best["Mean Score"],    "var(--amber)"),
            ]
            for col, (label, val, color) in zip(cols, metrics):
                col.markdown(f"""
                <div class="stat-card">
                  <div style="font-size:0.65rem; text-transform:uppercase; letter-spacing:0.1em;
                              color:var(--text3); margin-bottom:4px;">{label}</div>
                  <div style="font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; color:{color};">
                    {val:.3f}
                  </div>
                  <div style="font-size:0.65rem; color:var(--text3); margin-top:2px;">
                    {best['Config'][:25]}
                  </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)

            # Per-config bars
            st.markdown('<div class="section-label">All Configurations</div>', unsafe_allow_html=True)
            for _, row in df.iterrows():
                mrr_pct  = int(row["MRR"] * 100)
                ndcg_pct = int(row["nDCG@5"] * 100)
                hit_pct  = int(row["Hit@1"] * 100)
                st.markdown(f"""
                <div class="stat-card" style="margin-bottom:10px; text-align:left;">
                  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span style="font-family:'Syne',sans-serif; font-size:0.85rem; font-weight:600; color:var(--text);">
                      {row['Config']}
                    </span>
                    <span class="chip">{row['Mean Score']:.3f} mean</span>
                  </div>
                  <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; font-size:0.72rem; color:var(--text3);">
                    <div>Hit@1 <strong style="color:var(--green);">{hit_pct}%</strong>{perf_bar(row['Hit@1'])}</div>
                    <div>MRR <strong style="color:var(--cyan);">{row['MRR']:.3f}</strong>{perf_bar(row['MRR'])}</div>
                    <div>nDCG@5 <strong style="color:var(--accent2);">{row['nDCG@5']:.3f}</strong>{perf_bar(row['nDCG@5'])}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # Raw table
            with st.expander("Raw data table"):
                st.dataframe(df.style.highlight_max(
                    subset=["MRR", "nDCG@5", "Mean Score"], color="#1f2c4d"
                ), use_container_width=True)

        except Exception as e:
            st.error(f"Failed to load benchmark: {e}")
    else:
        st.markdown("""
        <div class="stat-card" style="text-align:center; padding:2rem;">
          <div style="font-size:2rem; margin-bottom:0.5rem;">⬡</div>
          <div style="font-family:'Syne',sans-serif; font-weight:700; color:var(--text); margin-bottom:0.5rem;">
            No benchmark data yet
          </div>
          <div style="font-size:0.82rem; color:var(--text3);">
            Run <code style="background:var(--bg3); padding:2px 6px; border-radius:4px;">python run_benchmark.py</code> to generate results
          </div>
        </div>
        """, unsafe_allow_html=True)