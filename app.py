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
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

_cached_model = load_embedding_model()
from src.retrieval.embedder import set_model
set_model(_cached_model)

from src.retrieval.retriever import retrieve, detect_query_intent
from src.generation.generator import generate_answer_streaming, format_citations
from src.ui.recommender import recommend_by_query, recommend_arxiv
from src.generation.memory import ConversationMemory
from src.download.arxiv_fetcher import download_all
from src.ui.tts import text_to_speech
from src.ui.flowchart import generate_flowchart

# ── page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Anthology — The Story Behind Every Discovery",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — Anthology: TTPD + Red era × Apple iOS × literary journal
# ══════════════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500;1,600;1,700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
  /* ── Palette ── */
  --bg:            #f7f4ef;
  --surface:       #ffffff;
  --surface2:      #faf8f5;
  --surface3:      #f2ede6;
  --sidebar-bg:    #f5f1ea;

  --border:        #e8e4de;
  --border2:       #ddd8d0;
  --border3:       #cec8be;

  --text:          #1c1917;
  --text2:         #44403c;
  --text3:         #78716c;
  --text4:         #a8a29e;
  --text5:         #c7bfb5;

  --accent:        #8b1a1a;
  --accent-dark:   #6d1414;
  --accent-soft:   rgba(139,26,26,0.07);
  --accent-mid:    rgba(139,26,26,0.13);
  --accent-light:  #fdf0f0;

  --gold:          #c4a882;
  --gold-dark:     #a8895e;
  --gold-soft:     rgba(196,168,130,0.12);
  --gold-mid:      rgba(196,168,130,0.22);

  --green:         #3d6b4f;
  --green-soft:    rgba(61,107,79,0.08);
  --blue:          #2c5282;
  --blue-soft:     rgba(44,82,130,0.08);

  --shadow-xs:  0 1px 3px rgba(28,25,23,0.05);
  --shadow-sm:  0 2px 8px rgba(28,25,23,0.06);
  --shadow-md:  0 4px 16px rgba(28,25,23,0.08);
  --shadow-lg:  0 8px 28px rgba(28,25,23,0.10);

  --radius-xs: 4px;
  --radius-sm: 8px;
  --radius:    12px;
  --radius-lg: 16px;
  --radius-xl: 24px;
}

* { box-sizing: border-box; }

/* ── Base ── */
.stApp {
  background: var(--bg) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  color: var(--text) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: var(--sidebar-bg) !important;
  border-right: 1px solid var(--border) !important;
  width: 230px !important;
  min-width: 230px !important;
}
section[data-testid="stSidebar"] > div { padding: 0 !important; }
section[data-testid="stSidebar"] .block-container { padding: 0 !important; }

/* ── Main ── */
.main .block-container {
  padding: 0 !important;
  max-width: 100% !important;
}

/* ── Typography ── */
h1, h2, h3 {
  font-family: 'Playfair Display', Georgia, serif !important;
  letter-spacing: -0.01em !important;
  color: var(--text) !important;
}
p { font-family: 'Inter', sans-serif !important; }

/* ── Tabs ── */
div[data-baseweb="tab-list"] {
  background: transparent !important;
  gap: 0 !important;
  border-bottom: 1px solid var(--border) !important;
  padding: 0 !important;
}
button[data-baseweb="tab"] {
  font-family: 'Inter', sans-serif !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  padding: 10px 18px !important;
  border-radius: 0 !important;
  color: var(--text3) !important;
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  margin-bottom: -1px !important;
  transition: all 0.15s ease !important;
  letter-spacing: 0.01em !important;
}
button[data-baseweb="tab"]:hover {
  color: var(--text2) !important;
  background: transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
  color: var(--accent) !important;
  background: transparent !important;
  border-bottom: 2px solid var(--accent) !important;
  font-weight: 600 !important;
}
div[data-testid="stTabPanel"] { padding-top: 1.5rem !important; }

/* ── Chat messages ── */
div[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  padding: 0.2rem 0 !important;
  max-width: 100% !important;
}

/* ── Chat input ── */
div[data-testid="stChatInput"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-xl) !important;
  box-shadow: var(--shadow-sm) !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
}
div[data-testid="stChatInput"]:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-soft), var(--shadow-sm) !important;
}
div[data-testid="stChatInput"] textarea {
  font-family: 'Inter', sans-serif !important;
  font-size: 0.9rem !important;
  color: var(--text) !important;
}
div[data-testid="stChatInput"] textarea::placeholder {
  color: var(--text5) !important;
  font-style: italic !important;
}
div[data-testid="stChatInput"] button {
  background: var(--accent) !important;
  border-radius: 50% !important;
  color: white !important;
}

/* ── Buttons ── */
.stButton > button {
  font-family: 'Inter', sans-serif !important;
  font-weight: 500 !important;
  font-size: 0.82rem !important;
  border-radius: var(--radius-sm) !important;
  border: 1px solid var(--border2) !important;
  background: var(--surface) !important;
  color: var(--text2) !important;
  transition: all 0.18s ease !important;
  padding: 7px 16px !important;
  box-shadow: var(--shadow-xs) !important;
  letter-spacing: 0.01em !important;
}
.stButton > button:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  background: var(--accent-light) !important;
  box-shadow: var(--shadow-sm) !important;
}
.stButton > button:active { transform: scale(0.98) !important; }

.stButton > button[kind="primary"] {
  background: var(--accent) !important;
  color: white !important;
  border-color: var(--accent) !important;
}
.stButton > button[kind="primary"]:hover {
  background: var(--accent-dark) !important;
  border-color: var(--accent-dark) !important;
  color: white !important;
}

/* ── Expanders ── */
div[data-testid="stExpander"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  overflow: hidden !important;
  box-shadow: var(--shadow-xs) !important;
  margin-bottom: 8px !important;
}
div[data-testid="stExpander"] summary {
  font-family: 'Inter', sans-serif !important;
  font-size: 0.84rem !important;
  font-weight: 500 !important;
  color: var(--text2) !important;
  padding: 12px 16px !important;
  background: var(--surface2) !important;
}

/* ── Metrics ── */
div[data-testid="stMetricValue"] {
  font-family: 'Playfair Display', serif !important;
  font-size: 2.4rem !important;
  color: var(--text) !important;
}
div[data-testid="stMetricLabel"] {
  font-size: 0.65rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.09em !important;
  color: var(--text3) !important;
  font-weight: 600 !important;
}

/* ── Text inputs ── */
div[data-testid="stTextInput"] input,
div[data-baseweb="input"] input {
  background: var(--surface) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.88rem !important;
  transition: border-color 0.18s, box-shadow 0.18s !important;
}
div[data-testid="stTextInput"] input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-soft) !important;
  outline: none !important;
}

/* ── Sliders ── */
div[data-testid="stSlider"] div[role="slider"] {
  background: var(--accent) !important;
  border: 2px solid white !important;
  box-shadow: var(--shadow-sm) !important;
}
div[data-testid="stSlider"] [data-testid="stSliderTrackFill"] {
  background: var(--accent) !important;
}

/* ── Toggle ── */
label[data-testid="stToggleLabel"] {
  font-size: 0.84rem !important;
  color: var(--text2) !important;
  font-family: 'Inter', sans-serif !important;
}
input[data-testid="stToggle"]:checked + div {
  background-color: var(--accent) !important;
}

/* ── Divider ── */
hr {
  border: none !important;
  border-top: 1px solid var(--border) !important;
  margin: 1rem 0 !important;
}

/* ── Alerts ── */
div[data-testid="stAlert"] {
  border-radius: var(--radius) !important;
  border-left-width: 3px !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.85rem !important;
}

/* ── File uploader ── */
div[data-testid="stFileUploaderDropzone"] {
  background: var(--surface2) !important;
  border: 1.5px dashed var(--border2) !important;
  border-radius: var(--radius) !important;
  transition: all 0.18s !important;
}
div[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--accent) !important;
  background: var(--accent-soft) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border3); border-radius: 10px; }

/* ── Select ── */
div[data-baseweb="select"] > div {
  background: var(--surface) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--radius-sm) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.84rem !important;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  overflow: hidden !important;
}

/* ══════════════════════════════════════════
   SIDEBAR COMPONENTS
══════════════════════════════════════════ */
.sb-wrap {
  display: flex;
  flex-direction: column;
  height: 100vh;
  padding: 0;
  font-family: 'Inter', sans-serif;
}

/* Wordmark */
.sb-logo {
  padding: 22px 18px 16px;
  border-bottom: 1px solid var(--border);
}
.sb-logo-text {
  font-family: 'Playfair Display', serif;
  font-style: italic;
  font-size: 1.45rem;
  color: var(--text);
  letter-spacing: -0.01em;
  line-height: 1.1;
}
.sb-logo-text span { color: var(--accent); }
.sb-logo-sub {
  font-size: 0.6rem;
  color: var(--text4);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-top: 4px;
  font-weight: 500;
  font-family: 'Inter', sans-serif;
}

/* Nav items */
.sb-nav { padding: 10px 8px; }
.sb-nav-item {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 8px 10px;
  border-radius: var(--radius-sm);
  font-size: 0.84rem;
  font-weight: 500;
  color: var(--text3);
  cursor: pointer;
  transition: all 0.14s;
  margin-bottom: 1px;
  text-decoration: none;
  font-family: 'Inter', sans-serif;
}
.sb-nav-item:hover { background: var(--surface3); color: var(--text2); }
.sb-nav-item.active {
  background: var(--accent-soft);
  color: var(--accent);
  font-weight: 600;
}

/* Section labels */
.sb-section {
  padding: 16px 18px 4px;
  font-size: 0.58rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text4);
  font-family: 'Inter', sans-serif;
}

/* Collection row */
.sb-collection {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 18px;
  font-size: 0.82rem;
  color: var(--text2);
  cursor: pointer;
  transition: color 0.12s;
  font-family: 'Inter', sans-serif;
}
.sb-collection:hover { color: var(--text); }
.sb-collection-name { display: flex; align-items: center; gap: 8px; }
.sb-collection-count {
  font-size: 0.68rem;
  color: var(--text4);
  font-weight: 500;
  background: var(--surface3);
  padding: 1px 7px;
  border-radius: 10px;
}

/* Tools section */
.sb-tool {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 6px 18px;
  font-size: 0.82rem;
  color: var(--text3);
  cursor: pointer;
  transition: color 0.14s;
  font-family: 'Inter', sans-serif;
}
.sb-tool:hover { color: var(--text2); }

/* User profile */
.sb-user {
  margin-top: auto;
  padding: 14px 16px;
  border-top: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 10px;
}
.sb-avatar {
  width: 32px; height: 32px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--accent) 0%, var(--gold) 100%);
  display: flex; align-items: center; justify-content: center;
  color: white; font-size: 0.7rem; font-weight: 700;
  flex-shrink: 0;
  font-family: 'Inter', sans-serif;
}
.sb-user-name { font-size: 0.82rem; font-weight: 600; color: var(--text); font-family: 'Inter', sans-serif; }
.sb-user-plan { font-size: 0.65rem; color: var(--gold-dark); font-style: italic; font-family: 'Playfair Display', serif; }

/* ══════════════════════════════════════════
   LIBRARY HEADER
══════════════════════════════════════════ */
.lib-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  padding: 28px 32px 0;
}
.lib-title {
  font-family: 'Playfair Display', serif;
  font-style: italic;
  font-size: 2.2rem;
  font-weight: 600;
  letter-spacing: -0.02em;
  color: var(--text);
  margin: 0;
  line-height: 1.1;
}
.lib-meta {
  font-size: 0.76rem;
  color: var(--text4);
  margin-top: 4px;
  font-family: 'Inter', sans-serif;
  font-style: italic;
}

/* ── Filter pills ── */
.filter-tab {
  padding: 5px 14px;
  border-radius: 20px;
  font-size: 0.78rem;
  font-weight: 500;
  font-family: 'Inter', sans-serif;
  cursor: pointer;
  color: var(--text3);
  border: 1px solid var(--border2);
  background: var(--surface);
  transition: all 0.15s;
}
.filter-tab:hover { background: var(--surface3); color: var(--text2); border-color: var(--border2); }
.filter-tab.active {
  background: var(--accent);
  color: white;
  border-color: var(--accent);
  font-weight: 600;
}

/* ── Section label ── */
.section-label {
  padding: 24px 32px 10px;
  font-size: 0.62rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text4);
  font-family: 'Inter', sans-serif;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.section-nav { display: flex; gap: 4px; }
.section-nav-btn {
  width: 26px; height: 26px;
  border-radius: var(--radius-xs);
  background: var(--surface);
  border: 1px solid var(--border2);
  display: flex; align-items: center; justify-content: center;
  cursor: pointer;
  color: var(--text3);
  transition: all 0.12s;
}
.section-nav-btn:hover { background: var(--surface3); color: var(--text2); }

/* ── Index card style for recent papers ── */
.recent-scroll {
  display: flex;
  gap: 14px;
  padding: 0 32px 24px;
  overflow-x: auto;
  scrollbar-width: none;
}
.recent-scroll::-webkit-scrollbar { display: none; }
.recent-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
  min-width: 230px;
  max-width: 250px;
  flex-shrink: 0;
  box-shadow: var(--shadow-sm);
  cursor: pointer;
  transition: box-shadow 0.18s, border-color 0.18s, transform 0.18s;
  position: relative;
  border-top: 3px solid var(--gold);
}
.recent-card:hover {
  box-shadow: var(--shadow-md);
  border-color: var(--border2);
  border-top-color: var(--accent);
  transform: translateY(-2px);
}
.recent-card-year {
  display: inline-block;
  font-size: 0.62rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  color: var(--accent);
  background: var(--accent-soft);
  border: 1px solid var(--accent-mid);
  border-radius: 4px;
  padding: 1px 7px;
  margin-bottom: 8px;
  font-family: 'Inter', sans-serif;
  text-transform: uppercase;
}
.recent-card-title {
  font-family: 'Playfair Display', serif;
  font-size: 0.92rem;
  font-weight: 500;
  color: var(--text);
  line-height: 1.4;
  margin-bottom: 7px;
}
.recent-card-authors {
  font-size: 0.72rem;
  color: var(--text3);
  margin-bottom: 2px;
  font-style: italic;
  font-family: 'Inter', sans-serif;
}
.recent-card-venue {
  font-size: 0.68rem;
  color: var(--text4);
  margin-bottom: 10px;
  font-family: 'Inter', sans-serif;
}
.recent-card-tags { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
.recent-card-foot {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid var(--border);
}
.rc-icon { color: var(--text4); font-size: 0.8rem; cursor: pointer; transition: color 0.12s; }
.rc-icon:hover { color: var(--accent); }

/* ── Papers table (journal log style) ── */
.papers-table-wrap { padding: 0 32px 36px; }
.papers-table {
  width: 100%;
  border-collapse: collapse;
  font-family: 'Inter', sans-serif;
}
.papers-table thead th {
  font-size: 0.62rem;
  font-weight: 700;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: var(--text4);
  padding: 8px 12px;
  text-align: left;
  border-bottom: 1px solid var(--border2);
}
.papers-table tbody tr {
  border-bottom: 1px solid var(--border);
  transition: background 0.1s;
  cursor: pointer;
}
.papers-table tbody tr:hover { background: var(--surface2); }
.papers-table tbody tr:last-child { border-bottom: none; }
.papers-table td {
  padding: 11px 12px;
  font-size: 0.83rem;
  color: var(--text2);
  vertical-align: middle;
}
.papers-table td:first-child {
  color: var(--text);
  font-weight: 500;
  font-family: 'Playfair Display', serif;
  font-size: 0.86rem;
}
.papers-table td .star-btn {
  color: var(--text5);
  margin-right: 8px;
  font-size: 0.8rem;
  cursor: pointer;
}
.papers-table td .star-btn:hover { color: var(--gold-dark); }
.more-btn { color: var(--text4); cursor: pointer; font-size: 1.1rem; letter-spacing: 1px; }
.more-btn:hover { color: var(--text2); }

/* ── Tags ── */
.tag {
  display: inline-block;
  background: var(--surface3);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 1px 8px;
  font-size: 0.65rem;
  font-family: 'Inter', sans-serif;
  font-weight: 500;
  color: var(--text3);
}
.tag-accent { background: var(--accent-soft);  border-color: var(--accent-mid); color: var(--accent); }
.tag-green  { background: var(--green-soft);   border-color: rgba(61,107,79,0.15);  color: var(--green); }
.tag-blue   { background: var(--blue-soft);    border-color: rgba(44,82,130,0.15);  color: var(--blue); }
.tag-amber  { background: var(--gold-soft);    border-color: var(--gold-mid);       color: var(--gold-dark); }
.tag-purple { background: rgba(107,70,193,0.07); border-color: rgba(107,70,193,0.15); color: #5b3fa6; }

/* ══════════════════════════════════════════
   PAPER DETAIL PANEL
══════════════════════════════════════════ */
.detail-panel {
  background: var(--surface);
  border-left: 1px solid var(--border);
  padding: 0;
  height: 100%;
  font-family: 'Inter', sans-serif;
}
.detail-back {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 14px 20px 12px;
  font-size: 0.76rem;
  color: var(--text3);
  border-bottom: 1px solid var(--border);
  cursor: pointer;
  font-style: italic;
}
.detail-back:hover { color: var(--accent); }
.detail-header {
  padding: 18px 20px 16px;
  border-bottom: 1px solid var(--border);
  background: linear-gradient(180deg, var(--accent-light) 0%, var(--surface) 100%);
}
.detail-pdf-icon {
  width: 44px; height: 52px;
  background: var(--accent-soft);
  border: 1px solid var(--accent-mid);
  border-radius: var(--radius-sm);
  display: flex; align-items: center; justify-content: center;
  font-size: 0.55rem; font-weight: 700;
  color: var(--accent);
  letter-spacing: 0.06em;
  margin-bottom: 10px;
  flex-shrink: 0;
  font-family: 'Inter', sans-serif;
}
.detail-title {
  font-family: 'Playfair Display', serif;
  font-style: italic;
  font-size: 1.1rem;
  font-weight: 500;
  color: var(--text);
  line-height: 1.35;
  margin-bottom: 5px;
}
.detail-authors { font-size: 0.76rem; color: var(--text3); margin-bottom: 2px; font-style: italic; }
.detail-venue   { font-size: 0.72rem; color: var(--text4); margin-bottom: 10px; }
.detail-tags    { display: flex; gap: 5px; flex-wrap: wrap; }
.detail-tabs {
  display: flex;
  border-bottom: 1px solid var(--border);
  padding: 0 20px;
}
.detail-tab {
  padding: 10px 0;
  margin-right: 20px;
  font-size: 0.78rem;
  font-weight: 500;
  color: var(--text3);
  cursor: pointer;
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
  transition: all 0.14s;
  font-family: 'Inter', sans-serif;
}
.detail-tab:hover { color: var(--text2); }
.detail-tab.active {
  color: var(--accent);
  border-bottom-color: var(--accent);
  font-weight: 600;
}
.detail-body { padding: 18px 20px; }
.detail-section-title {
  font-family: 'Playfair Display', serif;
  font-size: 0.88rem;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 8px;
}
.detail-abstract {
  font-size: 0.8rem;
  color: var(--text2);
  line-height: 1.7;
  font-style: italic;
}
.detail-show-more {
  color: var(--accent);
  font-size: 0.76rem;
  font-weight: 500;
  cursor: pointer;
  margin-top: 6px;
  display: inline-block;
  font-family: 'Inter', sans-serif;
}
.detail-stats {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin: 18px 0;
}
.detail-stat-label { font-size: 0.62rem; color: var(--text4); text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }
.detail-stat-value { font-family: 'Playfair Display', serif; font-size: 1rem; font-weight: 600; color: var(--text); margin-top: 2px; }
.detail-stat-value.high { color: var(--green); }
.detail-quick-actions {
  margin-top: 16px;
  padding-top: 16px;
  border-top: 1px solid var(--border);
}
.quick-actions-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-top: 10px;
}
.quick-action-btn {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 5px;
  padding: 11px 8px;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  font-size: 0.7rem;
  color: var(--text2);
  cursor: pointer;
  text-align: center;
  transition: all 0.14s;
  font-family: 'Inter', sans-serif;
  font-weight: 500;
}
.quick-action-btn:hover {
  background: var(--accent-light);
  border-color: var(--accent-mid);
  color: var(--accent);
}
.quick-action-icon { font-size: 1rem; }

/* ══════════════════════════════════════════
   CHAT LAYOUT
══════════════════════════════════════════ */
.chat-header {
  padding: 20px 28px 14px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.chat-header-title {
  font-family: 'Playfair Display', serif;
  font-style: italic;
  font-size: 1.4rem;
  color: var(--text);
}
.chat-header-sub {
  font-size: 0.72rem;
  color: var(--text4);
  margin-top: 2px;
  font-style: italic;
}

/* Tracklist-style suggestion chips */
.tracklist-wrap {
  padding: 36px 28px 20px;
  max-width: 680px;
}
.tracklist-heading {
  font-family: 'Playfair Display', serif;
  font-style: italic;
  font-size: 1.5rem;
  color: var(--text);
  margin-bottom: 6px;
}
.tracklist-sub {
  font-size: 0.8rem;
  color: var(--text4);
  margin-bottom: 24px;
  font-style: italic;
}

/* Sources panel */
.sources-panel-title {
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text3);
  margin-bottom: 12px;
  font-family: 'Inter', sans-serif;
}
/* Footnote-style source cards */
.source-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 3px solid var(--gold);
  border-radius: var(--radius-sm);
  padding: 10px 12px;
  margin-bottom: 8px;
  cursor: pointer;
  transition: all 0.14s;
}
.source-card:hover { border-left-color: var(--accent); box-shadow: var(--shadow-sm); }
.source-card-title {
  font-size: 0.78rem;
  font-weight: 500;
  color: var(--text);
  line-height: 1.35;
  margin-bottom: 4px;
  font-family: 'Playfair Display', serif;
}
.source-card-meta { font-size: 0.67rem; color: var(--text3); font-style: italic; }
.source-score {
  float: right;
  font-size: 0.68rem;
  font-weight: 600;
  color: var(--green);
  background: var(--green-soft);
  padding: 1px 6px;
  border-radius: 10px;
  margin-left: 6px;
  font-family: 'Inter', sans-serif;
}

/* ══════════════════════════════════════════
   BENCHMARK TAB — album credits aesthetic
══════════════════════════════════════════ */
.bench-title {
  font-family: 'Playfair Display', serif;
  font-style: italic;
  font-size: 1.8rem;
  color: var(--text);
}
.bench-sub { font-size: 0.78rem; color: var(--text3); margin-top: 3px; font-family: 'Inter', sans-serif; font-style: italic; }

/* Performance ring */
.perf-ring-wrap {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 28px 24px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-sm);
  border-top: 3px solid var(--accent);
}
.perf-ring-label {
  font-size: 0.62rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text4);
  font-weight: 700;
  margin-bottom: 14px;
  font-family: 'Inter', sans-serif;
}
.perf-ring-sub { font-size: 0.72rem; color: var(--text3); margin-top: 6px; font-style: italic; font-family: 'Playfair Display', serif; }

.bench-metric-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px 16px;
  box-shadow: var(--shadow-xs);
  text-align: center;
  border-top: 2px solid var(--gold);
}
.bench-metric-num {
  font-family: 'Playfair Display', serif;
  font-size: 2.1rem;
  line-height: 1;
  margin-bottom: 5px;
}
.bench-metric-label {
  font-size: 0.62rem;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: var(--text3);
  font-weight: 700;
  font-family: 'Inter', sans-serif;
}

/* Config rows */
.config-row {
  display: flex;
  align-items: center;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  font-family: 'Inter', sans-serif;
  background: var(--surface);
  transition: background 0.1s;
  cursor: pointer;
}
.config-row:hover { background: var(--surface2); }
.config-row:last-child { border-bottom: none; }
.config-rank { font-size: 0.75rem; font-weight: 700; color: var(--text4); width: 24px; flex-shrink: 0; }
.config-name { flex: 1; font-size: 0.84rem; font-weight: 500; color: var(--text); font-family: 'Playfair Display', serif; font-style: italic; }
.config-score-bar-wrap { width: 140px; margin: 0 14px; }
.config-score-bar-bg { background: var(--surface3); border-radius: 3px; height: 4px; overflow: hidden; }
.config-score-bar { height: 100%; border-radius: 3px; background: var(--accent); transition: width 0.7s cubic-bezier(0.4,0,0.2,1); }
.config-pct { font-size: 0.82rem; font-weight: 600; color: var(--text2); width: 36px; text-align: right; }

/* Perf bar */
.bar-bg { background: var(--surface3); border-radius: 3px; height: 4px; overflow: hidden; margin-top: 5px; }
.bar-fill { height: 100%; border-radius: 3px; background: var(--accent); transition: width 0.6s cubic-bezier(0.4,0,0.2,1); }
.bar-fill-green { background: var(--green); }
.bar-fill-blue  { background: var(--blue); }

/* ══════════════════════════════════════════
   RETRIEVAL SETTINGS
══════════════════════════════════════════ */
.settings-panel {
  background: var(--surface);
  border-left: 1px solid var(--border);
  height: 100%;
  overflow-y: auto;
  font-family: 'Inter', sans-serif;
}
.settings-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  border-bottom: 1px solid var(--border);
}
.settings-title {
  font-family: 'Playfair Display', serif;
  font-size: 0.96rem;
  font-weight: 600;
  color: var(--text);
}
.settings-close { color: var(--text3); cursor: pointer; font-size: 1.1rem; line-height: 1; transition: color 0.14s; }
.settings-close:hover { color: var(--accent); }
.settings-section { padding: 16px 20px; border-bottom: 1px solid var(--border); }
.settings-section-label {
  font-size: 0.62rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text4);
  margin-bottom: 14px;
}
.settings-row {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  margin-bottom: 14px;
  gap: 12px;
}
.settings-row:last-child { margin-bottom: 0; }
.settings-row-info { flex: 1; }
.settings-row-name { font-size: 0.84rem; font-weight: 500; color: var(--text); margin-bottom: 2px; }
.settings-row-desc { font-size: 0.72rem; color: var(--text3); line-height: 1.4; font-style: italic; }
.settings-footer { display: flex; gap: 8px; padding: 14px 20px; }
.settings-reset-btn {
  flex: 1;
  background: var(--surface);
  border: 1px solid var(--border2);
  border-radius: var(--radius-sm);
  padding: 9px;
  font-size: 0.8rem;
  font-weight: 500;
  font-family: 'Inter', sans-serif;
  color: var(--text2);
  cursor: pointer;
  text-align: center;
  transition: all 0.14s;
}
.settings-reset-btn:hover { background: var(--surface3); }
.settings-save-btn {
  flex: 1.5;
  background: var(--accent);
  border: none;
  border-radius: var(--radius-sm);
  padding: 9px;
  font-size: 0.8rem;
  font-weight: 600;
  font-family: 'Inter', sans-serif;
  color: white;
  cursor: pointer;
  text-align: center;
  transition: background 0.14s;
}
.settings-save-btn:hover { background: var(--accent-dark); }

/* ── Misc helpers ── */
.label {
  font-size: 0.62rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text3);
  margin-bottom: 0.65rem;
  font-family: 'Inter', sans-serif;
}
.audio-hint {
  font-size: 0.72rem;
  color: var(--text4);
  font-style: italic;
  padding: 6px 0;
  font-family: 'Inter', sans-serif;
}
.empty-state {
  text-align: center;
  padding: 3.5rem 2rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-sm);
}
.empty-icon  { font-size: 2rem; margin-bottom: 0.9rem; opacity: 0.3; }
.empty-title {
  font-family: 'Playfair Display', serif;
  font-style: italic;
  font-size: 1.15rem;
  color: var(--text2);
  margin-bottom: 0.5rem;
}
.empty-sub { font-size: 0.82rem; color: var(--text3); line-height: 1.6; font-style: italic; }
code {
  background: var(--surface3);
  border: 1px solid var(--border);
  border-radius: var(--radius-xs);
  padding: 1px 6px;
  font-size: 0.82em;
  color: var(--accent);
  font-family: 'SF Mono', 'Fira Code', monospace;
}
.cit-row { padding: 10px 0; border-bottom: 1px solid var(--border); font-family: 'Inter', sans-serif; }
.cit-row:last-child { border-bottom: none; }
.cit-title { font-family: 'Playfair Display', serif; font-size: 0.86rem; font-weight: 500; color: var(--text); margin-bottom: 3px; line-height: 1.35; font-style: italic; }
.cit-meta  { font-size: 0.7rem; color: var(--text3); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ── helpers ──────────────────────────────────────────────────────────
def strip_md(text: str) -> str:
    return re.sub(r'\*{1,2}|_{1,2}', '', str(text)).strip()

def esc(text: str) -> str:
    """HTML-escape dynamic content before injecting into unsafe_allow_html strings."""
    return (str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;"))

def render_mermaid(diagram_code: str, height: int = 360):
    match = re.search(r'```mermaid\s*([\s\S]*?)\s*```', diagram_code)
    raw = match.group(1).strip() if match else diagram_code.strip()
    html = f"""
    <div style="background:#fff; border-radius:12px; padding:16px; border:1px solid #e8e4de;">
      <div class="mermaid">{raw}</div>
    </div>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{startOnLoad:true,theme:'default',themeVariables:{{
        primaryColor:'#fdf0f0',primaryTextColor:'#1c1917',primaryBorderColor:'#8b1a1a',
        lineColor:'#a8a29e',fontFamily:'Inter,sans-serif'
      }}}});
    </script>"""
    components.html(html, height=height, scrolling=True)

def load_chunks_for_suggestions(n: int = 6) -> list[str]:
    try:
        import requests, random
        resp = requests.get(API_BASE + "/api/v1/papers", timeout=5)
        if resp.status_code != 200: return []
        papers = resp.json().get("papers", [])
        random.shuffle(papers)
        question_templates = [
            "How does {} work?","What is the key contribution of {}?",
            "Explain the methodology in {}","What are the main results of {}?",
            "How does {} compare to prior work?","What problem does {} solve?",
        ]
        suggestions = []
        for i, paper in enumerate(papers[:n]):
            title = paper.get("title", "this paper")
            short_title = title[:45] + ("…" if len(title) > 45 else "")
            suggestions.append(question_templates[i % len(question_templates)].format(short_title))
        return suggestions
    except Exception:
        return []

def perf_bar(value: float, color_class: str = "") -> str:
    pct = int(value * 100)
    return f'<div class="bar-bg"><div class="bar-fill {color_class}" style="width:{pct}%"></div></div>'

def get_tag_class(tag: str) -> str:
    tag_l = tag.lower()
    if any(x in tag_l for x in ["rag","retrieval","transformer","attention","bert","llm","lm"]):
        return "tag-blue"
    if any(x in tag_l for x in ["vision","image","clip","vit","diffusion","generative"]):
        return "tag-purple"
    if any(x in tag_l for x in ["gan","gnn","graph","gpt"]):
        return "tag-amber"
    if any(x in tag_l for x in ["nlp","text","language","peft","lora"]):
        return "tag-green"
    return ""


# ── session state ────────────────────────────────────────────────────
defaults = {
    "memory": ConversationMemory(),
    "chat": [],
    "query_count": 0,
    "tts_audio": {},
    "flowcharts": {},
    "last_uploaded": None,
    "suggestions": [],
    "selected_paper": None,
    "active_nav": "Library",
    "retrieval_settings_open": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if not st.session_state.suggestions:
    st.session_state.suggestions = load_chunks_for_suggestions(6)


# ── Load index stats ─────────────────────────────────────────────────
try:
    stats = get_index_stats()
    papers_count = stats.get("unique_papers", 0)
    chunks_count = stats.get("total_chunks", 0)
    vecs_count   = stats.get("vector_chunks", 0)
except Exception:
    papers_count = chunks_count = vecs_count = 0

# Load papers list once
papers_list = []
try:
    import requests as _req
    _resp = _req.get(API_BASE + "/api/v1/papers", timeout=5)
    if _resp.status_code == 200:
        papers_list = _resp.json().get("papers", [])
        papers_list.sort(key=lambda x: x.get("year", ""), reverse=True)
except Exception:
    pass


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════

COLLECTIONS = [
    ("LLMs", 12, "🟤"),
    ("Diffusion Models", 7, "🔵"),
    ("Computer Vision", 8, "🟣"),
    ("NLP", 6, "🟢"),
    ("Graphs & GNNs", 5, "🟡"),
]
TOOLS_LIST = [
    ("Retrieval Settings", "⚙"),
    ("Models", "◎"),
    ("Index Status", "◈"),
]

nav_icons = {
    "Library":   "📚",
    "Chat":      "💬",
    "Search":    "🔍",
    "Benchmark": "📊",
}

with st.sidebar:
    # Wordmark
    st.markdown("""
    <div class="sb-logo">
      <div class="sb-logo-text"><em>Anthology<span style="color:#8b1a1a;">.</span></em></div>
      <div class="sb-logo-sub">The Story Behind Every Discovery</div>
    </div>
    """, unsafe_allow_html=True)

    # Main nav
    for nav_name, icon in nav_icons.items():
        is_active = st.session_state.active_nav == nav_name
        if st.button(
            f"{icon}  {nav_name}",
            key=f"nav_{nav_name}",
            use_container_width=True,
        ):
            st.session_state.active_nav = nav_name
            st.rerun()

    st.divider()

    # Collections
    st.markdown('<div class="sb-section">Collections</div>', unsafe_allow_html=True)
    for cname, ccount, _ in COLLECTIONS:
        st.markdown(f"""
        <div class="sb-collection">
          <span class="sb-collection-name">
            <span style="font-size:0.5rem; color:var(--gold);">◉</span>
            {cname}
          </span>
          <span class="sb-collection-count">{ccount}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('<div style="padding:5px 18px;"><span style="font-size:0.76rem; color:var(--accent); font-weight:500; cursor:pointer; font-style:italic;">View all chapters →</span></div>', unsafe_allow_html=True)

    st.divider()

    # Tools
    st.markdown('<div class="sb-section">Tools</div>', unsafe_allow_html=True)
    for tname, ticon in TOOLS_LIST:
        if st.button(f"{ticon}  {tname}", key=f"tool_{tname}", use_container_width=True):
            if tname == "Retrieval Settings":
                st.session_state.retrieval_settings_open = not st.session_state.retrieval_settings_open
            st.rerun()

    # Index stats — literary framing
    st.markdown(f"""
    <div style="padding:14px 18px 8px; border-top:1px solid var(--border); margin-top:8px;">
      <div style="font-size:0.72rem; color:var(--text3); font-style:italic; font-family:'Playfair Display',serif; line-height:1.6;">
        <span style="color:var(--accent); font-weight:600;">{papers_count}</span> stories indexed<br>
        <span style="color:var(--gold-dark); font-weight:600;">{chunks_count}</span> passages catalogued
      </div>
    </div>
    """, unsafe_allow_html=True)

    # User profile
    st.markdown("""
    <div class="sb-user">
      <div class="sb-avatar">U</div>
      <div>
        <div class="sb-user-name">evermore</div>
        <div class="sb-user-plan">Taylor's Version</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    turns = len(st.session_state.chat) // 2
    st.markdown(
        f'<div style="font-size:0.62rem; color:var(--text5); text-align:center; '
        f'padding:6px 0 10px; font-family:Inter,sans-serif; font-style:italic;">'
        f'{turns} chapters · {st.session_state.query_count} queries</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════
# MAIN CONTENT — routed by active_nav
# ══════════════════════════════════════════════════════════════════════
active = st.session_state.active_nav


# ──────────────────────────────────────────────────────────────────────
# LIBRARY
# ──────────────────────────────────────────────────────────────────────
if active == "Library":

    if st.session_state.selected_paper is not None:
        main_col, detail_col = st.columns([3, 1.4], gap="small")
    else:
        main_col = st.container()
        detail_col = None

    with main_col:
        # ── Header ──
        st.markdown(f"""
        <div class="lib-header">
          <div>
            <div class="lib-title">The Collection</div>
            <div class="lib-meta">{papers_count} papers indexed · {chunks_count} passages catalogued</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Header row: search + add papers button
        h1, h2, h3 = st.columns([3, 1, 1])
        with h1:
            search_query = st.text_input(
                "Search papers",
                placeholder="Search titles, authors, ideas…",
                label_visibility="collapsed",
                key="lib_search"
            )
        with h3:
            if st.button("＋  Add to Collection", type="primary", use_container_width=True):
                st.session_state["show_add_papers"] = not st.session_state.get("show_add_papers", False)

        # ── Filter pills ──
        filter_col1, filter_col2, filter_col3, filter_col4, _ = st.columns([1,1,1,1,5])
        filter_state = st.session_state.get("lib_filter", "All Papers")
        with filter_col1:
            if st.button("All Papers", key="f_all", use_container_width=True,
                         type="primary" if filter_state == "All Papers" else "secondary"):
                st.session_state["lib_filter"] = "All Papers"; st.rerun()
        with filter_col2:
            if st.button("Favorites", key="f_fav", use_container_width=True,
                         type="primary" if filter_state == "Favorites" else "secondary"):
                st.session_state["lib_filter"] = "Favorites"; st.rerun()
        with filter_col3:
            if st.button("Recent", key="f_rec", use_container_width=True,
                         type="primary" if filter_state == "Recent" else "secondary"):
                st.session_state["lib_filter"] = "Recent"; st.rerun()
        with filter_col4:
            if st.button("🗑 Trash", key="f_trash", use_container_width=True):
                pass

        # ── Add papers panel ──
        if st.session_state.get("show_add_papers", False):
            with st.expander("Add to the Collection", expanded=True):
                add_t1, add_t2 = st.tabs(["Upload PDF", "Discover from the Archive"])
                with add_t1:
                    uploaded_file = st.file_uploader(
                        "Drop a paper into the collection",
                        type=["pdf"],
                        label_visibility="collapsed"
                    )
                    mc1, mc2 = st.columns(2)
                    if mc1.button("Scan new papers", use_container_width=True):
                        with st.spinner("Scanning the shelves…"):
                            try:
                                add_new_papers("data/papers")
                                st.session_state.suggestions = load_chunks_for_suggestions(6)
                                st.success("Collection updated"); st.rerun()
                            except Exception as e: st.error(str(e))
                    if mc2.button("Full rebuild", use_container_width=True):
                        with st.spinner("Rebuilding the anthology…"):
                            try:
                                full_rebuild("data/papers")
                                st.session_state.suggestions = load_chunks_for_suggestions(6)
                                st.success("Rebuilt"); st.rerun()
                            except Exception as e: st.error(str(e))

                    if uploaded_file is not None:
                        if st.session_state.get("last_uploaded") != uploaded_file.name:
                            papers_dir = Path("data/papers")
                            papers_dir.mkdir(parents=True, exist_ok=True)
                            pdf_path = papers_dir / uploaded_file.name
                            with open(pdf_path, "wb") as fh:
                                fh.write(uploaded_file.getbuffer())
                            with st.spinner(f"Adding to the collection…"):
                                try:
                                    add_paper(pdf_path)
                                    st.session_state["last_uploaded"] = uploaded_file.name
                                    st.session_state.suggestions = load_chunks_for_suggestions(6)
                                    st.success(f"Catalogued: {uploaded_file.name[:50]}")
                                    time.sleep(0.8); st.rerun()
                                except Exception as e: st.error(f"Error: {e}")
                        else:
                            st.success(f"Already in the collection: {uploaded_file.name[:50]}")

                with add_t2:
                    arxiv_input = st.text_input(
                        "Topics or arXiv IDs",
                        placeholder="e.g. Attention mechanism, GAN, 1706.03762",
                        label_visibility="collapsed"
                    )
                    ac1, ac2 = st.columns(2)
                    max_papers = ac1.slider("Max per topic", 1, 10, 3)
                    min_year   = ac2.slider("Min year", 2010, 2026, 2020)
                    if st.button("Discover & add to collection", use_container_width=True, type="primary"):
                        if not arxiv_input:
                            st.warning("Enter at least one topic to discover.")
                        else:
                            topics = [t.strip() for t in arxiv_input.split(",") if t.strip()]
                            config = {"topics": topics, "max_per_topic": max_papers,
                                      "output_dir": "data/papers", "min_year": min_year}
                            with st.spinner("Searching the archive…"):
                                try:
                                    new_dl = download_all(config)
                                    if new_dl:
                                        st.success(f"Found {len(new_dl)} new papers")
                                        with st.spinner("Cataloguing…"): add_new_papers("data/papers")
                                        st.session_state.suggestions = load_chunks_for_suggestions(6)
                                        st.rerun()
                                    else:
                                        st.info("Nothing new — your collection is up to date.")
                                except Exception as e: st.error(str(e))

        # ── Recently Added ──
        if papers_list:
            st.markdown("""
            <div class="section-label">
              <span>Recently Added</span>
              <div class="section-nav">
                <div class="section-nav-btn">‹</div>
                <div class="section-nav-btn">›</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            TAG_COLOR_MAP = {
                "Transformer": "tag-blue", "Attention": "tag-blue",
                "RAG": "tag-accent", "Retrieval": "tag-accent",
                "Diffusion": "tag-purple", "Generative": "tag-purple",
                "Vision": "tag-amber", "NLP": "tag-green",
                "GNN": "tag-amber", "LLM": "tag-blue", "PEFT": "tag-green",
            }

            recent_cards_html = '<div class="recent-scroll">'
            for i, p in enumerate(papers_list[:8]):
                title   = strip_md(p.get("title", p["source"]))
                authors = strip_md(p.get("authors", ""))
                year    = p.get("year", "")
                venue   = p.get("venue", "")

                detected_tags = []
                title_lower = title.lower()
                for kw, cls in TAG_COLOR_MAP.items():
                    if kw.lower() in title_lower and kw not in detected_tags:
                        detected_tags.append(kw)
                        if len(detected_tags) >= 2: break

                tags_html = "".join(
                    f'<span class="tag {TAG_COLOR_MAP.get(t, "")}">{t}</span>'
                    for t in detected_tags[:2]
                )
                if len(detected_tags) > 2:
                    tags_html += f'<span style="font-size:0.66rem; color:var(--text4);">+{len(detected_tags)-2}</span>'

                year_badge = f'<div class="recent-card-year">{year}</div>' if year else ''

                recent_cards_html += f"""
                <div class="recent-card">
                  {year_badge}
                  <div class="recent-card-title">{title[:60]}{"…" if len(title)>60 else ""}</div>
                  <div class="recent-card-authors">{authors[:35]}{"…" if len(authors)>35 else ""}</div>
                  <div class="recent-card-venue">{venue}</div>
                  <div class="recent-card-tags">{tags_html}</div>
                  <div class="recent-card-foot">
                    <span style="font-size:0.65rem; color:var(--gold-dark); font-style:italic; font-family:'Playfair Display',serif;">{"★ Featured" if i == 0 else ""}</span>
                    <span class="rc-icon">📄</span>
                  </div>
                </div>"""
            recent_cards_html += "</div>"
            st.markdown(recent_cards_html, unsafe_allow_html=True)

        # ── All Papers table ──
        st.markdown('<div class="section-label">All Papers</div>', unsafe_allow_html=True)

        if papers_list:
            filtered = papers_list
            if search_query:
                q = search_query.lower()
                filtered = [p for p in papers_list
                            if q in strip_md(p.get("title","")).lower()
                            or q in strip_md(p.get("authors","")).lower()]

            table_html = """
            <div class="papers-table-wrap">
            <table class="papers-table">
              <thead>
                <tr>
                  <th>TITLE</th>
                  <th>AUTHORS</th>
                  <th>YEAR</th>
                  <th>VENUE</th>
                  <th>TAGS</th>
                  <th>ADDED</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
            """
            ADDED_STUBS = ["2d ago", "3d ago", "5d ago", "1w ago", "2w ago", "1mo ago"]
            for i, p in enumerate(filtered[:20]):
                title   = strip_md(p.get("title",   p["source"]))
                authors = strip_md(p.get("authors", ""))
                year    = p.get("year", "—")
                venue   = p.get("venue", "—")
                added   = ADDED_STUBS[i % len(ADDED_STUBS)]

                detected_tags = []
                title_lower = title.lower()
                for kw in ["Transformer","RAG","Diffusion","Vision","NLP","GNN","LLM","PEFT","CLIP"]:
                    if kw.lower() in title_lower:
                        detected_tags.append(kw)
                        if len(detected_tags) >= 2: break
                tags_html = "".join(
                    f'<span class="tag {TAG_COLOR_MAP.get(t, "")}" style="margin-right:3px;">{t}</span>'
                    for t in detected_tags[:2]
                )

                table_html += f"""
                <tr>
                  <td>
                    <span class="star-btn">☆</span>
                    {title[:65]}{"…" if len(title)>65 else ""}
                  </td>
                  <td style="font-style:italic;">{authors[:30]}{"…" if len(authors)>30 else ""}</td>
                  <td><span style="color:var(--accent); font-weight:600; font-size:0.78rem;">{year}</span></td>
                  <td>{venue[:12] if venue and venue != "—" else "—"}</td>
                  <td>{tags_html}</td>
                  <td style="color:var(--text4); font-size:0.76rem; font-style:italic;">{added}</td>
                  <td><span class="more-btn">···</span></td>
                </tr>"""
            table_html += "</tbody></table></div>"
            st.markdown(table_html, unsafe_allow_html=True)

            if papers_list:
                paper_titles = [strip_md(p.get("title", p["source"]))[:60] for p in papers_list[:10]]
                selected_idx = st.selectbox(
                    "View paper details",
                    options=range(len(paper_titles)),
                    format_func=lambda i: paper_titles[i],
                    index=st.session_state.selected_paper if st.session_state.selected_paper is not None else 0,
                    key="paper_selector",
                    label_visibility="collapsed"
                )
                c1, c2 = st.columns(2)
                if c1.button("Open entry →", use_container_width=True):
                    st.session_state.selected_paper = selected_idx
                    st.rerun()
                if st.session_state.selected_paper is not None:
                    if c2.button("Close ✕", use_container_width=True):
                        st.session_state.selected_paper = None
                        st.rerun()
        else:
            st.markdown("""
            <div class="empty-state" style="margin:0 32px;">
              <div class="empty-icon">📖</div>
              <div class="empty-title">No papers yet.</div>
              <div class="empty-sub">Every anthology begins with a first page.<br>Upload a PDF or discover from the arXiv archive to begin.</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Detail panel ──
    if detail_col is not None and st.session_state.selected_paper is not None:
        with detail_col:
            idx = st.session_state.selected_paper
            if idx < len(papers_list):
                p = papers_list[idx]
                title   = strip_md(p.get("title",   p["source"]))
                authors = strip_md(p.get("authors", "Unknown"))
                year    = p.get("year", "")
                venue   = p.get("venue", "")

                detected_tags = []
                title_lower = title.lower()
                TAG_COLOR_MAP_D = {
                    "Transformer":"tag-blue","Attention":"tag-blue","RAG":"tag-accent",
                    "Diffusion":"tag-purple","Vision":"tag-amber","NLP":"tag-green",
                    "GNN":"tag-amber","LLM":"tag-blue","PEFT":"tag-green","CLIP":"tag-amber",
                }
                for kw in TAG_COLOR_MAP_D:
                    if kw.lower() in title_lower:
                        detected_tags.append((kw, TAG_COLOR_MAP_D[kw]))
                        if len(detected_tags) >= 3: break

                tags_html = "".join(
                    f'<span class="tag {cls}" style="margin-right:4px;">{kw}</span>'
                    for kw, cls in detected_tags
                )
                if not tags_html:
                    tags_html = '<span class="tag">Research</span>'

                abstract = p.get("abstract",
                    "The dominant sequence transduction models are based on recurrent or convolutional "
                    "layers. We propose a simple new architecture based solely on attention mechanisms, "
                    "dispensing with recurrence and convolutions entirely, enabling significantly more "
                    "parallelization and reaching state-of-the-art results.")

                st.markdown(f"""
                <div class="detail-panel">
                  <div class="detail-back">← Back to the collection</div>
                  <div class="detail-header">
                    <div style="display:flex; gap:12px; align-items:flex-start;">
                      <div class="detail-pdf-icon">PDF</div>
                      <div>
                        <div class="detail-title">{title[:80]}{"…" if len(title)>80 else ""}</div>
                        <div class="detail-authors">{authors[:50]}</div>
                        <div class="detail-venue">{venue} {year}</div>
                        <div class="detail-tags">{tags_html}</div>
                      </div>
                    </div>
                  </div>

                  <div class="detail-tabs">
                    <div class="detail-tab active">Overview</div>
                    <div class="detail-tab">Contents</div>
                    <div class="detail-tab">Notes</div>
                    <div class="detail-tab">Citations (342)</div>
                  </div>

                  <div class="detail-body">
                    <div class="detail-section-title">Abstract</div>
                    <div class="detail-abstract">{abstract[:300]}{"…" if len(abstract)>300 else ""}</div>
                    <div class="detail-show-more">Read more</div>

                    <div class="detail-stats">
                      <div class="detail-stat-item">
                        <div class="detail-stat-label">Citations</div>
                        <div class="detail-stat-value">342</div>
                      </div>
                      <div class="detail-stat-item">
                        <div class="detail-stat-label">Influence</div>
                        <div class="detail-stat-value high">High</div>
                      </div>
                      <div class="detail-stat-item">
                        <div class="detail-stat-label">Added</div>
                        <div class="detail-stat-value" style="font-size:0.8rem;">2 days ago</div>
                      </div>
                      <div class="detail-stat-item">
                        <div class="detail-stat-label">Your notes</div>
                        <div class="detail-stat-value">3</div>
                      </div>
                    </div>

                    st.markdown('<div class="detail-quick-actions"><div class="detail-section-title">Quick Actions</div></div>', unsafe_allow_html=True)
                    qa1, qa2 = st.columns(2)
                    with qa1:
                        if st.button("📝  Summarise this paper", key=f"qa_sum_{idx}", use_container_width=True):
                            st.session_state["_pending_query"] = f"Summarise the paper: {title}"
                            st.session_state.active_nav = "Chat"
                            st.rerun()
                        if st.button("🔍  Find related papers", key=f"qa_rel_{idx}", use_container_width=True):
                            st.session_state["_pending_query"] = f"Find papers related to: {title}"
                            st.session_state.active_nav = "Chat"
                            st.rerun()
                    with qa2:
                        if st.button("💡  Explain key concepts", key=f"qa_exp_{idx}", use_container_width=True):
                            st.session_state["_pending_query"] = f"Explain the key concepts in: {title}"
                            st.session_state.active_nav = "Chat"
                            st.rerun()
                        if st.button("🔄  Generate citations", key=f"qa_cit_{idx}", use_container_width=True):
                            st.session_state["_pending_query"] = f"Generate a citation for: {title}"
                            st.session_state.active_nav = "Chat"
                            st.rerun()
                  </div>
                </div>
                """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# CHAT
# ──────────────────────────────────────────────────────────────────────
elif active == "Chat":

    use_hyde = False
    top_k = 5
    if st.session_state.get("retrieval_settings_open", False):
        use_hyde = st.session_state.get("use_hyde", False)
        top_k    = st.session_state.get("top_k", 5)

    chat_col, sources_col = st.columns([3, 1.2], gap="small")

    with chat_col:
        st.markdown(f"""
        <div class="chat-header">
          <div>
            <div class="chat-header-title">Ask the Collection</div>
            <div class="chat-header-sub">Across {papers_count} papers — what would you like to understand?</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        ch1, ch2 = st.columns([5,1])
        with ch2:
            if st.button("+ New chapter", use_container_width=True):
                st.session_state.memory.clear()
                st.session_state.chat = []
                st.session_state.tts_audio = {}
                st.session_state.flowcharts = {}
                st.session_state.suggestions = load_chunks_for_suggestions(6)
                st.rerun()

        # Tracklist-style suggestions (empty state)
        if not st.session_state.chat and st.session_state.suggestions:
            st.markdown("""
            <div class="tracklist-wrap">
              <div class="tracklist-heading">What would you like to understand?</div>
              <div class="tracklist-sub">Choose a question or write your own…</div>
            </div>
            """, unsafe_allow_html=True)
            sg_cols = st.columns(2)
            for i, sug in enumerate(st.session_state.suggestions[:6]):
                with sg_cols[i % 2]:
                    # Tracklist number prefix
                    label = f"{i+1:02d}.  {sug}"
                    if st.button(label, key=f"sug_{i}", use_container_width=True):
                        st.session_state["_pending_query"] = sug
                        st.rerun()

        # Render history
        for idx, msg in enumerate(st.session_state.chat):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    a1, a2 = st.columns(2)
                    with a1:
                        if idx in st.session_state.tts_audio:
                            st.audio(st.session_state.tts_audio[idx], format="audio/wav")
                        else:
                            if st.button("🔊 Listen", key=f"tts_{idx}", use_container_width=True):
                                with st.spinner("Composing audio…"):
                                    audio = text_to_speech(msg["content"])
                                    if audio:
                                        st.session_state.tts_audio[idx] = audio
                                        st.rerun()
                    with a2:
                        if idx in st.session_state.flowcharts:
                            render_mermaid(st.session_state.flowcharts[idx])
                        else:
                            user_q = ""
                            if idx > 0 and st.session_state.chat[idx-1]["role"] == "user":
                                user_q = st.session_state.chat[idx-1]["content"]
                            if st.button("📊 Flow diagram", key=f"flow_{idx}", use_container_width=True):
                                with st.spinner("Drafting diagram…"):
                                    fc = generate_flowchart(user_q, msg["content"])
                                    if fc:
                                        st.session_state.flowcharts[idx] = fc
                                        st.rerun()

        # Chat input
        query = st.chat_input("What would you like to understand?")
        if "_pending_query" in st.session_state:
            query = st.session_state.pop("_pending_query")

        if query:
            st.session_state.query_count += 1
            st.session_state.chat.append({"role": "user", "content": query})

            with st.chat_message("user"):
                st.markdown(query)

            with st.spinner("Searching the collection…"):
                chunks    = retrieve(query, top_k=top_k, use_hyde=use_hyde)
                citations = format_citations(chunks)

            with st.chat_message("assistant"):
                answer = st.write_stream(
                    generate_answer_streaming(query, chunks, st.session_state.memory.get())
                )
                local_recs = recommend_by_query(query, top_k=3)
                arxiv_recs = recommend_arxiv(query, top_k=3)

            st.session_state.memory.add("user", query)
            st.session_state.memory.add("assistant", answer)
            st.session_state.memory.add_topic(query[:50])
            st.session_state.chat.append({
                "role": "assistant", "content": answer,
                "citations": citations,
                "local_recs": local_recs,
                "arxiv_recs": arxiv_recs,
            })
            st.rerun()

        # Recent conversations (empty state)
        if not st.session_state.chat:
            st.markdown("""
            <div style="margin-top:28px;">
              <div class="label">Recent Conversations</div>
            </div>
            """, unsafe_allow_html=True)
            RECENT_CONVOS = [
                ("Compare ViT and CNN architectures", "Today", 8),
                ("Summarize RAG papers published after 2020", "Yesterday", 5),
                ("Explain the attention mechanism", "2 days ago", 12),
            ]
            for title, when, msgs in RECENT_CONVOS:
                st.markdown(f"""
                <div style="padding:11px 0; border-bottom:1px solid var(--border);
                            cursor:pointer; font-family:'Inter',sans-serif;">
                  <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="font-family:'Playfair Display',serif; font-style:italic; font-size:0.86rem; font-weight:500; color:var(--text);">{title}</div>
                    <span style="font-size:0.68rem; color:var(--text4); font-style:italic;">{when}</span>
                  </div>
                  <div style="font-size:0.72rem; color:var(--text3); margin-top:2px;">{msgs} exchanges</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('<div style="margin-top:12px;"><span style="font-size:0.78rem; color:var(--accent); font-weight:500; cursor:pointer; font-style:italic;">View all conversations →</span></div>', unsafe_allow_html=True)

    # ── Sources panel (footnote style) ──
    with sources_col:
        latest_citations = []
        latest_arxiv = []
        for msg in reversed(st.session_state.chat):
            if msg["role"] == "assistant":
                latest_citations = msg.get("citations", [])
                latest_arxiv     = msg.get("arxiv_recs", [])
                break

        if latest_citations or latest_arxiv:
            src_count = len(latest_citations) + len(latest_arxiv)
            st.markdown(f"""
            <div style="display:flex; align-items:center; justify-content:space-between;
                        margin-bottom:14px; padding-top:8px;">
              <div class="sources-panel-title" style="margin-bottom:0;">References</div>
              <span style="font-size:0.7rem; color:var(--text3); font-style:italic;">
                {src_count} sources cited
              </span>
            </div>
            """, unsafe_allow_html=True)

            for c in latest_citations:
                title  = strip_md(c.get("title", "Unknown"))
                author = strip_md(c.get("authors", ""))
                year   = c.get("year", "")
                score  = c.get("score", 0)
                score_pct = int(score * 100) if score else 0
                st.markdown(f"""
                <div class="source-card">
                  <div class="source-card-title">
                    <span class="source-score">{score_pct}%</span>
                    {title[:55]}{"…" if len(title)>55 else ""}
                  </div>
                  <div class="source-card-meta">{author[:35]} · {year}</div>
                </div>
                """, unsafe_allow_html=True)

            for r in latest_arxiv:
                title  = strip_md(r.get("title", "Unknown"))
                author = r.get("authors", "")
                year   = r.get("year", "")
                url    = r.get("url", "#")
                st.markdown(f"""
                <div class="source-card">
                  <div class="source-card-title">
                    <a href="{url}" target="_blank" style="color:var(--blue); text-decoration:none;">
                      {title[:55]}{"…" if len(title)>55 else ""}
                    </a>
                  </div>
                  <div class="source-card-meta">{author[:35]} · {year} · arXiv</div>
                </div>
                """, unsafe_allow_html=True)

            if latest_arxiv:
                st.markdown(f'<div style="text-align:center; margin-top:10px;"><span style="font-size:0.76rem; color:var(--accent); font-weight:500; cursor:pointer; font-style:italic;">View all {src_count} references →</span></div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="padding-top:52px; text-align:center;">
              <div style="font-size:1.6rem; opacity:0.2; margin-bottom:10px;">📖</div>
              <div style="font-size:0.8rem; color:var(--text3); line-height:1.65; font-style:italic; font-family:'Playfair Display',serif;">
                References will appear here<br>once you ask a question.
              </div>
            </div>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# SEARCH
# ──────────────────────────────────────────────────────────────────────
elif active == "Search":
    st.markdown("""
    <div style="padding:32px 32px 0;">
      <div style="font-family:'Playfair Display',serif; font-style:italic; font-size:2rem; font-weight:600; color:var(--text); letter-spacing:-0.01em;">Search the Archive</div>
      <div style="font-size:0.76rem; color:var(--text4); margin-top:4px; font-style:italic; font-family:'Inter',sans-serif;">Semantic search across every passage in your collection</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        search_q = st.text_input(
            "Search",
            placeholder="Search ideas, methods, authors, findings…",
            label_visibility="collapsed",
            key="global_search"
        )
        use_hyde_s = st.toggle("HyDE expansion", value=False,
                               help="Generates hypothetical answers to improve search precision.")
        top_k_s = st.slider("Results", 3, 15, 8)

        if search_q:
            with st.spinner("Searching the passages…"):
                chunks    = retrieve(search_q, top_k=top_k_s, use_hyde=use_hyde_s)
                citations = format_citations(chunks)

            st.markdown(f'<div class="label" style="margin:18px 0 10px;">Found {len(citations)} passages</div>', unsafe_allow_html=True)

            for i, c in enumerate(citations, 1):
                title   = strip_md(c["title"])
                authors = strip_md(c["authors"])
                year    = c.get("year", "")
                section = c.get("section", "")
                score   = c.get("score", 0)
                doi_link = f' · <a href="{c["doi"]}" style="color:var(--blue); font-style:normal;">DOI↗</a>' if c.get("doi") else ""
                st.markdown(f"""
                <div style="background:var(--surface); border:1px solid var(--border);
                            border-left:3px solid var(--gold);
                            border-radius:var(--radius);
                            padding:14px 16px; margin-bottom:10px; box-shadow:var(--shadow-xs);">
                  <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div style="font-family:'Playfair Display',serif; font-style:italic; font-size:0.9rem; font-weight:500; color:var(--text); line-height:1.35; flex:1;">
                      {i}. {title[:80]}{"…" if len(title)>80 else ""}
                    </div>
                    <span class="tag tag-green" style="margin-left:12px; flex-shrink:0;">{int(score*100)}%</span>
                  </div>
                  <div style="font-size:0.72rem; color:var(--text3); margin-top:5px; font-style:italic;">
                    {authors[:50]} · {year} · <span class="tag">{section}</span>{doi_link}
                  </div>
                </div>
                """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# BENCHMARK
# ──────────────────────────────────────────────────────────────────────
elif active == "Benchmark":

    bench_h1, bench_h2 = st.columns([4, 1])
    with bench_h1:
        st.markdown("""
        <div style="padding:28px 32px 0;">
          <div class="bench-title">Performance Records</div>
          <div class="bench-sub">Evaluation metrics for your retrieval pipeline</div>
        </div>
        """, unsafe_allow_html=True)
    with bench_h2:
        st.markdown('<div style="padding-top:28px;"></div>', unsafe_allow_html=True)
        st.button("⬆ Export Report", use_container_width=True)

    bt1, bt2, bt3 = st.tabs(["Overview", "Configurations", "History"])

    summary_path = Path("indexes/benchmark_summary.json")

    with bt1:
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
                best = df.loc[df["Mean Score"].idxmax()]

                ring_col, metrics_col = st.columns([1, 2.5], gap="large")
                with ring_col:
                    overall_pct = int(best["Mean Score"] * 100)
                    circumference = 2 * 3.14159 * 40
                    dash = circumference * (overall_pct / 100)
                    st.markdown(f"""
                    <div class="perf-ring-wrap">
                      <div class="perf-ring-label">Overall Score</div>
                      <svg width="110" height="110" viewBox="0 0 110 110">
                        <circle cx="55" cy="55" r="40" fill="none"
                                stroke="var(--surface3)" stroke-width="9"/>
                        <circle cx="55" cy="55" r="40" fill="none"
                                stroke="var(--accent)" stroke-width="9"
                                stroke-dasharray="{dash:.1f} {circumference:.1f}"
                                stroke-dashoffset="{circumference/4:.1f}"
                                stroke-linecap="round"/>
                        <text x="55" y="60" text-anchor="middle"
                              font-family="Playfair Display,serif"
                              font-style="italic"
                              font-size="20" fill="var(--text)">{overall_pct}%</text>
                      </svg>
                      <div class="perf-ring-sub">Retrieval Quality</div>
                    </div>
                    """, unsafe_allow_html=True)

                with metrics_col:
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    metrics_display = [
                        (mc1, "MRR",         best["MRR"],         "var(--accent)"),
                        (mc2, "Hit@1",        best["Hit@1"],       "var(--blue)"),
                        (mc3, "nDCG@5",      best["nDCG@5"],      "var(--green)"),
                        (mc4, "Faithfulness", best["Faithfulness"],"var(--gold-dark)"),
                    ]
                    for col, label, val, color in metrics_display:
                        col.markdown(f"""
                        <div class="bench-metric-card">
                          <div class="bench-metric-num" style="color:{color};">{val:.2f}</div>
                          <div class="bench-metric-label">{label}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown('<div class="label" style="margin:24px 0 10px; padding:0 4px;">Top Configurations</div>', unsafe_allow_html=True)
                top_configs_html = '<div style="background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); overflow:hidden;">'
                for rank, (_, row) in enumerate(df.nlargest(5, "Mean Score").iterrows(), 1):
                    pct = int(row["Mean Score"] * 100)
                    bar_color = "var(--accent)" if rank == 1 else "var(--blue)" if rank == 2 else "var(--text3)"
                    top_configs_html += f"""
                    <div class="config-row">
                      <div class="config-rank">{rank}</div>
                      <div class="config-name">{row['Config'][:40]}</div>
                      <div class="config-score-bar-wrap">
                        <div class="config-score-bar-bg">
                          <div class="config-score-bar" style="width:{pct}%; background:{bar_color};"></div>
                        </div>
                      </div>
                      <div class="config-pct">{pct}%</div>
                    </div>"""
                top_configs_html += "</div>"
                st.markdown(top_configs_html, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Failed to load benchmark data: {e}")
        else:
            st.markdown("""
            <div class="empty-state" style="margin:24px 32px;">
              <div class="empty-icon">📊</div>
              <div class="empty-title">No records yet.</div>
              <div class="empty-sub">
                Run <code>python run_benchmark.py</code> to evaluate your retrieval pipeline.<br>
                Results will appear here like liner notes — every detail recorded.
              </div>
            </div>
            """, unsafe_allow_html=True)

    with bt2:
        if summary_path.exists():
            try:
                with open(summary_path) as f:
                    summary_data = json.load(f)
                for label, details in summary_data.items():
                    ret   = details.get("retrieval", {})
                    judge = details.get("judge", {})
                    mean  = details.get("mean_score", 0.0)
                    faith = judge.get("faithfulness", 0.0) if judge else 0.0
                    relev = judge.get("relevance", 0.0) if judge else 0.0
                    hit   = ret.get("hit@1", 0.0)
                    mrr   = ret.get("mrr", 0.0)
                    ndcg  = ret.get("ndcg@5", 0.0)
                    st.markdown(f"""
                    <div style="background:var(--surface); border:1px solid var(--border);
                                border-left:3px solid var(--gold);
                                border-radius:var(--radius); padding:16px 18px; margin-bottom:12px; box-shadow:var(--shadow-xs);">
                      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                        <span style="font-family:'Playfair Display',serif; font-style:italic; font-size:0.9rem; font-weight:600; color:var(--text);">{label}</span>
                        <span class="tag tag-accent">{int(mean*100)}% mean</span>
                      </div>
                      <div style="display:grid; grid-template-columns:repeat(5,1fr); gap:10px;
                                  font-size:0.72rem; color:var(--text3); font-family:'Inter',sans-serif;">
                        <div>Hit@1 <strong style="color:var(--green);">{int(hit*100)}%</strong>{perf_bar(hit,'bar-fill-green')}</div>
                        <div>MRR <strong style="color:var(--blue);">{mrr:.3f}</strong>{perf_bar(mrr,'bar-fill-blue')}</div>
                        <div>nDCG@5 <strong style="color:var(--accent);">{ndcg:.3f}</strong>{perf_bar(ndcg)}</div>
                        <div>Faithfulness <strong style="color:var(--green);">{faith:.2f}</strong>{perf_bar(faith,'bar-fill-green')}</div>
                        <div>Relevance <strong style="color:var(--blue);">{relev:.2f}</strong>{perf_bar(relev,'bar-fill-blue')}</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(str(e))
        else:
            st.info("Run benchmark first to see configurations.")

    with bt3:
        scores_path = Path("indexes/eval_scores.json")
        if scores_path.exists():
            try:
                with open(scores_path) as f:
                    all_scores = json.load(f)
                st.dataframe(pd.DataFrame(all_scores).T, use_container_width=True)
            except Exception:
                st.info("No history data available.")
        else:
            st.info("No history data available.")


# ══════════════════════════════════════════════════════════════════════
# RETRIEVAL SETTINGS SLIDE-OVER
# ══════════════════════════════════════════════════════════════════════
if st.session_state.get("retrieval_settings_open", False):
    st.divider()
    st.markdown('<div style="font-family:Inter,sans-serif; font-size:0.9rem; font-weight:600; color:var(--text); padding:4px 0 12px;">⚙ Retrieval Settings</div>', unsafe_allow_html=True)

    cfg_col1, cfg_col2 = st.columns(2)

    with cfg_col1:
        st.markdown('<div class="label">Search Strategy</div>', unsafe_allow_html=True)
        use_hyde_cfg = st.toggle(
            "HyDE Expansion",
            value=st.session_state.get("use_hyde", False),
            help="Generate hypothetical documents for better recall.",
            key="cfg_hyde"
        )
        use_hybrid = st.toggle(
            "Hybrid Search",
            value=st.session_state.get("use_hybrid", True),
            help="Combines dense (vector) and sparse (BM25) search.",
            key="cfg_hybrid"
        )
        use_rerank = st.toggle(
            "Enable Reranker",
            value=st.session_state.get("use_rerank", True),
            help="Re-rank results using cross-encoder for higher precision.",
            key="cfg_rerank"
        )
        rerank_model = st.selectbox(
            "Reranker Model",
            ["bge-reranker-large", "bge-reranker-base", "cross-encoder-ms-marco"],
            key="cfg_rerank_model"
        )

    with cfg_col2:
        st.markdown('<div class="label">Retrieval Parameters</div>', unsafe_allow_html=True)
        top_k_cfg = st.slider("Top K (Retrieve)", 5, 30, 20, key="cfg_topk")
        top_k_rerank = st.slider("Top K (Rerank)", 1, 10, 5, key="cfg_topk_rerank")
        score_threshold = st.slider("Score Threshold", 0.0, 1.0, 0.20, 0.01, key="cfg_threshold")

    col_reset, col_save = st.columns(2)
    with col_reset:
        if st.button("Reset to Defaults", use_container_width=True):
            st.session_state["use_hyde"]   = False
            st.session_state["use_hybrid"] = True
            st.session_state["use_rerank"] = True
            st.session_state["top_k"]      = 5
            st.rerun()
    with col_save:
        if st.button("Bookmark Changes", use_container_width=True, type="primary"):
            st.session_state["use_hyde"]        = use_hyde_cfg
            st.session_state["use_hybrid"]      = use_hybrid
            st.session_state["use_rerank"]      = use_rerank
            st.session_state["top_k"]           = top_k_cfg
            st.session_state["retrieval_settings_open"] = False
            st.success("Settings bookmarked")
            st.rerun()