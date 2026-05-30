import streamlit as st
import json
from pathlib import Path
from src.retriever import retrieve, detect_query_intent
from src.hyde import expand_query_with_hyde
from src.generator import generate_answer
from src.recommender import recommend_local, recommend_arxiv, recommend_by_query
from src.memory import ConversationMemory
from src.indexer import get_index_stats

# ── page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Research RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── session state ─────────────────────────────────────────────
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()
if "chat" not in st.session_state:
    st.session_state.chat = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# ── sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 Research RAG")
    st.caption("Academic paper intelligence system")
    st.divider()

    st.subheader("⚙️ Retrieval settings")
    use_hyde = st.toggle("HyDE query expansion", value=True,
                         help="LLM expands your query before searching — improves recall")
    top_k    = st.slider("Chunks to retrieve", 3, 10, 5,
                         help="More chunks = more context but slower")
    st.divider()

    st.subheader("📊 Index stats")
    try:
        stats = get_index_stats()
        col1, col2 = st.columns(2)
        col1.metric("Papers", stats.get("unique_papers", "—"))
        col2.metric("Chunks", stats.get("total_chunks", "—"))
        st.caption(f"FAISS vectors: {stats.get('faiss_vectors', '—')}")
    except Exception:
        st.caption("Run build_index.py first")

    st.divider()

    st.subheader("📈 Eval scores")
    scores_path = Path("indexes/eval_scores.json")
    if scores_path.exists():
        with open(scores_path) as f:
            all_scores = json.load(f)
        for label, s in list(all_scores.items())[-2:]:
            if s:
                st.caption(f"**{label}**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Faith", f"{s.get('faithfulness', 0):.2f}")
                c2.metric("Relev", f"{s.get('answer_relevancy', 0):.2f}")
                c3.metric("Prec",  f"{s.get('context_precision', 0):.2f}")
    else:
        st.caption("No eval scores yet")

    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("Clear chat"):
        st.session_state.memory.clear()
        st.session_state.chat = []
        st.rerun()
    if col2.button("Save session"):
        st.session_state.memory.save()
        st.success("Saved")

    st.caption(st.session_state.memory.summary())
    st.caption(f"Queries this session: {st.session_state.query_count}")

# ── main area ─────────────────────────────────────────────────
st.header("Ask about your research papers")
st.caption("Get concept explanations, citations, and paper recommendations")

# display chat history
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# chat input
query = st.chat_input("e.g. How does the GAN discriminator loss work?")

if query:
    st.session_state.query_count += 1

    # show user message
    st.session_state.chat.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # detect intent for display
    intent = detect_query_intent(query)

    with st.chat_message("assistant"):
        with st.spinner(f"Searching ({intent} query)..."):

            # retrieval
            search_query = expand_query_with_hyde(query) if use_hyde else query
            chunks       = retrieve(search_query, top_k=top_k)
            result       = generate_answer(query, chunks,
                                           st.session_state.memory.get())

        # ── answer ────────────────────────────────────────────
        st.markdown(result["answer"])

        # ── citations ─────────────────────────────────────────
        if result["citations"]:
            with st.expander(f"📄 Sources ({len(result['citations'])} papers)", expanded=True):
                for i, c in enumerate(result["citations"], 1):
                    doi_str   = f" · [DOI]({c['doi']})" if c.get("doi") else ""
                    score_str = f" · score: {c['score']:.3f}" if c.get("score") else ""
                    st.markdown(
                        f"**{i}. {c['title']}**{doi_str}  \n"
                        f"*{c['authors']}* ({c['year']}) · "
                        f"Section: `{c['section']}`{score_str}"
                    )

        # ── local recommendations ──────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            local_recs = recommend_by_query(query, top_k=3)
            if local_recs:
                with st.expander("🔗 Similar papers in your folder"):
                    for r in local_recs:
                        sim_pct = int(r["similarity"] * 100)
                        st.markdown(
                            f"**{r['title'][:55]}**  \n"
                            f"*{r['authors'][:40]}* ({r['year']})  \n"
                            f"Similarity: `{sim_pct}%`"
                        )

        # ── arxiv recommendations ──────────────────────────────
        with col2:
            arxiv_recs = recommend_arxiv(query, top_k=3)
            if arxiv_recs:
                with st.expander("🌐 Related papers on Arxiv"):
                    for r in arxiv_recs:
                        st.markdown(
                            f"**[{r['title'][:55]}]({r['url']})**  \n"
                            f"*{r['authors']}* ({r['year']})"
                        )

        # ── debug info ────────────────────────────────────────
        with st.expander("🔍 Debug info"):
            st.json({
                "intent":        intent,
                "hyde_used":     use_hyde,
                "chunks_used":   result["chunks_used"],
                "tokens_used":   result.get("tokens_used", "—"),
                "response_type": result["response_type"]
            })

    # update memory
    st.session_state.memory.add("user",      query)
    st.session_state.memory.add("assistant", result["answer"])
    st.session_state.memory.add_topic(query[:50])
    st.session_state.chat.append({
        "role":    "assistant",
        "content": result["answer"]
    })