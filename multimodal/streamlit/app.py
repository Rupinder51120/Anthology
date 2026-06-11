import streamlit as st
import requests
import os
from PIL import Image
from pathlib import Path

API_URL = os.getenv("API_URL", "http://localhost:8001")

st.set_page_config(
    page_title="Anthology Multimodal",
    page_icon="📚",
    layout="wide",
)

# Sidebar navigation
st.sidebar.title("📚 Anthology")
st.sidebar.markdown("*Multimodal Research Assistant*")
page = st.sidebar.radio("Navigate", [
    "💬 Chat",
    "📄 Papers",
    "🖼️ Figures",
    "📊 Tables",
    "⬆️ Upload",
])


def get_papers():
    try:
        r = requests.get(f"{API_URL}/api/v2/papers", timeout=5)
        return r.json().get("papers", [])
    except Exception:
        return []


# ── Chat ──────────────────────────────────────────────────────────────
if page == "💬 Chat":
    st.title("💬 Research Chat")

    col1, col2 = st.columns([3, 1])
    with col2:
        modality = st.selectbox("Filter by", ["All", "text", "table", "figure"])
        top_k = st.slider("Results", 3, 10, 5)

    with col1:
        question = st.text_input("Ask anything about the papers...",
                                  placeholder="What does the benchmark table show?")

    if st.button("Ask", type="primary") and question:
        with st.spinner("Retrieving and generating answer..."):
            payload = {
                "question": question,
                "top_k": top_k,
                "content_type": None if modality == "All" else modality,
            }
            try:
                r = requests.post(f"{API_URL}/api/v2/query", json=payload, timeout=60)
                data = r.json()

                st.markdown("### Answer")
                st.markdown(data.get("answer", "No answer generated."))

                st.markdown(f"*{data.get('chunks_used', 0)} sources · {data.get('latency_ms', 0):.0f}ms*")

                st.markdown("### Sources")
                for chunk in data.get("sources", []):
                    ctype = chunk["content_type"]
                    icon = {"text": "📝", "table": "📊", "figure": "🖼️"}.get(ctype, "📄")
                    label = f"{icon} {chunk.get('figure_number') or ctype.title()} · p.{chunk.get('page_number', '?')} · {chunk.get('section_title', '')}"

                    with st.expander(label):
                        if ctype == "figure" and chunk.get("image_path"):
                            img_url = f"{API_URL}/figures/{Path(chunk['image_path']).name}"
                            try:
                                img_r = requests.get(img_url, timeout=5)
                                if img_r.status_code == 200:
                                    from io import BytesIO
                                    st.image(Image.open(BytesIO(img_r.content)), use_column_width=True)
                            except Exception:
                                pass
                        elif ctype == "table" and chunk.get("table_markdown"):
                            st.markdown(chunk["table_markdown"])
                            if chunk.get("table_summary"):
                                st.info(chunk["table_summary"])
                        else:
                            st.write(chunk.get("content", ""))
            except Exception as e:
                st.error(f"Query failed: {e}")


# ── Papers ────────────────────────────────────────────────────────────
elif page == "📄 Papers":
    st.title("📄 Paper Library")
    papers = get_papers()

    if not papers:
        st.info("No papers indexed yet. Upload PDFs to get started.")
    else:
        st.markdown(f"**{len(papers)} papers indexed**")
        for paper in papers:
            with st.expander(f"**{paper['title']}** ({paper.get('year', 'N/A')})"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Text chunks", paper.get("chunk_count", 0))
                col2.metric("Figures", paper.get("figure_count", 0))
                col3.metric("Tables", paper.get("table_count", 0))
                if paper.get("authors"):
                    st.caption(paper["authors"])


# ── Figures ───────────────────────────────────────────────────────────
elif page == "🖼️ Figures":
    st.title("🖼️ Figure Explorer")
    papers = get_papers()

    if not papers:
        st.info("No papers indexed yet.")
    else:
        selected = st.selectbox("Select paper", [p["title"] for p in papers])
        paper = next(p for p in papers if p["title"] == selected)

        try:
            r = requests.get(f"{API_URL}/api/v2/papers/{paper['id']}/figures", timeout=5)
            figures = r.json().get("figures", [])
            if not figures:
                st.info("No figures extracted from this paper.")
            else:
                st.markdown(f"**{len(figures)} figures**")
                cols = st.columns(3)
                for i, fig in enumerate(figures):
                    with cols[i % 3]:
                        if fig.get("image_path"):
                            img_url = f"{API_URL}/figures/{Path(fig['image_path']).name}"
                            try:
                                img_r = requests.get(img_url, timeout=5)
                                if img_r.status_code == 200:
                                    from io import BytesIO
                                    st.image(Image.open(BytesIO(img_r.content)), use_column_width=True)
                            except Exception:
                                st.markdown("🖼️ *Image unavailable*")
                        st.caption(f"**{fig.get('figure_number', 'Figure')}** · p.{fig.get('page_number', '?')}")
                        st.caption(fig.get("content", "")[:200])
        except Exception as e:
            st.error(f"Failed to load figures: {e}")


# ── Tables ────────────────────────────────────────────────────────────
elif page == "📊 Tables":
    st.title("📊 Table Explorer")
    papers = get_papers()

    if not papers:
        st.info("No papers indexed yet.")
    else:
        selected = st.selectbox("Select paper", [p["title"] for p in papers])
        paper = next(p for p in papers if p["title"] == selected)

        try:
            r = requests.get(f"{API_URL}/api/v2/papers/{paper['id']}/tables", timeout=5)
            tables = r.json().get("tables", [])
            if not tables:
                st.info("No tables extracted from this paper.")
            else:
                st.markdown(f"**{len(tables)} tables**")
                for table in tables:
                    with st.expander(f"**{table.get('figure_number', 'Table')}** · p.{table.get('page_number', '?')} · {table.get('section_title', '')}"):
                        if table.get("table_markdown"):
                            st.markdown(table["table_markdown"])
                        if table.get("table_summary"):
                            st.info(f"**Summary:** {table['table_summary']}")
        except Exception as e:
            st.error(f"Failed to load tables: {e}")


# ── Upload ────────────────────────────────────────────────────────────
elif page == "⬆️ Upload":
    st.title("⬆️ Upload Paper")
    st.markdown("Upload a research PDF. The system will extract text, tables, and figures automatically.")

    use_vlm = st.checkbox("Enable figure captioning (requires Ollama)", value=True)

    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded and st.button("Ingest", type="primary"):
        with st.spinner("Uploading and starting ingestion..."):
            try:
                r = requests.post(
                    f"{API_URL}/api/v2/ingest",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                    timeout=10,
                )
                data = r.json()
                if data.get("status") == "ingestion_started":
                    st.success(f"Ingestion started for **{uploaded.name}**. Check Papers tab when complete.")
                else:
                    st.error(f"Unexpected response: {data}")
            except Exception as e:
                st.error(f"Upload failed: {e}")
