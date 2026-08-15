"""Streamlit UI for the paper RAG pipeline.

Launch with `run_app.bat` (or `.venv\\Scripts\\python.exe -m streamlit run app.py`).
A bare `streamlit run app.py` may pick up a *global* streamlit whose interpreter
can't see this venv's packages -- see the import guard below.
"""
import json
import sys

import streamlit as st

st.set_page_config(page_title="Paper RAG", layout="wide")


def _missing_dep_message(exc):
    """A missing dep here almost always means the wrong interpreter, not a
    genuinely uninstalled package -- say so, and name the interpreter in use."""
    dep = getattr(exc, "name", None) or str(exc)
    return (
        f"**Missing dependency: `{dep}`**\n\n"
        f"Running under: `{sys.executable}`\n\n"
        "If that is not this project's `.venv`, you are on the wrong interpreter — "
        "start the app with **`run_app.bat`** rather than a bare "
        "`streamlit run app.py`.\n\n"
        "If it *is* the venv, install the dependencies:\n"
        "```\n.venv\\Scripts\\python.exe -m pip install "
        "-r requirements.txt -r requirements-app.txt\n```"
    )


# Deps like torchvision/peft are imported lazily when a model is built, so the
# wrong interpreter usually survives this import and fails later -- guard both.
try:
    from config import INDEX_DIR
    from src.pipeline import RAGPipeline
except ImportError as e:
    st.error(_missing_dep_message(e))
    st.stop()
st.title("📄 Research Paper RAG")
st.markdown(
    "Ask questions about research papers. Uses semantic search, reranking, and multimodal retrieval."
)

# Nicer labels for the papers shipped with the repo. Anything else falls back to
# its index name, so a newly built paper shows up here without touching this file.
KNOWN_TITLES = {
    "imagenet": "ImageNet: A Large-Scale Hierarchical Image Database",
    "transformer": "Attention Is All You Need (Transformer)",
    "resnet": "Deep Residual Learning (ResNet)",
    "vgg": "Very Deep Convolutional Networks (VGG)",
    "bert": "BERT",
}


@st.cache_data
def discover_indices():
    """Every built index, newest first. Built by `scripts.build_index`."""
    found = {}
    for faiss_path in sorted(INDEX_DIR.glob("*.faiss"), key=lambda p: -p.stat().st_mtime):
        name = faiss_path.stem
        label = KNOWN_TITLES.get(name)
        if not label:
            # prefer the PDF's own title if ingest managed to pull one
            meta_path = INDEX_DIR / f"{name}.meta.json"
            if meta_path.exists():
                title = (json.loads(meta_path.read_text(encoding="utf-8")).get("title") or "").strip()
                label = title if title and title != "Untitled" else name
            else:
                label = name
        found[f"{label}  ·  {name}"] = name
    return found


INDICES = discover_indices()
if not INDICES:
    st.error(
        "No indices found. Build one first:\n\n"
        "```\npython -m scripts.fetch_papers 1706.03762\n"
        "python -m scripts.build_index data/1706.03762.pdf transformer\n```"
    )
    st.stop()

# Session state
if "rag" not in st.session_state:
    st.session_state.rag = None
if "current_index" not in st.session_state:
    st.session_state.current_index = None

# Sidebar: paper selection
st.sidebar.header("Select Paper")
selected_paper = st.sidebar.selectbox("Paper", list(INDICES.keys()))
index_name = INDICES[selected_paper]

# Load RAG pipeline for selected paper.
# torchvision/peft are imported lazily by sentence-transformers when a model is
# actually built, so a wrong-interpreter run gets this far and only fails HERE.
if st.session_state.current_index != index_name:
    with st.spinner(f"Loading {selected_paper}..."):
        try:
            st.session_state.rag = RAGPipeline(index_name)
        # Catch ImportError, not just ModuleNotFoundError: sentence-transformers
        # swallows the missing peft module and re-raises a plain ImportError.
        except ImportError as e:
            st.error(_missing_dep_message(e))
            st.stop()
        st.session_state.current_index = index_name

# Main query interface
st.header(f"📖 {selected_paper}")
question = st.text_input("Ask a question about this paper:")

if question:
    with st.spinner("Retrieving..."):
        result = st.session_state.rag.ask(question)

    # Display answer
    st.subheader("Answer")
    st.write(result["answer"])

    # Display metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Route", result["route"].upper())
    with col2:
        chunk_types = {}
        for c in result["chunks"]:
            t = c.get("type", "text")
            chunk_types[t] = chunk_types.get(t, 0) + 1
        st.metric("Chunks", len(result["chunks"]), f"{chunk_types}")
    with col3:
        has_images = sum(1 for c in result["chunks"] if c.get("type") == "figure")
        st.metric("Images", has_images)

    # Display retrieved chunks
    st.subheader("Retrieved Chunks")
    for i, chunk in enumerate(result["chunks"], 1):
        with st.expander(f"Chunk {i} ({chunk.get('type', 'text').upper()}) — {chunk['id']}"):
            if chunk.get("type") == "figure":
                try:
                    from PIL import Image

                    img = Image.open(chunk["image_path"])
                    st.image(img, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not load image: {e}")
            st.write(chunk["text"][:500] + ("..." if len(chunk["text"]) > 500 else ""))

st.sidebar.markdown("---")
with st.sidebar.expander("➕ Add a new paper"):
    st.markdown(
        "Run these in the repo, then refresh this page — the paper appears in the "
        "dropdown automatically.\n\n"
        "**From arXiv** (use the id from the URL):\n"
        "```\n"
        "python -m scripts.fetch_papers 2005.14165\n"
        "python -m scripts.build_index data/2005.14165.pdf gpt3\n"
        "```\n"
        "**Your own PDF** — drop it in `data/`, then:\n"
        "```\n"
        "python -m scripts.build_index data/mypaper.pdf mypaper\n"
        "```\n"
        "The last argument is the index name you'll see listed here."
    )
    if st.button("🔄 Refresh paper list"):
        discover_indices.clear()
        st.rerun()

st.sidebar.markdown(
    """
**How it works:**
1. Semantic search (bi-encoder) finds candidates
2. Cross-encoder reranks to top-5
3. LLM generates answer from the chunks

**Routes:**
- **metadata** — page count, from the PDF itself
- **metadata+text** — authors/title read off the title page
- **section** — pulls a named section directly
- **figure** — "what does Figure 3 show" → that caption + image
- **summary** — abstract + intro + conclusion
- **qa** — semantic search (the normal path)
"""
)
