"""Streamlit UI for the paper RAG pipeline."""
import streamlit as st
from pathlib import Path

from src.pipeline import RAGPipeline

st.set_page_config(page_title="Paper RAG", layout="wide")
st.title("📄 Research Paper RAG")
st.markdown(
    "Ask questions about research papers. Uses semantic search, reranking, and multimodal retrieval."
)

# Available indices (papers)
INDICES = {
    "ImageNet": "imagenet",
    "Attention Is All You Need (Transformer)": "transformer",
    "ResNet": "resnet",
    "VGG": "vgg",
    "BERT": "bert",
}

# Session state
if "rag" not in st.session_state:
    st.session_state.rag = None
if "current_index" not in st.session_state:
    st.session_state.current_index = None

# Sidebar: paper selection
st.sidebar.header("Select Paper")
selected_paper = st.sidebar.selectbox("Paper", list(INDICES.keys()))
index_name = INDICES[selected_paper]

# Load RAG pipeline for selected paper
if st.session_state.current_index != index_name:
    with st.spinner(f"Loading {selected_paper}..."):
        st.session_state.rag = RAGPipeline(index_name)
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
                    st.image(img, use_column_width=True)
                except Exception as e:
                    st.warning(f"Could not load image: {e}")
            st.write(chunk["text"][:500] + ("..." if len(chunk["text"]) > 500 else ""))

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
**How it works:**
1. Semantic search (bi-encoder) finds candidates
2. Cross-encoder reranks to top-5
3. LLM generates answer from chunks

**Route:**
- **metadata**: direct lookup (title, authors, etc.)
- **section**: retrieves a specific section
- **summary**: whole-paper summary
- **qa**: standard Q&A via semantic search
"""
)
