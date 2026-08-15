"""Build (or rebuild) the index for one paper using the LangChain pipeline.

    python -m scripts.build_index data/mypaper.pdf paper

Phase 6 refactor: uses PaperLoader + LoRAEmbeddings + LangChain FAISS
vectorstore instead of the raw ingest/embed/store calls. Same result,
LangChain plumbing.
"""
import sys

from src.lc.loader import PaperLoader
from src.lc.embeddings import LoRAEmbeddings
from src.lc.store import build_vectorstore


def main(pdf_path, name="paper"):
    print(f"Ingesting {pdf_path} ...")
    loader = PaperLoader(pdf_path, index_name=name)
    docs = loader.load()
    meta = loader.load_meta()

    text_count = sum(1 for d in docs if d.metadata.get("type") != "figure")
    fig_count = sum(1 for d in docs if d.metadata.get("type") == "figure")
    print(f"  {text_count} text chunks + {fig_count} figures, {meta.get('page_count', '?')} pages")

    print("Embedding (first run downloads the model, ~90MB) ...")
    embeddings = LoRAEmbeddings()
    build_vectorstore(docs, embeddings, name=name, meta=meta)
    print(f"Saved index '{name}' to ./index/{name}/")


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/paper.pdf"
    name = sys.argv[2] if len(sys.argv) > 2 else "paper"
    main(pdf, name)