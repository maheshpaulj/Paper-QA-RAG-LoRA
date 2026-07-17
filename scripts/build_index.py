"""Build (or rebuild) the index for one paper.

    python -m scripts.build_index data/mypaper.pdf paper
"""
import sys

from src.ingest import ingest_pdf, extract_meta, extract_figures
from src.embed import Embedder
from src.store import VectorStore
from config import FIGURE_DIR


def main(pdf_path, name="paper"):
    print(f"Ingesting {pdf_path} ...")
    chunks = ingest_pdf(pdf_path)
    meta = extract_meta(pdf_path)
    figures = extract_figures(pdf_path, FIGURE_DIR / name)  # per-paper: ids restart at F0
    # figures ride in the same index; their caption text is what gets embedded
    all_chunks = chunks + figures
    print(f"  {len(chunks)} text chunks + {len(figures)} figures, {meta['page_count']} pages")

    print("Embedding (first run downloads the model, ~90MB) ...")
    embedder = Embedder()
    vecs = embedder.encode([c["text"] for c in all_chunks])

    store = VectorStore()
    store.build(vecs, all_chunks, meta)
    store.save(name)
    print(f"Saved index '{name}' to ./index/")


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/paper.pdf"
    name = sys.argv[2] if len(sys.argv) > 2 else "paper"
    main(pdf, name)