"""LangChain FAISS vectorstore management — build, save, load paper indexes.

Replaces src/store.py with LangChain's FAISS wrapper, which pairs the FAISS
index with an InMemoryDocstore (chunk text + metadata in one object). The raw
.faiss file is identical to before; the difference is a .pkl docstore replaces
the manual .chunks.json.

Document metadata (meta.json) is still saved separately since it's document-
level, not per-chunk. The vectorstore handles chunk-level data.

Storage layout per paper:
    index/<name>/index.faiss     FAISS binary index
    index/<name>/index.pkl       Docstore (pickled Document objects)
    index/<name>/meta.json       Document metadata (page_count, title, etc.)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import INDEX_DIR


def build_vectorstore(
    documents: List[Document],
    embeddings,  # LoRAEmbeddings instance
    name: str = "paper",
    meta: dict | None = None,
) -> FAISS:
    """Build a FAISS vectorstore from Documents and save it to disk.

    Args:
        documents: LangChain Documents (text chunks + figure chunks).
        embeddings: LoRAEmbeddings instance for encoding.
        name: Index name (paper identifier).
        meta: Document-level metadata to save alongside.

    Returns:
        The built FAISS vectorstore (also persisted to index/<name>/).
    """
    store_dir = INDEX_DIR / name
    store_dir.mkdir(parents=True, exist_ok=True)

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(str(store_dir))

    # Meta is document-level, not per-chunk — save separately.
    if meta:
        meta_path = store_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    return vectorstore


def load_vectorstore(name: str, embeddings) -> FAISS:
    """Load a saved vectorstore from disk.

    Args:
        name: Index name (paper identifier).
        embeddings: LoRAEmbeddings instance (needed for query embedding).

    Returns:
        FAISS vectorstore ready for similarity search.
    """
    store_dir = INDEX_DIR / name
    return FAISS.load_local(
        str(store_dir),
        embeddings,
        allow_dangerous_deserialization=True,  # required for pickle loading
    )


def load_meta(name: str) -> dict:
    """Load document-level metadata for a paper."""
    meta_path = INDEX_DIR / name / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def list_indexes() -> list[str]:
    """Discover all built indexes — each is a subdirectory of INDEX_DIR
    containing index.faiss."""
    found = []
    for d in sorted(INDEX_DIR.iterdir()):
        if d.is_dir() and (d / "index.faiss").exists():
            found.append(d.name)
    return found
