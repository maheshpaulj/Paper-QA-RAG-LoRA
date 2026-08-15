"""LangChain DocumentLoader wrapping the existing ingest.py logic.

Rather than rewriting the PDF parsing, section-aware chunking, and figure
extraction (which took three phases to get right), PaperLoader delegates to
the battle-tested functions in src/ingest.py and converts their output into
LangChain Document objects.

Each chunk becomes a Document with:
    page_content: the chunk text (or figure caption)
    metadata:     {id, page, section, type, image_path (if figure)}

Usage:
    loader = PaperLoader("data/paper.pdf", index_name="imagenet")
    docs = loader.load()       # list[Document]
    meta = loader.load_meta()  # dict (page_count, title, reference_count, source_pdf)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from src.ingest import ingest_pdf, extract_meta, extract_figures
from config import FIGURE_DIR


class PaperLoader(BaseLoader):
    """Load a research paper PDF into LangChain Documents.

    Uses the existing ingest.py pipeline:
      - extract_pages() + chunk_pages() for section-aware text chunks
      - extract_figures() for figure crops + captions
      - extract_meta() for document-level metadata

    The figure directory is per-paper (FIGURE_DIR / index_name) because figure
    ids restart at F0 for every document — a shared directory would silently
    overwrite one paper's figures with another's.
    """

    def __init__(self, pdf_path: str | Path, index_name: str = "paper"):
        self.pdf_path = str(pdf_path)
        self.index_name = index_name
        self._fig_dir = FIGURE_DIR / index_name

    def lazy_load(self) -> Iterator[Document]:
        """Yield Documents one at a time — text chunks then figure chunks."""
        # Text chunks
        for chunk in ingest_pdf(self.pdf_path):
            yield Document(
                page_content=chunk["text"],
                metadata={
                    "id": chunk["id"],
                    "page": chunk["page"],
                    "section": chunk["section"],
                    "type": "text",
                },
            )

        # Figure chunks — caption as page_content, image path in metadata
        for fig in extract_figures(self.pdf_path, self._fig_dir):
            yield Document(
                page_content=fig["text"],
                metadata={
                    "id": fig["id"],
                    "page": fig["page"],
                    "section": fig.get("section", "Figure"),
                    "type": "figure",
                    "image_path": fig.get("image_path", ""),
                },
            )

    def load_meta(self) -> dict:
        """Document-level metadata (page count, title, ref count, source PDF).

        Stored separately from chunks — metadata answers (page count, etc.)
        come from here, not from retrieval.
        """
        meta = extract_meta(self.pdf_path)
        meta["source_pdf"] = str(self.pdf_path).replace("\\", "/")
        return meta
