"""Retrieval = embed the query, find nearest chunks.

Phase 2 note: reranking plugs in right here. You'll fetch a larger candidate
set (e.g. k=20), pass it through a cross-encoder, and keep the top 5.
"""
import re

from src.embed import Embedder
from src.store import VectorStore
from config import TOP_K, RERANK_ENABLED, RERANK_CANDIDATES


class Retriever:
    def __init__(self, index_name="paper", embedder=None, rerank=None, reranker=None):
        self.embedder = embedder or Embedder()
        self.store = VectorStore().load(index_name)
        self.rerank_enabled = RERANK_ENABLED if rerank is None else rerank
        self.reranker = reranker  # pass one in to share it across indices
        if self.rerank_enabled and self.reranker is None:
            from src.rerank import Reranker  # lazy: only load the model if used
            self.reranker = Reranker()

    def retrieve(self, query, k=TOP_K):
        if self.reranker:
            # fetch a wide candidate set, then let the cross-encoder pick the top k
            cands = self.store.search(self.embedder.encode([query]), k=RERANK_CANDIDATES)
            return self.reranker.rerank(query, cands, top_k=k)
        return self.store.search(self.embedder.encode([query]), k=k)

    def by_section(self, section, limit=6):
        """All chunks tagged with a section, in reading order. Bypasses search
        so structural queries ('what is the abstract') can't be missed."""
        hits = [dict(c, score=1.0) for c in self.store.chunks
                if c.get("section") == section]
        return hits[:limit]

    def by_figure(self, kind, number, limit=3):
        """The chunk whose caption *is* 'Figure 3' / 'Table 2.1'.

        Semantic search can't do this: the number carries almost none of the
        caption's meaning, so 'what does Figure 1.1 show' retrieves whatever is
        topically nearest instead. Match the caption text directly.

        'fig' and 'figure' are the same thing; 'table' is not.
        """
        want = "table" if kind.startswith("tab") else "fig"
        pat = re.compile(
            rf"^\s*(?:{'table' if want == 'table' else 'figure|fig'})\s*\.?\s*"
            rf"{re.escape(number)}\s*[.:]\s",
            re.IGNORECASE,
        )
        hits = [dict(c, score=1.0) for c in self.store.chunks if pat.match(c["text"])]
        return hits[:limit]

    def front_matter(self, limit=3):
        """The paper's opening text chunks -- the title block, author list and
        affiliations. Academic PDFs usually ship with empty or junk metadata, so
        this is where 'who wrote this' is actually answerable from."""
        text = [c for c in self.store.chunks if c.get("type") != "figure"]
        return [dict(c, score=1.0) for c in text[:limit]]

    def summary_chunks(self, limit=8):
        """Chunks to summarize the whole paper: the high-signal sections if we
        tagged them, else a spread across the document as a fallback."""
        wanted = ("Abstract", "Introduction", "Conclusion")
        picked = [dict(c, score=1.0) for c in self.store.chunks
                  if c.get("section") in wanted]
        if not picked:
            n = len(self.store.chunks)
            idxs = sorted({0, n // 2, n - 1} | set(range(min(3, n))))
            picked = [dict(self.store.chunks[i], score=1.0) for i in idxs if i < n]
        return picked[:limit]