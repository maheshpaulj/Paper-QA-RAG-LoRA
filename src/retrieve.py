"""Retrieval = embed the query, find nearest chunks.

Phase 2 note: reranking plugs in right here. You'll fetch a larger candidate
set (e.g. k=20), pass it through a cross-encoder, and keep the top 5.
"""
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