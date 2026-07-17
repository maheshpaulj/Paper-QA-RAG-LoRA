"""Phase 2: cross-encoder reranking.

The bi-encoder (Embedder) embeds query and chunk separately -- fast, but coarse.
A cross-encoder scores each (query, chunk) pair *jointly*, which is more accurate
but too slow to run over every chunk. So we let FAISS fetch a wide candidate set
cheaply, then rerank just those and keep the best few.
"""
from sentence_transformers import CrossEncoder

from config import RERANK_MODEL


class Reranker:
    def __init__(self, model_name=RERANK_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, chunks, top_k):
        if not chunks:
            return chunks
        scores = self.model.predict([(query, c["text"]) for c in chunks])
        ranked = sorted(zip(chunks, scores), key=lambda cs: cs[1], reverse=True)
        return [dict(c, rerank_score=float(s)) for c, s in ranked[:top_k]]
