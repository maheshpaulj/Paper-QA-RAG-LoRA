"""Cross-encoder reranker wrapped for LangChain Documents.

Uses the same ms-marco-MiniLM cross-encoder as Phase 2, but works with
LangChain Document objects instead of raw chunk dicts. This replaces the
LangChain CrossEncoderReranker which was removed in newer versions.

The approach is deliberately simple: take a list of Documents + a query,
score each with the cross-encoder, sort by score, return top-k. No fancy
abstraction — the cross-encoder is a pure scoring function.
"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import RERANK_MODEL, RERANK_ENABLED, TOP_K, RERANK_CANDIDATES


class Reranker:
    """Cross-encoder reranker for LangChain Documents.

    Usage:
        reranker = Reranker()
        reranked = reranker.rerank(query, docs, top_n=5)
    """

    def __init__(self, model_name: str = RERANK_MODEL):
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Document], top_n: int = TOP_K) -> List[Document]:
        """Score each doc against the query and return the top-n by score."""
        if not docs:
            return []

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self._model.predict(pairs)

        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_n]]


def build_reranking_retriever(vectorstore, reranker: Reranker | None = None):
    """Build a retriever that optionally reranks results.

    If reranking is enabled, fetches RERANK_CANDIDATES and narrows to TOP_K.
    If disabled, fetches TOP_K directly from FAISS.

    Returns a callable: query -> list[Document]
    """
    if RERANK_ENABLED and reranker:
        base_retriever = vectorstore.as_retriever(
            search_kwargs={"k": RERANK_CANDIDATES}
        )

        def reranking_retrieve(query: str) -> List[Document]:
            docs = base_retriever.invoke(query)
            return reranker.rerank(query, docs, top_n=TOP_K)

        return reranking_retrieve

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    return lambda query: retriever.invoke(query)
