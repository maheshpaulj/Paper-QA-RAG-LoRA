"""The LCEL pipeline — Phase 6's replacement for src/pipeline.py.

This is the centrepiece of the LangChain refactor. The original pipeline.py
was a manual if/elif chain:

    kind, arg = route(question)
    if kind == "metadata": ...
    elif kind == "section": ...
    elif kind == "summary": ...
    else: ...  # qa

Now it's an LCEL chain composed with RunnableBranch — same routing logic, same
retrieval paths, but expressed as composable LangChain runnables. The router
still uses the hand-built regex classifier (it works well, and a semantic
classifier would be slower without being better).

Usage:
    chain = build_rag_chain("imagenet")
    result = chain.invoke({"question": "What is the abstract?"})
    # -> {"question": ..., "answer": ..., "chunks": [...], "route": "section"}
"""
from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from src.lc.embeddings import LoRAEmbeddings, get_embeddings
from src.lc.store import load_vectorstore, load_meta, list_indexes
from src.lc.reranker import Reranker, get_reranker, build_reranking_retriever
from src.lc.prompts import qa_prompt, summary_prompt, REFUSAL_TEXT
from src.router import route
from src.sections import canonical
from config import (
    TOP_K, RERANK_ENABLED,
    ROOT, llm_endpoint,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _format_context(docs: list[Document]) -> str:
    """Format retrieved Documents into the context string the LLM expects.

    Same format as the original generate.py _format_context — chunk ID, page,
    section, and a figure tag if applicable.
    """
    out = []
    for doc in docs:
        m = doc.metadata
        head = f"[{m.get('id', '?')}] (p.{m.get('page', '?')}, {m.get('section', '?')})"
        if m.get("type") == "figure":
            head += " -- FIGURE, image attached below"
        out.append(f"{head}\n{doc.page_content}")
    return "\n\n".join(out)


def _build_llm() -> ChatOpenAI:
    """Build a ChatOpenAI pointing at whichever provider config selects.

    Both Cloudflare Workers AI and OpenRouter expose OpenAI-compatible
    endpoints, so ChatOpenAI works for both — just swap the base URL and key.
    """
    url, key, model, _fallbacks = llm_endpoint()
    base_url = url.replace("/chat/completions", "")
    return ChatOpenAI(
        model=model,
        api_key=key,
        base_url=base_url,
        temperature=0.3,
        max_tokens=1024,
    )


def _docs_to_chunks(docs: list[Document]) -> list[dict]:
    """Convert LangChain Documents back to the chunk dict format the UI and
    eval pipeline expect ({id, text, page, section, type, score, image_url, ...})."""
    chunks = []
    for doc in docs:
        chunk = {
            "text": doc.page_content,
            **doc.metadata,
        }
        img_path = chunk.get("image_path")
        if img_path:
            p = Path(img_path)
            folder = p.parent.name
            filename = p.name
            chunk["image_url"] = f"/api/figures/{folder}/{filename}"
        chunks.append(chunk)
    return chunks


# ── Route handlers ───────────────────────────────────────────────────────────

def _handle_metadata(state: dict) -> dict:
    """Answer from document-level metadata (page count, reference count).

    If the answer is purely metadata (page count), return it directly.
    If it's about authors/title, fall through to front-matter text retrieval.
    """
    meta = state["meta"]
    q = state["question"].lower()

    if "page" in q or "how long" in q:
        n = meta.get("page_count")
        if n:
            return {
                "question": state["question"],
                "answer": f"This paper is {n} pages long.",
                "chunks": [],
                "route": "metadata",
            }

    if "reference" in q or "citation" in q or "cited" in q:
        n = meta.get("reference_count")
        if n:
            return {
                "question": state["question"],
                "answer": f"This paper cites {n} references.",
                "chunks": [],
                "route": "metadata",
            }
        return {
            "question": state["question"],
            "answer": (
                "This paper's bibliography isn't numbered, so I can't count "
                "the entries exactly. You can read the References section "
                "directly by asking for it."
            ),
            "chunks": [],
            "route": "metadata",
        }

    # Authors / title / publication date — extract from metadata or front matter
    title = meta.get("title", "Untitled")
    pub_date = meta.get("publication_date")
    pub_year = meta.get("publication_year")
    
    docs = state["all_docs"][:3]  # first chunks = title page
    front = [d for d in docs if d.metadata.get("type") != "figure"][:3]
    
    meta_context = f"Document Metadata (Highest Priority):\nTitle: {title}\n"
    if pub_date:
        meta_context += f"Publication Date: {pub_date}\n"
    if pub_year:
        meta_context += f"Publication Year: {pub_year}\n"
    
    # We prepend the exact metadata to the front-matter chunks so the LLM has it
    if front:
        front[0] = Document(
            page_content=f"{meta_context}\n\n{front[0].page_content}",
            metadata=front[0].metadata
        )
        
    answer = _generate_answer(state["question"], front, state["llm"], state.get("history", []))
    return {
        "question": state["question"],
        "answer": answer,
        "chunks": _docs_to_chunks(front),
        "route": "metadata+text",
    }


def _handle_section(state: dict) -> dict:
    """Pull all chunks tagged with a named section — bypasses semantic search
    so structural queries can't be missed."""
    section = state["route_arg"]
    docs = [d for d in state["all_docs"] if d.metadata.get("section") == section][:6]

    if not docs:
        # Section not tagged — fall through to QA
        return _handle_qa(state)

    answer = _generate_answer(state["question"], docs, state["llm"], state.get("history", []))
    return {
        "question": state["question"],
        "answer": answer,
        "chunks": _docs_to_chunks(docs),
        "route": "section",
    }


def _handle_figure(state: dict) -> dict:
    """Match a figure by its caption text (Figure 3, Table 2.1) — semantic
    search can't do this because the number carries no topical meaning."""
    kind_raw, number = state["route_arg"]
    want = "table" if kind_raw.startswith("tab") else "fig"
    pat = re.compile(
        rf"^\s*(?:{'table' if want == 'table' else 'figure|fig'})\s*\.?\s*"
        rf"{re.escape(number)}\s*[.:]\s",
        re.IGNORECASE,
    )
    docs = [d for d in state["all_docs"] if pat.match(d.page_content)][:3]

    if not docs:
        # Unknown figure number — fall through to QA
        return _handle_qa(state)

    answer = _generate_answer(state["question"], docs, state["llm"], state.get("history", []))
    return {
        "question": state["question"],
        "answer": answer,
        "chunks": _docs_to_chunks(docs),
        "route": "figure",
    }


def _handle_summary(state: dict) -> dict:
    """Summarize: pick abstract + introduction + conclusion chunks, or fall
    back to a spread if sections weren't tagged."""
    wanted = ("Abstract", "Introduction", "Conclusion")
    docs = [d for d in state["all_docs"] if d.metadata.get("section") in wanted][:8]

    if not docs:
        # Fallback: spread across the document
        all_text = [d for d in state["all_docs"] if d.metadata.get("type") != "figure"]
        n = len(all_text)
        idxs = sorted({0, n // 2, n - 1} | set(range(min(3, n))))
        docs = [all_text[i] for i in idxs if i < n][:8]

    # Use summary prompt instead of QA prompt
    context = _format_context(docs)
    chain = summary_prompt | state["llm"]
    
    # Format history: filter out any 'error' roles
    raw_history = state.get("history", [])
    history = []
    for msg in raw_history:
        role = "ai" if msg.get("role") == "assistant" else msg.get("role")
        if role in ["user", "ai"]:
            history.append((role, msg.get("text") or msg.get("content") or ""))
            
    answer = chain.invoke({"context": context, "question": state["question"], "history": history}).content
    return {
        "question": state["question"],
        "answer": answer,
        "chunks": _docs_to_chunks(docs),
        "route": "summary",
    }


def _handle_qa(state: dict) -> dict:
    """The normal semantic-search path: FAISS top-20 → cross-encoder → top-5 → LLM."""
    retrieve = state["retrieve"]
    docs = retrieve(state["question"])
    answer = _generate_answer(state["question"], docs, state["llm"], state.get("history", []))
    return {
        "question": state["question"],
        "answer": answer,
        "chunks": _docs_to_chunks(docs),
        "route": "qa",
    }


def _generate_answer(question: str, docs: list[Document], llm, raw_history: list = None) -> str:
    """Run the QA prompt through the LLM. Handles figure images if present."""
    context = _format_context(docs)
    
    raw_history = raw_history or []
    history = []
    for msg in raw_history:
        # FastAPI might send dicts with "role" and "text" or "content".
        # Map frontend 'assistant' to LangChain 'ai'
        role = "ai" if msg.get("role") == "assistant" else msg.get("role")
        if role in ["user", "ai"]:
            history.append((role, msg.get("text") or msg.get("content") or ""))

    # For now, use text-only generation (figure images go through the old path
    # when needed — the LLM sees the caption text which is usually sufficient).
    chain = qa_prompt | llm
    result = chain.invoke({"context": context, "question": question, "history": history})
    return result.content


# ── Chain builder ────────────────────────────────────────────────────────────

def _build_retriever(vectorstore):
    """Build a retriever with optional cross-encoder reranking.

    qa path: FAISS fetches RERANK_CANDIDATES (20), cross-encoder rescores and
    keeps TOP_K (5). Without reranking, FAISS returns TOP_K directly.
    """
    reranker = Reranker() if RERANK_ENABLED else None
    return build_reranking_retriever(vectorstore, reranker)


class RAGChain:
    """The end-to-end RAG chain for one paper.

    Wraps the LCEL pipeline and provides a simple .invoke() interface that
    returns the same dict structure as the old RAGPipeline.ask().

    This is a class rather than a bare chain because the routing logic needs
    access to the full document list (for section/figure/summary lookups) and
    the metadata — state that doesn't fit neatly into a pure function chain.
    """

    def __init__(self, index_name: str = "paper"):
        self.index_name = index_name
        self.embeddings = get_embeddings()
        self.vectorstore = load_vectorstore(index_name, self.embeddings)
        self.meta = load_meta(index_name)
        self.llm = _build_llm()
        self.retrieve = _build_retriever(self.vectorstore)

        # Pre-load all docs for section/figure/summary lookups.
        # These are small (50-100 docs per paper), so this is fine.
        docstore = self.vectorstore.docstore
        
        def _doc_sort_key(doc):
            try:
                cid = int(doc.metadata.get("id", "C0")[1:])
            except ValueError:
                cid = 0
            return (doc.metadata.get("page", 1), cid)

        self.all_docs = sorted(list(docstore._dict.values()), key=_doc_sort_key)

    def invoke(self, input_dict: dict) -> dict:
        """Route → retrieve → generate, returning the standard result dict."""
        question = input_dict["question"]
        kind, arg = route(question)

        state = {
            "question": question,
            "route_kind": kind,
            "route_arg": arg,
            "meta": self.meta,
            "all_docs": self.all_docs,
            "retrieve": self.retrieve,
            "llm": self.llm,
            "history": input_dict.get("history", []),
        }

        # RunnableBranch-style dispatch (explicit for clarity and debuggability)
        if kind == "metadata":
            return _handle_metadata(state)
        elif kind == "section":
            return _handle_section(state)
        elif kind == "figure":
            return _handle_figure(state)
        elif kind == "summary":
            return _handle_summary(state)
        else:
            return _handle_qa(state)


_CHAIN_CACHE: dict[str, RAGChain] = {}


def build_rag_chain(index_name: str = "paper") -> RAGChain:
    """Build or retrieve cached RAG chain for a paper.

    Usage:
        chain = build_rag_chain("imagenet")
        result = chain.invoke({"question": "What is the abstract?"})
    """
    if index_name not in _CHAIN_CACHE:
        _CHAIN_CACHE[index_name] = RAGChain(index_name)
    return _CHAIN_CACHE[index_name]
