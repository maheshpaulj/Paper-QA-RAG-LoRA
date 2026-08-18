"""LoRA-aware embeddings wrapped as a LangChain Embeddings class.

This is the same bi-encoder from Phase 1 (all-MiniLM-L6-v2) with the same
LoRA toggle from Phase 4 — but now it speaks LangChain's Embeddings interface
so it plugs straight into FAISS vectorstores, retrieval chains, and anything
else in the LC ecosystem.

Why a custom class instead of HuggingFaceEmbeddings:
  - We need to transparently swap between base and LoRA-merged weights based on
    the LORA env flag — HuggingFaceEmbeddings doesn't know about that.
  - Vectors MUST be L2-normalized (so FAISS inner-product == cosine). The base
    class doesn't guarantee this.
  - We want the same model-loading semantics as the original embed.py: if
    models/minilm-lora/ exists and LORA=1, use it; otherwise fall back to the
    HuggingFace hub model.
"""
from __future__ import annotations

from typing import List

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from config import EMBED_MODEL, LORA_MODEL_DIR, USE_LORA


def active_model_name() -> str:
    """Which model path to load — LoRA-merged or base."""
    if USE_LORA and LORA_MODEL_DIR.exists():
        return str(LORA_MODEL_DIR)
    return EMBED_MODEL


class LoRAEmbeddings(Embeddings):
    """Bi-encoder embeddings with optional LoRA fine-tuned weights.

    Drop-in replacement for the original src/embed.py Embedder, exposing the
    LangChain Embeddings interface (embed_documents / embed_query).

    Vectors are L2-normalized so FAISS IndexFlatIP gives cosine similarity.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or active_model_name()
        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception:
            self._model = SentenceTransformer(self.model_name, local_files_only=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts (chunks/documents)."""
        vecs = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.astype("float32").tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        return self.embed_documents([text])[0]


_GLOBAL_EMBEDDINGS: LoRAEmbeddings | None = None


def get_embeddings(model_name: str | None = None) -> LoRAEmbeddings:
    """Singleton getter for the LoRAEmbeddings instance."""
    global _GLOBAL_EMBEDDINGS
    target = model_name or active_model_name()
    if _GLOBAL_EMBEDDINGS is None or _GLOBAL_EMBEDDINGS.model_name != target:
        _GLOBAL_EMBEDDINGS = LoRAEmbeddings(target)
    return _GLOBAL_EMBEDDINGS
