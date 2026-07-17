"""The retriever's embedding model -- the ONLY component Phase 4 fine-tunes.

If a LoRA-trained model has been unzipped into models/minilm-lora it is used
automatically; env LORA=0 forces the base model, which is how you measure the
before/after. Re-run build_index after switching: the vectors change.
"""
from sentence_transformers import SentenceTransformer

from config import EMBED_MODEL, LORA_MODEL_DIR, USE_LORA


def active_model_name():
    if USE_LORA and LORA_MODEL_DIR.exists():
        return str(LORA_MODEL_DIR)
    return EMBED_MODEL


class Embedder:
    def __init__(self, model_name=None):
        self.model_name = model_name or active_model_name()
        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts):
        # normalize -> cosine similarity becomes a plain inner product in FAISS
        vecs = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.astype("float32")