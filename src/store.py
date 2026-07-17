"""FAISS index + the chunk metadata it maps back to. Saved to disk so you
build the index once per paper and reuse it."""
import json

import faiss

from config import INDEX_DIR


class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.meta = {}

    def build(self, vectors, chunks, meta=None):
        self.index = faiss.IndexFlatIP(vectors.shape[1])  # inner product = cosine (normalized)
        self.index.add(vectors)
        self.chunks = chunks
        self.meta = meta or {}

    def search(self, query_vec, k=5):
        scores, idxs = self.index.search(query_vec, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            c = dict(self.chunks[idx])
            c["score"] = float(score)
            results.append(c)
        return results

    def save(self, name="paper"):
        faiss.write_index(self.index, str(INDEX_DIR / f"{name}.faiss"))
        with open(INDEX_DIR / f"{name}.chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        with open(INDEX_DIR / f"{name}.meta.json", "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def load(self, name="paper"):
        self.index = faiss.read_index(str(INDEX_DIR / f"{name}.faiss"))
        with open(INDEX_DIR / f"{name}.chunks.json", encoding="utf-8") as f:
            self.chunks = json.load(f)
        meta_path = INDEX_DIR / f"{name}.meta.json"
        self.meta = json.load(open(meta_path, encoding="utf-8")) if meta_path.exists() else {}
        return self