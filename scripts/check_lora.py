"""Is the LoRA actually doing anything?

    python -m scripts.check_lora

A LoRA whose keys didn't match on load is re-initialised to lora_B = 0, which
makes it mathematically identical to the base model -- no error, no warning, just
a silent no-op that would show up as "LoRA gave exactly 0 improvement".

This compares LoRA vs base embeddings. cosine 1.0 == dead adapter.
Run it after every retrain, before spending time on rebuild_all + run_eval.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBED_MODEL, LORA_MODEL_DIR

# in-domain (should move most) vs off-domain (should barely move)
PROBES = [
    "what optimizer does the paper propose?",
    "how does batch normalization reduce internal covariate shift?",
    "what is the capital of France?",
]


def main():
    if not LORA_MODEL_DIR.exists():
        raise SystemExit(f"No LoRA model at {LORA_MODEL_DIR} -- train it first (colab/train_lora.ipynb)")

    lora = SentenceTransformer(str(LORA_MODEL_DIR))
    base = SentenceTransformer(EMBED_MODEL)
    a = lora.encode(PROBES, normalize_embeddings=True)
    b = base.encode(PROBES, normalize_embeddings=True)

    cosines = [float(np.dot(x, y)) for x, y in zip(a, b)]
    for c, q in zip(cosines, PROBES):
        print(f"  cos={c:.4f}  {q}")
    mean = float(np.mean(cosines))
    print(f"\nmean cosine vs base: {mean:.4f}")

    if mean > 0.9999:
        raise SystemExit(
            "\nDEAD ADAPTER: identical to the base model.\n"
            "The saved keys probably didn't match on load. Fix with:\n"
            "  python -m scripts.fix_adapter_keys"
        )
    print("LoRA is active.")


if __name__ == "__main__":
    main()
