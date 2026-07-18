"""Rebuild every built index with whichever embedder is currently active.

    python -m scripts.rebuild_all

You MUST run this after adding or removing the LoRA model: the stored FAISS
vectors come from the old embedder, and a query embedded by a new model searched
against old vectors is nonsense. It prints the model it used so a stale index is
obvious.

Papers are discovered from the indices on disk (each meta.json records the PDF it
was built from). It used to be a hardcoded name->pdf map, which silently skipped
any paper you added yourself and left it on stale vectors -- exactly the failure
this script exists to prevent.

Papers whose source PDF is missing are reported and skipped, not fatal: one gone
PDF shouldn't abort the rebuild and leave the remaining indices half-updated.
"""
import json

from config import INDEX_DIR, ROOT
from src.embed import active_model_name
from scripts.build_index import main


def discover():
    """[(index_name, pdf_path)] for every built index, from its meta.json."""
    found = []
    for faiss_path in sorted(INDEX_DIR.glob("*.faiss")):
        name = faiss_path.stem
        meta_path = INDEX_DIR / f"{name}.meta.json"
        src = None
        if meta_path.exists():
            src = json.loads(meta_path.read_text(encoding="utf-8")).get("source_pdf")
        found.append((name, src))
    return found


if __name__ == "__main__":
    print(f"Embedder: {active_model_name()}\n")

    rebuilt, skipped = 0, []
    for name, src in discover():
        if not src:
            skipped.append((name, "no source_pdf recorded (built before this was tracked)"))
            continue
        if not (ROOT / src).exists():
            skipped.append((name, f"missing {src}"))
            continue
        try:
            main(src, name)
            rebuilt += 1
        except Exception as e:  # one bad paper must not abandon the rest
            skipped.append((name, f"{type(e).__name__}: {str(e)[:80]}"))

    print(f"\nRebuilt {rebuilt} indices with {active_model_name()}")
    if skipped:
        print(f"\nSkipped {len(skipped)} -- these keep STALE vectors and will give")
        print("bad results until rebuilt with:  python -m scripts.build_index <pdf> <name>")
        for name, why in skipped:
            print(f"  {name}: {why}")
