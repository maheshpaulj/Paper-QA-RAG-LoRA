"""Rebuild every built index with whichever embedder is currently active.

    python -m scripts.rebuild_all

You MUST run this after adding or removing the LoRA model: the stored FAISS
vectors come from the old embedder, and a query embedded by a new model searched
against old vectors is nonsense. It prints the model it used so a stale index is
obvious.

Phase 6 refactor: discovers indexes from both old format (*.faiss flat files)
and new format (subdirectories with index.faiss). Old-format indexes are
rebuilt into the new LangChain FAISS format.
"""
import json

from config import INDEX_DIR, ROOT
from src.lc.embeddings import active_model_name
from scripts.build_index import main


def discover():
    """[(index_name, pdf_path)] for every built index.

    Looks in two places:
      - New format: index/<name>/meta.json (subdirectory per paper)
      - Old format: index/<name>.meta.json (flat files, pre-Phase 6)
    """
    found = {}

    # New format: subdirectories
    for d in sorted(INDEX_DIR.iterdir()):
        if d.is_dir() and (d / "index.faiss").exists():
            meta_path = d / "meta.json"
            src = None
            if meta_path.exists():
                src = json.loads(meta_path.read_text(encoding="utf-8")).get("source_pdf")
            found[d.name] = src

    # Old format: flat .faiss files (pre-Phase 6 migration)
    for faiss_path in sorted(INDEX_DIR.glob("*.faiss")):
        name = faiss_path.stem
        if name in found:
            continue  # already found in new format
        meta_path = INDEX_DIR / f"{name}.meta.json"
        src = None
        if meta_path.exists():
            src = json.loads(meta_path.read_text(encoding="utf-8")).get("source_pdf")
        found[name] = src

    return list(found.items())


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
