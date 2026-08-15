"""Remove a paper's index so it stops showing up in the app.

    python -m scripts.remove_paper gpt3            # remove one
    python -m scripts.remove_paper gpt3 vgg        # remove several
    python -m scripts.remove_paper --list          # show what's built

Deletes the index files and rendered figures for that paper. The source PDF in
data/ is left alone, so `build_index` can recreate it.

Refuses to touch a paper that has an eval set (eval/eval_set_<name>.json) unless
--force is given: dropping one of those silently changes what `run_eval` measures
and makes the before/after numbers in the README non-reproducible.

Phase 6 refactor: handles both old format (flat .faiss files) and new format
(subdirectory per paper).
"""
import shutil
import sys

from config import INDEX_DIR, FIGURE_DIR, ROOT

EVAL_DIR = ROOT / "eval"


def built_indices():
    """All index names, from both old and new format."""
    found = set()
    # New format: subdirectories with index.faiss
    for d in INDEX_DIR.iterdir():
        if d.is_dir() and (d / "index.faiss").exists():
            found.add(d.name)
    # Old format: flat .faiss files
    for p in INDEX_DIR.glob("*.faiss"):
        found.add(p.stem)
    return sorted(found)


def has_eval_set(name):
    return (EVAL_DIR / f"eval_set_{name}.json").exists()


def remove(name, force=False):
    # New format targets
    new_dir = INDEX_DIR / name
    # Old format targets
    old_targets = [
        INDEX_DIR / f"{name}.faiss",
        INDEX_DIR / f"{name}.chunks.json",
        INDEX_DIR / f"{name}.meta.json",
    ]
    figures = FIGURE_DIR / name

    has_old = any(t.exists() for t in old_targets)
    has_new = new_dir.is_dir() and (new_dir / "index.faiss").exists()

    if not has_old and not has_new:
        print(f"[{name}] no index found -- nothing to remove")
        return False

    if has_eval_set(name) and not force:
        print(f"[{name}] SKIPPED: it has an eval set (eval/eval_set_{name}.json).")
        print("         Removing it would change what run_eval measures and make")
        print("         the README's before/after numbers non-reproducible.")
        print("         Re-run with --force if you really mean it.")
        return False

    # Remove new format
    if has_new:
        shutil.rmtree(new_dir)
    # Remove old format
    for t in old_targets:
        if t.exists():
            t.unlink()
    # Remove figures
    if figures.is_dir():
        shutil.rmtree(figures)

    print(f"[{name}] removed index" + (" + figures" if figures.parent.exists() else ""))
    if has_eval_set(name):
        print(f"         note: eval/eval_set_{name}.json still exists; run_eval will")
        print(f"         fail on it until you rebuild or delete that file too.")
    return True


def main(argv):
    if "--list" in argv or not [a for a in argv if not a.startswith("--")]:
        names = built_indices()
        if not names:
            print("No indices built.")
            return
        print("Built papers:")
        for n in names:
            print(f"  {n}{'   (has eval set)' if has_eval_set(n) else ''}")
        if not any(not a.startswith("--") for a in argv):
            print("\nusage: python -m scripts.remove_paper <name> [<name> ...] [--force]")
        return

    force = "--force" in argv
    for name in [a for a in argv if not a.startswith("--")]:
        remove(name, force=force)


if __name__ == "__main__":
    main(sys.argv[1:])
