"""Generate LoRA training triplets (query, positive, hard negative).

    python -m scripts.make_train tr_adam tr_batchnorm --n 50

Writes train/triplets.jsonl -- upload that one file to Colab.

Two rules this script enforces, because getting them wrong invalidates Phase 4:

  1. TRAIN PAPERS MUST BE DISJOINT FROM EVAL PAPERS. Not just different
     questions -- different *documents*. Training on the papers you evaluate on
     teaches the embedder those exact chunks, and the "improvement" is
     memorisation. Any index with an eval set is refused outright.

  2. Hard negatives, not random ones. A random chunk is trivially separable and
     teaches nothing. We embed the query, search the paper's own index, and take
     the top-ranked chunks that are NOT the positive -- the ones the current
     model already confuses. That is what the fine-tune has to fix.

Phase 6 refactor: uses LangChain LoRAEmbeddings + FAISS vectorstore. Same
triplet generation logic, LangChain plumbing.
"""
import json
import re
import sys
import glob

from src.llm import chat
from src.lc.embeddings import LoRAEmbeddings
from src.lc.store import load_vectorstore
from config import INDEX_DIR, ROOT

BATCH = 8
NEGATIVES_PER_QUERY = 1
CANDIDATE_DEPTH = 8  # how deep to look for a hard negative

PROMPT = """You are building training data for a research-paper retrieval model.
For EACH excerpt below, write one specific, self-contained question that a
researcher might ask, which this excerpt answers. Do not mention "the excerpt".
Return ONLY a JSON array, one object per line, no prose:
[{"id":"C3","question":"..."}]

EXCERPTS:
%s
"""


def _clean_json(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def _parse_rows(text):
    text = _clean_json(text)
    try:
        data = json.loads(text, strict=False)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        pass
    rows = []
    for m in re.finditer(r"\{[^{}]*\}", text, re.S):
        try:
            rows.append(json.loads(m.group(0), strict=False))
        except json.JSONDecodeError:
            continue
    return rows


def eval_indices():
    """Papers that have an eval set are off-limits for training."""
    names = set()
    for p in glob.glob(str(ROOT / "eval" / "eval_set_*.json")):
        with open(p, encoding="utf-8") as f:
            for it in json.load(f):
                names.add(it.get("index"))
    return {n for n in names if n}


def text_chunks_from_vectorstore(name, embeddings):
    """Load all text chunks from a vectorstore, filtering out figures and refs."""
    vectorstore = load_vectorstore(name, embeddings)
    all_docs = list(vectorstore.docstore._dict.values())
    return [
        {"id": d.metadata.get("id", "?"), "text": d.page_content, **d.metadata}
        for d in all_docs
        if d.metadata.get("type") != "figure"
        and d.metadata.get("section") != "References"
        and len(d.page_content) > 220
    ]


def build(names, n_per_paper=50, out=None):
    contaminated = sorted(set(names) & eval_indices())
    if contaminated:
        raise SystemExit(
            f"REFUSED: {contaminated} have eval sets. Training on a paper you "
            f"evaluate on makes the Phase 4 number meaningless. Use other papers."
        )

    embeddings = LoRAEmbeddings()
    triplets = []
    out = out or (ROOT / "train" / "triplets.jsonl")
    out.parent.mkdir(exist_ok=True)

    def flush():
        """Save after every paper. These cost LLM calls under a tight rate limit;
        losing them to a later failure is not acceptable."""
        with open(out, "w", encoding="utf-8") as f:
            for t in triplets:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

    for name in names:
        try:
            chunks = text_chunks_from_vectorstore(name, embeddings)
        except Exception as e:
            print(f"[{name}] no index -- run build_index first; skipping ({e})")
            continue
        by_id = {c["id"]: c for c in chunks}

        # Use vectorstore retriever without reranking -- we want the bi-encoder's
        # own confusions as negatives
        vectorstore = load_vectorstore(name, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": CANDIDATE_DEPTH})
        made = 0

        for i in range(0, len(chunks), BATCH):
            if made >= n_per_paper:
                break
            batch = chunks[i:i + BATCH]
            excerpts = "\n\n".join(f"[{c['id']}] {c['text'][:500]}" for c in batch)
            try:
                rows = _parse_rows(chat(PROMPT % excerpts))
            except Exception as e:
                print(f"  [{name}] batch {i // BATCH} failed: {str(e)[:70]}")
                continue

            for r in rows:
                pos = by_id.get(r.get("id"))
                q = (r.get("question") or "").strip()
                if not (pos and q):
                    continue
                neg_docs = retriever.invoke(q)
                negs = [
                    d for d in neg_docs
                    if d.metadata.get("id") != pos["id"]
                    and d.metadata.get("type") != "figure"
                ]
                for neg in negs[:NEGATIVES_PER_QUERY]:
                    triplets.append({
                        "query": q,
                        "positive": pos["text"],
                        "negative": neg.page_content,
                        "paper": name,
                    })
                    made += 1
                if made >= n_per_paper:
                    break
        flush()
        print(f"[{name}] {made} triplets  (saved, {len(triplets)} total)")

    print(f"\nWrote {len(triplets)} triplets from {len(names)} papers -> {out}")
    print(f"Eval papers (excluded): {sorted(eval_indices())}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    n = 50
    if "--n" in argv:  # consume the flag AND its value, else it looks like an index name
        i = argv.index("--n")
        n = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    names = [a for a in argv if not a.startswith("--")]
    if not names:
        raise SystemExit("usage: python -m scripts.make_train <index> [<index> ...] [--n 50]")
    build(names, n)
