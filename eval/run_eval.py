"""Run the eval set and print the numbers you'll quote on your resume.

    python -m eval.run_eval                       # every eval/eval_set_*.json
    python -m eval.run_eval eval/eval_set_imagenet.json
    RERANK=0 python -m eval.run_eval              # baseline, no reranking
    python -m eval.run_eval --refusal             # also test refusals (uses LLM quota)

Metrics:
  precision@k  -- of the k retrieved chunks, what fraction are relevant
  hit@k        -- did at least one relevant chunk make the top k
  refusal acc  -- of the out-of-scope questions, how many were correctly refused

A chunk counts as relevant if it contains any of the item's gold_snippets.
Each item carries an "index" naming the paper it belongs to, so one run can span
many papers: items are grouped by index and a retriever is built per paper (the
embedder and reranker are shared, so the models load once).

Retrieval metrics call only the retriever (no LLM), so they're free and isolate
retrieval quality -- exactly what reranking and LoRA change. Run before/after
each phase to get real deltas. The refusal check needs generation, so it's opt-in.
"""
import glob
import json
import sys
from collections import defaultdict

from src.embed import Embedder
from src.retrieve import Retriever
from config import TOP_K, RERANK_ENABLED, ROOT


def chunk_is_relevant(chunk, gold_snippets):
    text = chunk["text"].lower()
    return any(g.lower() in text for g in gold_snippets)


def load_items(paths):
    items = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            items.extend(json.load(f))
    return items


def run(paths, with_refusal=False):
    items = load_items(paths)
    by_index = defaultdict(list)
    for it in items:
        by_index[it.get("index", "paper")].append(it)

    embedder = Embedder()          # load the models once, reuse across papers
    reranker = None
    if RERANK_ENABLED:
        from src.rerank import Reranker
        reranker = Reranker()

    rows, all_prec, all_hit = [], [], []
    for index_name, group in sorted(by_index.items()):
        retriever = Retriever(index_name, embedder=embedder, reranker=reranker)
        prec, hit = [], []
        for it in group:
            if it["type"] != "in_scope" or not it.get("gold_snippets"):
                continue
            chunks = retriever.retrieve(it["question"], k=TOP_K)
            rel = [chunk_is_relevant(c, it["gold_snippets"]) for c in chunks]
            prec.append(sum(rel) / max(len(chunks), 1))
            hit.append(1.0 if any(rel) else 0.0)
        if prec:
            rows.append((index_name, len(prec), sum(prec) / len(prec), sum(hit) / len(hit)))
            all_prec += prec
            all_hit += hit

    print(f"\n=== Metrics (rerank={'on' if RERANK_ENABLED else 'off'}) ===")
    print(f"{'paper':22} {'n':>3}  {'precision@' + str(TOP_K):>12} {'hit@' + str(TOP_K):>8}")
    for name, n, p, h in rows:
        print(f"{name:22} {n:>3}  {p:>12.3f} {h:>8.3f}")
    if all_prec:
        print(f"{'-' * 48}")
        print(f"{'OVERALL':22} {len(all_prec):>3}  "
              f"{sum(all_prec) / len(all_prec):>12.3f} {sum(all_hit) / len(all_hit):>8.3f}")

    if with_refusal:
        _run_refusal(by_index)


def _run_refusal(by_index):
    """Out-of-scope refusal accuracy. Needs the LLM, so it spends API quota."""
    from src.pipeline import RAGPipeline
    from src.generate import REFUSAL_TEXT

    correct = total = 0
    for index_name, group in sorted(by_index.items()):
        oos = [it for it in group if it["type"] == "out_of_scope"]
        if not oos:
            continue
        rag = RAGPipeline(index_name)
        for it in oos:
            total += 1
            if REFUSAL_TEXT.lower() in rag.ask(it["question"])["answer"].lower():
                correct += 1
    if total:
        print(f"refusal acc:  {correct}/{total} = {correct / total:.3f}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    paths = args or sorted(glob.glob(str(ROOT / "eval" / "eval_set_*.json")))
    if not paths:
        raise SystemExit("No eval sets found. Run: python -m scripts.make_eval <index>")
    run(paths, with_refusal="--refusal" in sys.argv)
