"""Generate an eval set from a built index using Gemini.

    python -m scripts.make_eval paper --n 15

Writes eval/eval_set.json in the schema eval/run_eval.py expects:
  {id, type:"in_scope",    question, gold_snippets:[verbatim substring], source_chunk}
  {id, type:"out_of_scope", question}

Every in_scope snippet is checked to be an exact substring of its source chunk,
so run_eval's relevance test is sound. Questions come from the paper's own text,
which is fine for a held-out *eval* set; never reuse these for LoRA *training*.
"""
import json
import re
import sys
import random

from src.llm import chat

from config import INDEX_DIR, ROOT

OUT_OF_SCOPE = [
    "What is the capital of France?",
    "Who won the 2018 FIFA World Cup?",
    "What is the boiling point of water at sea level?",
    "How do I bake sourdough bread?",
]

PROMPT = """You are building a retrieval eval set for one research paper.
For EACH excerpt below, write one specific, self-contained factual question the
excerpt answers, then copy the EXACT verbatim sentence (word-for-word from the
excerpt, under 220 characters) that contains the answer.
Return ONLY a JSON array, no prose. Keep each object on a single line and escape
any quotes inside strings:
[{"id":"C3","question":"...","snippet":"..."}]

EXCERPTS:
%s
"""

# Ask for a few excerpts at a time: one big array invites truncated replies, and
# a single malformed row would otherwise cost us the whole paper.
BATCH = 8


def _clean_json(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def _parse_rows(text):
    """Parse the model's reply, salvaging what we can.

    strict=False tolerates the raw newlines that show up in snippets copied out
    of a PDF. If the reply is still broken (usually truncated mid-array), pull
    out the objects that did survive instead of losing the whole batch.
    """
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


def candidates(chunks, k):
    pool = [c for c in chunks if c.get("section") != "References" and len(c["text"]) > 220]
    random.Random(0).shuffle(pool)
    return pool[:k]


def generate(name="paper", n=15, out=None):
    chunks = json.load(open(INDEX_DIR / f"{name}.chunks.json", encoding="utf-8"))
    by_id = {c["id"]: c for c in chunks}
    cands = candidates(chunks, n + 12)  # oversample; some snippets won't verify

    items, qn = [], 0
    for i in range(0, len(cands), BATCH):
        if qn >= n:
            break
        batch = cands[i:i + BATCH]
        excerpts = "\n\n".join(f"[{c['id']}] {c['text'][:500]}" for c in batch)
        try:
            rows = _parse_rows(chat(PROMPT % excerpts))
        except Exception as e:  # a dead batch shouldn't kill the paper
            print(f"  [{name}] batch {i // BATCH} failed: {str(e)[:80]}")
            continue

        for r in rows:
            c = by_id.get(r.get("id"))
            q = (r.get("question") or "").strip()
            snip = " ".join((r.get("snippet") or "").split())
            if not (c and q and snip):
                continue
            # exactly the test run_eval applies -- pass here, pass there
            if snip.lower() not in c["text"].lower():
                continue
            qn += 1
            items.append({
                "id": f"{name}_q{qn}", "type": "in_scope", "index": name, "question": q,
                "gold_snippets": [snip], "source_chunk": c["id"],
            })
            if qn >= n:
                break

    for i, q in enumerate(OUT_OF_SCOPE, 1):
        items.append({"id": f"{name}_oos{i}", "type": "out_of_scope",
                      "index": name, "question": q})

    out = out or (ROOT / "eval" / f"eval_set_{name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"[{name}] wrote {qn} in-scope + {len(OUT_OF_SCOPE)} out-of-scope -> {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    name = args[0] if args and not args[0].startswith("--") else "paper"
    n = 15
    if "--n" in args:
        n = int(args[args.index("--n") + 1])
    generate(name, n)
