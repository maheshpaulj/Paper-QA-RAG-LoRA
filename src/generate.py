"""The LLM turns retrieved chunks into a grounded, cited answer.

Two things baked in now that pay off later:
  * citation-forcing (every claim tagged [C#]) -> Phase 5 verification reads these
  * a fixed refusal string -> your eval harness can detect correct refusals
"""
from src.llm import chat, text_part, image_part
from config import ROOT

REFUSAL_TEXT = "The paper does not address this."

SYSTEM = f"""You answer questions about a single research paper using ONLY the provided context chunks.

Rules:
- Use only information in the context. Do not use outside knowledge.
- After each sentence, cite the chunk(s) it came from like [C3] or [C3][C7].
- If the context does not contain the answer, reply with exactly this and nothing else:
  "{REFUSAL_TEXT}"
- Be concise and precise."""

SUMMARY_SYSTEM = """You summarize a single research paper using ONLY the provided context chunks.

Rules:
- Use only information in the context. Do not use outside knowledge.
- Write a concise summary (3-6 sentences) covering the problem, approach, and key result.
- After each sentence, cite the chunk(s) it came from like [C3] or [C3][C7]."""


def _format_context(chunks):
    out = []
    for c in chunks:
        head = f"[{c['id']}] (p.{c['page']}, {c['section']})"
        if c.get("type") == "figure":
            head += " -- FIGURE, image attached below"
        out.append(f"{head}\n{c['text']}")
    return "\n\n".join(out)


def _image_parts(chunks):
    """Attach the rendered image for any figure chunk we retrieved."""
    parts = []
    for c in chunks:
        if c.get("type") != "figure" or not c.get("image_path"):
            continue
        path = ROOT / c["image_path"]
        if path.exists():
            parts.append(image_part(path.read_bytes()))
    return parts


class Generator:
    def __init__(self, model=None):
        self.model = model  # None -> whatever config.llm_endpoint() selects

    def answer(self, question, chunks):
        prompt = (
            f"{SYSTEM}\n\nCONTEXT:\n{_format_context(chunks)}\n\n"
            f"QUESTION: {question}\n\nANSWER:"
        )
        return chat([text_part(prompt)] + _image_parts(chunks), model=self.model)

    def summarize(self, question, chunks):
        prompt = (
            f"{SUMMARY_SYSTEM}\n\nCONTEXT:\n{_format_context(chunks)}\n\n"
            f"REQUEST: {question}\n\nSUMMARY:"
        )
        return chat(prompt, model=self.model)