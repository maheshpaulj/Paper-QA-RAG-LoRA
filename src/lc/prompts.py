"""Prompt templates for the RAG pipeline — same wording as generate.py, now as
LangChain ChatPromptTemplates.

Two key rules baked into these prompts (and tested by eval):
  1. Citation forcing: every claim must be tagged [C#] or [F#].
  2. Refusal: if the context can't answer, reply with EXACTLY the refusal string.
     The eval harness (run_eval --refusal) matches on this string, so changing
     it silently breaks refusal accuracy measurement.

The prompts are separated from the chain so they can be read, tested, and
modified independently.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Must match what eval/run_eval.py checks for in refusal accuracy.
REFUSAL_TEXT = "The paper does not address this."

QA_SYSTEM = (
    "You answer questions about a single research paper using ONLY the "
    "provided context chunks.\n\n"
    "Rules:\n"
    "- Use only information in the context. Do not use outside knowledge.\n"
    "- (Exception: You may logically infer the paper's title and authors from the front matter chunks even if not explicitly labeled.)\n"
    "- After each sentence, cite the chunk(s) it came from like [C3] or [C3][C7].\n"
    "- If the context does not contain the answer, reply with exactly this and "
    f'nothing else:\n  "{REFUSAL_TEXT}"\n'
    "- Be concise and precise."
)

SUMMARY_SYSTEM = (
    "You summarize a single research paper using ONLY the provided context "
    "chunks.\n\n"
    "Rules:\n"
    "- Use only information in the context. Do not use outside knowledge.\n"
    "- Write a concise summary (3-6 sentences) covering the problem, approach, "
    "and key result.\n"
    "- After each sentence, cite the chunk(s) it came from like [C3] or [C3][C7]."
)

# ── QA prompt (the normal path) ──────────────────────────────────────────────

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM),
    MessagesPlaceholder(variable_name="history", optional=True),
    ("human", "CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"),
])

# ── Summary prompt ───────────────────────────────────────────────────────────

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", SUMMARY_SYSTEM),
    MessagesPlaceholder(variable_name="history", optional=True),
    ("human", "CONTEXT:\n{context}\n\nREQUEST: {question}\n\nSUMMARY:"),
])
