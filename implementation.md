## Problem Statement

Research papers pack their most important findings into dense text and visual elements (figures, charts, tables), but general-purpose PDF chat tools treat papers as flat text — ignoring figures, using generic retrieval not tuned for academic content, and answering confidently even when the paper doesn't address the question at all.

**This project builds a system that lets a user upload a single research paper and ask natural-language questions — including questions requiring understanding of a chart or figure — and receive answers that are explicitly grounded, cited, and verified against the paper's actual content, with retrieval quality improved via domain-specific fine-tuning and generation constrained to refuse ungrounded questions rather than hallucinate.**

**Scope boundary:** single-paper Q&A (not a multi-paper corpus) — keeps retrieval bounded, verification tractable, and fine-tuning data collection feasible within a student timeline.

## Goals (each maps to a specific technique you can defend in an interview)

**Goal 1 — Multimodal Retrieval-Augmented Generation**
Retrieve and reason over both text sections and figures/charts jointly, not treat the paper as text-only.
- *Technique*: separate text and CLIP image embeddings, dual-index retrieval, multimodal LLM generation (Gemini/GPT-4o taking retrieved text + images together)

**Goal 2 — Domain-Adapted Retrieval via LoRA Fine-Tuning**
Improve retrieval precision beyond what an off-the-shelf general-purpose embedding model achieves, by adapting it specifically to academic-paper Q&A patterns.
- *Technique*: LoRA fine-tuning of a small `sentence-transformers` model on a query–chunk triplet dataset (built from your eval set + synthetic augmentation), trained on free Colab/Kaggle GPU
- *Success metric*: retrieval precision@5 before vs. after fine-tuning — this comparison is the actual deliverable, not the fine-tuning itself

**Goal 3 — Hallucination Mitigation via Grounded Generation + Verification**
Prevent the system from answering questions the paper doesn't cover, and catch ungrounded claims before they reach the user.
- *Technique*: citation-forced generation (every claim tagged to its source chunk/figure) + a second LLM-as-judge verification pass that checks each claim against its cited source and flags/strips unsupported ones
- *Success metric*: correct-refusal rate on deliberately out-of-scope test questions, and citation-faithfulness rate on your eval set
- *Explicit non-goal*: this project does not claim to solve hallucination generally (an open research problem) — it demonstrates measured mitigation on a bounded task, which is the honest and defensible framing

**Goal 4 — Reranking for Precision**
Reduce noise from near-duplicate or superficially-similar chunks that pure vector similarity retrieves incorrectly.
- *Technique*: cross-encoder reranker (`bge-reranker-base`) applied to initial retrieval candidates before generation

**Goal 5 — Rigorous, Reproducible Evaluation**
Every claim above ("improved precision," "reduced hallucination") needs to be backed by a number, not asserted.
- *Technique*: hand-built eval set (~30 Q&A pairs across several test papers, including deliberate trick/out-of-scope questions), measuring retrieval precision@5, citation accuracy, and refusal accuracy — run both before and after each major change (baseline retriever vs. fine-tuned, with/without verification pass) so you have real before/after deltas to cite

## What this project demonstrates, mapped to skills employers actually screen for

| Skill area | How this project proves it |
|---|---|
| RAG system design | Multimodal dual-index retrieval + reranking pipeline |
| Applied fine-tuning (LoRA/PEFT) | Adapted embedding model with measured before/after gain |
| Hallucination-aware engineering | Citation-forcing + independent verification pass |
| ML evaluation rigor | Hand-built eval set, quantified metrics, ablation-style before/after comparisons |
| Production-adjacent backend | FastAPI, streaming, deployed service |
