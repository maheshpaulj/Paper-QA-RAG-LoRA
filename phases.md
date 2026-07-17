## Six things I'd change or add

**1. Build it in phases where each phase is independently demoable *and* produces a number.** This is the single most important change. The failure mode for a project this size is "built everything, nothing quite works, no clean before/after." If you build a text-only baseline + eval harness first, then every later goal becomes a measured delta on top of a working system.

**2. Separate your eval data from your training data.** ~30 hand-built Q&A pairs is right for a *held-out eval set* — but it's far too little to *train* a LoRA on. For training, generate a few hundred synthetic query–chunk triplets: feed Gemini a chunk, ask it to write questions that chunk answers (positives), and mine hard negatives (chunks that are similar but wrong, via BM25 or embedding similarity). Never let a training pair touch your eval set.

**3. Don't over-engineer figure retrieval in v1.** CLIP image embeddings for figures sound impressive but are fiddly (separating real figures from logos/equations is messy). Pragmatic version that still fully demos multimodal: extract each figure *with its caption*, embed the **caption text** for retrieval (captions are gold), and pass the **actual figure image** to Gemini at generation time. Gemini Flash reads charts well and is free. CLIP embeddings can be a later enhancement, not a blocker.

**4. Set LoRA expectations, and pick one paper domain.** The before/after precision gain can be small and noisy on a single domain. Make it clean by choosing eval papers from *one* area (e.g. all ML/CS papers) so the fine-tune has a coherent target. Keep LoRA/PEFT as the resume story — but know that full fine-tuning of a tiny model (`all-MiniLM-L6-v2`) on free Colab is trivial and a valid fallback if LoRA gains are marginal.

**5. Add section-aware chunking** (missing from the spec). Naive fixed-size chunks hurt on papers. Respecting section headers and keeping tables/captions intact is a cheap win and a good thing to talk about in an interview.

**6. Batch the verification pass.** LLM-as-judge doubles API calls. Send all claims from one answer in a *single* judge call, not one per claim — matters because Gemini's free tier has per-minute/per-day limits.

## Recommended free stack

Parsing: **PyMuPDF** (text + image extraction). Embeddings/reranker: **sentence-transformers** (`all-MiniLM-L6-v2`, `bge-reranker-base`) — both run fine on your CPU for a single paper. Vector store: **FAISS-CPU** or **Chroma**. Generation + multimodal: **Gemini 2.x Flash** free API. LoRA training: **Colab free T4** with PEFT. Backend/UI: **FastAPI** + **Streamlit**. Deploy: **Hugging Face Spaces** (free CPU, best portfolio fit). Eval logging: plain Python + CSV.

## Suggested roadmap

0. Repo skeleton, env, eval-set scaffold, Gemini key working
1. **Text-only baseline RAG** end-to-end + eval harness (precision@5, citation, refusal) ← first working demo + baseline numbers
2. Reranking → measure the delta
3. Multimodal (figure+caption extraction, images to Gemini) → figure-question accuracy
4. LoRA fine-tune on Colab → precision@5 before/after
5. Verification pass → refusal + citation-faithfulness numbers
6. FastAPI/Streamlit → deploy to HF Spaces
