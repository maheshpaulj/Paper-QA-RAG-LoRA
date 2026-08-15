"""LangChain component wrappers for the paper RAG pipeline.

Phase 6 wraps the hand-built modules from Phases 0-5 in LangChain interfaces
so they compose with LCEL chains, RunnableBranch routing, and the broader
LangChain ecosystem — while preserving the custom logic (LoRA embeddings,
section-aware chunking, figure extraction) that makes this project unique.
"""
