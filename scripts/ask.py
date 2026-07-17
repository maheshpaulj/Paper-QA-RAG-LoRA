"""Interactive Q&A against the built index.

    python -m scripts.ask
"""
from src.pipeline import RAGPipeline


def main():
    rag = RAGPipeline()
    print("Ask questions about the paper (Ctrl+C to quit).")
    while True:
        try:
            q = input("\nQ: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        out = rag.ask(q)
        print(f"\nA: {out['answer']}")
        print("\nRetrieved chunks:")
        for c in out["chunks"]:
            print(f"  [{c['id']}] p.{c['page']} {c['section']} (score {c['score']:.2f})")


if __name__ == "__main__":
    main()