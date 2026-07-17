"""Rebuild every index with whichever embedder is currently active.

    python -m scripts.rebuild_all

You MUST run this after dropping in (or removing) the LoRA model: the stored
FAISS vectors come from the old embedder, and a query embedded by a new model
searched against old vectors is nonsense. Prints which model it used so a stale
index is obvious.
"""
from src.embed import active_model_name
from scripts.build_index import main

# index name -> pdf.  tr_* are the LoRA training papers; the rest are eval papers.
PAPERS = {
    "imagenet": "data/paper.pdf",
    "transformer": "data/1706.03762.pdf",
    "resnet": "data/1512.03385.pdf",
    "vgg": "data/1409.1556.pdf",
    "bert": "data/1810.04805.pdf",
    "tr_adam": "data/1412.6980.pdf",
    "tr_batchnorm": "data/1502.03167.pdf",
    "tr_densenet": "data/1608.06993.pdf",
    "tr_mobilenet": "data/1704.04861.pdf",
    "tr_vit": "data/2010.11929.pdf",
    "tr_efficientnet": "data/1905.11946.pdf",
}


if __name__ == "__main__":
    print(f"Embedder: {active_model_name()}\n")
    for name, pdf in PAPERS.items():
        main(pdf, name)
    print(f"\nRebuilt {len(PAPERS)} indices with {active_model_name()}")
