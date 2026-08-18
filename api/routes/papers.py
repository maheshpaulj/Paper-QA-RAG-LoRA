"""API routes for managing papers."""

import shutil
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, Form
from fastapi.responses import FileResponse
from typing import List

from api.schemas import PaperInfo, IngestResponse
from config import INDEX_DIR, DATA_DIR, FIGURE_DIR, ROOT

try:
    from src.lc.store import list_indexes, load_meta
except ImportError:
    # Fallback if not available
    def list_indexes():
        return []
    def load_meta(name):
        return {}
        
try:
    from scripts.build_index import main as build_index_main
except ImportError:
    def build_index_main(pdf_path, index_name):
        pass

router = APIRouter()

KNOWN_TITLES = {
    "imagenet": "ImageNet: A Large-Scale Hierarchical Image Database",
    "transformer": "Attention Is All You Need (Transformer)",
    "resnet": "Deep Residual Learning for Image Recognition (ResNet)",
    "vgg": "Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG)",
    "bert": "BERT: Pre-training of Deep Bidirectional Transformers",
    "gpt3": "Language Models are Few-Shot Learners (GPT-3)",
    "tr_adam": "Adam: A Method for Stochastic Optimization",
    "tr_batchnorm": "Batch Normalization: Accelerating Deep Network Training",
    "tr_densenet": "Densely Connected Convolutional Networks (DenseNet)",
    "tr_efficientnet": "EfficientNet: Rethinking Model Scaling for CNNs",
    "tr_mobilenet": "MobileNets: Efficient Convolutional Neural Networks",
    "tr_vit": "An Image is Worth 16x16 Words: Transformers for Image Recognition",
    "AR": "Conference Paper Revised",
}


@router.get("", response_model=List[PaperInfo])
def list_papers():
    """List all indexed papers with clean metadata."""
    try:
        indexes = list_indexes()
        papers = []
        for name in indexes:
            meta = load_meta(name)
            raw_title = (meta.get("title") or "").strip()
            title = KNOWN_TITLES.get(name) or (raw_title if raw_title and raw_title != "Untitled" else name)
            page_count = meta.get("page_count", None)
            ref_count = meta.get("reference_count", None)
            papers.append(PaperInfo(
                index_name=name,
                title=title,
                page_count=page_count,
                reference_count=ref_count
            ))
        return papers
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest", response_model=IngestResponse)
async def ingest_paper(file: UploadFile, index_name: str = Form(...)):
    """Ingest a PDF file and build its index."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        pdf_path = DATA_DIR / file.filename
        with open(pdf_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        build_index_main(str(pdf_path), index_name)
        
        return IngestResponse(index_name=index_name, chunk_count=0, figure_count=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{name}")
def delete_paper(name: str):
    """Delete a paper's index."""
    try:
        index_path = Path(INDEX_DIR) / name
        figures_path = Path(FIGURE_DIR) / name
        
        if index_path.exists():
            shutil.rmtree(index_path)
        if figures_path.exists():
            shutil.rmtree(figures_path)
            
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{name}/pdf")
def get_paper_pdf(name: str):
    """Serve the original PDF file of a paper."""
    try:
        meta_path = Path(INDEX_DIR) / name / "meta.json"
        if not meta_path.exists():
            raise HTTPException(status_code=404, detail="Meta file not found.")
            
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        source_pdf = meta.get("source_pdf")
        if not source_pdf:
            raise HTTPException(status_code=404, detail="Source PDF path not found in meta.")
            
        pdf_path = ROOT / source_pdf
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail="PDF file not found on disk.")
            
        return FileResponse(path=pdf_path, media_type="application/pdf", filename=pdf_path.name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/figures/{name}/{filename}")
def get_figure_image(name: str, filename: str):
    """Serve a cropped figure/table image from a paper."""
    fig_path = Path(FIGURE_DIR) / name / filename
    if not fig_path.exists():
        raise HTTPException(status_code=404, detail="Figure image not found.")
    return FileResponse(path=fig_path, media_type="image/png")
