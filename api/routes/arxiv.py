"""API routes for fetching arXiv papers."""

from fastapi import APIRouter, HTTPException
from api.schemas import ArxivFetchRequest, ArxivFetchResponse

try:
    from scripts.fetch_papers import download
except ImportError:
    download = None

router = APIRouter()

@router.post("/fetch", response_model=ArxivFetchResponse)
def fetch_arxiv(req: ArxivFetchRequest):
    """Fetch an open-access paper from arXiv."""
    try:
        if not download:
            raise HTTPException(status_code=501, detail="Arxiv fetcher not implemented")
            
        pdf_path = download(req.arxiv_id)
        return ArxivFetchResponse(
            pdf_path=str(pdf_path),
            arxiv_id=req.arxiv_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
