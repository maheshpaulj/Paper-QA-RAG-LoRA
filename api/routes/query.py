"""API routes for querying the papers."""

from fastapi import APIRouter, HTTPException
from api.schemas import AskRequest, AskResponse

try:
    from src.lc.chain import build_rag_chain
except ImportError:
    build_rag_chain = None

router = APIRouter()

@router.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    """Ask a question to the RAG system."""
    try:
        if not build_rag_chain:
            raise HTTPException(status_code=501, detail="RAG chain not implemented")
            
        chain = build_rag_chain(req.index_name)
        result = chain.invoke({
            "question": req.question,
            "history": req.history
        })
        
        return AskResponse(
            question=req.question,
            answer=result.get("answer", ""),
            route=result.get("route", "default"),
            chunks=result.get("chunks", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
