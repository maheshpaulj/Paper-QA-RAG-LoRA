"""Pydantic models for the Paper RAG API."""

from pydantic import BaseModel
from typing import Optional, List, Dict

class AskRequest(BaseModel):
    """Request model for asking a question."""
    question: str
    index_name: str
    history: Optional[List[Dict[str, str]]] = []

class ChunkSchema(BaseModel):
    """Schema representing a text chunk from a paper."""
    id: str
    text: str
    page: int
    section: str
    type: str = 'text'
    score: Optional[float] = None
    image_path: Optional[str] = None
    image_url: Optional[str] = None

class AskResponse(BaseModel):
    """Response model for a question."""
    question: str
    answer: str
    route: str
    chunks: List[ChunkSchema]

class PaperInfo(BaseModel):
    """Information about an indexed paper."""
    index_name: str
    title: str
    page_count: Optional[int] = None
    reference_count: Optional[int] = None

class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    index_name: str
    chunk_count: int
    figure_count: int

class ArxivFetchRequest(BaseModel):
    """Request model for fetching an arXiv paper."""
    arxiv_id: str

class ArxivFetchResponse(BaseModel):
    """Response model for an arXiv fetch."""
    pdf_path: str
    arxiv_id: str
