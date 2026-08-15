"""Main FastAPI application module."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import papers, query, arxiv

app = FastAPI(
    title='Paper RAG API',
    version='6.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(papers.router, prefix="/api/papers", tags=["papers"])
app.include_router(query.router, prefix="/api", tags=["query"])
app.include_router(arxiv.router, prefix="/api/arxiv", tags=["arxiv"])

@app.get("/")
async def root():
    """Root endpoint for status check."""
    return {"status": "ok", "version": "6.0"}
