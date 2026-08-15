@echo off
echo Starting Paper RAG API on http://localhost:8000
echo Docs at http://localhost:8000/docs
.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
