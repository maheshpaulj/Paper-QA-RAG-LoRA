FROM python:3.11-slim

# System deps for PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cache-friendly)
COPY requirements.txt requirements-langchain.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-langchain.txt

# Copy the app
COPY config.py ./
COPY src/ ./src/
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY eval/ ./eval/

# Data, index, models are mounted as volumes
RUN mkdir -p data index models

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
