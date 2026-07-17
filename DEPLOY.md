# Deployment to Hugging Face Spaces

The Streamlit app runs on **HF Spaces** with zero configuration. It auto-detects and uses:
- All built indices (5 papers by default)
- The LoRA model if `models/minilm-lora/adapter_config.json` exists
- The reranker (automatically loaded by `src/retrieve.py`)

## Prerequisites

1. **Built indices** — run locally:
   ```bash
   python -m scripts.build_index data/paper.pdf imagenet
   python -m scripts.build_index data/1706.03762.pdf transformer
   # ... (all 5 papers)
   ```

2. **Cloudflare key** (for LLM queries) — add to `.env`:
   ```
   LLM_PROVIDER=cloudflare
   CLOUDFLARE_ACCOUNT_ID=<id>
   CLOUDFLARE_API_TOKEN=<token>
   ```

## Deploy to HF Spaces

1. Create a new **Streamlit** space at https://huggingface.co/new-space
   - Owner: your username
   - Space name: `paper-rag` (or whatever)
   - License: MIT or your choice
   - Private: uncheck (or keep private if you prefer)

2. Clone the space:
   ```bash
   git clone https://huggingface.co/spaces/<username>/paper-rag
   cd paper-rag
   ```

3. Copy files from this repo:
   ```bash
   cp <repo>/app.py .
   cp <repo>/requirements.txt .
   cp <repo>/.streamlit/config.toml .streamlit/
   cp -r <repo>/index .
   cp -r <repo>/models .
   cp -r <repo>/src .
   cp -r <repo>/config.py .
   cp <repo>/.env .  # ← HF Spaces can't read .env in the UI, so add secrets via the Space settings
   ```

4. **IMPORTANT**: Add secrets in the HF Spaces UI:
   - Go to **Settings → Repository secrets**
   - Add `CLOUDFLARE_ACCOUNT_ID=<value>`
   - Add `CLOUDFLARE_API_TOKEN=<value>`
   - HF Spaces injects these as env vars automatically

5. Commit and push:
   ```bash
   git add -A
   git commit -m "add paper-rag app"
   git push
   ```

   Spaces auto-deploys on push. It will download dependencies from `requirements.txt` and start the Streamlit server.

## What's included

- **5 papers**: ImageNet, Transformer, ResNet, VGG, BERT
- **Reranking**: cross-encoder enabled by default
- **LoRA**: auto-loaded if present
- **Multimodal**: figures render as images in the UI

## First run

First request will be slow (~30s) — the embedder and reranker models download and cache. Subsequent queries are fast.

## Troubleshooting

**"No such file: index/imagenet.faiss"** — run `build_index` locally for all 5 papers and commit the `index/` dir.

**LLM requests timing out** — check CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN are set in Spaces Settings → Secrets.

**Out of memory** — HF Spaces free tier has 2GB RAM. FAISS + embedder + reranker use ~1.5GB together. Should fit, but if it's tight, reduce the number of papers.

## Local testing before deploy

```bash
pip install -r requirements-app.txt
streamlit run app.py
# Visit http://localhost:8501
```
