import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"
FIGURE_DIR = INDEX_DIR / "figures"
DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

# --- Models ---
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- LoRA embedder (Phase 4) ---
# Colab trains a LoRA on the base model, merges it, and you unzip the result
# here. Used automatically once it exists. Set env LORA=0 to force the base
# model -- that's how you get the before/after number.
LORA_MODEL_DIR = ROOT / "models" / "minilm-lora"
USE_LORA = os.getenv("LORA", "1") == "1"

# --- Generation ---
# Both providers speak the OpenAI chat-completions shape, so src/llm.py has one
# code path. Whichever you pick, the model MUST be VISION-capable: Phase 3 sends
# figure images, and a text-only model drops them silently rather than erroring.
#
#   cloudflare -- Workers AI. Free tier is ~10k Neurons/day (thousands of calls).
#   openrouter -- free tier is only 50 requests/DAY, account-wide across every
#                 free model, which is why it can't carry the training-data step.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "cloudflare")

# Cloudflare Workers AI (OpenAI-compatible endpoint)
CF_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CF_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
# Vision-capable, ungated, and on the Workers Free plan. Cloudflare's other
# vision models (llama-3.2-11b-vision, moondream) are gated behind a licence
# agreement you must accept yourself; llava-1.5 is ungated but weak at text.
CF_MODEL = "@cf/mistralai/mistral-small-3.1-24b-instruct"
CF_FALLBACKS = []

# OpenRouter
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "google/gemma-4-26b-a4b-it:free"
OPENROUTER_FALLBACKS = [
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
]
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def llm_endpoint():
    """(url, api_key, model, fallbacks) for the active provider."""
    if LLM_PROVIDER == "cloudflare":
        if not CF_ACCOUNT_ID:
            raise RuntimeError("CLOUDFLARE_ACCOUNT_ID not set -- see .env.example")
        url = (f"https://api.cloudflare.com/client/v4/accounts/"
               f"{CF_ACCOUNT_ID}/ai/v1/chat/completions")
        return url, CF_API_TOKEN, CF_MODEL, CF_FALLBACKS
    return OPENROUTER_URL, OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_FALLBACKS

# --- Chunking (characters; ~4 chars per token) ---
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# --- Retrieval ---
TOP_K = 5

# --- Reranking (Phase 2) ---
# Cross-encoder reorders a wider candidate set. ms-marco-MiniLM is ~90MB and
# CPU-friendly. Set env RERANK=0 to disable (used for before/after eval).
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 20
RERANK_ENABLED = os.getenv("RERANK", "1") == "1"