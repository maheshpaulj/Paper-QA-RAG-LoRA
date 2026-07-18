"""Chat transport -- the one place that knows how to call the LLM.

Cloudflare Workers AI and OpenRouter both expose an OpenAI-compatible
chat-completions endpoint, so a single code path covers both: only the base URL,
key and model name differ (see config.llm_endpoint). A message's content is a
list of parts -- text parts and image parts as base64 data URLs.

Free tiers rate-limit, so calls try each model in turn before backing off.
"""
import base64
import time

import requests

from config import llm_endpoint

RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}


def text_part(text):
    return {"type": "text", "text": text}


def image_part(data, mime="image/png"):
    b64 = base64.b64encode(data).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def _post(url, key, model, parts, timeout, max_tokens):
    """One attempt. Returns (text, None), or (None, reason) if worth retrying.
    Raises on failures that retrying won't fix."""
    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": parts}],
                # Set explicitly: provider defaults are small and undocumented,
                # and a silently truncated answer looks like a bad answer.
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
    except requests.RequestException as e:
        # dropped connections / timeouts are transient -- retry, don't crash the run
        return None, f"{model}: {type(e).__name__}"
    data = resp.json() if resp.content else {}

    # OpenRouter reports upstream failures in the body even on HTTP 200.
    # Cloudflare uses {"success": false, "errors": [...]} on its native paths.
    err = data.get("error") or {}
    if not err and data.get("errors"):
        err = {"message": str(data["errors"])}
    if resp.status_code in RETRY_STATUS or err.get("code") in RETRY_STATUS:
        return None, f"{model}: {err.get('message') or resp.status_code}"
    if err:
        raise RuntimeError(f"LLM error: {err}")
    resp.raise_for_status()
    return (data["choices"][0]["message"]["content"] or "").strip(), None


def chat(parts, model=None, tries=3, timeout=180, max_tokens=1024):
    url, key, default_model, fallbacks = llm_endpoint()
    if not key:
        raise RuntimeError("No API key set for the active LLM provider -- see .env.example")
    if isinstance(parts, str):
        parts = [text_part(parts)]

    primary = model or default_model
    models = [primary] + [m for m in fallbacks if m != primary]

    reasons = []
    for attempt in range(tries):
        for m in models:
            text, reason = _post(url, key, m, parts, timeout, max_tokens)
            if text is not None:
                return text
            reasons.append(reason)
        if attempt < tries - 1:
            time.sleep(min(60, 5 * 2 ** attempt))  # 5s, 10s
    # Don't call this "rate-limited" -- ConnectionError and a 429 are very
    # different problems and saying the wrong one sends you debugging the wrong
    # thing. Report what actually happened.
    raise RuntimeError("LLM request failed after retries: " + "; ".join(reasons[-3:]))
