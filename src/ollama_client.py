import os
import requests

def _base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

def ollama_embed(text: str, model: str) -> list[float]:
    """
    Uses Ollama embeddings endpoint:
    POST /api/embeddings  { "model": "...", "prompt": "..." }
    Returns: { "embedding": [...] }
    """
    url = f"{_base_url()}/api/embeddings"
    payload = {"model": model, "prompt": text}
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    emb = data.get("embedding")
    if not emb:
        raise RuntimeError(f"Ollama embeddings returned no embedding. Response: {data}")
    return emb

def ollama_chat(prompt: str, model: str) -> str:
    """
    Uses Ollama generate endpoint:
    POST /api/generate { "model": "...", "prompt": "...", "stream": false }
    Returns: { "response": "..." }
    """
    url = f"{_base_url()}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()
