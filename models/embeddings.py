# models/embeddings.py
"""
Embedding utility for the project.

"""

import os
import time
import json
from typing import List, Iterable, Optional, Union, Dict
import numpy as np
from pathlib import Path
from openai import OpenAI
from config.config import settings

# Client
_client = OpenAI(api_key=settings.OPENAI_API_KEY)

# Model & defaults
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))
MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "3"))
CACHE_DIR = Path(os.getenv("EMBED_CACHE_DIR", "data/emb_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _cache_key(text: str) -> str:
    """Return a simple filename-safe key for caching small texts."""
    # simple safe filename (hashing avoids extremely long filenames)
    import hashlib
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return h

def _load_from_cache(key: str) -> Optional[List[float]]:
    p = CACHE_DIR / f"{key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def _save_to_cache(key: str, embedding: List[float]):
    p = CACHE_DIR / f"{key}.json"
    try:
        p.write_text(json.dumps(embedding), encoding="utf-8")
    except Exception:
        pass

def _call_openai_embeddings(inputs: List[str]) -> List[List[float]]:
    """
    Call OpenAI embeddings API in batch with retry/backoff.
    Returns list of embedding vectors (list of floats) in the same order as inputs.
    """
    attempt = 0
    backoff = 1.0
    while attempt < MAX_RETRIES:
        try:
            resp = _client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=inputs
            )
            # Resp may contain data list with embeddings in same order
            out = [item.embedding for item in resp.data]
            return out
        except Exception as e:
            attempt += 1
            if attempt >= MAX_RETRIES:
                raise RuntimeError(f"Embedding API failed after {attempt} attempts: {e}")
            time.sleep(backoff)
            backoff *= 2.0
    raise RuntimeError("Unreachable")

def embed_texts(texts: Iterable[str], use_cache: bool = True) -> List[List[float]]:
    """
    Embed an iterable of texts. Returns list of embeddings corresponding to texts.
    - Batches requests for efficiency.
    - Uses a simple cache keyed by content hash.
    """
    texts = list(texts)
    embeddings: List[List[float]] = [None] * len(texts)
    # first, try cache
    if use_cache:
        for i, t in enumerate(texts):
            if not t or not t.strip():
                embeddings[i] = []  # empty input -> empty embedding placeholder
                continue
            key = _cache_key(t)
            cached = _load_from_cache(key)
            if cached is not None:
                embeddings[i] = cached

    # find indices to compute
    to_compute = []
    idx_map = []
    for i, emb in enumerate(embeddings):
        if emb is None:
            to_compute.append(texts[i])
            idx_map.append(i)

    # batch compute
    for i in range(0, len(to_compute), BATCH_SIZE):
        batch_texts = to_compute[i:i+BATCH_SIZE]
        try:
            results = _call_openai_embeddings(batch_texts)
        except Exception as e:
            # fallback: fill with empty vectors to keep ordering stable
            results = [[] for _ in batch_texts]
            print("Embedding error for batch:", str(e))

        # assign results back into embeddings list and cache
        for j, emb in enumerate(results):
            idx = idx_map[i + j]
            embeddings[idx] = emb
            if use_cache and emb:
                try:
                    key = _cache_key(texts[idx])
                    _save_to_cache(key, emb)
                except Exception:
                    pass

    # Final normalization/validation: ensure all entries are lists
    final = []
    for emb in embeddings:
        if not emb:
            final.append([])
        else:
            final.append(list(emb))
    return final

def embed_text(text: str, use_cache: bool = True) -> List[float]:
    """
    Convenience wrapper to embed a single text and return a list[float].
    """
    out = embed_texts([text], use_cache=use_cache)
    return out[0] if out else []

def embed_text_as_numpy(text: str, use_cache: bool = True) -> np.ndarray:
    """
    Return embedding as numpy array (float32) suitable for FAISS.
    """
    vec = embed_text(text, use_cache=use_cache)
    if not vec:
        return np.array([], dtype="float32")
    return np.array(vec, dtype="float32")

def embed_texts_as_numpy(texts: Iterable[str], use_cache: bool = True) -> np.ndarray:
    """
    Embed multiple texts and return a 2D numpy array (n x dim) float32 for FAISS indexing.
    """
    embs = embed_texts(texts, use_cache=use_cache)
    if not embs:
        return np.zeros((0, 0), dtype="float32")
    arr = np.vstack([np.array(e, dtype="float32") if e else np.zeros((len(embs[0]),), dtype="float32") for e in embs])
    return arr
