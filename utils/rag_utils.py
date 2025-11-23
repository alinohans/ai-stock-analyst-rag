# utils/rag_utils.py


import os
import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import faiss
from PyPDF2 import PdfReader

from config.config import settings
from models.embeddings import embed_texts_as_numpy, embed_text  # your local functions

# Config defaults (tweak as needed)
CHUNK_SIZE_WORDS = int(os.getenv("CHUNK_SIZE_WORDS", "1200"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "150"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))

DATA_ROOT = Path("data/raw")
INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"

# Demo fallback file (local path present in your environment)
DEMO_LOCAL_PDF = "/mnt/data/NeoStats AI Engineer Case Study.pdf"


# -------------------------
# PDF -> pages -> text
# -------------------------
def pdf_to_pages(path: Path) -> List[Dict]:
    """
    Extract text per page from a PDF.
    Returns list of {"page": int, "text": str}
    """
    pages = []
    try:
        reader = PdfReader(str(path))
        for i, p in enumerate(reader.pages):
            try:
                text = p.extract_text() or ""
            except Exception:
                text = ""
            pages.append({"page": i + 1, "text": text})
    except Exception as e:
        print(f"[pdf_to_pages] Failed to read {path}: {e}")
    return pages


# -------------------------
# chunking
# -------------------------
def chunk_text(text: str, chunk_size_words: int = CHUNK_SIZE_WORDS, overlap_words: int = CHUNK_OVERLAP_WORDS) -> List[Dict]:
    """
    Split text into chunks with overlap (word-based).
    Returns list of {"text": str}
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    N = len(words)
    while i < N:
        end = i + chunk_size_words
        chunk_words = words[i:end]
        chunk_text_str = " ".join(chunk_words).strip()
        if chunk_text_str:
            chunks.append({"text": chunk_text_str})
        i = end - overlap_words
        if i <= 0:
            i = end
    return chunks


# -------------------------
# Company / report inference helper (basic)
# -------------------------
def infer_company_and_report_type(pdf_path: Path) -> Tuple[str, str]:
    """
    Infer company and report type from file path.
    ex: data/raw/reliance/annual/Reliance_AR_FY25.pdf -> ("reliance", "annual")
    """
    parts = pdf_path.parts
    # naive extraction: look for "data/raw/<company>/<report_type>/file.pdf"
    try:
        idx = parts.index("raw")
        company = parts[idx + 1]
        report_type = parts[idx + 2] if len(parts) > idx + 2 else "unknown"
        return company, report_type
    except ValueError:
        return ("misc", "unknown")
    except Exception:
        return ("misc", "unknown")


# -------------------------
# Build FAISS index & metadata
# -------------------------
def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index (cosine similarity via normalized inner product).
    """
    d = vectors.shape[1]
    # Normalize embeddings to use inner product == cosine
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(d)
    index.add(vectors)
    return index


def save_index_and_meta(index: faiss.Index, metadata: List[Dict]):
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False)


def load_index_and_meta() -> Tuple[Optional[faiss.Index], List[Dict]]:
    if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
        return None, []
    idx = faiss.read_index(str(FAISS_INDEX_PATH))
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception:
        meta = []
    return idx, meta


# -------------------------
# Ingest pipeline
# -------------------------
def ingest_file(path: Path) -> List[Dict]:
    """
    Ingest a single PDF and return metadata entries for its chunks
    (but does NOT persist index; index built in ingest_all).
    """
    company, report_type = infer_company_and_report_type(path)
    pages = pdf_to_pages(path)
    out_meta = []
    chunk_counter = 0

    for p in pages:
        page_num = p["page"]
        page_text = p["text"]
        if not page_text or len(page_text.strip()) < 10:
            continue
        chunks = chunk_text(page_text)
        for c in chunks:
            chunk_id = f"{path.name}::p{page_num}::{chunk_counter}"
            out_meta.append({
                "chunk_id": chunk_id,
                "company": company,
                "report_type": report_type,
                "fiscal_year": None,
                "source": str(path.name),
                "page": page_num,
                "text": c["text"],
                "text_preview": c["text"][:800]
            })
            chunk_counter += 1
    return out_meta


def ingest_all(data_root: Path = DATA_ROOT, index_dir: Path = INDEX_DIR, rebuild: bool = False):
    """
    Walk data_root for PDFs, create embeddings and build faiss index + metadata.
    If rebuild is True, old index/metadata will be overwritten.
    """
    # find pdf files
    pdfs = list(data_root.rglob("*.pdf"))
    if not pdfs:
        # if no PDFs found, try demo file
        if Path(DEMO_LOCAL_PDF).exists():
            pdfs = [Path(DEMO_LOCAL_PDF)]
        else:
            print("[ingest_all] No pdf files found in", data_root)
            return

    print(f"[ingest_all] Found {len(pdfs)} files. Processing...")
    all_meta = []
    all_texts = []

    for p in pdfs:
        print(f"[ingest_all] Processing: {p}  (company={infer_company_and_report_type(p)[0]}, report_type={infer_company_and_report_type(p)[1]})")
        metas = ingest_file(p)
        print(f"[ingest_all] -> extracted {len(metas)} chunks")
        for m in metas:
            all_meta.append(m)
            all_texts.append(m["text"])

    if not all_texts:
        print("[ingest_all] No text chunks extracted. Aborting index build.")
        return

    # Create embeddings in batches
    vectors = []
    n = len(all_texts)
    for i in range(0, n, EMBED_BATCH_SIZE):
        batch_texts = all_texts[i:i + EMBED_BATCH_SIZE]
        try:
            arr = embed_texts_as_numpy(batch_texts)  # expected shape (batch, dim)
        except Exception as e:
            # try single embeddings fallback
            print(f"[ingest_all] embed_texts_as_numpy failed for batch starting {i}: {e}. Falling back to single calls.")
            arr_list = []
            for t in batch_texts:
                vec = embed_text(t)
                arr_list.append(np.array(vec, dtype="float32"))
            arr = np.vstack(arr_list).astype("float32")

        if arr.dtype != np.float32:
            arr = arr.astype("float32")
        vectors.append(arr)

        print(f"[ingest_all] Embedded batch {i // EMBED_BATCH_SIZE + 1} / {math.ceil(n / EMBED_BATCH_SIZE)}")

    vectors = np.vstack(vectors)
    print(f"[ingest_all] Embeddings shape: {vectors.shape}")

    # Build faiss index
    index = build_faiss_index(vectors)
    save_index_and_meta(index, all_meta)
    print(f"[ingest_all] Ingest completed. Indexed {len(all_meta)} chunks.")
    print(f"[ingest_all] FAISS index saved to: {FAISS_INDEX_PATH}")
    print(f"[ingest_all] Metadata saved to: {METADATA_PATH}")


# -------------------------
# Retrieval
# -------------------------
def _embed_query(query: str) -> np.ndarray:
    """
    Use your models.embeddings.embed_text(s) to vectorize the query.
    Must produce a normalized float32 vector (1,d)
    """
    try:
        qv = embed_texts_as_numpy([query])
        if qv.dtype != np.float32:
            qv = qv.astype("float32")
        faiss.normalize_L2(qv)
        return qv
    except Exception:
        v = np.array(embed_text(query), dtype="float32")[None, :]
        faiss.normalize_L2(v)
        return v


def retrieve(query: str, k: int = 4, company_filter: Optional[str] = None) -> List[Dict]:
    """
    Retrieve top-k chunks for the query. Returns list of chunks with keys:
    { "score": float, "source": str, "page": int, "company": str, "text": str }
    """
    idx, meta = load_index_and_meta()
    if idx is None or not meta:
        print("[retrieve] Index or metadata missing. Run ingest_all() first.")
        return []

    qv = _embed_query(query)  # shape (1, d)
    D, I = idx.search(qv, k)  # D: scores, I: indices
    D = D[0].tolist()
    I = I[0].tolist()

    results = []
    for score, ix in zip(D, I):
        if ix < 0 or ix >= len(meta):
            continue
        m = meta[ix].copy()
        results.append({
            "score": float(score),
            "company": m.get("company"),
            "report_type": m.get("report_type"),
            "fiscal_year": m.get("fiscal_year"),
            "source": m.get("source"),
            "page": m.get("page"),
            "chunk_id": m.get("chunk_id"),
            "text": m.get("text"),
            "text_preview": m.get("text_preview"),
        })

    # optional company filtering
    if company_filter:
        results = [r for r in results if r.get("company","").lower() == company_filter.lower() or company_filter.lower() in str(r.get("source","")).lower()]

    return results


# -------------------------
# CLI entrypoint
# -------------------------
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild", action="store_true", help="Rebuild index from data/raw (default)")
    p.add_argument("--file", type=str, help="Ingest a single PDF file path")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.file:
        fp = Path(args.file)
        if not fp.exists():
            print(f"[main] file not found: {fp}")
        else:
            meta = ingest_file(fp)
            if meta:
                # build index for this file only
                texts = [m["text"] for m in meta]
                arr = embed_texts_as_numpy(texts)
                if arr.dtype != np.float32:
                    arr = arr.astype("float32")
                faiss.normalize_L2(arr)
                idx = faiss.IndexFlatIP(arr.shape[1])
                idx.add(arr)
                save_index_and_meta(idx, meta)
                print(f"[main] Ingested single file and saved index/metadata for {fp.name}")
    elif args.rebuild:
        ingest_all(DATA_ROOT, INDEX_DIR, rebuild=True)
    else:
        print("Usage: python -m utils.rag_utils --rebuild OR --file <pdf_path>")