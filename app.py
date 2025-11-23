import os
import sys
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import streamlit as st

# ensure project root is importable when running streamlit
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.config import settings
from utils.rag_utils import retrieve
from models.llm import generate_from_chunks
from models.embeddings import embed_texts_as_numpy, embed_text

# websearch import
try:
    from utils.websearch import search as web_search
    _HAS_WEBSEARCH = True
except Exception:
    web_search = None
    _HAS_WEBSEARCH = False

DATA_RAW = Path("data/raw")
SAMPLE_LOCAL_PATH = "data/fallback/fallback.pdf"  # uploaded demo fallback

st.set_page_config(page_title="AI Stock Analyst (RAG)", layout="wide")


@st.cache_data
def list_companies() -> List[str]:
    """Discover company folders under data/raw/"""
    if not DATA_RAW.exists():
        return []
    companies = [p.name for p in DATA_RAW.iterdir() if p.is_dir()]
    companies = sorted(companies)
    return companies


def call_rag_and_llm(query: str, company: Optional[str], k: int, mode: str) -> Tuple[Optional[str], List[Dict], bool, dict]:
    """
    Retrieve local chunks, optionally fallback to web search (DuckDuckGo/SerpAPI via utils.websearch),
    re-run LLM if needed. Returns (answer, used_chunks, used_web_fallback, debug).
    """
    raw_chunks = retrieve(query, k=k)

    normalized: List[Dict] = []
    for r in raw_chunks:
        text = r.get("text") or r.get("text_preview") or r.get("preview") or ""
        normalized.append({
            "text": text,
            "source": r.get("source"),
            "page": r.get("page"),
            "company": r.get("company", "")
        })

    if company:
        normalized = [
            c for c in normalized
            if (c.get("company","").lower() == company.lower()) or (company.lower() in str(c.get("source","")).lower())
        ]

    used_web = False
    debug_info = {"retrieved_local": len(normalized)}

    # If no local chunks found, try websearch
    if not normalized and _HAS_WEBSEARCH:
        web_results = web_search(query, max_results=3)
        normalized = []
        for r in web_results:
            t = f"{r.get('title','')}\n{r.get('snippet','')}\n{r.get('url','')}"
            normalized.append({"text": t, "source": r.get("url"), "page": None, "company": "web"})
        used_web = True
        debug_info["web_rows"] = len(normalized)

    if not normalized:
        debug_info["note"] = "no_chunks"
        return None, [], used_web, debug_info

    answer, debug = generate_from_chunks(query, normalized, mode=mode)

    empty_or_insufficient = False
    if not answer or (isinstance(answer, str) and "insufficient information" in answer.lower()):
        empty_or_insufficient = True

    if empty_or_insufficient and (not used_web) and _HAS_WEBSEARCH:
        web_results = web_search(query, max_results=4)
        web_chunks = []
        for r in web_results:
            t = f"{r.get('title','')}\n{r.get('snippet','')}\n{r.get('url','')}"
            web_chunks.append({"text": t, "source": r.get("url"), "page": None, "company": "web"})
        if web_chunks:
            used_web = True
            normalized = web_chunks + normalized
            answer, debug = generate_from_chunks(query, normalized, mode=mode)
            debug_info["web_retry"] = len(web_chunks)

    debug_info["llm_debug"] = debug if isinstance(debug, dict) else {}
    return answer, normalized, used_web, debug_info


# ---------- UI ----------
st.title("AI Stock Analyst — RAG-powered Chatbot")
st.markdown("Ask questions about company annual & quarterly reports. Responses are grounded in retrieved documents (provenance shown).")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    companies = list_companies()
    company = st.selectbox("Select company (or leave blank to search all)", [""] + companies, index=0)
    k = st.slider("Retrieval: top k chunks", min_value=1, max_value=12, value=4)
    mode = st.radio("Response mode", ("Concise", "Detailed"))
    show_provenance = st.checkbox("Show retrieved chunks (provenance)", value=True)

    st.markdown("---")
    st.markdown("Upload a PDF (optional)")
    uploaded_file = st.file_uploader("Upload Annual/Quarterly PDF", type=["pdf"], accept_multiple_files=False)
    if uploaded_file:
        save_dir = DATA_RAW / "misc"
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / uploaded_file.name
        with open(out_path, "wb") as fh:
            fh.write(uploaded_file.getbuffer())
        st.success(f"Saved to {out_path}. To include it in the FAISS index, run ingestion (`python -m utils.rag_utils --rebuild`) from your shell.")
        # Demo: extract first page text and compute embeddings (preview only)
        try:
            from PyPDF2 import PdfReader
            rdr = PdfReader(str(out_path))
            ptext = rdr.pages[0].extract_text() or ""
            demo_chunks = ptext.split()[:500]
            demo_text = " ".join(demo_chunks)
            arr = embed_texts_as_numpy([demo_text])
            st.write("Preview embedding shape:", arr.shape)
        except Exception as e:
            st.warning("Could not run embedding preview: " + str(e))

# Query box
query = st.text_input("Ask a question about the reports (e.g. 'Summarize Reliance FY25 performance')")

if st.button("Run Query") and query.strip():
    with st.spinner("Retrieving relevant chunks and calling the LLM..."):
        answer, used_chunks, used_web_fallback, debug = call_rag_and_llm(
            query,
            company=company if company else None,
            k=k,
            mode=mode.lower()
        )

    if answer is None:
        st.error("No local results and no web fallback available. Try a simpler query or re-index documents.")
    else:
        if used_web_fallback:
            st.info("Used live web search fallback (SerpAPI).")

        st.subheader("Answer")
        if mode == "Concise":
            st.write(answer)
        else:
            st.markdown(answer)

        # provenance panel
        if show_provenance:
            st.subheader("Provenance / Retrieved Chunks")
            for i, c in enumerate(used_chunks):
                src = c.get("source") or "unknown"
                page = c.get("page")
                st.markdown(f"**Chunk {i+1}** — Source: `{src}`  Page: {page}")
                st.write(c.get("text")[:800])
                st.markdown("---")

    # optional: show debug collapsed
    with st.expander("Debug info"):
        st.write(debug)

# End of app.py