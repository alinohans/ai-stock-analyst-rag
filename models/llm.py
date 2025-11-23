# models/llm.py
"""
LLM wrapper specialized for Stock Market / Company Report analysis.

"""

import time
import os
from typing import List, Optional, Tuple, Dict, Any
from openai import OpenAI
from config.config import settings

# Demo/sample file path (local). Your environment / tests can use this as a quick demo input.
SAMPLE_DOC_LOCAL_PATH = "/mnt/data/NeoStats AI Engineer Case Study.pdf"

# Initialize client
_client = OpenAI(api_key=settings.OPENAI_API_KEY)

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "800"))


def _build_prompt_for_finance(
    query: str,
    context: Optional[str],
    mode: str,
    sources: Optional[List[dict]] = None,
    instructions: Optional[str] = None
) -> Tuple[str, str]:
    """
    Build a system & user prompt tailored for financial / company report analysis.

    - query: user's question
    - context: concatenated retrieved chunks (RAG)
    - mode: "concise" or "detailed"
    - sources: provenance metadata list
    - instructions: additional instructions for specific tasks (e.g., "output CSV")
    """
    system = (
        "You are a careful, conservative financial research assistant. "
        "You must base your answers strictly on the provided context (company annual reports, "
        "quarterly results, investor presentations, and concall transcripts). "
        "Avoid hallucinations. If the context does not contain enough information to answer, "
        "say 'Insufficient information in context' and indicate which documents or specific sections "
        "would likely contain the answer. When asked to extract numerical data (revenue, EBITDA, PAT, "
        "margins, debt, cash flow), return values with units and fiscal year/period if available. "
        "When asked to compare companies, provide a concise comparison table and highlight key differences."
    )

    # Build user message
    parts = [f"Question: {query}"]

    if context:
        parts.append("Context (extracted from company documents):")
        parts.append(context)
    else:
        parts.append("Context: (no RAG context provided)")

    if sources:
        # compact provenance list to help the model cite
        prov_lines = []
        for s in sources:
            name = s.get("source") or s.get("filename") or s.get("doc") or "unknown"
            page = s.get("page")
            if page:
                prov_lines.append(f"- {name} (page {page})")
            else:
                prov_lines.append(f"- {name}")
        parts.append("Provenance:\n" + "\n".join(prov_lines))

    # Mode-specific output instructions
    if mode == "concise":
        parts.append(
            "Output format: Provide a short answer. For factual questions give 1-4 bullet points. "
            "If numerical data is requested, return the numbers and their units. "
            "When possible, append a one-line citation to sources (e.g., [Source: file.pdf page 34])."
        )
    else:
        parts.append(
            "Output format: Provide a detailed explanation, list steps to verify the data, and include "
            "explicit citations to the provenance. If requested, include a short structured table or CSV "
            "for numeric data. Clearly separate analysis from facts."
        )

    if instructions:
        parts.append(f"Additional instructions: {instructions}")

    user = "\n\n".join(parts)
    return system, user


def _call_model(system: str, user: str, model: str = DEFAULT_MODEL, max_retries: int = 3, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """
    Call OpenAI chat completion endpoint with simple retries.
    """
    attempt = 0
    backoff = 1.0
    while attempt < max_retries:
        try:
            resp = _client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,  # deterministic for finance
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            attempt += 1
            if attempt >= max_retries:
                raise RuntimeError(f"LLM call failed after {attempt} attempts: {e}")
            time.sleep(backoff)
            backoff *= 2.0


def generate_response(
    query: str,
    context: Optional[str] = None,
    mode: str = "concise",
    sources: Optional[List[dict]] = None,
    model: Optional[str] = None,
    instructions: Optional[str] = None,
) -> Tuple[str, dict]:
    """
    General-purpose response generator for finance questions.

    Returns:
      - assistant_text: the model's answer (string)
      - debug: dict with prompt snippets, model used, mode, and provenance
    """
    if mode not in ("concise", "detailed"):
        mode = "concise"

    system, user = _build_prompt_for_finance(query=query, context=context, mode=mode, sources=sources, instructions=instructions)
    chosen_model = model or DEFAULT_MODEL

    assistant_text = _call_model(system, user, model=chosen_model)

    debug = {
        "model": chosen_model,
        "mode": mode,
        "system_prompt_snippet": system[:400],
        "user_prompt_snippet": user[:2000],
        "sources": sources or [],
    }
    return assistant_text, debug


def generate_from_chunks(query: str, chunks: List[dict], mode: str = "concise") -> Tuple[str, dict]:
    """
    Use retrieved chunks (from FAISS) as RAG context and call the LLM.
    Each chunk dict is expected to have 'text' or 'text_preview', 'source', 'page', etc.
    """
    texts = []
    sources = []
    total_chars = 0
    for c in chunks:
        t = c.get("text") or c.get("text_preview") or ""
        if not t:
            continue
        texts.append(t)
        total_chars += len(t)
        sources.append({"source": c.get("source"), "page": c.get("page")})
        # Prevent extremely long context; stop at a conservative char limit
        if total_chars > 30000:
            break

    context = "\n\n---\n\n".join(texts)
    return generate_response(query=query, context=context, mode=mode, sources=sources)


# -------------------------
# Financial extraction helpers
# -------------------------
def generate_financial_summary(
    company_name: str,
    chunks: List[dict],
    fiscal_periods: Optional[List[str]] = None,
    required_metrics: Optional[List[str]] = None,
    mode: str = "concise"
) -> Tuple[Dict[str, Any], dict]:
    """
    Extract structured financial metrics from the provided chunks.
    - company_name: human-readable company name for prompts
    - chunks: retrieved chunks from that company's documents
    - fiscal_periods: e.g., ["FY2023", "FY2022"] or ["Q1 2025"]
    - required_metrics: list like ["Revenue", "EBITDA", "PAT", "Gross Margin", "Net Debt"]
    Returns:
      - summary: dict mapping metric -> {period: value, unit: str, provenance: [...]}
      - debug: debug info (prompt, raw_text_answer)
    """
    if required_metrics is None:
        required_metrics = ["Revenue", "EBITDA", "PAT", "Operating Margin", "Net Debt", "Cash Flow from Operations"]

    # Prepare context
    texts = []
    sources = []
    total_chars = 0
    for c in chunks:
        t = c.get("text") or c.get("text_preview") or ""
        if not t:
            continue
        texts.append(t)
        total_chars += len(t)
        sources.append({"source": c.get("source"), "page": c.get("page")})
        if total_chars > 35000:
            break
    context = "\n\n---\n\n".join(texts)

    # Instruction to output JSON with provenance
    metrics_list = ", ".join(required_metrics)
    periods_text = ""
    if fiscal_periods:
        periods_text = f"for these periods: {', '.join(fiscal_periods)}."

    instruction = (
        "Extract the following financial metrics: " + metrics_list + ". "
        "Return the result as a JSON object with keys as metrics and values as objects mapping fiscal period -> {value, unit, provenance_list}. "
        "Provenance_list should include source filename and page. If a metric is not found in the context, return null for that period. "
        "Do not invent numbers. Keep numeric values as plaintext numbers and include currency units. "
        "Example response:\n"
        '{ "Revenue": {"FY2023": {"value": 123456, "unit": "INR crore", "provenance": ["file.pdf page 34"]}, "FY2022": null}, ... }\n'
    )

    q = f"Extract financial metrics for {company_name} {periods_text} Use only the context. Provide JSON output as described."
    user_instructions = instruction

    assistant_text, debug = generate_response(query=q, context=context, mode=mode, sources=sources, instructions=user_instructions)
    # Try to parse JSON from assistant_text (best-effort)
    import json
    parsed = None
    try:
        # The assistant may include text around JSON; attempt to locate the first '{' and parse
        start = assistant_text.find("{")
        if start != -1:
            candidate = assistant_text[start:]
            parsed = json.loads(candidate)
    except Exception:
        parsed = None

    summary = parsed if isinstance(parsed, dict) else {"raw": assistant_text}
    debug_out = {"raw_text": assistant_text, **debug}
    return summary, debug_out


def compare_companies(
    company_a: str,
    chunks_a: List[dict],
    company_b: str,
    chunks_b: List[dict],
    metrics: Optional[List[str]] = None,
    mode: str = "concise"
) -> Tuple[str, dict]:
    """
    Compare two companies across specified metrics and produce a structured comparison.
    Returns (textual_comparison, debug)
    """
    if metrics is None:
        metrics = ["Revenue", "EBITDA", "PAT", "Operating Margin", "Net Debt", "ROE"]

    # Build compact contexts
    def combine_chunks(chunks):
        texts = []
        total = 0
        for c in chunks:
            t = c.get("text") or c.get("text_preview") or ""
            if not t:
                continue
            texts.append(t)
            total += len(t)
            if total > 18000:
                break
        return "\n\n---\n\n".join(texts)

    context_a = combine_chunks(chunks_a)
    context_b = combine_chunks(chunks_b)

    instruction = (
        f"Compare {company_a} and {company_b} on the following metrics: {', '.join(metrics)}. "
        "Provide a concise comparison table and 3-4 bullet insights highlighting the most important differences. "
        "Cite provenance inline using [filename page N]. If a metric can't be found, mark it as 'N/A'."
    )

    query = f"Compare {company_a} vs {company_b} using the provided contexts."
    # pack contexts as part of the combined context for the model
    combined_context = f"=== Context for {company_a} ===\n{context_a}\n\n=== Context for {company_b} ===\n{context_b}"

    return generate_response(query=query, context=combined_context, mode=mode, sources=None, instructions=instruction)


# Utility: simple wrapper for free-text Q&A when no chunks available
def answer_without_rag(query: str, mode: str = "concise") -> Tuple[str, dict]:
    """
    Use the LLM to answer general finance questions without RAG context.
    The model should be conservative and say if it does not "know" the specific company facts.
    """
    system = (
        "You are a helpful financial research assistant. If you don't know the exact data for a company, "
        "say 'Insufficient information in context' and recommend authoritative sources (e.g., annual report, investor presentation)."
    )
    user = f"Question: {query}\n\nResponse format: {'short bullet points' if mode=='concise' else 'detailed explanation'}."
    assistant_text = _call_model(system, user)
    debug = {"model": DEFAULT_MODEL, "mode": mode}
    return assistant_text, debug


