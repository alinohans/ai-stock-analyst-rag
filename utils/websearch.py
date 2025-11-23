"""
SerpAPI-only web search helper for RAG.
"""

import os
import requests
from typing import List, Dict
from config.config import settings

# Local fallback file (your uploaded file)
LOCAL_FALLBACK = "/mnt/data/AD84528D-4374-444A-A711-AA7F09A74178.png"


# -------------------------------
# SerpAPI Search (Primary & Only)
# -------------------------------
def serpapi_search(query: str, max_results: int = 3) -> List[Dict]:
    api_key = settings.SERPAPI_KEY or os.getenv("SERPAPI_KEY", "")
    if not api_key:
        raise RuntimeError("SERPAPI_KEY not configured in .env")

    endpoint = "https://serpapi.com/search.json"
    params = {
        "q": query,
        "engine": "google",
        "api_key": api_key,
        "num": max_results,
        # optional: "google_domain": "google.co.in"
    }

    resp = requests.get(endpoint, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    results = []
    organic = data.get("organic_results") or []
    for item in organic[:max_results]:
        results.append({
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "url": item.get("link", ""),
        })

    return results


# -------------------------------
# Local Fallback (always 1 result)
# -------------------------------
def local_fallback(query: str) -> List[Dict]:
    if os.path.exists(LOCAL_FALLBACK):
        return [{
            "title": "Local Fallback Document",
            "snippet": f"SerpAPI returned no results for: '{query}'",
            "url": LOCAL_FALLBACK
        }]
    else:
        return [{
            "title": "No Results",
            "snippet": f"SerpAPI failed and fallback file not found for: {query}",
            "url": ""
        }]


# -------------------------------
# Public API used by app.py
# -------------------------------
def search(query: str, max_results: int = 3) -> List[Dict]:
    try:
        results = serpapi_search(query, max_results=max_results)
        if results:
            return results
    except Exception as e:
        print("SerpAPI error:", e)

    # fallback if any error or empty results
    return local_fallback(query)


# -------------------------------
# Test from CLI
# -------------------------------
if __name__ == "__main__":
    q = "Zomato profit margin FY25"
    print(search(q, max_results=3))