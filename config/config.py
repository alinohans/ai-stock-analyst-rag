import os
from dataclasses import dataclass
from dotenv import load_dotenv

# load local .env for dev
load_dotenv()

# Helper to read from Streamlit secrets if available
def _get_secret(key: str):
    # use st.secrets when available at runtime, but don't import streamlit at top-level
    try:
        import streamlit as st
        # st.secrets returns a dict-like object
        val = st.secrets.get(key)
        if val:
            return val
    except Exception:
        # streamlit not available or not running (local scripts)
        pass
    # fallback to environment variables
    return os.getenv(key, "")

@dataclass
class Settings:
    OPENAI_API_KEY: str = _get_secret("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    SERPAPI_KEY: str = _get_secret("SERPAPI_KEY") or os.getenv("SERPAPI_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    # other settings...
    S3_BUCKET: str = os.getenv("S3_BUCKET", "")

settings = Settings()
