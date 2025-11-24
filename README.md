AI Stock Analyst — RAG-Powered Financial Chatbot

A Retrieval-Augmented Generation (RAG) system that answers financial questions based on annual and quarterly reports of Indian public companies. The system uses FAISS for vector search, OpenAI models for embeddings and LLM reasoning, and SerpAPI for live web search fallback.

⸻

Overview

AI Stock Analyst is an end-to-end intelligent financial analysis assistant.
It retrieves relevant content directly from real company reports and generates grounded, explainable answers. When information is not available locally, it automatically performs a live web search.

This project demonstrates practical application of RAG, document intelligence, and LLM reasoning for financial analysis.

⸻

Use Case Objective
	•	Build an AI chatbot capable of answering questions about company performance, financial metrics, and insights.
	•	Use Retrieval-Augmented Generation to ground responses in real, verifiable documents.
	•	Provide fallback capability through live web search when local documents are insufficient.
	•	Deploy a user-friendly web interface for interactive financial exploration.

⸻

How the Problem Was Approached
	1.	Selected the use case: an “AI Stock Market Analyst” for Indian companies.
	2.	Collected annual and quarterly reports (PDFs) for companies such as Reliance, TCS, HDFC Bank, SBI, Infosys, and more.
	3.	Extracted, cleaned, and chunked text from PDFs.
	4.	Generated embeddings using OpenAI’s embeddings API.
	5.	Built a FAISS vector index for efficient semantic retrieval.
	6.	Designed a RAG flow combining retrieval and LLM-based answer generation.
	7.	Integrated SerpAPI for web search fallback.
	8.	Built and deployed the frontend using Streamlit.

⸻

Solution Overview
	•	End-to-end Retrieval-Augmented Generation pipeline.
	•	Local PDF document understanding combined with external web intelligence.
	•	FAISS vector search for fast and accurate chunk retrieval.
	•	OpenAI GPT models for summarization, insight generation, and question answering.
	•	Streamlit-based UI supporting document upload, provenance display, retrieval tuning, and customizable response mode.
	•	Automatic fallback to SerpAPI for questions not covered in local reports.

⸻

Features Implemented
	•	PDF ingestion pipeline: extraction, chunking, metadata tagging.
	•	FAISS vector search index for semantic retrieval.
	•	RAG pipeline (retrieve → generate → cite sources).
	•	SerpAPI integration for real-time web search when no local context is available.
	•	Streamlit UI with:
	•	Retrieval k-value slider
	•	Concise/Detailed response modes
	•	Provenance viewer (source + page number)
	•	PDF upload support
	•	Company context filters
	•	Deployment-ready architecture with Git LFS support for large FAISS files.

⸻

Challenges Faced
	•	Large PDF sizes (200+ pages) requiring optimized chunking and memory management.
	•	High embedding count, requiring batching and API quota management.
	•	FAISS index often exceeded 100 MB, requiring Git LFS for repository storage.
	•	Ensuring FAISS and OpenAI SDK compatibility during Streamlit Cloud deployment.
	•	Handling inconsistent web search responses (initially from DuckDuckGo), leading to the switch to SerpAPI.
	•	Ensuring the fallback system behaves predictably when both local retrieval and web search return no results.
	•	Managing secrets securely using Streamlit TOML-based secrets.


Project Structure
  .
├── app.py                     # Streamlit application
├── config/
│   └── config.py              # Secrets and configuration loader
├── models/
│   ├── embeddings.py          # Embedding utilities using OpenAI API
│   └── llm.py                 # LLM interface for answer generation
├── utils/
│   ├── rag_utils.py           # PDF ingestion, chunking, FAISS indexing
│   └── websearch.py           # SerpAPI-based fallback search module
├── data/
│   ├── raw/                   # PDF annual/quarterly reports
│   ├── index/                 # faiss.index and metadata.json
│   └── fallback/              # fallback PDF/image
├── requirements.txt           # Dependency list
└── runtime.txt                # Python version pin

Deployment

Streamlit Cloud:

https://ai-stock-analyst-rag-chatbot.streamlit.app/

The app is production-ready and successfully deployable using Streamlit Cloud with Git LFS.

⸻

Future Enhancements
	•	Comparative analysis across multiple companies.
	•	Financial ratio extraction and visualization.
	•	Real-time stock market API integration.
	•	Automatic quarterly/annual updates via web scrapers.
	•	Analyst-style multi-step reasoning chains.
	•	Exportable PDF financial reports.

⸻

Acknowledgements
	•	OpenAI — embeddings and LLM APIs
	•	FAISS — high-performance vector search
	•	SerpAPI — reliable web search
	•	Streamlit — deployment platform
	•	PyPDF2 and supporting libraries used in the ingestion pipeline
