import os
from pathlib import Path


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", "data/index/default"))
UPLOAD_DIR = Path(os.getenv("RAG_UPLOAD_DIR", "data/uploads"))
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "160"))
TOP_K = int(os.getenv("RAG_TOP_K", "4"))
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.15"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
