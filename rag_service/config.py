import os
from pathlib import Path


class ConfigError(ValueError):
    pass


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CONFIG_ERRORS = []


def read_str(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    return value or default


def read_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got '{value}'") from exc


def read_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got '{value}'") from exc


def read_int_or_default(name: str, default: int) -> int:
    try:
        return read_int(name, default)
    except ConfigError as error:
        CONFIG_ERRORS.append(str(error))
        return default


def read_float_or_default(name: str, default: float) -> float:
    try:
        return read_float(name, default)
    except ConfigError as error:
        CONFIG_ERRORS.append(str(error))
        return default


INDEX_DIR = Path(read_str("RAG_INDEX_DIR", "data/index/default"))
UPLOAD_DIR = Path(read_str("RAG_UPLOAD_DIR", "data/uploads"))
EMBEDDING_MODEL = read_str("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
CHUNK_SIZE = read_int_or_default("RAG_CHUNK_SIZE", 900)
CHUNK_OVERLAP = read_int_or_default("RAG_CHUNK_OVERLAP", 160)
TOP_K = read_int_or_default("RAG_TOP_K", 4)
MIN_SCORE = read_float_or_default("RAG_MIN_SCORE", 0.25)
OLLAMA_URL = read_str("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = read_str("OLLAMA_MODEL", "llama3.1")
