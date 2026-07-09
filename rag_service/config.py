import os
from pathlib import Path


class ConfigError(ValueError):
    pass


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CONFIG_ERRORS = []


def read_str(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    return value or default


def read_choice(name: str, default: str, choices: set) -> str:
    value = read_str(name, default)
    if value in choices:
        return value
    CONFIG_ERRORS.append(f"{name} must be one of {sorted(choices)}, got '{value}'")
    return default


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


def require_range(name: str, value, default, is_valid, description: str):
    if is_valid(value):
        return value
    CONFIG_ERRORS.append(f"{name} must be {description}, got '{value}'")
    return default


INDEX_DIR = Path(read_str("RAG_INDEX_DIR", "data/index/default"))
UPLOAD_DIR = Path(read_str("RAG_UPLOAD_DIR", "data/uploads"))
EMBEDDING_MODEL = read_str("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
CHUNK_SIZE = require_range(
    "RAG_CHUNK_SIZE",
    read_int_or_default("RAG_CHUNK_SIZE", 900),
    900,
    lambda value: value > 0,
    "greater than 0",
)
CHUNK_OVERLAP = require_range(
    "RAG_CHUNK_OVERLAP",
    read_int_or_default("RAG_CHUNK_OVERLAP", 160),
    min(160, CHUNK_SIZE - 1),
    lambda value: 0 <= value < CHUNK_SIZE,
    f"between 0 and RAG_CHUNK_SIZE - 1 ({CHUNK_SIZE - 1})",
)
TOP_K = require_range(
    "RAG_TOP_K",
    read_int_or_default("RAG_TOP_K", 4),
    4,
    lambda value: value > 0,
    "greater than 0",
)
MIN_SCORE = require_range(
    "RAG_MIN_SCORE",
    read_float_or_default("RAG_MIN_SCORE", 0.25),
    0.25,
    lambda value: 0 <= value <= 1,
    "between 0 and 1",
)
OLLAMA_URL = read_str("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = read_str("OLLAMA_MODEL", "llama3.1")
OLLAMA_TIMEOUT = require_range(
    "OLLAMA_TIMEOUT",
    read_int_or_default("OLLAMA_TIMEOUT", 120),
    120,
    lambda value: value > 0,
    "greater than 0",
)
RERANK_MODE = read_choice("RAG_RERANK_MODE", "None", {"None", "Keyword", "CrossEncoder"})
RERANK_CANDIDATES = require_range(
    "RAG_RERANK_CANDIDATES",
    read_int_or_default("RAG_RERANK_CANDIDATES", 12),
    12,
    lambda value: value > 0,
    "greater than 0",
)
RERANK_MODEL = read_str(
    "RAG_RERANK_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
