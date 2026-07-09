import importlib

from rag_service import config


def test_invalid_numeric_env_uses_default_and_records_error(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "large")

    reloaded = importlib.reload(config)

    assert reloaded.CHUNK_SIZE == 900
    assert "RAG_CHUNK_SIZE must be an integer" in reloaded.CONFIG_ERRORS[0]


def test_valid_env_overrides_defaults(monkeypatch):
    monkeypatch.setenv("RAG_MIN_SCORE", "0.4")

    reloaded = importlib.reload(config)

    assert reloaded.MIN_SCORE == 0.4
