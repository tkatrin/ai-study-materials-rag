import importlib

from rag_service import config


def test_invalid_numeric_env_uses_default_and_records_error(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "large")

    reloaded = importlib.reload(config)

    assert reloaded.CHUNK_SIZE == 900
    assert "RAG_CHUNK_SIZE must be an integer" in reloaded.CONFIG_ERRORS[0]
    monkeypatch.delenv("RAG_CHUNK_SIZE")
    importlib.reload(config)


def test_valid_env_overrides_defaults(monkeypatch):
    monkeypatch.setenv("RAG_MIN_SCORE", "0.4")

    reloaded = importlib.reload(config)

    assert reloaded.MIN_SCORE == 0.4
    monkeypatch.delenv("RAG_MIN_SCORE")
    importlib.reload(config)


def test_out_of_range_env_uses_default_and_records_error(monkeypatch):
    monkeypatch.setenv("RAG_MIN_SCORE", "7")
    monkeypatch.setenv("RAG_TOP_K", "-5")
    monkeypatch.setenv("OLLAMA_TIMEOUT", "0")

    reloaded = importlib.reload(config)

    assert reloaded.MIN_SCORE == 0.25
    assert reloaded.TOP_K == 4
    assert reloaded.OLLAMA_TIMEOUT == 120
    assert len(reloaded.CONFIG_ERRORS) == 3
    monkeypatch.delenv("RAG_MIN_SCORE")
    monkeypatch.delenv("RAG_TOP_K")
    monkeypatch.delenv("OLLAMA_TIMEOUT")
    importlib.reload(config)
