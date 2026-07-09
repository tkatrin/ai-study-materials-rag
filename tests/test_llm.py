import io
from urllib import error

import pytest

from rag_service.llm import OllamaGenerator, make_generator


def test_ollama_generator_reports_invalid_json(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return b"not-json"

    monkeypatch.setattr("rag_service.llm.request.urlopen", lambda *args, **kwargs: Response())

    with pytest.raises(RuntimeError, match="invalid JSON"):
        OllamaGenerator()("prompt")


def test_ollama_generator_reports_http_error(monkeypatch):
    def raise_http_error(*args, **kwargs):
        raise error.HTTPError(
            url="http://localhost:11434/api/generate",
            code=404,
            msg="not found",
            hdrs=None,
            fp=io.BytesIO(b"missing model"),
        )

    monkeypatch.setattr("rag_service.llm.request.urlopen", raise_http_error)

    with pytest.raises(RuntimeError, match="HTTP 404"):
        OllamaGenerator()("prompt")


def test_make_generator_passes_timeout():
    generator = make_generator("Ollama", "llama3.1", "http://localhost:11434", ollama_timeout=240)

    assert isinstance(generator, OllamaGenerator)
    assert generator.timeout == 240
