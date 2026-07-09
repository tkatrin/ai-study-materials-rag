import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("faiss")

from rag_service.pipeline import ask_question, build_index


class FakeEmbedder:
    model_name = "fake-embedder"
    dimension = 3

    def embed(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "градиент" in lowered or "спуск" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "база" in lowered or "ключ" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return np.array(vectors, dtype="float32")


def test_build_index_then_ask_question(tmp_path):
    path = tmp_path / "ml.md"
    path.write_text(
        "Градиентный спуск уменьшает функцию потерь.\n"
        "База данных хранит таблицы и ключи.",
        encoding="utf-8",
    )
    embedder = FakeEmbedder()

    store = build_index([path], embedder, chunk_size=70, chunk_overlap=10)
    answer, results = ask_question("Что делает градиентный спуск?", store, embedder, top_k=1)

    assert store.embedding_model == "fake-embedder"
    assert results[0][0].metadata["source"] == "ml.md"
    assert "Градиентный спуск" in answer.answer
