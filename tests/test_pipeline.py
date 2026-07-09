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


class PreferDatabaseReranker:
    def rerank(self, question, results, top_k):
        sorted_results = sorted(
            results,
            key=lambda item: "База данных" in item[0].text,
            reverse=True,
        )
        return sorted_results[:top_k]


class CountingStore:
    def __init__(self):
        self.requested_top_k = None

    def search(self, query_embedding, top_k=4, min_score=None):
        self.requested_top_k = top_k
        return []


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


def test_ask_question_can_rerank_candidates(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text(
        "Градиентный спуск уменьшает функцию потерь.\n\n"
        "База данных хранит таблицы и ключи.",
        encoding="utf-8",
    )
    embedder = FakeEmbedder()
    store = build_index([path], embedder, chunk_size=55, chunk_overlap=5)

    answer, results = ask_question(
        "Что делает градиентный спуск?",
        store,
        embedder,
        top_k=1,
        candidate_k=2,
        reranker=PreferDatabaseReranker(),
    )

    assert "База данных" in results[0][0].text
    assert "База данных" in answer.answer


def test_candidate_k_is_ignored_without_reranker():
    store = CountingStore()
    embedder = FakeEmbedder()

    ask_question("Что известно?", store, embedder, top_k=3, candidate_k=10)

    assert store.requested_top_k == 3
