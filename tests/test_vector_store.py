import pytest

faiss = pytest.importorskip("faiss")
np = pytest.importorskip("numpy")

from rag_service.models import Chunk
from rag_service.vector_store import FaissVectorStore


def test_empty_index_returns_no_results():
    store = FaissVectorStore(dimension=3)

    assert store.search(np.array([1.0, 0.0, 0.0], dtype="float32")) == []


def test_vector_store_search_threshold_and_persistence(tmp_path):
    store = FaissVectorStore(dimension=3, embedding_model="test-embedder")
    chunks = [
        Chunk(text="машинное обучение", metadata={"source": "ml.md", "chunk_id": 1}),
        Chunk(text="базы данных", metadata={"source": "db.txt", "chunk_id": 1}),
    ]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype="float32",
    )

    store.add(chunks, embeddings)
    results = store.search(np.array([0.9, 0.1, 0.0], dtype="float32"), top_k=1)

    assert results[0][0].text == "машинное обучение"
    assert store.search(np.array([0.1, 0.1, 0.0], dtype="float32"), top_k=2, min_score=0.5) == []

    store.save(tmp_path)
    loaded = FaissVectorStore.load(tmp_path, expected_embedding_model="test-embedder")

    assert loaded.embedding_model == "test-embedder"
    assert loaded.chunks[0].metadata["source"] == "ml.md"
    assert loaded.search(np.array([0.0, 1.0, 0.0], dtype="float32"), top_k=1)[0][0].text == "базы данных"


def test_vector_store_rejects_embedding_model_mismatch(tmp_path):
    store = FaissVectorStore(dimension=2, embedding_model="model-a")
    store.add(
        [Chunk(text="пример", metadata={"source": "a.txt", "chunk_id": 1})],
        np.array([[1.0, 0.0]], dtype="float32"),
    )
    store.save(tmp_path)

    with pytest.raises(ValueError, match="model-a"):
        FaissVectorStore.load(tmp_path, expected_embedding_model="model-b")
