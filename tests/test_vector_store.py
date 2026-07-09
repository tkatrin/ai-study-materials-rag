import pytest

faiss = pytest.importorskip("faiss")
np = pytest.importorskip("numpy")

from rag_service.models import Chunk
from rag_service.vector_store import FaissVectorStore


def test_empty_index_returns_no_results():
    store = FaissVectorStore(dimension=3)

    assert store.search(np.array([1.0, 0.0, 0.0], dtype="float32")) == []


def test_vector_store_search_and_persistence(tmp_path):
    store = FaissVectorStore(dimension=3)
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

    store.save(tmp_path)
    loaded = FaissVectorStore.load(tmp_path)

    assert loaded.chunks[0].metadata["source"] == "ml.md"
    assert loaded.search(np.array([0.0, 1.0, 0.0], dtype="float32"), top_k=1)[0][0].text == "базы данных"
