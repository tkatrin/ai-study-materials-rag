import json
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

from .models import Chunk


PathLike = Union[str, Path]
METADATA_FILENAME = "metadata.json"


class FaissVectorStore:
    def __init__(self, dimension: int, embedding_model: Optional[str] = None):
        self.dimension = dimension
        self.embedding_model = embedding_model
        self.index = _faiss().IndexFlatIP(dimension)
        self.chunks: List[Chunk] = []

    def add(self, chunks: Sequence[Chunk], embeddings: Any) -> None:
        if len(chunks) == 0:
            return
        vectors = _as_float32(embeddings)
        if vectors.shape != (len(chunks), self.dimension):
            raise ValueError(
                f"Expected embeddings shape {(len(chunks), self.dimension)}, got {vectors.shape}"
            )
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding: Any,
        top_k: int = 4,
        min_score: Optional[float] = None,
    ) -> List[Tuple[Chunk, float]]:
        if self.index.ntotal == 0:
            return []
        vector = _as_float32(query_embedding)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        scores, indices = self.index.search(vector, min(top_k, self.index.ntotal))
        results = []
        for score, index in zip(scores[0], indices[0]):
            if index >= 0 and (min_score is None or float(score) >= min_score):
                results.append((self.chunks[int(index)], float(score)))
        return results

    def save(self, directory: PathLike) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        _faiss().write_index(self.index, str(path / "index.faiss"))
        payload = [
            {"text": chunk.text, "metadata": chunk.metadata}
            for chunk in self.chunks
        ]
        (path / "chunks.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        metadata = {
            "dimension": self.dimension,
            "embedding_model": self.embedding_model,
            "chunk_count": len(self.chunks),
        }
        (path / METADATA_FILENAME).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        directory: PathLike,
        expected_embedding_model: Optional[str] = None,
    ) -> "FaissVectorStore":
        path = Path(directory)
        index_path = path / "index.faiss"
        chunks_path = path / "chunks.json"
        metadata_path = path / METADATA_FILENAME
        if not index_path.exists() or not chunks_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Saved index is incomplete. Expected index.faiss, chunks.json and {METADATA_FILENAME} in {path}"
            )

        index = _faiss().read_index(str(index_path))
        payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        embedding_model = metadata.get("embedding_model")
        if expected_embedding_model and embedding_model != expected_embedding_model:
            raise ValueError(
                "Saved index was built with embedding model "
                f"'{embedding_model}', but current model is '{expected_embedding_model}'."
            )

        store = cls(index.d, embedding_model=embedding_model)
        store.index = index
        store.chunks = [Chunk(text=item["text"], metadata=item["metadata"]) for item in payload]
        if store.index.ntotal != len(store.chunks):
            raise ValueError(
                f"Index/chunk mismatch: FAISS has {store.index.ntotal} vectors, "
                f"but chunks.json has {len(store.chunks)} chunks."
            )
        return store


def _as_float32(vectors: Any) -> Any:
    import numpy as np

    array = np.asarray(vectors, dtype="float32")
    return np.ascontiguousarray(array)


def _faiss() -> Any:
    import faiss

    return faiss
