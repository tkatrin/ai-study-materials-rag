import json
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union

import numpy as np

from .models import Chunk


PathLike = Union[str, Path]


class FaissVectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = _faiss().IndexFlatIP(dimension)
        self.chunks: List[Chunk] = []

    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) == 0:
            return
        vectors = _as_float32(embeddings)
        if vectors.shape != (len(chunks), self.dimension):
            raise ValueError(
                f"Expected embeddings shape {(len(chunks), self.dimension)}, got {vectors.shape}"
            )
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        if self.index.ntotal == 0:
            return []
        vector = _as_float32(query_embedding)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        scores, indices = self.index.search(vector, min(top_k, self.index.ntotal))
        results = []
        for score, index in zip(scores[0], indices[0]):
            if index >= 0:
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

    @classmethod
    def load(cls, directory: PathLike) -> "FaissVectorStore":
        path = Path(directory)
        index = _faiss().read_index(str(path / "index.faiss"))
        payload = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
        store = cls(index.d)
        store.index = index
        store.chunks = [Chunk(text=item["text"], metadata=item["metadata"]) for item in payload]
        return store


def _as_float32(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype="float32")
    return np.ascontiguousarray(array)


def _faiss() -> Any:
    import faiss

    return faiss
