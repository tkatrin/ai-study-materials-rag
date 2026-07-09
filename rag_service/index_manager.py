import shutil
from pathlib import Path
from typing import List

from .config import INDEX_DIR, UPLOAD_DIR
from .vector_store import FaissVectorStore

DEFAULT_INDEX_DIR = INDEX_DIR
DEFAULT_UPLOAD_DIR = UPLOAD_DIR


def save_index(
    store: FaissVectorStore,
    index_dir: Path = DEFAULT_INDEX_DIR,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> None:
    metadata = {}
    if chunk_size is not None:
        metadata["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        metadata["chunk_overlap"] = chunk_overlap
    store.save(index_dir, metadata=metadata)


def load_index(
    embedding_model: str,
    index_dir: Path = DEFAULT_INDEX_DIR,
) -> FaissVectorStore:
    return FaissVectorStore.load(index_dir, expected_embedding_model=embedding_model)


def has_saved_index(index_dir: Path = DEFAULT_INDEX_DIR) -> bool:
    return (
        (index_dir / "index.faiss").exists()
        and (index_dir / "chunks.json").exists()
        and (index_dir / "metadata.json").exists()
    )


def indexed_filenames(store: FaissVectorStore) -> List[str]:
    names = []
    for chunk in store.chunks:
        source = chunk.metadata.get("source", "unknown")
        if source not in names:
            names.append(source)
    return names


def clear_project_state(
    index_dir: Path = DEFAULT_INDEX_DIR,
    upload_dir: Path = DEFAULT_UPLOAD_DIR,
) -> None:
    for directory in (index_dir, upload_dir):
        if directory.exists():
            shutil.rmtree(directory)
