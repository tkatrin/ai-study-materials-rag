from pathlib import Path
from typing import Iterable, List, Tuple, Union

from .chunker import chunk_documents
from .document_loader import load_documents
from .embedder import SentenceTransformerEmbedder
from .models import Answer, Chunk
from .rag_chain import build_answer
from .vector_store import FaissVectorStore


PathLike = Union[str, Path]


def build_index(
    file_paths: Iterable[PathLike],
    embedder: SentenceTransformerEmbedder,
    chunk_size: int = 900,
    chunk_overlap: int = 160,
) -> FaissVectorStore:
    documents = load_documents(file_paths)
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embeddings = embedder.embed(chunk.text for chunk in chunks)
    store = FaissVectorStore(embedder.dimension)
    store.add(chunks, embeddings)
    return store


def ask_question(
    question: str,
    store: FaissVectorStore,
    embedder: SentenceTransformerEmbedder,
    top_k: int = 4,
) -> Tuple[Answer, List[Tuple[Chunk, float]]]:
    query_embedding = embedder.embed([question])
    results = store.search(query_embedding, top_k=top_k)
    chunks = [chunk for chunk, _score in results]
    return build_answer(question, chunks), results
