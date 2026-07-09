from typing import Iterable, List, Optional, Protocol, Tuple

from .chunker import chunk_documents
from .document_loader import DocumentInput, load_documents
from .models import Answer, Chunk
from .rag_chain import TextGenerator, build_answer
from .reranker import RetrievalResult
from .vector_store import FaissVectorStore


class Embedder(Protocol):
    model_name: str

    @property
    def dimension(self) -> int:
        ...

    def embed(self, texts: Iterable[str]):
        ...


class Reranker(Protocol):
    def rerank(
        self,
        question: str,
        results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        ...


def build_index(
    file_paths: Iterable[DocumentInput],
    embedder: Embedder,
    chunk_size: int = 900,
    chunk_overlap: int = 160,
) -> FaissVectorStore:
    documents = load_documents(file_paths)
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError("No text chunks were extracted from the selected documents")
    embeddings = embedder.embed(chunk.text for chunk in chunks)
    store = FaissVectorStore(embedder.dimension, embedding_model=embedder.model_name)
    store.add(chunks, embeddings)
    return store


def ask_question(
    question: str,
    store: FaissVectorStore,
    embedder: Embedder,
    top_k: int = 4,
    generator: Optional[TextGenerator] = None,
    min_score: Optional[float] = None,
    candidate_k: Optional[int] = None,
    reranker: Optional[Reranker] = None,
) -> Tuple[Answer, List[Tuple[Chunk, float]]]:
    query_embedding = embedder.embed([question])
    retrieval_k = max(top_k, candidate_k or top_k)
    results = store.search(query_embedding, top_k=retrieval_k, min_score=min_score)
    if reranker is not None:
        results = reranker.rerank(question, results, top_k=top_k)
    else:
        results = results[:top_k]
    chunks = [chunk for chunk, _score in results]
    return build_answer(question, chunks, generator=generator), results
