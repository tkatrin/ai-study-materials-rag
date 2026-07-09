from functools import lru_cache
from typing import List, Optional, Tuple

from .models import Chunk


RetrievalResult = Tuple[Chunk, float]


class KeywordOverlapReranker:
    def rerank(
        self,
        question: str,
        results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        question_terms = _terms(question)
        rescored = []
        for chunk, score in results:
            chunk_terms = _terms(chunk.text)
            overlap = len(question_terms & chunk_terms)
            rerank_score = float(score) + overlap
            rescored.append((chunk, rerank_score))
        rescored.sort(key=lambda item: item[1], reverse=True)
        return rescored[:top_k]


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = _load_cross_encoder(model_name)

    def rerank(
        self,
        question: str,
        results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        if not results:
            return []
        pairs = [(question, chunk.text) for chunk, _score in results]
        scores = self.model.predict(pairs)
        rescored = [
            (chunk, float(score))
            for (chunk, _old_score), score in zip(results, scores)
        ]
        rescored.sort(key=lambda item: item[1], reverse=True)
        return rescored[:top_k]


def make_reranker(mode: str, model_name: str = ""):
    if mode == "Keyword":
        return KeywordOverlapReranker()
    if mode == "CrossEncoder":
        if not model_name.strip():
            raise ValueError("CrossEncoder reranker requires a model name")
        return CrossEncoderReranker(model_name)
    return None


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def _terms(text: str) -> set:
    return {
        token.strip(".,:;!?()[]{}\"'").lower()
        for token in text.split()
        if len(token.strip(".,:;!?()[]{}\"'")) > 3
    }
