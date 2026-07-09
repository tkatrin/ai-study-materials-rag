from rag_service.models import Chunk
from rag_service.reranker import KeywordOverlapReranker, make_reranker


def test_keyword_overlap_reranker_promotes_term_match():
    results = [
        (Chunk(text="База данных хранит таблицы.", metadata={}), 0.9),
        (Chunk(text="Градиентный спуск уменьшает функцию потерь.", metadata={}), 0.2),
    ]

    reranked = KeywordOverlapReranker().rerank(
        "Что делает градиентный спуск?",
        results,
        top_k=1,
    )

    assert reranked[0][0].text.startswith("Градиентный спуск")


def test_keyword_overlap_reranker_uses_faiss_score_as_tiebreaker():
    results = [
        (Chunk(text="Градиентный метод.", metadata={}), 0.2),
        (Chunk(text="Градиентный алгоритм.", metadata={}), 0.8),
    ]

    reranked = KeywordOverlapReranker().rerank("Что такое градиентный подход?", results, top_k=1)

    assert reranked[0][0].text == "Градиентный алгоритм."


def test_make_reranker_returns_none_for_disabled_mode():
    assert make_reranker("None") is None


def test_make_reranker_requires_cross_encoder_model_name():
    try:
        make_reranker("CrossEncoder", "")
    except ValueError as error:
        assert "model name" in str(error)
    else:
        raise AssertionError("Expected ValueError")
