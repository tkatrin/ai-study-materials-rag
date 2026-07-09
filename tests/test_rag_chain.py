from rag_service.models import Chunk
from rag_service.rag_chain import build_answer, format_sources


def test_build_answer_uses_retrieved_chunks():
    chunks = [
        Chunk(
            text="Градиентный спуск обновляет параметры модели в направлении уменьшения функции потерь.",
            metadata={"source": "ml.md", "chunk_id": 1},
        )
    ]

    answer = build_answer("Как работает градиентный спуск?", chunks)

    assert "Градиентный спуск" in answer.answer
    assert answer.sources == chunks


def test_format_sources_includes_file_and_fragment():
    text = format_sources([Chunk(text="Небольшой фрагмент", metadata={"source": "a.txt", "chunk_id": 2})])

    assert "a.txt" in text
    assert "фрагмент 2" in text
