from rag_service.chunker import chunk_documents, split_text
from rag_service.models import Document


def test_split_text_keeps_overlap_and_size():
    text = " ".join(f"sentence {index}." for index in range(80))
    chunks = split_text(text, chunk_size=120, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk) <= 140 for chunk in chunks)
    assert "sentence 0." in chunks[0]


def test_split_text_rejects_bad_overlap():
    try:
        split_text("hello", chunk_size=10, chunk_overlap=10)
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("Expected ValueError")


def test_chunk_documents_preserves_source_metadata():
    documents = [
        Document(
            text="Первый абзац.\nВторой абзац про базы данных.",
            metadata={"source": "notes.md"},
        )
    ]

    chunks = chunk_documents(documents, chunk_size=40, chunk_overlap=5)

    assert chunks
    assert chunks[0].metadata["source"] == "notes.md"
    assert chunks[0].metadata["chunk_id"] == 1


def test_chunk_documents_preserves_existing_page_metadata():
    documents = [
        Document(
            text="Текст страницы.",
            metadata={"source": "lecture.pdf", "page_number": 3},
        )
    ]

    chunks = chunk_documents(documents)

    assert chunks[0].metadata["page_number"] == 3
