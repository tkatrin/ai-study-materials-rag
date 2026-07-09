from rag_service.chunker import split_text


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
