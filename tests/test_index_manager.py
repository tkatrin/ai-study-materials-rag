from rag_service.index_manager import has_saved_index


def test_has_saved_index_requires_all_files(tmp_path):
    assert not has_saved_index(tmp_path)

    (tmp_path / "index.faiss").write_bytes(b"fake")
    (tmp_path / "chunks.json").write_text("[]", encoding="utf-8")
    assert not has_saved_index(tmp_path)

    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    assert has_saved_index(tmp_path)
