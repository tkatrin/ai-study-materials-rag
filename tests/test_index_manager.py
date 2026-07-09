from rag_service.index_manager import clear_project_state, has_saved_index


def test_has_saved_index_requires_all_files(tmp_path):
    assert not has_saved_index(tmp_path)

    (tmp_path / "index.faiss").write_bytes(b"fake")
    (tmp_path / "chunks.json").write_text("[]", encoding="utf-8")
    assert not has_saved_index(tmp_path)

    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    assert has_saved_index(tmp_path)


def test_clear_project_state_removes_index_and_uploads(tmp_path):
    index_dir = tmp_path / "index"
    upload_dir = tmp_path / "uploads"
    index_dir.mkdir()
    upload_dir.mkdir()
    (index_dir / "index.faiss").write_bytes(b"fake")
    (upload_dir / "a.txt").write_text("hello", encoding="utf-8")

    clear_project_state(index_dir=index_dir, upload_dir=upload_dir)

    assert not index_dir.exists()
    assert not upload_dir.exists()
