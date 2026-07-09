from rag_service.document_loader import load_document, load_documents


def test_load_txt_document(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Первая строка.\n\nВторая строка.", encoding="utf-8")

    document = load_document(path)

    assert document.text == "Первая строка.\nВторая строка."
    assert document.metadata["source"] == "notes.txt"
    assert document.metadata["extension"] == ".txt"


def test_load_markdown_with_original_source_name(tmp_path):
    path = tmp_path / "safe_name.md"
    path.write_text("# Тема\n\nОписание", encoding="utf-8")

    document = load_documents([(path, "original name.md")])[0]

    assert "# Тема" in document.text
    assert document.metadata["source"] == "original name.md"
