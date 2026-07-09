from types import SimpleNamespace

from rag_service.document_loader import load_document, load_document_blocks, load_documents


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


def test_load_pdf_as_page_documents(monkeypatch, tmp_path):
    class Page:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class FakeReader:
        def __init__(self, path):
            self.pages = [Page("Первая страница"), Page("Вторая страница")]

    monkeypatch.setitem(
        __import__("sys").modules,
        "pypdf",
        SimpleNamespace(PdfReader=FakeReader),
    )
    path = tmp_path / "lecture.pdf"
    path.write_bytes(b"fake pdf")

    documents = load_documents([path])

    assert len(documents) == 2
    assert documents[0].metadata["page_number"] == 1
    assert documents[1].metadata["page_number"] == 2


def test_load_document_pdf_returns_single_combined_document(monkeypatch, tmp_path):
    class Page:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class FakeReader:
        def __init__(self, path):
            self.pages = [Page("Первая страница"), Page("Вторая страница")]

    monkeypatch.setitem(
        __import__("sys").modules,
        "pypdf",
        SimpleNamespace(PdfReader=FakeReader),
    )
    path = tmp_path / "lecture.pdf"
    path.write_bytes(b"fake pdf")

    document = load_document(path)
    blocks = load_document_blocks(path)

    assert len(blocks) == 2
    assert "[page 1]" in document.text
    assert "[page 2]" in document.text
    assert "page_number" not in document.metadata
