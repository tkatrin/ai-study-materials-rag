from pathlib import Path
from typing import Iterable, List, Union

from .models import Document


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


PathLike = Union[str, Path]


def load_document(path: PathLike) -> Document:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{suffix}'. Supported: {supported}")

    if suffix == ".pdf":
        text = _read_pdf(file_path)
    elif suffix == ".docx":
        text = _read_docx(file_path)
    else:
        text = file_path.read_text(encoding="utf-8")

    return Document(
        text=_normalize_text(text),
        metadata={
            "source": file_path.name,
            "path": str(file_path),
            "extension": suffix,
        },
    )


def load_documents(paths: Iterable[PathLike]) -> List[Document]:
    return [load_document(path) for path in paths]


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(f"[page {page_number}]\n{page_text}")
    return "\n\n".join(pages)


def _read_docx(path: Path) -> str:
    from docx import Document as DocxDocument

    document = DocxDocument(str(path))
    paragraphs = [paragraph.text for paragraph in document.paragraphs]
    table_cells = []
    for table in document.tables:
        for row in table.rows:
            table_cells.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(paragraphs + table_cells)


def _normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    normalized_lines = [line for line in lines if line]
    return "\n".join(normalized_lines)
