from typing import Iterable, List

from .models import Chunk, Document


def chunk_documents(
    documents: Iterable[Document],
    chunk_size: int = 900,
    chunk_overlap: int = 160,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    for document in documents:
        for index, text in enumerate(split_text(document.text, chunk_size, chunk_overlap), start=1):
            metadata = dict(document.metadata)
            metadata["chunk_id"] = index
            chunks.append(Chunk(text=text, metadata=metadata))
    return chunks


def split_text(text: str, chunk_size: int = 900, chunk_overlap: int = 160) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    clean_text = _normalize_for_chunking(text)
    if not clean_text:
        return []
    if len(clean_text) <= chunk_size:
        return [clean_text]

    chunks = []
    start = 0
    while start < len(clean_text):
        end = min(start + chunk_size, len(clean_text))
        split_at = _best_split(clean_text, start, end)
        chunk = clean_text[start:split_at].strip()
        if chunk:
            chunks.append(chunk)
        if split_at >= len(clean_text):
            break
        start = max(split_at - chunk_overlap, 0)
    return chunks


def _best_split(text: str, start: int, end: int) -> int:
    if end >= len(text):
        return len(text)

    window = text[start:end]
    for separator in (". ", "? ", "! ", "\n", "; ", ", "):
        split = window.rfind(separator)
        if split > int(len(window) * 0.55):
            return start + split + len(separator)
    return end


def _normalize_for_chunking(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)
