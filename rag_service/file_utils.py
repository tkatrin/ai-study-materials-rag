from pathlib import Path
from typing import List, Tuple
from uuid import uuid4


def save_uploaded_files(uploaded_files, upload_dir: Path) -> List[Tuple[Path, str]]:
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []
    for uploaded_file in uploaded_files:
        original_name = Path(uploaded_file.name).name
        safe_name = safe_filename(original_name)
        destination = upload_dir / f"{uuid4().hex}_{safe_name}"
        destination.write_bytes(uploaded_file.getbuffer())
        saved_files.append((destination, original_name))
    return saved_files


def safe_filename(filename: str) -> str:
    allowed = []
    for char in filename:
        if char.isalnum() or char in {".", "-", "_"}:
            allowed.append(char)
        else:
            allowed.append("_")
    safe = "".join(allowed).strip("._")
    return safe or "uploaded_document"
