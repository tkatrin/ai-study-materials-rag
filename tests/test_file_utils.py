from rag_service.file_utils import safe_filename, save_uploaded_files


class UploadedFile:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getbuffer(self):
        return self._content


def test_safe_filename_removes_path_and_special_characters():
    assert safe_filename("../lecture notes?.md") == "lecture_notes_.md"
    assert safe_filename("...") == "uploaded_document"


def test_save_uploaded_files_uses_safe_unique_name(tmp_path):
    uploaded = UploadedFile("../bad name.txt", b"hello")

    saved_files = save_uploaded_files([uploaded], tmp_path)

    saved_path, original_name = saved_files[0]
    assert original_name == "bad name.txt"
    assert saved_path.parent == tmp_path
    assert saved_path.name.endswith("_bad_name.txt")
    assert saved_path.read_bytes() == b"hello"
