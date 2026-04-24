"""
File reader tool — reads uploaded files with truncation (from Final-Project).
"""

from pathlib import Path

from langchain_core.tools import tool

from src.config import settings


@tool
def read_uploaded_file(filename: str) -> str:
    """Read contents of a previously uploaded file (PDF, TXT, DOCX, MD).

    Args:
        filename: Name of the file in the uploads directory.
    """
    upload_root = settings.upload_dir.resolve()
    file_path = (upload_root / filename).resolve()
    if not file_path.is_relative_to(upload_root):
        return "Access denied: path outside upload directory"
    if not file_path.exists():
        return f"File not found: {filename}"

    suffix = file_path.suffix.lower()
    try:
        if suffix in (".txt", ".md"):
            content = file_path.read_text(encoding="utf-8")
        elif suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(file_path))
            content = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif suffix == ".docx":
            from docx import Document
            doc = Document(str(file_path))
            content = "\n".join(p.text for p in doc.paragraphs)
        else:
            return f"Unsupported file type: {suffix}"

        max_chars = 5000
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n\n... (total {len(content)} chars, showing first {max_chars})"
        return content
    except Exception as e:
        return f"File read error: {e}"
