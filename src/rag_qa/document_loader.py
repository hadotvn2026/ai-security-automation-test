"""Đọc tài liệu nguồn (.docx) thành văn bản thuần.

Ứng với hộp số 2 trong sơ đồ kiến trúc.
"""

from __future__ import annotations

from pathlib import Path

from docx import Document

__all__ = ["load_docx", "load_document"]


def load_docx(path: str | Path) -> str:
    """Đọc file .docx và trả về toàn bộ nội dung dạng text.

    Mỗi paragraph trong Word trở thành một dòng. Paragraph rỗng được giữ lại
    làm ranh giới đoạn — chunker.py dựa vào ranh giới này để không cắt giữa ý.

    Raises:
        FileNotFoundError: nếu file không tồn tại.
        ValueError: nếu file không phải .docx.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy tài liệu: {path}")
    if path.suffix.lower() != ".docx":
        raise ValueError(f"Chỉ hỗ trợ .docx, nhận được: {path.suffix}")

    document = Document(str(path))
    lines: list[str] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            # Heading được đánh dấu để chunk giữ được ngữ cảnh tiêu đề
            if paragraph.style.name.startswith("Heading"):
                lines.append(f"\n\n## {text}\n")
            else:
                lines.append(text)
        else:
            lines.append("")
    return "\n".join(lines)


def load_document(path: str | Path) -> str:
    """Điểm vào chung — sau này muốn thêm .pdf/.md thì mở rộng ở đây."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return load_docx(path)
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8")
    raise ValueError(f"Định dạng chưa hỗ trợ: {suffix}")
