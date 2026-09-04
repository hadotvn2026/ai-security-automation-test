"""Cắt văn bản dài thành các chunk nhỏ để đưa vào vector store.

Đây là module duy nhất trong ứng dụng KHÔNG phụ thuộc thư viện ngoài nào.
Vì vậy nó là "system under test" của Part 1: bạn học pytest trên một module
thật sự được Part 2 dùng lại, chứ không phải một ví dụ calculator vứt đi.

Ứng với hộp số 3 trong sơ đồ kiến trúc.
"""

from __future__ import annotations

import re

__all__ = [
    "normalize_whitespace",
    "split_paragraphs",
    "chunk_text",
    "chunk_legal_document",
]


def normalize_whitespace(text: str) -> str:
    """Gom mọi khoảng trắng liên tiếp thành 1 dấu cách, bỏ khoảng trắng 2 đầu.

    Xuống dòng đôi (\\n\\n) được giữ lại vì nó đánh dấu ranh giới đoạn văn.
    """
    if not isinstance(text, str):
        raise TypeError(f"text phải là str, nhận được {type(text).__name__}")
    # Chuẩn hoá xuống dòng kiểu Windows trước
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # >=2 xuống dòng -> giữ đúng 2 (ranh giới đoạn)
    text = re.sub(r"\n{2,}", "\n\n", text)
    # Khoảng trắng khác trong cùng 1 dòng -> 1 dấu cách
    text = re.sub(r"[ \t]+", " ", text)
    # Bỏ khoảng trắng thừa quanh mỗi dòng
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    """Tách văn bản thành danh sách đoạn văn, bỏ đoạn rỗng."""
    normalized = normalize_whitespace(text)
    if not normalized:
        return []
    return [p for p in normalized.split("\n\n") if p]


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Cắt `text` thành các chunk tối đa `chunk_size` ký tự, chồng lấn `overlap`.

    Quy tắc:
      - Ưu tiên không cắt giữa chừng một đoạn văn: đoạn nào còn vừa thì ghép tiếp.
      - Đoạn dài hơn `chunk_size` sẽ bị cắt cứng theo cửa sổ trượt có overlap.
      - Kết quả không bao giờ chứa chunk rỗng.

    Raises:
        ValueError: nếu chunk_size <= 0, overlap < 0, hoặc overlap >= chunk_size.
        TypeError: nếu text không phải str.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size phải > 0, nhận được {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap phải >= 0, nhận được {overlap}")
    if overlap >= chunk_size:
        # Nếu overlap >= chunk_size thì cửa sổ trượt không bao giờ tiến lên
        # -> vòng lặp vô hạn. Chặn từ đầu thay vì treo máy.
        raise ValueError(
            f"overlap ({overlap}) phải nhỏ hơn chunk_size ({chunk_size})"
        )

    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[str] = []
    buffer = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            # Đoạn quá dài: đẩy buffer hiện tại ra trước rồi cắt cứng đoạn này
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.extend(_sliding_window(para, chunk_size, overlap))
            continue

        candidate = f"{buffer}\n\n{para}" if buffer else para
        if len(candidate) <= chunk_size:
            buffer = candidate
        else:
            chunks.append(buffer)
            buffer = para

    if buffer:
        chunks.append(buffer)

    return [c for c in chunks if c.strip()]


def _sliding_window(text: str, size: int, overlap: int) -> list[str]:
    """Cắt cứng một chuỗi dài bằng cửa sổ trượt. Dùng khi 1 đoạn > chunk_size."""
    step = size - overlap  # đã được đảm bảo > 0 bởi chunk_text
    out: list[str] = []
    start = 0
    while start < len(text):
        out.append(text[start : start + size])
        start += step
    return out


# ---------------------------------------------------------------------------
# Cắt theo CẤU TRÚC tài liệu, không theo số ký tự
# ---------------------------------------------------------------------------
# Vì sao cần hàm riêng cho văn bản luật?
#
# `chunk_text` ở trên cắt theo số ký tự. Với văn bản luật, nó tách "Điều 44.
# Hiệu lực thi hành" ra khỏi nội dung của chính điều đó. Kết quả đo được trên
# Luật 116/2025: hỏi "luật có hiệu lực từ ngày nào" thì hệ thống trả lời
# "Nguyên thủ tịch Quốc hội" — retriever lấy đúng trang chữ ký.
#
# Hàm dưới đây cắt theo ranh giới ĐIỀU, và quan trọng nhất: khi một điều dài
# phải chia nhỏ, TIÊU ĐỀ ĐIỀU ĐƯỢC LẶP LẠI ở đầu mỗi mảnh. Nhờ vậy mọi chunk
# đều tự mang ngữ cảnh "đây là Điều mấy, nói về cái gì".

TIEU_DE_DIEU = re.compile(r"^(?:##\s*)?(Điều\s+\d+\.\s*.*)$", re.MULTILINE)
TIEU_DE_CHUONG = re.compile(r"^(?:##\s*)?(Chương\s+[IVXLC]+.*)$", re.MULTILINE)


def chunk_legal_document(
    text: str,
    max_size: int = 1200,
    overlap: int = 100,
) -> list[str]:
    """Cắt văn bản luật theo Điều, giữ tiêu đề Chương và Điều trong mọi chunk.

    Args:
        text: toàn văn đã chuẩn hoá.
        max_size: độ dài tối đa một chunk. Điều dài hơn sẽ bị chia nhỏ, mỗi
            mảnh vẫn mang tiêu đề điều ở đầu.
        overlap: độ chồng lấn khi phải chia nhỏ một điều.

    Returns:
        Danh sách chunk. Mỗi chunk bắt đầu bằng dòng ngữ cảnh dạng
        "Chương II | Điều 12. ...".
    """
    if max_size <= 0:
        raise ValueError(f"max_size phải > 0, nhận được {max_size}")
    if not 0 <= overlap < max_size:
        raise ValueError(f"overlap ({overlap}) phải nằm trong [0, {max_size})")

    text = normalize_whitespace(text)
    if not text:
        return []

    # Đánh dấu vị trí mọi tiêu đề Điều
    moc = [(m.start(), m.group(1).strip()) for m in TIEU_DE_DIEU.finditer(text)]
    if not moc:
        # Không phải văn bản luật -> quay về cách cắt thông thường
        return chunk_text(text, max_size, overlap)

    chuong_theo_vi_tri = [
        (m.start(), m.group(1).strip()) for m in TIEU_DE_CHUONG.finditer(text)
    ]

    def chuong_cua(vi_tri: int) -> str:
        ten = ""
        for bat_dau, nhan in chuong_theo_vi_tri:
            if bat_dau <= vi_tri:
                ten = nhan
            else:
                break
        return ten

    chunks: list[str] = []
    for i, (bat_dau, tieu_de) in enumerate(moc):
        ket_thuc = moc[i + 1][0] if i + 1 < len(moc) else len(text)
        than = text[bat_dau:ket_thuc].strip()
        nhan = f"{chuong_cua(bat_dau)} | {tieu_de}".strip(" |")

        if len(than) <= max_size:
            chunks.append(f"[{nhan}]\n{than}")
            continue

        # Điều quá dài: chia nhỏ nhưng LẶP LẠI nhãn ở mọi mảnh
        con_lai = than[len(tieu_de) :].strip()
        for manh in _sliding_window(con_lai, max_size - len(nhan) - 16, overlap):
            chunks.append(f"[{nhan}]\n{tieu_de}\n{manh}")

    return [c for c in chunks if c.strip()]
