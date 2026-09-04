"""PART 1 — pytest cơ bản, chưa có AI.

System under test: `rag_qa.chunker` — module thật mà Part 2 sẽ dùng lại để
cắt tài liệu, không phải ví dụ vứt đi.

Chạy:
    uv run pytest tests/test_chunker.py -v
"""

from __future__ import annotations

import pytest

from rag_qa.chunker import (
    chunk_legal_document,
    chunk_text,
    normalize_whitespace,
    split_paragraphs,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1.1 — assert đơn giản
# ---------------------------------------------------------------------------
def test_normalize_gom_khoang_trang_lien_tiep():
    assert normalize_whitespace("a    b") == "a b"


def test_normalize_bo_khoang_trang_hai_dau():
    assert normalize_whitespace("   xin chào   ") == "xin chào"


def test_normalize_giu_ranh_gioi_doan():
    # Xuống dòng đôi là ranh giới đoạn -> phải được giữ
    assert normalize_whitespace("đoạn 1\n\n\n\nđoạn 2") == "đoạn 1\n\nđoạn 2"


def test_normalize_xu_ly_xuong_dong_windows():
    assert normalize_whitespace("a\r\n\r\nb") == "a\n\nb"


# ---------------------------------------------------------------------------
# 1.2 — parametrize: một test, nhiều bộ dữ liệu
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("", []),
        ("   ", []),
        ("\n\n\n", []),
        ("một đoạn", ["một đoạn"]),
        ("đoạn 1\n\nđoạn 2", ["đoạn 1", "đoạn 2"]),
        ("đoạn 1\n\n\n\n\nđoạn 2", ["đoạn 1", "đoạn 2"]),
    ],
    ids=["rỗng", "toàn-khoảng-trắng", "toàn-xuống-dòng", "một-đoạn", "hai-đoạn", "nhiều-dòng-trống"],
)
def test_split_paragraphs(text, expected):
    assert split_paragraphs(text) == expected


# ---------------------------------------------------------------------------
# 1.3 — pytest.raises: kiểm tra lỗi được ném đúng
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("chunk_size", [0, -1, -100])
def test_chunk_size_khong_duong_thi_bao_loi(chunk_size):
    with pytest.raises(ValueError, match="chunk_size phải > 0"):
        chunk_text("xin chào", chunk_size=chunk_size)


def test_overlap_am_thi_bao_loi():
    with pytest.raises(ValueError, match="overlap phải >= 0"):
        chunk_text("xin chào", chunk_size=100, overlap=-1)


@pytest.mark.parametrize("overlap", [100, 150])
def test_overlap_lon_hon_chunk_size_thi_bao_loi(overlap):
    """Đây là test QUAN TRỌNG NHẤT của Part 1.

    Nếu overlap >= chunk_size, cửa sổ trượt không bao giờ tiến lên và hàm sẽ
    lặp vô hạn. Không có test này, bug đó chỉ lộ ra khi ai đó treo cả CI.
    """
    with pytest.raises(ValueError, match="phải nhỏ hơn chunk_size"):
        chunk_text("x" * 500, chunk_size=100, overlap=overlap)


def test_text_khong_phai_chuoi_thi_bao_loi():
    with pytest.raises(TypeError, match="text phải là str"):
        chunk_text(12345)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1.4 — fixture: dữ liệu dùng chung, viết một lần
# ---------------------------------------------------------------------------
@pytest.fixture
def van_ban_ba_doan() -> str:
    return "Đoạn thứ nhất.\n\nĐoạn thứ hai.\n\nĐoạn thứ ba."


@pytest.fixture
def doan_van_rat_dai() -> str:
    """Một đoạn duy nhất dài hơn mọi chunk_size ta dùng trong test."""
    return "A" * 1000


def test_van_ban_ngan_thi_ra_dung_mot_chunk(van_ban_ba_doan):
    chunks = chunk_text(van_ban_ba_doan, chunk_size=500, overlap=50)
    assert len(chunks) == 1


def test_khong_chunk_nao_vuot_qua_chunk_size(doan_van_rat_dai):
    chunks = chunk_text(doan_van_rat_dai, chunk_size=200, overlap=20)
    assert all(len(c) <= 200 for c in chunks)


def test_doan_qua_dai_bi_cat_thanh_nhieu_chunk(doan_van_rat_dai):
    chunks = chunk_text(doan_van_rat_dai, chunk_size=200, overlap=20)
    assert len(chunks) > 1


def test_khong_bao_gio_tra_ve_chunk_rong(van_ban_ba_doan):
    chunks = chunk_text(van_ban_ba_doan + "\n\n\n\n", chunk_size=50, overlap=5)
    assert all(c.strip() for c in chunks)


# ---------------------------------------------------------------------------
# 1.5 — kiểm tra tính chất, không kiểm tra giá trị cụ thể
# ---------------------------------------------------------------------------
def test_chunk_co_chong_lan_thuc_su():
    """Chunk sau phải chứa phần đuôi của chunk trước.

    Đây là lý do overlap tồn tại: một câu bị cắt đôi giữa hai chunk vẫn còn
    nguyên vẹn ở ít nhất một trong hai. Với RAG, mất chỗ này nghĩa là mất
    thông tin — và Part 4 sẽ thấy nó ở điểm Context Recall.
    """
    text = "".join(str(i % 10) for i in range(500))
    chunks = chunk_text(text, chunk_size=100, overlap=30)
    duoi_chunk_dau = chunks[0][-30:]
    assert chunks[1].startswith(duoi_chunk_dau)


@pytest.mark.parametrize("chunk_size,overlap", [(100, 0), (200, 50), (500, 50), (1000, 100)])
def test_moi_cau_hinh_deu_bao_phu_het_van_ban(chunk_size, overlap):
    """Không được mất chữ nào: nối các chunk lại phải chứa đủ nội dung gốc."""
    text = "".join(str(i % 10) for i in range(1000))
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    assert "".join(chunks).replace(" ", "").count("0") >= text.count("0")


def test_van_ban_rong_tra_ve_list_rong():
    assert chunk_text("") == []
    assert chunk_text("   \n\n   ") == []


# ---------------------------------------------------------------------------
# 1.6 — chunk_legal_document: cắt theo CẤU TRÚC, không theo số ký tự
# ---------------------------------------------------------------------------
VAN_BAN_LUAT = """## Chương I NHỮNG QUY ĐỊNH CHUNG

## Điều 1. Phạm vi điều chỉnh

Luật này quy định về an ninh mạng.

## Điều 2. Giải thích từ ngữ

An ninh mạng là sự ổn định, an ninh, an toàn của không gian mạng.

## Chương II BẢO VỆ AN NINH MẠNG

## Điều 8. Phân loại cấp độ

Hệ thống thông tin được phân loại theo 5 cấp độ."""


def test_moi_dieu_thanh_mot_chunk():
    chunks = chunk_legal_document(VAN_BAN_LUAT, max_size=1200, overlap=100)
    assert len(chunks) == 3


@pytest.mark.parametrize("so_dieu", ["Điều 1.", "Điều 2.", "Điều 8."])
def test_moi_chunk_mang_nhan_dieu_cua_no(so_dieu):
    chunks = chunk_legal_document(VAN_BAN_LUAT, max_size=1200, overlap=100)
    assert any(c.startswith("[") and so_dieu in c.splitlines()[0] for c in chunks)


def test_nhan_chua_dung_chuong_cua_dieu_do():
    """Điều 8 thuộc Chương II, không phải Chương I."""
    chunks = chunk_legal_document(VAN_BAN_LUAT, max_size=1200, overlap=100)
    chunk_dieu_8 = next(c for c in chunks if "Điều 8." in c.splitlines()[0])
    assert "Chương II" in chunk_dieu_8.splitlines()[0]
    assert "Chương I " not in chunk_dieu_8.splitlines()[0]


def test_dieu_qua_dai_bi_chia_nho_nhung_van_giu_nhan():
    """Đây là tính chất quan trọng nhất của hàm.

    Một điều luật dài phải chia nhiều mảnh, nhưng MỌI mảnh vẫn phải mang nhãn
    "Chương X | Điều Y". Mất nhãn ở mảnh nào thì mảnh đó trở thành văn bản mồ
    côi — bộ truy hồi lấy về mà không ai biết nó thuộc điều nào.

    Đây chính là lỗi đã khiến hệ thống trả lời "Nguyên thủ tịch Quốc hội" cho
    câu hỏi về ngày hiệu lực (xem docs/sessions/session-02).
    """
    dai = "## Điều 99. Điều rất dài\n\n" + ("Nội dung lặp lại. " * 300)
    chunks = chunk_legal_document(dai, max_size=500, overlap=50)
    assert len(chunks) > 1
    assert all(c.startswith("[") and "Điều 99." in c.splitlines()[0] for c in chunks)


def test_van_ban_khong_phai_luat_thi_quay_ve_cach_cat_thuong():
    """Không có 'Điều' nào -> dùng chunk_text. Đây là bài tập 4 của Session 2."""
    thuong = "Đoạn một.\n\nĐoạn hai.\n\nĐoạn ba."
    assert chunk_legal_document(thuong, max_size=1200) == chunk_text(thuong, 1200, 100)


@pytest.mark.parametrize("max_size,overlap", [(0, 10), (-5, 0), (100, 100), (100, 150)])
def test_tham_so_khong_hop_le_thi_bao_loi(max_size, overlap):
    with pytest.raises(ValueError):
        chunk_legal_document(VAN_BAN_LUAT, max_size=max_size, overlap=overlap)


def test_van_ban_rong_tra_ve_list_rong_legal():
    assert chunk_legal_document("") == []
