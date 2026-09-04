"""Test trí nhớ tự sửa — tất định, không gọi model.

Cơ chế trí nhớ mà không test được thì không đáng tin hơn cái nó thay thế.
"""

from __future__ import annotations

import pytest

from rag_qa import tri_nho as tn

pytestmark = pytest.mark.unit

VAN_BAN = """Điều 8. Phân loại cấp độ hệ thống thông tin
1. Hệ thống thông tin được phân loại theo 5 cấp độ.

Điều 9. Hệ thống thông tin quan trọng
1. Danh mục do Thủ tướng ban hành.

Điều 44. Hiệu lực thi hành
1. Luật này có hiệu lực thi hành từ ngày 01 tháng 7 năm 2026.
"""


@pytest.fixture
def nguon(tmp_path):
    p = tmp_path / "luat.txt"
    p.write_text(VAN_BAN, encoding="utf-8")
    return p


@pytest.fixture
def kho_mau(nguon):
    return [
        tn.KhangDinh(ma="hieu-luc", cau_hoi="Hiệu lực khi nào?",
                     noi_dung="Có hiệu lực từ ngày 01 tháng 7 năm 2026.",
                     dieu=44, van_tay_bang_chung=tn.van_tay_dieu(44, nguon)),
        tn.KhangDinh(ma="cap-do", cau_hoi="Mấy cấp độ?",
                     noi_dung="Phân loại theo 5 cấp độ.",
                     dieu=8, van_tay_bang_chung=tn.van_tay_dieu(8, nguon)),
    ]


# ---------------------------------------------------------------------------
# Bằng chứng: trích và băm
# ---------------------------------------------------------------------------
def test_trich_dung_pham_vi_mot_dieu(nguon):
    d8 = tn.trich_dieu(8, nguon)
    assert "5 cấp độ" in d8
    assert "Điều 9" not in d8, "Trích lẫn sang điều kế tiếp"


def test_dieu_khong_ton_tai_tra_ve_none(nguon):
    assert tn.trich_dieu(999, nguon) is None
    assert tn.van_tay_dieu(999, nguon) is None


def test_dieu_cuoi_cung_van_trich_duoc(nguon):
    """Điều 44 là điều cuối — không có 'Điều 45' để làm mốc dừng."""
    assert "01 tháng 7 năm 2026" in tn.trich_dieu(44, nguon)


def test_van_tay_on_dinh_qua_thay_doi_khoang_trang(tmp_path):
    """Chi tiết quan trọng: OCR lại cùng một trang cho khoảng trắng khác nhau.

    Không được báo lạc hậu chỉ vì chuyện đó — chỉ NỘI DUNG đổi mới tính.
    """
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("Điều 44. Hiệu lực\n1. Có hiệu lực từ 01/7/2026.\n", encoding="utf-8")
    b.write_text("Điều 44.   Hiệu lực\n\n1.   Có hiệu lực    từ 01/7/2026.\n", encoding="utf-8")
    assert tn.van_tay_dieu(44, a) == tn.van_tay_dieu(44, b)


def test_van_tay_doi_khi_noi_dung_doi(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("Điều 44. Hiệu lực\n1. Có hiệu lực từ 01/7/2026.\n", encoding="utf-8")
    b.write_text("Điều 44. Hiệu lực\n1. Có hiệu lực từ 01/01/2027.\n", encoding="utf-8")
    assert tn.van_tay_dieu(44, a) != tn.van_tay_dieu(44, b)


# ---------------------------------------------------------------------------
# Phát hiện lạc hậu
# ---------------------------------------------------------------------------
def test_bang_chung_nguyen_ven_thi_khong_bao_lac_hau(kho_mau, nguon):
    assert tn.phat_hien_lac_hau(kho_mau, nguon) == []


def test_ban_kinh_anh_huong_chinh_xac(kho_mau, tmp_path):
    """Sửa Điều 44 thì CHỈ khẳng định neo vào Điều 44 bị gắn cờ.

    Đây là toàn bộ giá trị của việc neo bằng chứng theo Điều: bạn biết chính xác
    câu trả lời nào cần xác thực lại, thay vì phải rà lại tất cả.
    """
    sua = tmp_path / "sua.txt"
    sua.write_text(VAN_BAN.replace("01 tháng 7 năm 2026", "01 tháng 01 năm 2027"),
                   encoding="utf-8")

    lac_hau = tn.phat_hien_lac_hau(kho_mau, sua)
    assert [k.ma for k, _ in lac_hau] == ["hieu-luc"]


def test_dieu_bi_xoa_khoi_van_ban_thi_bao_lac_hau(kho_mau, tmp_path):
    xoa = tmp_path / "xoa.txt"
    xoa.write_text("Điều 8. Phân loại\n1. Theo 5 cấp độ.\n", encoding="utf-8")
    ly_do = dict((k.ma, l) for k, l in tn.phat_hien_lac_hau(kho_mau, xoa))
    assert "KHÔNG CÒN" in ly_do["hieu-luc"]


def test_khang_dinh_chua_co_van_tay_bi_bao_lac_hau(nguon):
    kho = [tn.KhangDinh(ma="x", cau_hoi="q", noi_dung="n", dieu=8, van_tay_bang_chung="")]
    assert "Chưa từng ghi vân tay" in tn.phat_hien_lac_hau(kho, nguon)[0][1]


# ---------------------------------------------------------------------------
# Vòng đối chiếu — ba nhánh
# ---------------------------------------------------------------------------
def test_bang_chung_khong_doi_tra_loi_nhat_quan_thi_tu_lam_moi(kho_mau, nguon):
    kq = tn.doi_chieu(kho_mau, tra_loi_lai=lambda q: "Có hiệu lực từ 01 tháng 7 năm 2026, phân loại 5 cấp độ.", nguon=nguon)
    assert set(kq["lam_moi"]) == {"hieu-luc", "cap-do"}
    assert all(k.trang_thai == "xac_thuc" for k in kho_mau)


def test_bang_chung_khong_doi_tra_loi_lech_thi_nghi_ngo_HE_THONG(kho_mau, nguon):
    """Văn bản y nguyên mà câu trả lời đổi -> lỗi ở prompt/model/chunking.

    Phân biệt được hai nguyên nhân này là điểm mấu chốt: một cái phải sửa hệ
    thống, một cái phải đọc lại luật.
    """
    kq = tn.doi_chieu(kho_mau, tra_loi_lai=lambda q: "Có hiệu lực từ năm 2030.", nguon=nguon)
    assert "hieu-luc" in kq["he_thong_lech"]
    k = next(x for x in kho_mau if x.ma == "hieu-luc")
    assert k.trang_thai == "nghi_ngo"
    assert "không phải văn bản" in k.ghi_chu.lower()


def test_bang_chung_doi_thi_CHUYEN_NGUOI_khong_tu_sua(kho_mau, tmp_path):
    """Khác biệt cố ý so với OpenWiki.

    Judge ở đây là model 8B chạy local, đã đo được là chấm sai có hệ thống.
    Để nó tự viết lại một khẳng định pháp lý là đổi rủi ro nhỏ lấy rủi ro lớn.
    """
    sua = tmp_path / "sua.txt"
    sua.write_text(VAN_BAN.replace("01 tháng 7 năm 2026", "01 tháng 01 năm 2027"),
                   encoding="utf-8")

    noi_dung_goc = kho_mau[0].noi_dung
    kq = tn.doi_chieu(kho_mau, tra_loi_lai=lambda q: "Có hiệu lực từ 01/01/2027.", nguon=sua)

    assert kq["can_nguoi_ra_soat"] == ["hieu-luc"]
    k = next(x for x in kho_mau if x.ma == "hieu-luc")
    assert k.trang_thai == "can_nguoi_ra_soat"
    assert k.noi_dung == noi_dung_goc, "KHÔNG được tự viết lại khẳng định pháp lý"


def test_khong_truyen_ham_tra_loi_thi_chi_kiem_bang_chung(kho_mau, nguon):
    kq = tn.doi_chieu(kho_mau, tra_loi_lai=None, nguon=nguon)
    assert len(kq["khong_doi"]) == 2
    assert kq["lam_moi"] == []


# ---------------------------------------------------------------------------
# So khớp nhất quán — con số là thứ phải giữ
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cu,moi,mong_doi", [
    ("hiệu lực từ 01/7/2026", "Luật có hiệu lực thi hành từ ngày 01/7/2026.", True),
    ("hiệu lực từ 01/7/2026", "Luật có hiệu lực từ 01/01/2027.", False),
    ("phân loại 5 cấp độ", "Hệ thống được chia thành 5 cấp độ.", True),
    ("phân loại 5 cấp độ", "Hệ thống được chia thành 3 cấp độ.", False),
    ("theo Điều 44", "Quy định tại Điều 44 của Luật.", True),
    ("theo Điều 44", "Quy định tại Điều 45 của Luật.", False),
])
def test_so_khop_giu_duoc_con_so(cu, moi, mong_doi):
    """Câu chữ đổi thì chấp nhận được. Con số đổi thì không."""
    assert tn._nhat_quan(cu, moi) is mong_doi


# ---------------------------------------------------------------------------
# Tính bền của sự nghi ngờ
# ---------------------------------------------------------------------------
def test_nghi_ngo_ton_tai_qua_luu_va_doc_lai(kho_mau, tmp_path):
    """Sự không chắc chắn không được biến mất âm thầm giữa các phiên chạy."""
    kho_mau[0].trang_thai = "can_nguoi_ra_soat"
    kho_mau[0].ghi_chu = "Điều 44 đã đổi"

    f = tmp_path / "kho.json"
    tn.ghi_kho(kho_mau, f)
    doc_lai = tn.doc_kho(f)

    k = next(x for x in doc_lai if x.ma == "hieu-luc")
    assert k.trang_thai == "can_nguoi_ra_soat"
    assert k.ghi_chu == "Điều 44 đã đổi"


def test_kho_trong_thi_tra_ve_list_rong(tmp_path):
    assert tn.doc_kho(tmp_path / "chua-co.json") == []


def test_thong_ke_dem_dung(kho_mau):
    kho_mau[0].trang_thai = "nghi_ngo"
    tk = tn.thong_ke(kho_mau)
    assert tk["xac_thuc"] == 1 and tk["nghi_ngo"] == 1
