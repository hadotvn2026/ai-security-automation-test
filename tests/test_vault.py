"""Test kho tài liệu (vault) — tất định, không gọi model, không gọi mạng.

Vault là thành phần vá **điểm mù lớn nhất của mọi hệ RAG**: vector store chỉ biết
những gì đã được nạp, nó không tự biết tài liệu nguồn đã có bản mới hơn.

Vì vault thuần dữ liệu, toàn bộ file này chạy trong mili-giây và bắt được lỗi
trước khi bất kỳ lời gọi model nào xảy ra.
"""

from __future__ import annotations

import json

import pytest

from rag_qa import vault

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Đọc manifest
# ---------------------------------------------------------------------------
def test_doc_duoc_vault_that():
    assert len(vault.danh_sach_tai_lieu()) >= 4


def test_thieu_file_thi_bao_loi_ro_rang(tmp_path):
    with pytest.raises(vault.VaultError, match="Không tìm thấy vault"):
        vault.doc_manifest(tmp_path / "khong-ton-tai.json")


def test_json_hong_thi_bao_loi_ro_rang(tmp_path):
    hong = tmp_path / "hong.json"
    hong.write_text("{ đây không phải json", encoding="utf-8")
    with pytest.raises(vault.VaultError, match="sai định dạng JSON"):
        vault.doc_manifest(hong)


def test_thieu_khoa_tai_lieu_thi_bao_loi(tmp_path):
    thieu = tmp_path / "thieu.json"
    thieu.write_text(json.dumps({"cap_nhat_luc": "2026-01-01"}), encoding="utf-8")
    with pytest.raises(vault.VaultError, match="thiếu khoá"):
        vault.doc_manifest(thieu)


# ---------------------------------------------------------------------------
# Tra cứu
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ma", ["LUAT-116-2025", "luat-116-2025", " Luat-116-2025 "])
def test_tra_cuu_khong_phan_biet_hoa_thuong_va_khoang_trang(ma):
    assert vault.lay_tai_lieu(ma)["so_hieu"] == "116/2025/QH15"


def test_ma_khong_ton_tai_tra_ve_none():
    assert vault.lay_tai_lieu("KHONG-CO-THAT") is None


def test_chi_co_dung_mot_van_ban_dang_duoc_dung():
    dang_dung = [t for t in vault.danh_sach_tai_lieu() if t.get("dang_dung_trong_rag")]
    assert len(dang_dung) == 1, "Vector store chỉ nạp được một văn bản nguồn"
    assert dang_dung[0]["ma"] == "LUAT-116-2025"


# ---------------------------------------------------------------------------
# Phát hiện bản cập nhật — lý do vault tồn tại
# ---------------------------------------------------------------------------
def test_phat_hien_duoc_du_thao_moi_hon():
    kq = vault.tim_ban_cap_nhat()
    assert kq["co_cap_nhat"] is True
    assert any(t["ma"] == "DUTHAO-ND-ANM-2026" for t in kq["cap_nhat"])


def test_khong_coi_van_ban_cu_hon_la_cap_nhat():
    """Luật 24/2018 liên quan tới 116/2025 nhưng CŨ HƠN — không phải bản cập nhật."""
    kq = vault.tim_ban_cap_nhat("LUAT-116-2025")
    ma_cap_nhat = {t["ma"] for t in kq["cap_nhat"]}
    assert "LUAT-24-2018" not in ma_cap_nhat
    assert "LUAT-86-2015" not in ma_cap_nhat


def test_canh_bao_neu_van_ban_chua_co_hieu_luc():
    """Luật 116/2025 chỉ có hiệu lực từ 01/7/2026 — người dùng phải được nhắc."""
    canh_bao = " ".join(vault.tim_ban_cap_nhat("LUAT-116-2025")["canh_bao"])
    assert "2026-07-01" in canh_bao


def test_ma_sai_khong_nem_loi_ma_tra_canh_bao():
    """Agent có thể truyền mã bịa. Vault phải trả lời tử tế, không được nổ."""
    kq = vault.tim_ban_cap_nhat("MA-BIA-RA")
    assert kq["co_cap_nhat"] is False
    assert kq["goc"] is None
    assert "Không tìm thấy" in kq["canh_bao"][0]


def test_van_ban_khong_lien_quan_thi_khong_bao_cap_nhat():
    kq = vault.tim_ban_cap_nhat("LUAT-86-2015")
    assert all(t["ma"] != "DUTHAO-ND-ANM-2026" for t in kq["cap_nhat"])


# ---------------------------------------------------------------------------
# Tầng công cụ — cái mà agent thực sự gọi
# ---------------------------------------------------------------------------
def test_tool_tra_ve_chuoi_doc_duoc():
    from rag_qa.tools import kiem_tra_cap_nhat

    ket_qua = kiem_tra_cap_nhat.invoke({"ma_tai_lieu": ""})
    assert "Dự thảo" in ket_qua
    assert isinstance(ket_qua, str)


def test_tool_khong_nem_loi_khi_vault_hong(monkeypatch):
    """Kho tài liệu hỏng thì agent phải nhận thông báo, không phải stack trace."""
    from rag_qa import tools

    def vault_hong(*a, **k):
        raise vault.VaultError("đĩa hỏng")

    monkeypatch.setattr(vault, "tim_ban_cap_nhat", vault_hong)
    assert "Không đọc được kho tài liệu" in tools.kiem_tra_cap_nhat.invoke(
        {"ma_tai_lieu": ""}
    )
