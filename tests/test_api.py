"""Test backend HTTP — mock agent, không gọi model.

Đây là ví dụ rõ nhất của nguyên tắc "mock đúng ranh giới" trong toàn khoá:

    Test API kiểm tra tầng HTTP: định tuyến, kiểm tra đầu vào, hình dạng JSON
    trả về, mã lỗi. Nó KHÔNG kiểm tra chất lượng câu trả lời — việc đó thuộc về
    test đánh giá (Bài 9-11).

Vì thế ở đây agent bị mock hoàn toàn, và bộ test chạy trong mili-giây.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag_qa import agent_graph, api, rag_graph, tri_nho, vault

pytestmark = pytest.mark.unit


@pytest.fixture
def client() -> TestClient:
    return TestClient(api.app)


@pytest.fixture
def agent_gia(monkeypatch):
    """Thay agent bằng một hàm tất định, trả về đúng cấu trúc mà API mong đợi."""

    def chay(cau_hoi, lich_su=None):
        return {
            "cau_tra_loi": "Luật có hiệu lực từ ngày 01/7/2026.",
            "quy_dao": ["tra_cuu_van_ban", "kiem_tra_cap_nhat"],
            "scratchpad": [
                {"tool": "tra_cuu_van_ban", "args": {"cau_hoi": "hiệu lực"},
                 "ket_qua": "[Điều 44] Luật này có hiệu lực từ 01/7/2026."},
                {"tool": "kiem_tra_cap_nhat", "args": {"ma_tai_lieu": ""},
                 "ket_qua": "Có văn bản mới hơn: Dự thảo Nghị định..."},
            ],
            "so_buoc": 3,
            "messages": [],
        }

    monkeypatch.setattr(agent_graph, "chay_agent", chay)


# ---------------------------------------------------------------------------
# Giao diện và điểm cuối tĩnh
# ---------------------------------------------------------------------------
def test_trang_chu_tra_ve_giao_dien(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Trợ lý pháp lý" in r.text


def test_suc_khoe_bao_dung_model_va_so_chunk(client):
    d = client.get("/api/suc-khoe").json()
    assert d["model_ung_dung"]
    assert d["top_k"] >= 1


def test_danh_sach_tai_lieu_dung_hinh_dang(client):
    ds = client.get("/api/tai-lieu").json()
    assert len(ds) >= 4
    assert {"ma", "ten", "so_hieu", "trang_thai"} <= set(ds[0])


def test_diem_cuoi_cap_nhat_bao_co_ban_moi(client):
    assert client.get("/api/cap-nhat").json()["co_cap_nhat"] is True


def test_vault_hong_thi_tra_ve_500_khong_phai_stack_trace(client, monkeypatch):
    def hong():
        raise vault.VaultError("đĩa hỏng")

    monkeypatch.setattr(vault, "danh_sach_tai_lieu", hong)
    r = client.get("/api/tai-lieu")
    assert r.status_code == 500
    assert "đĩa hỏng" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Kiểm tra đầu vào — tầng HTTP phải chặn trước khi tới model
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("than", [{}, {"noi_dung": ""}, {"noi_dung": "x" * 2001}])
def test_dau_vao_khong_hop_le_thi_tra_422(client, than):
    assert client.post("/api/hoi", json=than).status_code == 422


def test_che_do_la_bay_thi_tra_422(client):
    r = client.post("/api/hoi", json={"noi_dung": "hỏi gì đó", "che_do": "linh_tinh"})
    assert r.status_code == 422


def test_cau_hoi_rong_khong_bao_gio_toi_duoc_agent(client, monkeypatch):
    """Chặn ở tầng HTTP là chặn rẻ nhất — đừng để rác đi tới model."""

    def khong_duoc_goi(*a, **k):
        raise AssertionError("Agent không được gọi với đầu vào rỗng")

    monkeypatch.setattr(agent_graph, "chay_agent", khong_duoc_goi)
    assert client.post("/api/hoi", json={"noi_dung": ""}).status_code == 422


# ---------------------------------------------------------------------------
# Chế độ agent
# ---------------------------------------------------------------------------
def test_che_do_agent_tra_ve_quy_dao(client, agent_gia):
    d = client.post("/api/hoi", json={"noi_dung": "Hiệu lực khi nào?"}).json()
    assert d["che_do"] == "agent"
    assert d["quy_dao"] == ["tra_cuu_van_ban", "kiem_tra_cap_nhat"]
    assert d["so_buoc"] == 3


def test_api_tra_ve_ca_ngu_canh_de_nguoi_dung_kiem_chung(client, agent_gia):
    """Người dùng phải xem được đoạn tài liệu, không phải tin một hộp đen."""
    d = client.post("/api/hoi", json={"noi_dung": "Hiệu lực khi nào?"}).json()
    assert len(d["ngu_canh"]) == 1
    assert "Điều 44" in d["ngu_canh"][0]


def test_ngu_canh_chi_lay_tu_tool_tra_cuu(client, agent_gia):
    """Kết quả của kiem_tra_cap_nhat KHÔNG phải ngữ cảnh tài liệu."""
    d = client.post("/api/hoi", json={"noi_dung": "Hiệu lực khi nào?"}).json()
    assert all("Dự thảo Nghị định" not in c for c in d["ngu_canh"])
    assert len(d["buoc"]) == 2


def test_co_do_thoi_gian_tra_loi(client, agent_gia):
    d = client.post("/api/hoi", json={"noi_dung": "Hiệu lực khi nào?"}).json()
    assert d["thoi_gian_ms"] >= 0


# ---------------------------------------------------------------------------
# Chế độ RAG — để so sánh trực tiếp với agent
# ---------------------------------------------------------------------------
def test_che_do_rag_khong_goi_agent(client, monkeypatch):
    monkeypatch.setattr(
        rag_graph, "answer_with_contexts",
        lambda q: ("Trả lời từ pipeline.", ["[Điều 44] ..."]),
    )

    def khong_duoc_goi(*a, **k):
        raise AssertionError("Chế độ rag không được gọi agent")

    monkeypatch.setattr(agent_graph, "chay_agent", khong_duoc_goi)

    d = client.post("/api/hoi", json={"noi_dung": "q", "che_do": "rag"}).json()
    assert d["che_do"] == "rag"
    assert d["quy_dao"] == ["retrieve", "generate"]
    assert d["buoc"] == []


# ---------------------------------------------------------------------------
# Trí nhớ tự sửa
# ---------------------------------------------------------------------------
def test_diem_cuoi_tri_nho_dung_hinh_dang(client):
    d = client.get("/api/tri-nho").json()
    assert {"tong", "thong_ke", "lac_hau"} <= set(d)
    assert isinstance(d["lac_hau"], list)


def test_cau_tra_loi_kem_canh_bao_khi_bang_chung_nghi_ngo(client, agent_gia, monkeypatch):
    """Người dùng phải được cảnh báo NGAY trong câu trả lời, không phải chờ script."""
    kho = [tri_nho.KhangDinh(ma="x", cau_hoi="q", noi_dung="n", dieu=44,
                             van_tay_bang_chung="cu", trang_thai="can_nguoi_ra_soat")]
    monkeypatch.setattr(tri_nho, "doc_kho", lambda *a, **k: kho)

    d = client.post("/api/hoi", json={"noi_dung": "Hiệu lực khi nào?"}).json()
    assert any("Điều 44" in c for c in d["canh_bao"])


def test_tri_nho_hong_khong_lam_hong_cau_tra_loi(client, agent_gia, monkeypatch):
    def no(*a, **k):
        raise RuntimeError("kho hỏng")

    monkeypatch.setattr(tri_nho, "canh_bao_cho_ngu_canh", no)
    d = client.post("/api/hoi", json={"noi_dung": "Hiệu lực khi nào?"}).json()
    assert d["cau_tra_loi"]
    assert d["canh_bao"] == []
