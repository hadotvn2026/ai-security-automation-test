"""Vault — kho văn bản pháp luật nội bộ (GIẢ LẬP).

Vì sao ứng dụng cần thành phần này?

Một hệ RAG chỉ biết đúng những gì đã được nạp vào vector store. Nó **không có
cách nào biết** tài liệu nguồn đã có bản mới hơn. Với văn bản pháp luật, đó là
lỗ hổng nghiêm trọng: trả lời đúng theo bản cũ vẫn là trả lời sai.

Vault mô phỏng kho tài liệu của doanh nghiệp: mỗi văn bản có phiên bản, ngày ban
hành, trạng thái hiệu lực, và biết văn bản nào thay thế văn bản nào. Agent tra
cứu vault để cảnh báo người dùng khi có bản cập nhật.

Toàn bộ dữ liệu trong `data/vault/manifest.json` là **bịa**, dùng để mô phỏng
tình huống. Không phải dữ liệu pháp lý thật.

Module này KHÔNG gọi LLM và KHÔNG gọi mạng — mọi hàm đều tất định. Đó là chủ ý:
khi test agent, thứ duy nhất không tất định nên là quyết định của agent.
"""

from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from rag_qa import config

__all__ = [
    "doc_manifest",
    "lay_tai_lieu",
    "tai_lieu_dang_dung",
    "tim_ban_cap_nhat",
    "danh_sach_tai_lieu",
    "VaultError",
]

MANIFEST_PATH = config.DATA_DIR / "vault" / "manifest.json"

NHAN_TRANG_THAI = {
    "hieu_luc": "đang có hiệu lực",
    "sap_hieu_luc": "sắp có hiệu lực",
    "sap_het_hieu_luc": "sắp hết hiệu lực",
    "het_hieu_luc": "đã hết hiệu lực",
    "du_thao": "dự thảo, chưa ban hành",
}


class VaultError(RuntimeError):
    """Vault không đọc được hoặc sai định dạng."""


@lru_cache(maxsize=1)
def doc_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Đọc manifest của vault. Cache lại vì file không đổi trong một phiên chạy."""
    p = Path(path or MANIFEST_PATH)
    if not p.exists():
        raise VaultError(f"Không tìm thấy vault: {p}")
    try:
        du_lieu = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VaultError(f"Vault sai định dạng JSON: {exc}") from exc
    if "tai_lieu" not in du_lieu:
        raise VaultError("Vault thiếu khoá 'tai_lieu'")
    return du_lieu


def danh_sach_tai_lieu() -> list[dict[str, Any]]:
    return list(doc_manifest()["tai_lieu"])


def lay_tai_lieu(ma: str) -> dict[str, Any] | None:
    """Tìm một văn bản theo mã. Trả về None nếu không có."""
    ma = (ma or "").strip().upper()
    for t in danh_sach_tai_lieu():
        if t["ma"].upper() == ma:
            return t
    return None


def tai_lieu_dang_dung() -> dict[str, Any] | None:
    """Văn bản mà vector store của ứng dụng đang dùng."""
    for t in danh_sach_tai_lieu():
        if t.get("dang_dung_trong_rag"):
            return t
    return None


def _ngay(chuoi: str | None) -> date | None:
    if not chuoi:
        return None
    try:
        return date.fromisoformat(chuoi)
    except ValueError:
        return None


def tim_ban_cap_nhat(ma: str | None = None) -> dict[str, Any]:
    """Kiểm tra văn bản `ma` có bản cập nhật nào trong vault không.

    Bỏ trống `ma` để kiểm tra chính văn bản mà ứng dụng đang dùng.

    Trả về dict gồm:
        co_cap_nhat  — True nếu tìm thấy văn bản liên quan mới hơn
        goc          — thông tin văn bản được kiểm tra
        cap_nhat     — danh sách văn bản liên quan mới hơn
        canh_bao     — danh sách cảnh báo dạng chữ, sẵn sàng đưa vào câu trả lời
    """
    goc = lay_tai_lieu(ma) if ma else tai_lieu_dang_dung()
    if goc is None:
        return {
            "co_cap_nhat": False,
            "goc": None,
            "cap_nhat": [],
            "canh_bao": [f"Không tìm thấy văn bản có mã '{ma}' trong vault."],
        }

    ngay_goc = _ngay(goc.get("ngay_ban_hanh"))
    lien_quan = []
    for t in danh_sach_tai_lieu():
        if t["ma"] == goc["ma"]:
            continue
        noi_voi_goc = (
            t.get("lien_quan_den") == goc["ma"]
            or t.get("bi_thay_the_boi") == goc["ma"]
            or goc.get("bi_thay_the_boi") == t["ma"]
        )
        if not noi_voi_goc:
            continue
        ngay_t = _ngay(t.get("ngay_ban_hanh"))
        if ngay_goc and ngay_t and ngay_t > ngay_goc:
            lien_quan.append(t)

    canh_bao = []
    for t in lien_quan:
        nhan = NHAN_TRANG_THAI.get(t.get("trang_thai", ""), t.get("trang_thai", ""))
        canh_bao.append(
            f"Có văn bản mới hơn: {t['ten']} ({t['so_hieu']}), "
            f"ban hành {t['ngay_ban_hanh']}, {nhan}."
        )
    if goc.get("trang_thai") == "sap_hieu_luc" and goc.get("ngay_hieu_luc"):
        canh_bao.append(
            f"Lưu ý: {goc['ten']} ({goc['so_hieu']}) chỉ có hiệu lực từ "
            f"{goc['ngay_hieu_luc']}."
        )

    return {
        "co_cap_nhat": bool(lien_quan),
        "goc": goc,
        "cap_nhat": lien_quan,
        "canh_bao": canh_bao,
    }
