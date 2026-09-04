"""Bốn công cụ của agent tra cứu pháp luật.

Bốn công cụ, bốn quyết định mà agent phải phân biệt:

    tra_cuu_van_ban    -> câu hỏi về nội dung luật
    kiem_tra_cap_nhat  -> tài liệu còn mới không (điểm mù của RAG)
    tinh_thoi_han      -> cần tính ngày đến hạn
    chuyen_chuyen_gia  -> ngoài phạm vi luật, phải chuyển người thật

Càng nhiều công cụ, quỹ đạo càng dễ sai — và đó chính là thứ Bài 12 đo.

Cả ba đều TẤT ĐỊNH — không có LLM nào bên trong. Đó là chủ ý: khi test agent,
bạn muốn phần duy nhất không tất định là *quyết định của agent*, không phải
kết quả của công cụ.
"""

from __future__ import annotations

import re
from datetime import date

from langchain_core.tools import tool

from rag_qa import retriever, vault

__all__ = [
    "tra_cuu_van_ban",
    "kiem_tra_cap_nhat",
    "tinh_thoi_han",
    "chuyen_chuyen_gia",
    "TAT_CA_TOOL",
    "TEN_TOOL",
]

_NGAY = re.compile(r"(\d{1,2})\s*[/-]\s*(\d{1,2})\s*[/-]\s*(\d{4})")


@tool
def tra_cuu_van_ban(cau_hoi: str) -> str:
    """Tra cứu nội dung Luật An ninh mạng số 116/2025/QH15.

    Dùng công cụ này cho MỌI câu hỏi về nội dung luật: điều khoản, hiệu lực,
    cấp độ hệ thống thông tin, trách nhiệm của doanh nghiệp, hành vi bị nghiêm
    cấm, điều khoản chuyển tiếp.
    """
    doan = retriever.retrieve(cau_hoi)
    if not doan:
        return "Không tìm thấy điều khoản liên quan."
    return "\n\n---\n\n".join(doan)


@tool
def kiem_tra_cap_nhat(ma_tai_lieu: str = "") -> str:
    """Kiểm tra văn bản đang dùng có bản cập nhật mới hơn trong kho tài liệu không.

    Dùng công cụ này khi người dùng hỏi về hiệu lực, về việc văn bản còn mới
    không, hoặc trước khi đưa ra kết luận quan trọng dựa trên nội dung luật.

    Bỏ trống `ma_tai_lieu` để kiểm tra chính văn bản mà hệ thống đang tra cứu.

    ĐÂY LÀ ĐIỂM MÙ CỦA MỌI HỆ RAG: vector store chỉ biết những gì đã được nạp.
    Nó không tự biết tài liệu nguồn đã có bản mới. Với văn bản pháp luật, trả
    lời đúng theo bản cũ vẫn là trả lời sai.
    """
    try:
        kq = vault.tim_ban_cap_nhat(ma_tai_lieu or None)
    except vault.VaultError as exc:
        return f"Không đọc được kho tài liệu: {exc}"
    if not kq["canh_bao"]:
        return "Không có bản cập nhật nào liên quan."
    return " ".join(kq["canh_bao"])


@tool
def tinh_thoi_han(ngay_bat_dau: str, so_thang: int) -> str:
    """Tính ngày đến hạn = ngày bắt đầu cộng thêm số tháng.

    `ngay_bat_dau` theo định dạng dd/mm/yyyy, ví dụ "01/07/2026".
    Dùng khi cần tính hạn chuyển tiếp, hạn tuân thủ, hạn nộp hồ sơ.
    KHÔNG dùng để tra cứu nội dung luật.
    """
    khop = _NGAY.search(ngay_bat_dau or "")
    if not khop:
        return f"Không đọc được ngày '{ngay_bat_dau}'. Cần định dạng dd/mm/yyyy."
    try:
        so_thang = int(so_thang)
    except (TypeError, ValueError):
        return f"Số tháng không hợp lệ: {so_thang!r}"

    ngay, thang, nam = (int(x) for x in khop.groups())
    tong = (nam * 12 + (thang - 1)) + so_thang
    nam_moi, thang_moi = divmod(tong, 12)
    thang_moi += 1
    # Lùi ngày nếu tháng đích không có ngày đó (ví dụ 31/01 + 1 tháng)
    for lui in range(4):
        try:
            ket_qua = date(nam_moi, thang_moi, ngay - lui)
        except ValueError:
            continue
        return ket_qua.strftime("%d/%m/%Y")
    return "Không tính được ngày đến hạn."


@tool
def chuyen_chuyen_gia(ly_do: str) -> str:
    """Chuyển câu hỏi cho chuyên gia pháp lý của công ty.

    Dùng khi: câu hỏi cần tư vấn pháp lý cho một tình huống cụ thể, hỏi về mức
    xử phạt hoặc trách nhiệm hình sự (không thuộc phạm vi Luật An ninh mạng),
    hoặc người dùng yêu cầu ý kiến pháp lý chính thức.
    """
    return f"Đã chuyển cho chuyên gia pháp lý. Lý do: {ly_do}"


TAT_CA_TOOL = [tra_cuu_van_ban, kiem_tra_cap_nhat, tinh_thoi_han, chuyen_chuyen_gia]
TEN_TOOL = {t.name for t in TAT_CA_TOOL}
