"""Quản lý trí nhớ tự sửa của trợ lý pháp lý.

    uv run python scripts/tri_nho.py khoi-tao     # dựng kho từ bộ câu hỏi vàng
    uv run python scripts/tri_nho.py kiem-tra     # phát hiện khẳng định lạc hậu (không gọi model)
    uv run python scripts/tri_nho.py doi-chieu    # chạy vòng đối chiếu (có gọi model)
    uv run python scripts/tri_nho.py xem          # xem kho

Mã thoát của `kiem-tra`:
    0  mọi bằng chứng còn nguyên vẹn
    1  có khẳng định lạc hậu  -> dùng làm cổng chặn CI sau mỗi lần nạp lại tài liệu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_qa import tri_nho  # noqa: E402

GOLDEN = ROOT / "tests" / "data" / "rag_test_cases.json"


def khoi_tao() -> int:
    """Dựng kho khẳng định từ bộ câu hỏi vàng.

    Bộ câu hỏi vàng ĐÃ LÀ một kho khẳng định — chỉ thiếu neo bằng chứng. Mỗi
    case có `user_input` (câu hỏi), `reference` (khẳng định do người viết) và
    `dieu` (bằng chứng). Việc còn lại là ghi vân tay của Điều đó.
    """
    cases = json.loads(GOLDEN.read_text(encoding="utf-8"))["cases"]
    kho: list[tri_nho.KhangDinh] = []
    bo_qua = 0

    for c in cases:
        if c.get("nhom") == "bay" or not c.get("dieu"):
            bo_qua += 1
            continue
        vt = tri_nho.van_tay_dieu(c["dieu"])
        if vt is None:
            print(f"  ! Điều {c['dieu']} không có trong văn bản — bỏ qua {c['id']}")
            bo_qua += 1
            continue
        kho.append(tri_nho.KhangDinh(
            ma=c["id"],
            cau_hoi=c["user_input"],
            noi_dung=c["reference"],
            dieu=c["dieu"],
            van_tay_bang_chung=vt,
            trang_thai="xac_thuc",
            xac_thuc_luc=tri_nho.datetime.now(tri_nho.timezone.utc).isoformat(),
        ))

    duong_dan = tri_nho.ghi_kho(kho)
    print(f"Đã dựng {len(kho)} khẳng định (bỏ qua {bo_qua} case không neo được bằng chứng)")
    print(f"  {duong_dan}")
    for k in kho:
        print(f"  {k.ma:24} Điều {k.dieu:<3} {k.van_tay_bang_chung}")
    return 0


def kiem_tra() -> int:
    """Phát hiện khẳng định lạc hậu. KHÔNG gọi model — chạy trong mili-giây."""
    kho = tri_nho.doc_kho()
    if not kho:
        print("Kho rỗng. Chạy: uv run python scripts/tri_nho.py khoi-tao")
        return 0

    lac_hau = tri_nho.phat_hien_lac_hau(kho)
    print(f"Kiểm tra {len(kho)} khẳng định…\n")
    if not lac_hau:
        print("✅ Mọi bằng chứng còn nguyên vẹn.")
        print("   (Điều này KHÔNG có nghĩa mọi khẳng định đều đúng — chỉ nghĩa là")
        print("    văn bản nguồn chưa đổi kể từ lần xác thực cuối.)")
        return 0

    print(f"⚠️  {len(lac_hau)} khẳng định cần xác thực lại:\n")
    for k, ly_do in lac_hau:
        print(f"  🔶 {k.ma}")
        print(f"     {ly_do}")
        print(f"     Khẳng định: {k.noi_dung[:100]}")
    print(f"\nChạy tiếp: uv run python scripts/tri_nho.py doi-chieu")
    return 1


def doi_chieu() -> int:
    """Vòng đối chiếu — có gọi model cho các khẳng định mà bằng chứng chưa đổi."""
    from rag_qa import rag_graph

    kho = tri_nho.doc_kho()
    if not kho:
        print("Kho rỗng. Chạy `khoi-tao` trước.")
        return 0

    print(f"Đối chiếu {len(kho)} khẳng định (có gọi model)…\n")
    kq = tri_nho.doi_chieu(kho, tra_loi_lai=rag_graph.answer)
    tri_nho.ghi_kho(kho)

    print(f"  ✅ tự làm mới          : {len(kq['lam_moi'])}")
    for m in kq["lam_moi"]:
        print(f"       {m}")
    print(f"  🔶 hệ thống trả lời lệch: {len(kq['he_thong_lech'])}")
    for m in kq["he_thong_lech"]:
        print(f"       {m}")
    print(f"  🔴 cần người rà soát    : {len(kq['can_nguoi_ra_soat'])}")
    for m in kq["can_nguoi_ra_soat"]:
        print(f"       {m}")

    if kq["he_thong_lech"]:
        print("\nBằng chứng KHÔNG đổi mà câu trả lời đổi → nghi ngờ prompt/model/chunking,")
        print("không phải nghi ngờ văn bản luật.")
    if kq["can_nguoi_ra_soat"]:
        print("\nBằng chứng ĐÃ đổi → hệ thống KHÔNG tự sửa khẳng định pháp lý.")
        print("Người phải đọc văn bản mới và cập nhật bằng tay.")
    return 1 if (kq["he_thong_lech"] or kq["can_nguoi_ra_soat"]) else 0


def xem() -> int:
    kho = tri_nho.doc_kho()
    if not kho:
        print("Kho rỗng.")
        return 0
    for k in kho:
        bieu = {"xac_thuc": "✅", "nghi_ngo": "🔶",
                "can_nguoi_ra_soat": "🔴", "sai": "❌"}.get(k.trang_thai, "·")
        print(f"{bieu} {k.ma:24} Điều {k.dieu:<3} {k.nhan()}")
        print(f"   {k.noi_dung[:110]}")
        if k.ghi_chu:
            print(f"   ↳ {k.ghi_chu[:110]}")
    print("\n" + " · ".join(f"{v} {tri_nho.NHAN_TRANG_THAI[k]}"
                            for k, v in tri_nho.thong_ke(kho).items() if v))
    return 0


LENH = {"khoi-tao": khoi_tao, "kiem-tra": kiem_tra, "doi-chieu": doi_chieu, "xem": xem}


def main() -> int:
    parser = argparse.ArgumentParser(description="Trí nhớ tự sửa")
    parser.add_argument("lenh", choices=sorted(LENH))
    return LENH[parser.parse_args().lenh]()


if __name__ == "__main__":
    raise SystemExit(main())
