"""OCR file PDF scan thành văn bản — CHỈ DÙNG KHI SOẠN TÀI LIỆU, không phải bước của học viên.

Luật 116/2025/QH15 tải từ cổng chính phủ là bản SCAN: 37 trang, 0 ký tự trích
xuất được bằng cách thông thường. Phải OCR.

Script này dùng Vision framework của macOS (chất lượng tiếng Việt tốt, chạy
offline, không cần cài gói hệ thống). Vì thế nó CHỈ CHẠY TRÊN macOS.

Học viên KHÔNG cần chạy script này — kết quả đã được commit sẵn thành
data/luat-116-2025-an-ninh-mang.docx.

    uv run --with pypdfium2 --with pyobjc-framework-Vision --with pyobjc-framework-Quartz \
        python scripts/ocr_pdf.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "luat-116-2025-an-ninh-mang.pdf"
OUT_TXT = ROOT / "data" / "luat-116-2025-an-ninh-mang.txt"
TMP = ROOT / "data" / ".ocr-tmp"
CACHE = TMP / "ocr-tho.json"   # OCR thô, để chỉnh bước làm sạch mà không OCR lại


def render_pages() -> list[Path]:
    import pypdfium2 as pdfium

    TMP.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(str(PDF))
    paths = []
    for i in range(len(doc)):
        p = TMP / f"trang{i + 1:03d}.png"
        if not p.exists():
            doc[i].render(scale=2).to_pil().save(p)
        paths.append(p)
    return paths


def ocr_image(path: Path) -> str:
    import Quartz
    import Vision
    from Foundation import NSURL

    url = NSURL.fileURLWithPath_(str(path))
    src = Quartz.CGImageSourceCreateWithURL(url, None)
    img = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(0)  # accurate
    req.setRecognitionLanguages_(["vi-VN", "en-US"])
    req.setUsesLanguageCorrection_(True)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(img, None)
    handler.performRequests_error_([req], None)
    dong = []
    for obs in req.results() or []:
        cand = obs.topCandidates_(1)
        if cand:
            dong.append(cand[0].string())
    return "\n".join(dong)


# --- Chuẩn hoá TIÊU ĐỀ trước tiên ---
# OCR đọc "Điêu 24" (mất dấu ề) làm cả Điều 24 biến mất khỏi cấu trúc. Tiêu đề
# sai một dấu là mất nguyên một điều luật khỏi chỉ mục — nên phải sửa trước.
CHUAN_HOA_TIEU_DE = [
    (r"^\s*Đi[êềéẽẻ]u\s+(\d+)\s*[.:]", r"Điều \1."),
    (r"^\s*Ch[uưừứ][oơöóờ]ng\s+([IVXLC]+)\b", r"Chương \1"),
    (r"^\s*M[uụ]c\s+(\d+)\b", r"Mục \1"),
]

# OCR hay nhầm dấu hỏi/ngã và ô/ố trên bản scan. Sửa các lỗi lặp lại nhiều nhất.
SUA_LOI = [
    (r"\bsửa đối\b", "sửa đổi"),
    (r"\bbố sung\b", "bổ sung"),
    (r"\btô chức\b", "tổ chức"),
    (r"\btố chức\b", "tổ chức"),
    (r"\bhô trợ\b", "hỗ trợ"),
    (r"\bquôc\b", "quốc"),
    (r"\bnhiêm vụ\b", "nhiệm vụ"),
    (r"\bcông thông tin\b", "cổng thông tin"),
]


def lam_sach(text: str) -> str:
    dong_chuan = []
    for d in text.split("\n"):
        for mau, thay in CHUAN_HOA_TIEU_DE:
            d = re.sub(mau, thay, d)
        dong_chuan.append(d)
    text = "\n".join(dong_chuan)

    for mau, thay in SUA_LOI:
        text = re.sub(mau, thay, text, flags=re.IGNORECASE)
    # Nối dòng bị OCR ngắt giữa câu: dòng không kết thúc bằng dấu câu và dòng
    # sau không bắt đầu bằng đánh số điều/khoản.
    dong = [d.strip() for d in text.split("\n")]
    ra: list[str] = []
    for d in dong:
        if not d:
            ra.append("")
            continue
        moi_muc = re.match(r"^(Chương|Điều|Mục)\s|^\d+\.\s|^[a-zđ]\)\s", d)
        if ra and ra[-1] and not moi_muc and not re.search(r"[.;:]$", ra[-1]):
            ra[-1] = f"{ra[-1]} {d}"
        else:
            ra.append(d)
    return "\n".join(ra)


def main() -> int:
    if not PDF.exists():
        print(f"Không thấy {PDF}")
        return 1
    if sys.platform != "darwin":
        print("Script này chỉ chạy trên macOS (dùng Vision framework).")
        print("Kết quả đã commit sẵn: data/luat-116-2025-an-ninh-mang.docx")
        return 1

    trang = render_pages()
    print(f"Đã render {len(trang)} trang")

    if CACHE.exists():
        tho = json.loads(CACHE.read_text(encoding="utf-8"))
        print(f"Dùng lại OCR thô đã cache ({len(tho)} trang)")
    else:
        tho = []
        for i, p in enumerate(trang, 1):
            tho.append(ocr_image(p))
            print(f"  OCR trang {i:2d}/{len(trang)}: {len(tho[-1]):5d} ký tự", flush=True)
        CACHE.write_text(json.dumps(tho, ensure_ascii=False), encoding="utf-8")

    phan = [lam_sach(t) for t in tho]
    OUT_TXT.write_text("\n\n".join(phan), encoding="utf-8")
    print(f"\nĐã ghi {OUT_TXT} ({OUT_TXT.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
