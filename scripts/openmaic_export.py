"""Biến sổ tay Markdown trong docs/sessions/ thành lớp học OpenMAIC (.maic.zip).

OpenMAIC (https://github.com/THU-MAIC/OpenMAIC) là lớp học đa tác tử: giảng viên
AI trình chiếu, thuyết minh, và thảo luận với học viên AI. Nó có sẵn đường sinh
bài bằng LLM, nhưng đường đó VIẾT LẠI nội dung của bạn. Script này đi đường khác:
đọc thẳng giáo án đã soạn và dựng đúng nội dung đó thành lớp học — không model,
không API key, chạy lại bao nhiêu lần cũng ra file y hệt.

Quy ước chuyển đổi (đúng triết lý slide của OpenMAIC — slide để NHÌN, lời giảng
để NGHE):

    ## Tiêu đề mục       -> một scene slide
    - gạch đầu dòng      -> bullet trên slide (tối đa 5, mỗi dòng <= 90 ký tự)
    đoạn văn xuôi        -> lời thuyết minh (speech action), KHÔNG lên slide
    ```code```           -> scene riêng, dùng code element
    | bảng |             -> scene riêng, dùng table element
    > 💡 / > ⚠️          -> hộp lưu ý màu trên slide
    ## Bài tập           -> câu tự luận trong scene quiz (judge của lớp học tự chấm)

Nội dung giảng dạy dành riêng cho lớp học nằm ở docs/95-ngan-hang-cau-hoi.md:
bốn câu trắc nghiệm và hai chủ đề thảo luận cho mỗi bài. Trắc nghiệm vào scene
quiz (chấm được ngay, không cần judge); chủ đề thảo luận thành hành động
`discussion` để giảng viên AI và hai học viên AI tranh luận — một ở slide bìa,
một ở slide kiểm tra cuối bài.

Dùng:

    uv run python scripts/openmaic_export.py                 # cả 12 bài
    uv run python scripts/openmaic_export.py --bai 1 3       # chọn bài
    uv run python scripts/openmaic_export.py --gop           # gộp 12 bài, 1 file

File .maic.zip nằm ở reports/openmaic/. Nạp vào lớp học: mở OpenMAIC, chọn
"Import Classroom" rồi chỉ tới file .zip.
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
import unicodedata
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NGUON = ROOT / "docs" / "sessions"
NGAN_HANG = ROOT / "docs" / "95-ngan-hang-cau-hoi.md"
DICH = ROOT / "reports" / "openmaic"

# Khung hình chuẩn của OpenMAIC: 1000 x 562.5 (16:9), lề an toàn 50px.
KHUNG_W = 1000.0
KHUNG_H = 562.5
LE = 60.0

MAU_NEN = "#ffffff"
MAU_CHU = "#1f2937"
MAU_NHAN = "#2563eb"
MAU_PHU = "#6b7280"
MAU_HOP_LUU_Y = "#fef3c7"
MAU_HOP_CANH_BAO = "#fee2e2"

# Mục điều hướng của sổ tay — không phải nội dung giảng, bỏ khỏi lớp học.
MUC_BO_QUA = {
    "Nội dung trang này",
    "Chọn bước tiếp theo",
    "Đọc tiếp",
    "Trang liên quan",
    "Cấu trúc dự án sau bài này",
}
MUC_MUC_LUC = "Nội dung trang này"
MUC_BAI_TAP = "Bài tập"
MUC_KIEM_TRA = "Kiểm tra trước khi sang bài sau"

NGON_NGU_CODE = {
    "bash": "bash",
    "sh": "bash",
    "shell": "bash",
    "console": "bash",
    "powershell": "powershell",
    "python": "python",
    "py": "python",
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
    "toml": "toml",
    "text": "text",
    "": "text",
}


# --------------------------------------------------------------------------- #
# Đọc Markdown
# --------------------------------------------------------------------------- #


@dataclass
class Khoi:
    """Một khối nội dung trong sổ tay: đoạn văn, gạch đầu dòng, code, bảng…"""

    loai: str  # para | bullets | code | table | callout
    van_ban: str = ""
    muc: list[str] = field(default_factory=list)
    ngon_ngu: str = ""
    hang: list[list[str]] = field(default_factory=list)
    canh_bao: bool = False


@dataclass
class Muc:
    tieu_de: str
    khoi: list[Khoi] = field(default_factory=list)


@dataclass
class Bai:
    ma: str
    tieu_de: str
    muc_tieu: str
    thoi_luong: str
    muc: list[Muc]


def _lam_sach(dong: str) -> str:
    """Bỏ cú pháp Markdown inline, giữ lại chữ."""
    dong = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", dong)
    dong = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", dong)
    dong = re.sub(r"<(https?://[^>]+)>", r"\1", dong)
    dong = dong.replace("**", "").replace("`", "")
    dong = re.sub(r"(?<!\w)\*([^*]+)\*(?!\w)", r"\1", dong)
    dong = re.sub(r"\s+", " ", dong)
    return dong.strip()


def _doc_bang(dong: list[str]) -> list[list[str]]:
    hang: list[list[str]] = []
    for d in dong:
        if re.fullmatch(r"\|[\s:|-]+\|", d.strip()):
            continue
        o = [_lam_sach(x) for x in d.strip().strip("|").split("|")]
        hang.append(o)
    return hang


def _tach_khoi(dong: list[str]) -> list[Khoi]:
    khoi: list[Khoi] = []
    i = 0
    while i < len(dong):
        d = dong[i]
        if not d.strip():
            i += 1
            continue

        # code fence
        if d.lstrip().startswith("```"):
            ngon_ngu = d.strip().strip("`").strip().lower()
            i += 1
            than: list[str] = []
            while i < len(dong) and not dong[i].lstrip().startswith("```"):
                than.append(dong[i].rstrip())
                i += 1
            i += 1
            khoi.append(
                Khoi(
                    loai="code",
                    van_ban="\n".join(than),
                    ngon_ngu=NGON_NGU_CODE.get(ngon_ngu, "text"),
                )
            )
            continue

        # bảng
        if d.lstrip().startswith("|"):
            than = []
            while i < len(dong) and dong[i].lstrip().startswith("|"):
                than.append(dong[i])
                i += 1
            khoi.append(Khoi(loai="table", hang=_doc_bang(than)))
            continue

        # trích dẫn / hộp lưu ý
        if d.lstrip().startswith(">"):
            than = []
            while i < len(dong) and (dong[i].lstrip().startswith(">") or (than and dong[i].strip() and not dong[i].lstrip().startswith(("-", "*", "#", "|", "```")))):
                than.append(re.sub(r"^\s*>\s?", "", dong[i]))
                i += 1
            noi_dung = _lam_sach(" ".join(than))
            khoi.append(
                Khoi(
                    loai="callout",
                    van_ban=noi_dung.lstrip("💡⚠️✅🎯⏱ ").strip(),
                    canh_bao="⚠️" in noi_dung,
                )
            )
            continue

        # danh sách
        if re.match(r"^\s*([-*]|\d+\.)\s+", d):
            muc: list[str] = []
            while i < len(dong):
                cur = dong[i]
                if re.match(r"^\s*([-*]|\d+\.)\s+", cur):
                    muc.append(_lam_sach(re.sub(r"^\s*([-*]|\d+\.)\s+", "", cur)))
                    i += 1
                elif cur.startswith(("  ", "\t")) and cur.strip() and muc:
                    muc[-1] = f"{muc[-1]} {_lam_sach(cur)}".strip()
                    i += 1
                elif not cur.strip():
                    # danh sách có thể cách dòng; dừng nếu dòng kế không phải mục
                    ke = dong[i + 1] if i + 1 < len(dong) else ""
                    if re.match(r"^\s*([-*]|\d+\.)\s+", ke):
                        i += 1
                        continue
                    break
                else:
                    break
            khoi.append(Khoi(loai="bullets", muc=[m for m in muc if m]))
            continue

        # tiêu đề con (### …)
        if d.lstrip().startswith("#"):
            khoi.append(Khoi(loai="sub", van_ban=_lam_sach(d.lstrip("#").strip())))
            i += 1
            continue

        # đoạn văn
        moc = i
        than = []
        while i < len(dong) and dong[i].strip() and not dong[i].lstrip().startswith(("```", "|", ">", "#")) and not re.match(r"^\s*([-*]|\d+\.)\s+", dong[i]):
            than.append(dong[i])
            i += 1
        van = _lam_sach(" ".join(than))
        if van and not set(van) <= {"-", "—", "*"}:
            khoi.append(Khoi(loai="para", van_ban=van))
        if i == moc:  # không nuốt được dòng nào: bỏ qua để khỏi lặp vô hạn
            i += 1
    return khoi


def doc_bai(duong_dan: Path) -> Bai:
    dong = duong_dan.read_text(encoding="utf-8").splitlines()

    tieu_de = ""
    for d in dong:
        if d.startswith("# "):
            tieu_de = _lam_sach(d[2:])
            break

    # dòng metadata "> ✅ … · ⏱ 90 phút · 🎯 mục tiêu"
    muc_tieu, thoi_luong = "", ""
    meta = " ".join(
        re.sub(r"^\s*>\s?", "", d) for d in dong[:12] if d.lstrip().startswith(">")
    )
    if "🎯" in meta:
        muc_tieu = _lam_sach(meta.split("🎯", 1)[1].split("·")[0])
    if "⏱" in meta:
        thoi_luong = _lam_sach(meta.split("⏱", 1)[1].split("·")[0])

    muc: list[Muc] = []
    hien_tai: Muc | None = None
    than: list[str] = []
    for d in dong:
        if d.startswith("## "):
            if hien_tai:
                hien_tai.khoi = _tach_khoi(than)
                muc.append(hien_tai)
            hien_tai = Muc(tieu_de=_lam_sach(d[3:]))
            than = []
        elif hien_tai is not None:
            than.append(d)
    if hien_tai:
        hien_tai.khoi = _tach_khoi(than)
        muc.append(hien_tai)

    ma = re.match(r"session-(\d+)", duong_dan.stem)
    return Bai(
        ma=ma.group(1) if ma else duong_dan.stem,
        tieu_de=tieu_de or duong_dan.stem,
        muc_tieu=muc_tieu,
        thoi_luong=thoi_luong or "90 phút",
        muc=muc,
    )


# --------------------------------------------------------------------------- #
# Đọc ngân hàng câu hỏi
# --------------------------------------------------------------------------- #


@dataclass
class TracNghiem:
    cau_hoi: str
    lua_chon: list[tuple[str, str]]  # (nhãn A/B/C/D, nội dung)
    dap_an: list[str]
    giai_thich: str = ""


@dataclass
class ThaoLuan:
    chu_de: str
    goi_y: str


@dataclass
class NoiDungLop:
    trac_nghiem: list[TracNghiem] = field(default_factory=list)
    thao_luan: list[ThaoLuan] = field(default_factory=list)


RE_BAI = re.compile(r"^## Bài (\d+)")
RE_CAU = re.compile(r"^(\d+)\.\s+(.*)$")
RE_LUA_CHON = re.compile(r"^\s+-\s+([A-Z])\.\s+(.*)$")
RE_GIAI_THICH = re.compile(r"^\s+>\s+(.*)$")
RE_THAO_LUAN = re.compile(r"^-\s+\*\*(.+?)\*\*\s+—\s+(.*)$")


def doc_ngan_hang(duong_dan: Path = NGAN_HANG) -> dict[str, NoiDungLop]:
    """Đọc docs/95-ngan-hang-cau-hoi.md. Thiếu file thì trả về rỗng, không chết."""
    if not duong_dan.exists():
        return {}

    ngan_hang: dict[str, NoiDungLop] = {}
    hien_tai: NoiDungLop | None = None
    phan = ""
    cau: TracNghiem | None = None

    def chot() -> None:
        nonlocal cau
        if cau and hien_tai is not None and cau.lua_chon:
            hien_tai.trac_nghiem.append(cau)
        cau = None

    for dong in duong_dan.read_text(encoding="utf-8").splitlines():
        bai = RE_BAI.match(dong)
        if bai:
            chot()
            hien_tai = NoiDungLop()
            ngan_hang[f"{int(bai.group(1)):02d}"] = hien_tai
            phan = ""
            continue
        if hien_tai is None:
            continue
        if dong.startswith("### "):
            chot()
            phan = _lam_sach(dong[4:]).lower()
            continue

        if phan.startswith("trắc nghiệm"):
            m = RE_CAU.match(dong)
            if m:
                chot()
                cau = TracNghiem(cau_hoi=_lam_sach(m.group(2)), lua_chon=[], dap_an=[])
                continue
            m = RE_LUA_CHON.match(dong)
            if m and cau:
                noi_dung = m.group(2)
                dung = "✅" in noi_dung
                cau.lua_chon.append((m.group(1), _lam_sach(noi_dung.replace("✅", ""))))
                if dung:
                    cau.dap_an.append(m.group(1))
                continue
            m = RE_GIAI_THICH.match(dong)
            if m and cau:
                cau.giai_thich = _lam_sach(m.group(1))
                continue
        elif phan.startswith("thảo luận"):
            m = RE_THAO_LUAN.match(dong)
            if m:
                hien_tai.thao_luan.append(
                    ThaoLuan(chu_de=_lam_sach(m.group(1)), goi_y=_lam_sach(m.group(2)))
                )

    chot()
    return ngan_hang


# --------------------------------------------------------------------------- #
# Dựng slide
# --------------------------------------------------------------------------- #

THEME = {
    "backgroundColor": MAU_NEN,
    "themeColors": [MAU_NHAN, "#10b981", "#f59e0b", "#ec4899", "#8b5cf6", "#06b6d4"],
    "fontColor": MAU_CHU,
    "fontName": "Helvetica",
}


class DemId:
    """Sinh id ổn định để chạy lại cho ra file y hệt."""

    def __init__(self) -> None:
        self.dem: dict[str, int] = {}

    def __call__(self, tien_to: str) -> str:
        self.dem[tien_to] = self.dem.get(tien_to, 0) + 1
        return f"{tien_to}_{self.dem[tien_to]:03d}"


def _cao_chu(chuoi_dong: list[str], rong: float, co_chu: int) -> float:
    """Ước lượng chiều cao hộp chữ: text element chừa 10px padding mỗi phía."""
    ky_tu_moi_dong = max(int((rong - 20) / (co_chu * 0.55)), 10)
    so_dong = sum(max(1, -(-len(d) // ky_tu_moi_dong)) for d in chuoi_dong)
    return 20 + so_dong * co_chu * 1.6


def _text(dem: DemId, *, trai, tren, rong, cao, html, mau=MAU_CHU, nen=None) -> dict:
    el = {
        "id": dem("text"),
        "type": "text",
        "left": round(trai, 1),
        "top": round(tren, 1),
        "width": round(rong, 1),
        "height": round(cao, 1),
        "rotate": 0,
        "content": html,
        "defaultFontName": "Helvetica",
        "defaultColor": mau,
        "lineHeight": 1.5,
    }
    if nen:
        el["fill"] = nen
    return el


def _p(chu: str, *, co_chu: int, mau: str, dam: bool = False, can: str = "left") -> str:
    kieu = f"font-size: {co_chu}px; color: {mau}; text-align: {can};"
    if dam:
        kieu += " font-weight: bold;"
    chu = (
        chu.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    return f'<p style="{kieu}">{chu}</p>'


def _slide(dem: DemId, phan_tu: list[dict]) -> dict:
    return {
        "id": dem("slide"),
        "viewportSize": 1000,
        "viewportRatio": 0.5625,
        "theme": THEME,
        "background": {"type": "solid", "color": MAU_NEN},
        "elements": phan_tu,
    }


def _slide_tieu_de(dem: DemId, tieu_de: str, tren: float = 48.0) -> dict:
    return _text(
        dem,
        trai=LE,
        tren=tren,
        rong=KHUNG_W - 2 * LE,
        cao=_cao_chu([tieu_de], KHUNG_W - 2 * LE, 30),
        html=_p(tieu_de, co_chu=30, mau=MAU_CHU, dam=True),
    )


def _hop_luu_y(dem: DemId, khoi: Khoi, tren: float) -> dict:
    chu = textwrap.shorten(khoi.van_ban, width=240, placeholder="…")
    nhan = "⚠️ Cảnh báo" if khoi.canh_bao else "💡 Ghi nhớ"
    cao = _cao_chu([nhan, chu], KHUNG_W - 2 * LE, 17) + 8
    return _text(
        dem,
        trai=LE,
        tren=tren,
        rong=KHUNG_W - 2 * LE,
        cao=min(cao, KHUNG_H - 50 - tren),
        html=_p(nhan, co_chu=15, mau="#92400e", dam=True) + _p(chu, co_chu=16, mau=MAU_CHU),
        nen=MAU_HOP_CANH_BAO if khoi.canh_bao else MAU_HOP_LUU_Y,
    )


def _code(dem: DemId, khoi: Khoi, tren: float) -> dict:
    dong = khoi.van_ban.splitlines()[:16]
    co_chu = 15 if max((len(d) for d in dong), default=0) <= 68 else 13
    # 38px là thanh tiêu đề của khung code, cộng thêm đệm trên dưới.
    cao = min(38 + len(dong) * co_chu * 1.62 + 20, KHUNG_H - 50 - tren)
    return {
        "id": dem("code"),
        "type": "code",
        "left": LE,
        "top": round(tren, 1),
        "width": KHUNG_W - 2 * LE,
        "height": round(cao, 1),
        "rotate": 0,
        "language": khoi.ngon_ngu,
        "fontSize": co_chu,
        "showLineNumbers": len(dong) > 3,
        "lines": [{"id": f"L{n + 1}", "content": d} for n, d in enumerate(dong)],
    }


def _bang(dem: DemId, khoi: Khoi, tren: float) -> dict:
    hang = khoi.hang[:9]
    so_cot = max(len(h) for h in hang)
    cao_o = 34.0
    cao = min(len(hang) * cao_o + 4, KHUNG_H - 50 - tren)
    du_lieu = []
    for chi_so, h in enumerate(hang):
        o = []
        for cot in range(so_cot):
            chu = textwrap.shorten(h[cot] if cot < len(h) else "", width=64, placeholder="…")
            kieu = {"fontsize": "14px"}
            if chi_so == 0:
                kieu |= {"bold": True, "backcolor": "#eff6ff", "color": MAU_NHAN}
            o.append(
                {
                    "id": f"c{chi_so}_{cot}",
                    "colspan": 1,
                    "rowspan": 1,
                    "text": chu,
                    "style": kieu,
                }
            )
        du_lieu.append(o)
    return {
        "id": dem("table"),
        "type": "table",
        "left": LE,
        "top": round(tren, 1),
        "width": KHUNG_W - 2 * LE,
        "height": round(cao, 1),
        "rotate": 0,
        "outline": {"width": 1, "style": "solid", "color": "#d1d5db"},
        "theme": {
            "color": "#eff6ff",
            "rowHeader": True,
            "rowFooter": False,
            "colHeader": False,
            "colFooter": False,
        },
        "colWidths": [round(1 / so_cot, 4)] * so_cot,
        "cellMinHeight": cao_o,
        "data": du_lieu,
    }


# --------------------------------------------------------------------------- #
# Dựng scene
# --------------------------------------------------------------------------- #

GIOI_HAN_BULLET = 5
DAI_BULLET = 90
DAI_LOI_GIANG = 900


def _cau_dau(van: str, so_cau: int = 2) -> str:
    """Lấy vài câu đầu. KHÔNG cắt ở dấu hai chấm giữa câu — "Hệ quả: agent có thể…"
    mà cắt thì bullet còn trơ lại hai chữ "Hệ quả"."""
    cau = re.split(r"(?<=[.!?])\s+", van)
    return " ".join(cau[:so_cau]).strip()


def _bullet_tu_muc(muc: Muc) -> list[tuple[str, bool]]:
    """Trả về (nội dung, in đậm). Tiêu đề con in đậm, gạch đầu dòng thì không."""
    ra: list[tuple[str, bool]] = []
    for k in muc.khoi:
        if k.loai == "sub":
            ra.append((textwrap.shorten(k.van_ban, width=DAI_BULLET, placeholder="…"), True))
        elif k.loai == "bullets":
            ra.extend(
                (textwrap.shorten(m, width=DAI_BULLET, placeholder="…"), False) for m in k.muc
            )
    if not any(not dam for _, dam in ra):
        # Mục không có gạch đầu dòng: rút ý từ văn xuôi, bỏ các câu dẫn
        # ("Chạy thử ngay:") vì chúng thuộc về khối code ngay sau, không phải slide.
        for k in muc.khoi:
            if k.loai == "para" and not _la_dan(k):
                y = textwrap.shorten(_cau_dau(k.van_ban, 1), width=DAI_BULLET, placeholder="…")
                y = y.rstrip(" :")
                if len(y) >= 25:  # bỏ mảnh vụn kiểu "Hệ quả", "Cách bắt"
                    ra.append((y, False))
    return [b for b in ra if b[0]][:GIOI_HAN_BULLET]


def _loi_giang(muc: Muc) -> str:
    ngat = next(
        (i for i, k in enumerate(muc.khoi) if k.loai in {"code", "table"}), len(muc.khoi)
    )
    dau = [
        k for k in muc.khoi[:ngat] if k.loai in {"para", "callout", "sub"} and not _la_dan(k)
    ]
    van = " ".join(k.van_ban for k in (dau or [k for k in muc.khoi if k.loai == "callout"]))
    if not van:
        # Mục chỉ có gạch đầu dòng: đọc chính các ý đó thay vì để giảng viên câm.
        y = [m for k in muc.khoi if k.loai == "bullets" for m in k.muc] + [
            k.van_ban for k in muc.khoi if k.loai == "sub"
        ]
        van = f"{muc.tieu_de}. " + " ".join(y) if y else ""
    return textwrap.shorten(van, width=DAI_LOI_GIANG, placeholder="…") if van else ""


def _la_dan(khoi: Khoi) -> bool:
    """Đoạn kết thúc bằng dấu hai chấm là lời dẫn cho khối ngay sau nó."""
    return khoi.loai == "para" and khoi.van_ban.rstrip().endswith(":")


def _day_du(loi: str, du_phong: str) -> str:
    """Lời dẫn quá ngắn ("Hỏi nó:") thì nối thêm câu chốt cho đủ một nhịp nói."""
    if not loi:
        return du_phong
    return loi if len(loi) >= 90 else f"{loi} {du_phong}"


def _dan_khoi(muc: Muc, chi_so: int) -> str:
    """Đoạn văn ngay trước khối code/bảng — chính là lời dẫn tác giả đã viết."""
    truoc = muc.khoi[chi_so - 1] if chi_so else None
    if truoc and _la_dan(truoc):
        return textwrap.shorten(truoc.van_ban, width=400, placeholder="…")
    return ""


def _speech(dem: DemId, chu: str) -> list[dict]:
    """Cắt lời giảng thành các speech action ~400 ký tự cho khớp nhịp TTS."""
    if not chu:
        return []
    manh, hien_tai = [], ""
    for cau in re.split(r"(?<=[.!?])\s+", chu):
        if len(hien_tai) + len(cau) > 400 and hien_tai:
            manh.append(hien_tai.strip())
            hien_tai = cau
        else:
            hien_tai = f"{hien_tai} {cau}".strip()
    if hien_tai:
        manh.append(hien_tai.strip())
    return [{"id": dem("action"), "type": "speech", "text": m} for m in manh]


def _discussion(dem: DemId, chu_de: ThaoLuan) -> dict:
    return {
        "id": dem("action"),
        "type": "discussion",
        "topic": chu_de.chu_de,
        "prompt": (
            f"{chu_de.goi_y} Tranh luận bằng tiếng Việt, bám vào nội dung buổi học, "
            "mỗi lượt nói ngắn và nêu ví dụ cụ thể thay vì nói chung chung."
        ),
    }


def _scene(loai: str, tieu_de: str, thu_tu: int, noi_dung: dict, hanh_dong: list[dict], da_tac_tu=None) -> dict:
    scene = {
        "type": loai,
        "title": tieu_de,
        "order": thu_tu,
        "content": noi_dung,
        "actions": hanh_dong,
    }
    if da_tac_tu:
        scene["multiAgent"] = da_tac_tu
    return scene


def _scene_slide(dem: DemId, tieu_de: str, thu_tu: int, phan_tu: list[dict], loi: str, da_tac_tu=None) -> dict:
    return _scene(
        "slide",
        tieu_de,
        thu_tu,
        {"type": "slide", "canvas": _slide(dem, phan_tu)},
        _speech(dem, loi),
        da_tac_tu,
    )


def scene_bia(dem: DemId, bai: Bai, thu_tu: int, thao_luan: ThaoLuan | None = None) -> dict:
    phan_tu = [
        _text(
            dem,
            trai=LE,
            tren=150,
            rong=KHUNG_W - 2 * LE,
            cao=_cao_chu([f"Bài {bai.ma}"], KHUNG_W - 2 * LE, 20),
            html=_p(f"BÀI {bai.ma}", co_chu=20, mau=MAU_NHAN, dam=True),
        ),
        _text(
            dem,
            trai=LE,
            tren=200,
            rong=KHUNG_W - 2 * LE,
            cao=_cao_chu([bai.tieu_de], KHUNG_W - 2 * LE, 40),
            html=_p(bai.tieu_de, co_chu=40, mau=MAU_CHU, dam=True),
        ),
    ]
    if bai.muc_tieu:
        phan_tu.append(
            _text(
                dem,
                trai=LE,
                tren=330,
                rong=KHUNG_W - 2 * LE,
                cao=_cao_chu([bai.muc_tieu], KHUNG_W - 2 * LE, 19),
                html=_p(f"🎯 {bai.muc_tieu}", co_chu=19, mau=MAU_PHU),
            )
        )
    phan_tu.append(
        _text(
            dem,
            trai=LE,
            tren=460,
            rong=KHUNG_W - 2 * LE,
            cao=40,
            html=_p(f"⏱ {bai.thoi_luong} · Kiểm thử tự động cho ứng dụng AI", co_chu=16, mau=MAU_PHU),
        )
    )
    loi = f"Chào cả lớp. Bài {bai.ma}: {bai.tieu_de}. "
    if bai.muc_tieu:
        loi += f"Hết {bai.thoi_luong} này, mục tiêu là: {bai.muc_tieu}."
    if thao_luan:
        loi += f" Trước khi vào nội dung, mở màn bằng một câu hỏi: {thao_luan.chu_de}"
    scene = _scene_slide(
        dem,
        f"Bài {bai.ma} — {bai.tieu_de}",
        thu_tu,
        phan_tu,
        loi,
        {"enabled": True, "agentIndices": [0, 1, 2]},
    )
    if thao_luan:
        scene["actions"].append(_discussion(dem, thao_luan))
    return scene


def scene_muc_luc(dem: DemId, bai: Bai, thu_tu: int) -> dict | None:
    muc = next((m for m in bai.muc if m.tieu_de == MUC_MUC_LUC), None)
    bang = next((k for k in muc.khoi if k.loai == "table"), None) if muc else None
    if not bang:
        return None
    phan_tu = [_slide_tieu_de(dem, "Nội dung buổi học"), _bang(dem, bang, 140)]
    loi = (
        "Đây là chặng đường của buổi hôm nay: "
        + ", ".join(h[0] for h in bang.hang[1:6])
        + ". Mỗi mục có mốc thời gian riêng, bám theo để không lệch nhịp."
    )
    return _scene_slide(dem, "Nội dung buổi học", thu_tu, phan_tu, loi)


def scene_tu_muc(dem: DemId, muc: Muc, thu_tu: int) -> list[dict]:
    ra: list[dict] = []
    bullet = _bullet_tu_muc(muc)
    luu_y = next((k for k in muc.khoi if k.loai == "callout"), None)

    phan_tu = [_slide_tieu_de(dem, muc.tieu_de)]
    tren = 150.0
    if bullet:
        cao = _cao_chu([b for b, _ in bullet], KHUNG_W - 2 * LE, 21)
        phan_tu.append(
            _text(
                dem,
                trai=LE,
                tren=tren,
                rong=KHUNG_W - 2 * LE,
                cao=cao,
                html="".join(
                    _p(b if dam else f"• {b}", co_chu=21, mau=MAU_NHAN if dam else MAU_CHU, dam=dam)
                    for b, dam in bullet
                ),
            )
        )
        tren += cao + 20
    if luu_y and tren < KHUNG_H - 130:
        phan_tu.append(_hop_luu_y(dem, luu_y, tren))
    ra.append(_scene_slide(dem, muc.tieu_de, thu_tu, phan_tu, _loi_giang(muc)))

    # Code và bảng lên scene riêng — chúng cần cả khung hình mới đọc được.
    code = [i for i, x in enumerate(muc.khoi) if x.loai == "code"][:3]
    for n, i in enumerate(code, start=1):
        k = muc.khoi[i]
        thu_tu += 1
        nhan = (
            "lệnh"
            if k.ngon_ngu in {"bash", "powershell"}
            else "kết quả"
            if k.ngon_ngu == "text"
            else "mã"
        )
        so = f" {n}" if len(code) > 1 else ""
        ra.append(
            _scene_slide(
                dem,
                f"{muc.tieu_de} — {nhan}{so}",
                thu_tu,
                [_slide_tieu_de(dem, muc.tieu_de), _code(dem, k, 130)],
                _day_du(
                    _dan_khoi(muc, i),
                    f"Đây là đoạn {nhan} của mục này. Đọc kỹ từng dòng trước khi chạy.",
                ),
            )
        )
    bang = [i for i, x in enumerate(muc.khoi) if x.loai == "table"][:1]
    for i in bang:
        thu_tu += 1
        ra.append(
            _scene_slide(
                dem,
                f"{muc.tieu_de} — bảng đối chiếu",
                thu_tu,
                [_slide_tieu_de(dem, muc.tieu_de), _bang(dem, muc.khoi[i], 130)],
                _day_du(
                    _dan_khoi(muc, i),
                    "Bảng này là phần cần nhớ kỹ nhất của mục vừa rồi, đọc theo từng hàng.",
                ),
            )
        )
    return ra


def scene_bai_tap(
    dem: DemId, bai: Bai, thu_tu: int, noi_dung: NoiDungLop | None = None
) -> dict | None:
    cau_hoi: list[dict] = []

    # Trắc nghiệm đi trước: chấm được ngay, không cần judge.
    for n, tn in enumerate((noi_dung.trac_nghiem if noi_dung else []), start=1):
        cau_hoi.append(
            {
                "id": f"tn{n}",
                "type": "multiple" if len(tn.dap_an) > 1 else "single",
                "question": tn.cau_hoi,
                "options": [{"label": f"{nhan}. {chu}", "value": nhan} for nhan, chu in tn.lua_chon],
                "answer": tn.dap_an,
                "analysis": tn.giai_thich,
                "hasAnswer": True,
                "points": 1,
            }
        )

    muc = next((m for m in bai.muc if m.tieu_de == MUC_BAI_TAP), None)
    for k in muc.khoi if muc else []:
        if k.loai != "bullets":
            continue
        for m in k.muc:
            cau_hoi.append(
                {
                    "id": f"q{len(cau_hoi) + 1}",
                    "type": "short_answer",
                    "question": m,
                    "analysis": "Chấm theo cách làm và bằng chứng chạy thật, không chấm câu chữ.",
                    "commentPrompt": (
                        f"Bài {bai.ma} — {bai.tieu_de}. Chấm câu trả lời của học viên: có nêu được "
                        "cách làm cụ thể không, có dẫn kết quả chạy thật không. Góp ý ngắn, bằng tiếng Việt."
                    ),
                    "hasAnswer": False,
                    "points": 1,
                }
            )
    if not cau_hoi:
        return None
    so_tn = sum(1 for c in cau_hoi if c["hasAnswer"])
    so_tl = len(cau_hoi) - so_tn
    if so_tn and so_tl:
        loi = (
            f"Đến phần kiểm tra. {so_tn} câu trắc nghiệm chấm được ngay, sau đó là "
            f"{so_tl} câu tự luận. Phần tự luận phải làm trên máy trước rồi mới trả lời — "
            "mỗi câu cần một bằng chứng chạy thật, không phải suy đoán."
        )
    elif so_tn:
        loi = (
            f"Đến phần kiểm tra: {so_tn} câu trắc nghiệm về đúng những gì vừa học. "
            "Chọn xong sẽ có ngay giải thích cho từng câu."
        )
    else:
        loi = (
            "Đến phần bài tập. Làm trên máy trước rồi hãy trả lời — mỗi câu đều cần một "
            "bằng chứng chạy thật, không phải suy đoán."
        )
    return _scene(
        "quiz",
        "Kiểm tra và bài tập" if so_tn and so_tl else "Trắc nghiệm" if so_tn else "Bài tập",
        thu_tu,
        {"type": "quiz", "questions": cau_hoi[:10]},
        _speech(dem, loi),
    )


def scene_kiem_tra(
    dem: DemId, bai: Bai, thu_tu: int, thao_luan: ThaoLuan | None = None
) -> dict | None:
    muc = next((m for m in bai.muc if m.tieu_de == MUC_KIEM_TRA), None)
    khoi = next((k for k in muc.khoi if k.loai == "code"), None) if muc else None
    if not khoi:
        return None
    loi = (
        "Trước khi đóng máy, chạy đoạn kiểm tra này. Nó xanh thì bài sau chạy được; "
        "nó đỏ thì dừng lại xử lý ngay, đừng mang lỗi sang buổi sau."
    )
    if thao_luan:
        loi += f" Còn mười phút cuối, cả lớp bàn tiếp: {thao_luan.chu_de}"
    scene = _scene_slide(
        dem,
        "Kiểm tra trước khi sang bài sau",
        thu_tu,
        [_slide_tieu_de(dem, "Kiểm tra trước khi sang bài sau"), _code(dem, khoi, 130)],
        loi,
        {"enabled": True, "agentIndices": [0, 1, 2]},
    )
    if thao_luan:
        scene["actions"].append(_discussion(dem, thao_luan))
    return scene


# --------------------------------------------------------------------------- #
# Đóng gói .maic.zip
# --------------------------------------------------------------------------- #

TAC_TU = [
    {
        "name": "Thầy Hoàng",
        "role": "teacher",
        "persona": (
            "Giảng viên kiểm thử ứng dụng AI, 10 năm làm QA rồi chuyển sang LLM. Nói ngắn, "
            "luôn kèm số đo thật và tên lệnh cụ thể. Ghét câu trả lời chung chung; hay hỏi "
            "ngược 'bạn đo bằng gì?'. Nhắc học viên rằng mock model của ứng dụng thì hợp lệ, "
            "còn mock judge thì vô nghĩa."
        ),
        "avatar": "/avatars/teacher.png",
        "color": "#3b82f6",
        "priority": 1,
    },
    {
        "name": "Vy",
        "role": "student",
        "persona": (
            "Lập trình viên backend mới chuyển sang mảng AI. Quen assert giá trị chính xác nên "
            "hay thắc mắc vì sao test LLM lại đỏ ngẫu nhiên. Hỏi thẳng, hỏi nhiều, thích ví dụ "
            "chạy được ngay trên máy."
        ),
        "avatar": "/avatars/curious.png",
        "color": "#10b981",
        "priority": 2,
    },
    {
        "name": "Đăng",
        "role": "student",
        "persona": (
            "Kỹ sư vận hành, quan tâm chi phí và độ ổn định. Luôn hỏi bài chạy mất bao lâu, tốn "
            "bao nhiêu RAM, hỏng thì khoanh vùng thế nào. Hay chỉ ra cái bẫy phiên bản thư viện."
        ),
        "avatar": "/avatars/thinker.png",
        "color": "#f59e0b",
        "priority": 3,
    },
]

MO_TA_KHOA = (
    "Khoá 12 buổi × 90 phút: xây ứng dụng agentic trên văn bản pháp luật thật (RAG + tool call "
    "+ backend + web) rồi viết bộ test tự động chấm chất lượng đầu ra bằng pytest, DeepEval và "
    "RAGAS. Toàn bộ chạy bằng model local qua Ollama."
)


def dung_manifest(danh_sach_bai: list[Bai], ten: str, mo_ta: str) -> dict:
    dem = DemId()
    scenes: list[dict] = []
    ngan_hang = doc_ngan_hang()

    for bai in danh_sach_bai:
        noi_dung = ngan_hang.get(bai.ma)
        chu_de = list(noi_dung.thao_luan) if noi_dung else []
        scenes.append(scene_bia(dem, bai, len(scenes), chu_de[0] if chu_de else None))
        muc_luc = scene_muc_luc(dem, bai, len(scenes))
        if muc_luc:
            scenes.append(muc_luc)
        for muc in bai.muc:
            if muc.tieu_de in MUC_BO_QUA or muc.tieu_de in {MUC_BAI_TAP, MUC_KIEM_TRA}:
                continue
            for scene in scene_tu_muc(dem, muc, len(scenes)):
                scene["order"] = len(scenes)
                scenes.append(scene)
        bai_tap = scene_bai_tap(dem, bai, len(scenes), noi_dung)
        if bai_tap:
            scenes.append(bai_tap)
        chu_de_cuoi = chu_de[1] if len(chu_de) > 1 else None
        kiem_tra = scene_kiem_tra(dem, bai, len(scenes), chu_de_cuoi)
        if kiem_tra:
            kiem_tra["order"] = len(scenes)
            scenes.append(kiem_tra)
        elif chu_de_cuoi and scenes:
            # Bài không có mục "Kiểm tra trước khi sang bài sau" (Bài 1 và Bài 12):
            # chủ đề thảo luận cuối vẫn phải được nói, gắn vào scene cuối cùng.
            cuoi = scenes[-1]
            cuoi.setdefault("actions", []).extend(
                _speech(dem, f"Mười phút cuối, cả lớp bàn tiếp: {chu_de_cuoi.chu_de}")
            )
            cuoi["actions"].append(_discussion(dem, chu_de_cuoi))
            cuoi["multiAgent"] = {"enabled": True, "agentIndices": [0, 1, 2]}

    bay_gio = int(datetime.now(timezone.utc).timestamp() * 1000)
    return {
        "formatVersion": 1,
        "exportedAt": datetime.now(timezone.utc).isoformat(),
        "appVersion": "openmaic_export.py",
        "stage": {
            "name": ten,
            "description": mo_ta,
            "language": "Tiếng Việt",
            "style": "Giảng dạy kỹ thuật: ngắn gọn, bám lệnh chạy được và số đo thật.",
            "createdAt": bay_gio,
            "updatedAt": bay_gio,
        },
        "agents": TAC_TU,
        "scenes": scenes,
        "mediaIndex": {},
    }


def ghi_zip(manifest: dict, duong_dan: Path) -> Path:
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(duong_dan, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return duong_dan


def _ten_file(chuoi: str) -> str:
    """Bỏ dấu để tên file chạy được trên mọi hệ tệp."""
    chuoi = unicodedata.normalize("NFD", chuoi).replace("đ", "d").replace("Đ", "D")
    chuoi = "".join(c for c in chuoi if unicodedata.category(c) != "Mn")
    chuoi = re.sub(r"[^A-Za-z0-9\s-]", "", chuoi).strip().lower()
    return re.sub(r"[\s_]+", "-", chuoi)


def _duong_ngan(duong_dan: Path) -> str:
    return str(duong_dan.relative_to(ROOT)) if duong_dan.is_relative_to(ROOT) else str(duong_dan)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bai", nargs="*", help="số bài cần xuất, ví dụ: --bai 1 3 7")
    parser.add_argument("--gop", action="store_true", help="gộp mọi bài vào một lớp học")
    parser.add_argument("--dich", type=Path, default=DICH, help="thư mục xuất")
    tham_so = parser.parse_args()

    tep = sorted(NGUON.glob("session-*.md"))
    if tham_so.bai:
        chon = {f"{int(b):02d}" for b in tham_so.bai}
        tep = [t for t in tep if re.match(r"session-(\d+)", t.stem).group(1) in chon]
    if not tep:
        print("Không tìm thấy bài nào trong docs/sessions/")
        return 1

    danh_sach = [doc_bai(t) for t in tep]

    if tham_so.gop:
        manifest = dung_manifest(danh_sach, "Kiểm thử tự động cho ứng dụng AI", MO_TA_KHOA)
        ra = ghi_zip(manifest, tham_so.dich / "khoa-kiem-thu-ung-dung-ai.maic.zip")
        print(f"{_duong_ngan(ra)}  ({len(manifest['scenes'])} scene, {len(danh_sach)} bài)")
        return 0

    for bai in danh_sach:
        manifest = dung_manifest([bai], f"Bài {bai.ma} — {bai.tieu_de}", bai.muc_tieu or MO_TA_KHOA)
        ten = f"bai-{bai.ma}-{_ten_file(bai.tieu_de)}.maic.zip"
        ra = ghi_zip(manifest, tham_so.dich / ten)
        quiz = sum(1 for s in manifest["scenes"] if s["type"] == "quiz")
        print(f"{_duong_ngan(ra)}  ({len(manifest['scenes'])} scene, {quiz} quiz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
