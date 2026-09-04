"""Trí nhớ tự sửa — khẳng định neo vào bằng chứng có phiên bản.

Ý tưởng lấy từ OpenWiki (LangChain): thay vì lưu câu trả lời như một chuỗi chữ,
lưu nó như một KHẲNG ĐỊNH có neo vào BẰNG CHỨNG cụ thể, kèm vân tay của bằng
chứng tại thời điểm xác thực. Mỗi chu kỳ, so vân tay cũ với văn bản hiện tại:
bằng chứng đổi thì khẳng định bị đánh dấu NGHI NGỜ cho tới khi được xác thực lại.

VÌ SAO ỨNG DỤNG NÀY CẦN NÓ
--------------------------
`vault.py` đã phát hiện được "có văn bản mới hơn" ở mức TOÀN VĂN BẢN. Nhưng khi
Luật 116/2025 được sửa đổi một điều, điều gì xảy ra?

    - Bạn chạy lại `ingest.py`, vector store có nội dung mới.
    - Mọi câu trả lời cũ vẫn nằm nguyên trong đầu người dùng và trong tài liệu nội bộ.
    - Không có gì cho biết câu trả lời nào vừa trở nên SAI.

Với văn bản pháp luật, đó là rủi ro nghiêm trọng: một câu trả lời đúng vào tháng
trước có thể sai hoàn toàn sau một nghị định sửa đổi.

BA KHÁC BIỆT SO VỚI BẢN GỐC CỦA OPENWIKI
----------------------------------------
1. **Bằng chứng neo theo ĐIỀU, không theo chunk.** Chunk là đơn vị kỹ thuật:
   đổi `CHUNK_SIZE` là mọi ranh giới chunk dịch chuyển và toàn bộ khẳng định bị
   báo lạc hậu giả. Điều luật là đơn vị ngữ nghĩa, ổn định qua mọi cách cắt.

2. **KHÔNG tự sửa khẳng định khi bằng chứng đã đổi.** OpenWiki để agent tự sửa.
   Ở đây judge là model 8B chạy local, và đã đo được nó chấm sai có hệ thống
   (xem Bài 11). Tự động viết lại một khẳng định pháp lý bằng một judge không
   đáng tin là đổi một rủi ro nhỏ lấy một rủi ro lớn hơn. Bằng chứng đổi thì
   chuyển cho NGƯỜI rà soát.

3. **Tự làm mới chỉ khi bằng chứng KHÔNG đổi.** Đó là trường hợp an toàn duy
   nhất: văn bản y nguyên, chỉ cần xác nhận hệ thống vẫn trả lời nhất quán.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

from rag_qa import config

__all__ = [
    "KhangDinh",
    "TrangThai",
    "van_tay_dieu",
    "trich_dieu",
    "doc_kho",
    "ghi_kho",
    "phat_hien_lac_hau",
    "doi_chieu",
    "thong_ke",
    "canh_bao_cho_ngu_canh",
]

KHO = config.DATA_DIR / "tri-nho" / "khang-dinh.json"
NGUON_TXT = config.DATA_DIR / "luat-116-2025-an-ninh-mang.txt"

TrangThai = Literal["xac_thuc", "nghi_ngo", "can_nguoi_ra_soat", "sai"]

NHAN_TRANG_THAI = {
    "xac_thuc": "đã xác thực",
    "nghi_ngo": "nghi ngờ — bằng chứng đã đổi",
    "can_nguoi_ra_soat": "chờ người rà soát",
    "sai": "đã xác định sai",
}


@dataclass
class KhangDinh:
    """Một khẳng định về nội dung văn bản, neo vào bằng chứng cụ thể."""

    ma: str
    cau_hoi: str
    noi_dung: str                      # khẳng định — câu trả lời đã được xác thực
    dieu: int                          # bằng chứng: số Điều
    van_ban: str = "LUAT-116-2025"
    van_tay_bang_chung: str = ""       # băm nội dung Điều lúc xác thực
    trang_thai: TrangThai = "xac_thuc"
    xac_thuc_luc: str = ""
    ghi_chu: str = ""
    lich_su: list[dict[str, Any]] = field(default_factory=list)

    def nhan(self) -> str:
        return NHAN_TRANG_THAI.get(self.trang_thai, self.trang_thai)


# ---------------------------------------------------------------------------
# Bằng chứng: trích và băm một Điều
# ---------------------------------------------------------------------------
def trich_dieu(so: int, nguon: Path | None = None) -> str | None:
    """Lấy toàn văn một Điều từ văn bản nguồn. None nếu không có."""
    p = Path(nguon or NGUON_TXT)
    if not p.exists():
        return None
    van_ban = p.read_text(encoding="utf-8")
    m = re.search(
        rf"^Điều {so}\..*?(?=^Điều {so + 1}\.|\Z)", van_ban, re.MULTILINE | re.DOTALL
    )
    return m.group(0).strip() if m else None


def van_tay_dieu(so: int, nguon: Path | None = None) -> str | None:
    """Băm nội dung một Điều, sau khi chuẩn hoá khoảng trắng.

    Chuẩn hoá trước khi băm là chi tiết quan trọng: OCR lại cùng một trang có
    thể cho khoảng trắng khác nhau, và ta không muốn báo lạc hậu vì chuyện đó.
    Chỉ NỘI DUNG đổi mới tính.
    """
    noi_dung = trich_dieu(so, nguon)
    if noi_dung is None:
        return None
    chuan = re.sub(r"\s+", " ", noi_dung).strip()
    return hashlib.sha256(chuan.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Kho khẳng định
# ---------------------------------------------------------------------------
def doc_kho(duong_dan: Path | None = None) -> list[KhangDinh]:
    p = Path(duong_dan or KHO)
    if not p.exists():
        return []
    du_lieu = json.loads(p.read_text(encoding="utf-8"))
    return [KhangDinh(**k) for k in du_lieu.get("khang_dinh", [])]


def ghi_kho(kho: list[KhangDinh], duong_dan: Path | None = None) -> Path:
    p = Path(duong_dan or KHO)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "_ghi_chu": "Trí nhớ tự sửa. Mỗi khẳng định neo vào một Điều luật "
                            "và vân tay nội dung Điều đó lúc xác thực.",
                "cap_nhat_luc": datetime.now(timezone.utc).isoformat(),
                "khang_dinh": [asdict(k) for k in kho],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return p


# ---------------------------------------------------------------------------
# Phát hiện lạc hậu
# ---------------------------------------------------------------------------
def phat_hien_lac_hau(
    kho: list[KhangDinh] | None = None, nguon: Path | None = None
) -> list[tuple[KhangDinh, str]]:
    """So vân tay đã lưu với văn bản hiện tại.

    Trả về danh sách (khẳng định, lý do). Danh sách rỗng nghĩa là mọi bằng chứng
    còn nguyên vẹn — KHÔNG có nghĩa là mọi khẳng định đều đúng.
    """
    kho = doc_kho() if kho is None else kho
    lac_hau: list[tuple[KhangDinh, str]] = []
    for k in kho:
        hien_tai = van_tay_dieu(k.dieu, nguon)
        if hien_tai is None:
            lac_hau.append((k, f"Điều {k.dieu} KHÔNG CÒN trong văn bản nguồn"))
        elif not k.van_tay_bang_chung:
            lac_hau.append((k, "Chưa từng ghi vân tay bằng chứng"))
        elif hien_tai != k.van_tay_bang_chung:
            lac_hau.append(
                (k, f"Điều {k.dieu} đã ĐỔI nội dung "
                    f"({k.van_tay_bang_chung} -> {hien_tai})")
            )
    return lac_hau


# ---------------------------------------------------------------------------
# Vòng đối chiếu
# ---------------------------------------------------------------------------
def doi_chieu(
    kho: list[KhangDinh] | None = None,
    tra_loi_lai=None,
    nguon: Path | None = None,
) -> dict[str, Any]:
    """Chạy một chu kỳ đối chiếu trên toàn bộ kho.

    Args:
        tra_loi_lai: hàm `(cau_hoi) -> str` để hỏi lại hệ thống. Truyền
            `rag_graph.answer` khi chạy thật; truyền hàm giả khi test.

    Ba nhánh xử lý, và ranh giới giữa chúng là phần quan trọng nhất:

        bằng chứng KHÔNG đổi + trả lời vẫn nhất quán  -> tự làm mới (an toàn)
        bằng chứng KHÔNG đổi + trả lời đã lệch        -> nghi ngờ hệ thống, không phải luật
        bằng chứng ĐÃ đổi                             -> CHUYỂN NGƯỜI, không tự sửa

    Nhánh thứ ba là điểm khác biệt cố ý so với OpenWiki: không để model tự viết
    lại một khẳng định pháp lý.
    """
    kho = doc_kho() if kho is None else kho
    hom_nay = datetime.now(timezone.utc).isoformat()

    ket_qua = {
        "tong": len(kho),
        "lam_moi": [],
        "he_thong_lech": [],
        "can_nguoi_ra_soat": [],
        "khong_doi": [],
    }

    for k in kho:
        hien_tai = van_tay_dieu(k.dieu, nguon)
        bang_chung_doi = hien_tai != k.van_tay_bang_chung

        if bang_chung_doi:
            k.trang_thai = "can_nguoi_ra_soat"
            k.ghi_chu = (
                f"Điều {k.dieu} đã đổi nội dung ({k.van_tay_bang_chung} -> {hien_tai}). "
                f"Khẳng định cần người đọc lại văn bản mới trước khi tin."
            )
            k.lich_su.append({"luc": hom_nay, "su_kien": "bang_chung_doi",
                              "van_tay_cu": k.van_tay_bang_chung, "van_tay_moi": hien_tai})
            ket_qua["can_nguoi_ra_soat"].append(k.ma)
            continue

        if tra_loi_lai is None:
            ket_qua["khong_doi"].append(k.ma)
            continue

        moi = (tra_loi_lai(k.cau_hoi) or "").strip()
        if _nhat_quan(k.noi_dung, moi):
            k.trang_thai = "xac_thuc"
            k.xac_thuc_luc = hom_nay
            k.ghi_chu = ""
            ket_qua["lam_moi"].append(k.ma)
        else:
            # Bằng chứng y nguyên mà câu trả lời đổi => nghi ngờ HỆ THỐNG
            # (prompt, model, cách cắt chunk), không phải nghi ngờ văn bản luật.
            k.trang_thai = "nghi_ngo"
            k.ghi_chu = (
                "Bằng chứng KHÔNG đổi nhưng hệ thống trả lời khác. Nghi ngờ "
                f"prompt/model/chunking, không phải văn bản. Trả lời mới: {moi[:120]}"
            )
            k.lich_su.append({"luc": hom_nay, "su_kien": "he_thong_lech",
                              "tra_loi_moi": moi[:200]})
            ket_qua["he_thong_lech"].append(k.ma)

    return ket_qua


def _nhat_quan(cu: str, moi: str) -> bool:
    """So khớp thô: mọi số và mọi mã điều trong khẳng định cũ còn xuất hiện không.

    Cố ý KHÔNG dùng LLM. Với văn bản pháp luật, thứ phải giữ nguyên là con số:
    ngày hiệu lực, số điều, thời hạn, số cấp độ. Câu chữ đổi thì chấp nhận được;
    con số đổi thì không.
    """
    so_cu = set(re.findall(r"\d+(?:[.,]\d+)?", cu))
    so_moi = set(re.findall(r"\d+(?:[.,]\d+)?", moi))
    return so_cu.issubset(so_moi) if so_cu else bool(moi)


def thong_ke(kho: list[KhangDinh] | None = None) -> dict[str, int]:
    kho = doc_kho() if kho is None else kho
    ra = {t: 0 for t in NHAN_TRANG_THAI}
    for k in kho:
        ra[k.trang_thai] = ra.get(k.trang_thai, 0) + 1
    return ra


# ---------------------------------------------------------------------------
# Nối vào luồng trả lời: cảnh báo tại chỗ
# ---------------------------------------------------------------------------
# Dấu chấm sau số là TUỲ CHỌN. Nhãn chunk thật có dạng "Điều 44. Hiệu lực",
# nhưng kết quả công cụ có thể là "[Điều 44] ...". Với cảnh báo an toàn, bỏ sót
# tệ hơn thừa — nên khớp rộng.
_SO_DIEU_TRONG_NHAN = re.compile(r"Điều\s+(\d+)\b")


def canh_bao_cho_ngu_canh(
    ngu_canh: list[str], kho: list[KhangDinh] | None = None
) -> list[str]:
    """Cảnh báo nếu câu trả lời đang dựa trên Điều có khẳng định đáng ngờ.

    Đây là chỗ trí nhớ tự sửa trả công: người dùng nhận cảnh báo NGAY trong câu
    trả lời, thay vì phải nhớ đi chạy một script kiểm tra định kỳ.

    Cách hoạt động: đọc số Điều từ nhãn của chunk lấy về, đối chiếu với kho.
    Không gọi model, chạy trong micro-giây.
    """
    kho = doc_kho() if kho is None else kho
    if not kho:
        return []

    dieu_dang_dung: set[int] = set()
    for c in ngu_canh or []:
        nhan = c.splitlines()[0] if c else ""
        dieu_dang_dung.update(int(m) for m in _SO_DIEU_TRONG_NHAN.findall(nhan))

    ra: list[str] = []
    for k in kho:
        if k.dieu not in dieu_dang_dung:
            continue
        if k.trang_thai == "can_nguoi_ra_soat":
            ra.append(
                f"⚠️ Điều {k.dieu} đã thay đổi kể từ lần xác thực cuối và chưa được "
                f"người rà soát. Hãy đối chiếu với văn bản gốc trước khi dùng."
            )
        elif k.trang_thai == "nghi_ngo":
            ra.append(
                f"⚠️ Hệ thống từng trả lời không nhất quán về Điều {k.dieu}. "
                f"Khuyến nghị kiểm chứng lại."
            )
    return sorted(set(ra))
