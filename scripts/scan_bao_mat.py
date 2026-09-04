"""Quét an toàn thông tin tự động cho ứng dụng agentic.

Chạy toàn bộ:
    uv run --with pip-audit --with bandit python scripts/scan_bao_mat.py

Chạy một nhóm:
    uv run python scripts/scan_bao_mat.py --chi cau-hinh

Mã thoát:
    0  không có phát hiện mức CAO
    1  có ít nhất một phát hiện mức CAO  -> dùng làm cổng chặn CI

Bảy nhóm kiểm tra, xếp theo thứ tự bề mặt tấn công của một ứng dụng agentic:

    1. phu-thuoc     CVE trong thư viện (pip-audit)
    2. ma-nguon      Lỗi lập trình nguy hiểm (bandit): eval, subprocess, ...
    3. bi-mat        Khoá/token lọt vào mã nguồn hoặc file cấu hình
    4. cau-hinh      Kill switch, danh sách nguồn MCP, trần tài nguyên
    5. cong-cu       Mô tả công cụ chứa chỉ thị đáng ngờ  <- riêng của agentic
    6. van-tay       MCP server đổi mô tả sau khi được duyệt (rug pull)
    7. du-lieu       Tài liệu nạp vào RAG chứa prompt injection

Nhóm 5-7 KHÔNG có trong bất kỳ bộ quét bảo mật truyền thống nào. Chúng là bề mặt
tấn công mới mà ứng dụng agentic tự tạo ra, và bạn phải tự viết.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_qa import an_toan  # noqa: E402

MUC_DO_UU_TIEN = {"cao": 0, "trung_binh": 1, "thap": 2, "thong_tin": 3}
MIEN_TRU_PATH = ROOT / "data" / "bao-mat-mien-tru.json"


def doc_mien_tru() -> tuple[dict[str, dict], list["KetQua"]]:
    """Đọc danh sách miễn trừ đã được đánh giá.

    Miễn trừ KHÔNG phải danh sách bỏ qua. Mỗi mục bắt buộc có đánh giá phơi
    nhiễm, biện pháp bù, ngày hết hạn và người duyệt. Hết hạn thì lỗ hổng tự
    bật lại thành mức CAO — đó là điểm mấu chốt, vì rủi ro chấp nhận được hôm
    nay có thể không còn chấp nhận được sau sáu tháng.
    """
    if not MIEN_TRU_PATH.exists():
        return {}, []
    du_lieu = json.loads(MIEN_TRU_PATH.read_text(encoding="utf-8"))
    hom_nay = date.today()
    theo_ma: dict[str, dict] = {}
    canh_bao: list[KetQua] = []
    for mt in du_lieu.get("mien_tru", []):
        het_han = date.fromisoformat(mt["het_han"])
        qua_han = het_han < hom_nay
        if qua_han:
            canh_bao.append(KetQua(
                "mien-tru", "cao",
                f"Miễn trừ ĐÃ HẾT HẠN cho {mt['goi']} ({', '.join(mt['ma'])})",
                f"Hết hạn {mt['het_han']}. Phải đánh giá lại, không được gia hạn máy móc.",
            ))
        if "(điền tên" in str(mt.get("nguoi_duyet", "")):
            canh_bao.append(KetQua(
                "mien-tru", "trung_binh",
                f"Miễn trừ {mt['goi']} chưa có người duyệt",
                "Miễn trừ không có người chịu trách nhiệm là miễn trừ vô chủ.",
            ))
        for ma in mt["ma"]:
            theo_ma[ma] = {**mt, "qua_han": qua_han}
    return theo_ma, canh_bao


def kiem_tra_dieu_kien_huy() -> list["KetQua"]:
    """Miễn trừ có ĐIỀU KIỆN. Kiểm tra bằng máy xem điều kiện còn đúng không.

    Miễn trừ chromadb dựa trên "ứng dụng chỉ dùng PersistentClient, không mở
    cổng mạng". Nếu một ngày ai đó thêm HttpClient, lập luận đó sụp đổ — và
    không ai nhớ ra. Nên máy phải nhớ hộ.
    """
    ra: list[KetQua] = []
    for f in (ROOT / "src").rglob("*.py"):
        noi_dung = f.read_text(encoding="utf-8", errors="ignore")
        if "chromadb.HttpClient" in noi_dung or "chromadb.AsyncHttpClient" in noi_dung:
            ra.append(KetQua(
                "mien-tru", "cao",
                f"HUỶ miễn trừ chromadb — tìm thấy HttpClient trong {f.name}",
                "Miễn trừ 4 CVE của chromadb dựa trên việc chỉ dùng PersistentClient. "
                "Dùng HttpClient là mở đúng bề mặt tấn công đó.",
            ))
    return ra


@dataclass
class KetQua:
    nhom: str
    muc_do: str
    tieu_de: str
    chi_tiet: str = ""

    def __str__(self) -> str:
        bieu_tuong = {"cao": "🔴", "trung_binh": "🟡", "thap": "🔵", "thong_tin": "⚪"}
        return f"{bieu_tuong.get(self.muc_do, '·')} [{self.nhom}] {self.tieu_de}"


# ---------------------------------------------------------------------------
# 1. Phụ thuộc — CVE đã biết
# ---------------------------------------------------------------------------
def quet_phu_thuoc() -> list[KetQua]:
    try:
        r = subprocess.run(
            ["pip-audit", "--format", "json", "--progress-spinner", "off"],
            capture_output=True, text=True, timeout=300, cwd=ROOT,
        )
    except FileNotFoundError:
        return [KetQua("phu-thuoc", "thong_tin", "Bỏ qua: chưa cài pip-audit",
                       "Chạy: uv run --with pip-audit python scripts/scan_bao_mat.py")]
    except subprocess.TimeoutExpired:
        return [KetQua("phu-thuoc", "trung_binh", "pip-audit quá thời gian")]

    try:
        du_lieu = json.loads(r.stdout or "{}")
    except json.JSONDecodeError:
        return [KetQua("phu-thuoc", "trung_binh", "Không đọc được kết quả pip-audit",
                       (r.stderr or "")[:200])]

    mien_tru, _ = doc_mien_tru()
    ra: list[KetQua] = []
    for goi in du_lieu.get("dependencies", []):
        for v in goi.get("vulns", []):
            ma = v.get("id", "?")
            mt = mien_tru.get(ma)
            if mt and not mt["qua_han"]:
                ra.append(KetQua(
                    "phu-thuoc", "thong_tin",
                    f"{goi['name']} {goi.get('version','?')} — {ma} (đã miễn trừ)",
                    f"Lý do: {mt['danh_gia_phoi_nhiem'][:130]}…",
                ))
                continue
            ra.append(KetQua(
                "phu-thuoc", "cao",
                f"{goi['name']} {goi.get('version','?')} — {ma}",
                (v.get("description") or "")[:160],
            ))
    if not ra:
        ra.append(KetQua("phu-thuoc", "thong_tin", "Không có CVE đã biết"))
    return ra


# ---------------------------------------------------------------------------
# 2. Mã nguồn — SAST
# ---------------------------------------------------------------------------
def quet_ma_nguon() -> list[KetQua]:
    try:
        r = subprocess.run(
            ["bandit", "-r", "src", "scripts", "-f", "json", "-q"],
            capture_output=True, text=True, timeout=300, cwd=ROOT,
        )
    except FileNotFoundError:
        return [KetQua("ma-nguon", "thong_tin", "Bỏ qua: chưa cài bandit",
                       "Chạy: uv run --with bandit python scripts/scan_bao_mat.py")]

    try:
        du_lieu = json.loads(r.stdout or "{}")
    except json.JSONDecodeError:
        return [KetQua("ma-nguon", "trung_binh", "Không đọc được kết quả bandit")]

    anh_xa = {"HIGH": "cao", "MEDIUM": "trung_binh", "LOW": "thap"}
    ra = [
        KetQua("ma-nguon", anh_xa.get(k.get("issue_severity", ""), "thap"),
               f"{k.get('test_id')} {Path(k.get('filename','')).name}:{k.get('line_number')}",
               (k.get("issue_text") or "")[:160])
        for k in du_lieu.get("results", [])
    ]
    return ra or [KetQua("ma-nguon", "thong_tin", "Không có phát hiện")]


# ---------------------------------------------------------------------------
# 3. Bí mật lọt vào repo
# ---------------------------------------------------------------------------
MAU_BI_MAT = (
    ("cao", "OpenAI key", r"sk-[A-Za-z0-9]{20,}"),
    ("cao", "Anthropic key", r"sk-ant-[A-Za-z0-9\-_]{20,}"),
    ("cao", "AWS access key", r"AKIA[0-9A-Z]{16}"),
    ("cao", "GitHub token", r"gh[pousr]_[A-Za-z0-9]{30,}"),
    ("cao", "Private key", r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ("trung_binh", "Gán mật khẩu cứng",
     r"(?i)\b(password|passwd|secret|token)\s*=\s*[\"'][^\"'\s]{8,}[\"']"),
)
BO_QUA = {".venv", "__pycache__", ".git", "data", "snapshots", "reports", ".pytest_cache"}


def quet_bi_mat() -> list[KetQua]:
    ra: list[KetQua] = []
    for f in ROOT.rglob("*"):
        if not f.is_file() or any(p in BO_QUA for p in f.parts):
            continue
        if f.suffix not in {".py", ".toml", ".json", ".yml", ".yaml", ".md", ".html", ".env"}:
            continue
        try:
            noi_dung = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for muc_do, ten, mau in MAU_BI_MAT:
            for m in re.finditer(mau, noi_dung):
                dong = noi_dung[: m.start()].count("\n") + 1
                ra.append(KetQua("bi-mat", muc_do, f"{ten} — {f.relative_to(ROOT)}:{dong}",
                                 m.group(0)[:24] + "…"))
    return ra or [KetQua("bi-mat", "thong_tin", "Không tìm thấy bí mật lộ")]


# ---------------------------------------------------------------------------
# 4. Cấu hình an toàn
# ---------------------------------------------------------------------------
def quet_cau_hinh() -> list[KetQua]:
    from rag_qa import config

    ra: list[KetQua] = []

    nguon = an_toan.nguon_duoc_phep()
    if not nguon:
        ra.append(KetQua("cau-hinh", "thong_tin",
                         "Danh sách MCP tin cậy rỗng — mọi nguồn bị từ chối",
                         "Đây là trạng thái AN TOÀN khi chưa tích hợp MCP."))
    else:
        for u in nguon:
            if u.startswith("http://"):
                ra.append(KetQua("cau-hinh", "cao",
                                 f"MCP server dùng HTTP không mã hoá: {u}",
                                 "Mô tả công cụ đi vào prompt — kẻ đứng giữa sửa được."))
            elif not (u.startswith("https://") or u.startswith("stdio:")):
                ra.append(KetQua("cau-hinh", "trung_binh",
                                 f"MCP server dùng giao thức lạ: {u}"))

    if an_toan.kill_switch_dang_bat():
        ra.append(KetQua("cau-hinh", "thong_tin", "Công tắc dừng khẩn ĐANG BẬT"))

    if config.REQUEST_TIMEOUT > 300:
        ra.append(KetQua("cau-hinh", "trung_binh",
                         f"REQUEST_TIMEOUT quá lớn: {config.REQUEST_TIMEOUT}s"))
    if config.MAX_TOKENS > 4096:
        ra.append(KetQua("cau-hinh", "trung_binh",
                         f"MAX_TOKENS quá lớn: {config.MAX_TOKENS}"))

    from rag_qa.agent_graph import SO_BUOC_TOI_DA
    if SO_BUOC_TOI_DA > 10:
        ra.append(KetQua("cau-hinh", "cao",
                         f"SO_BUOC_TOI_DA quá lớn: {SO_BUOC_TOI_DA}",
                         "Agent không có trần bước chặt là agent đốt tiền không giới hạn."))

    return ra or [KetQua("cau-hinh", "thong_tin", "Cấu hình an toàn hợp lệ")]


# ---------------------------------------------------------------------------
# 5. Mô tả công cụ — bề mặt tấn công riêng của agentic
# ---------------------------------------------------------------------------
def quet_mo_ta_cong_cu() -> list[KetQua]:
    from rag_qa.tools import TAT_CA_TOOL

    ra: list[KetQua] = []
    for t in TAT_CA_TOOL:
        for d in an_toan.quet_chi_thi_dang_ngo(t.description or "", nguon=t.name):
            ra.append(KetQua("cong-cu", d.muc_do,
                             f"Mô tả công cụ '{t.name}' — {d.loai}", d.trich_doan))
    return ra or [KetQua("cong-cu", "thong_tin",
                         f"{len(TAT_CA_TOOL)} mô tả công cụ đều sạch")]


# ---------------------------------------------------------------------------
# 6. Vân tay MCP — chống rug pull
# ---------------------------------------------------------------------------
def quet_van_tay() -> list[KetQua]:
    if not an_toan.VAN_TAY.exists():
        return [KetQua("van-tay", "thong_tin",
                       "Chưa có vân tay MCP nào được duyệt",
                       "Bình thường khi chưa tích hợp MCP server.")]
    from rag_qa.tools import TAT_CA_TOOL

    hien_tai = an_toan.van_tay_cong_cu(
        [{"name": t.name, "description": t.description} for t in TAT_CA_TOOL]
    )
    canh_bao = an_toan.kiem_tra_van_tay(hien_tai)
    return [KetQua("van-tay", "cao", c) for c in canh_bao] or [
        KetQua("van-tay", "thong_tin", "Vân tay công cụ khớp bản đã duyệt")
    ]


# ---------------------------------------------------------------------------
# 7. Dữ liệu nạp vào RAG
# ---------------------------------------------------------------------------
def quet_du_lieu() -> list[KetQua]:
    from rag_qa import config

    ra: list[KetQua] = []
    nguon_txt = config.DATA_DIR / "luat-116-2025-an-ninh-mang.txt"
    if nguon_txt.exists():
        noi_dung = nguon_txt.read_text(encoding="utf-8")
        for d in an_toan.quet_chi_thi_dang_ngo(noi_dung, nguon=nguon_txt.name):
            if d.muc_do in {"cao", "trung_binh"}:
                ra.append(KetQua("du-lieu", d.muc_do,
                                 f"{nguon_txt.name} — {d.loai} @ {d.vi_tri}",
                                 d.trich_doan))

    manifest = config.DATA_DIR / "vault" / "manifest.json"
    if manifest.exists():
        for d in an_toan.quet_chi_thi_dang_ngo(
            manifest.read_text(encoding="utf-8"), nguon="vault"
        ):
            if d.muc_do == "cao":
                ra.append(KetQua("du-lieu", "cao", f"vault — {d.loai}", d.trich_doan))

    return ra or [KetQua("du-lieu", "thong_tin", "Dữ liệu nạp vào RAG không có dấu hiệu injection")]


def quet_mien_tru() -> list[KetQua]:
    _, canh_bao = doc_mien_tru()
    canh_bao += kiem_tra_dieu_kien_huy()
    return canh_bao or [KetQua("mien-tru", "thong_tin",
                               "Miễn trừ còn hạn và điều kiện vẫn đúng")]


NHOM = {
    "phu-thuoc": quet_phu_thuoc,
    "mien-tru": quet_mien_tru,
    "ma-nguon": quet_ma_nguon,
    "bi-mat": quet_bi_mat,
    "cau-hinh": quet_cau_hinh,
    "cong-cu": quet_mo_ta_cong_cu,
    "van-tay": quet_van_tay,
    "du-lieu": quet_du_lieu,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Quét an toàn thông tin")
    parser.add_argument("--chi", choices=sorted(NHOM), help="Chỉ chạy một nhóm")
    parser.add_argument("--json", action="store_true", help="Xuất JSON")
    args = parser.parse_args()

    can_chay = {args.chi: NHOM[args.chi]} if args.chi else NHOM

    tat_ca: list[KetQua] = []
    for ten, ham in can_chay.items():
        try:
            tat_ca.extend(ham())
        except Exception as exc:  # một nhóm hỏng không được làm hỏng cả bộ quét
            tat_ca.append(KetQua(ten, "trung_binh", f"Nhóm '{ten}' lỗi: {type(exc).__name__}",
                                 str(exc)[:160]))

    tat_ca.sort(key=lambda k: (MUC_DO_UU_TIEN.get(k.muc_do, 9), k.nhom))

    if args.json:
        print(json.dumps([k.__dict__ for k in tat_ca], ensure_ascii=False, indent=2))
    else:
        print("\n" + "=" * 74)
        print("QUÉT AN TOÀN THÔNG TIN — ứng dụng agentic")
        print("=" * 74)
        for k in tat_ca:
            print(k)
            if k.chi_tiet:
                print(f"    {k.chi_tiet}")
        print("-" * 74)
        dem = {m: sum(1 for k in tat_ca if k.muc_do == m) for m in MUC_DO_UU_TIEN}
        print(f"CAO {dem['cao']} · TRUNG BÌNH {dem['trung_binh']} · "
              f"THẤP {dem['thap']} · THÔNG TIN {dem['thong_tin']}")

    return 1 if any(k.muc_do == "cao" for k in tat_ca) else 0


if __name__ == "__main__":
    raise SystemExit(main())
