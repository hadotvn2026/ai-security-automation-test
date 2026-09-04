"""Kiểm soát an toàn cho ứng dụng agentic có tích hợp MCP.

BỐI CẢNH RỦI RO
---------------
Khi agent chỉ dùng công cụ do bạn tự viết (Bài 5), bề mặt tấn công nhỏ: bạn kiểm
soát cả mã lẫn mô tả công cụ.

Tích hợp một **MCP server** đảo ngược điều đó. MCP server ở phía bên kia quyết
định:

    - TÊN công cụ
    - MÔ TẢ công cụ  ← đi thẳng vào prompt của model
    - SCHEMA tham số
    - NỘI DUNG kết quả trả về  ← cũng đi thẳng vào ngữ cảnh

Hai dòng có mũi tên là điểm mấu chốt: **kẻ tấn công không cần khai thác lỗi mã
nguồn nào cả**. Chỉ cần viết chỉ thị vào phần mô tả công cụ, và model sẽ đọc nó
như đọc chỉ thị của bạn. Đây là *prompt injection qua kênh công cụ*.

Module này cài bốn lớp phòng thủ, mỗi lớp đều TẤT ĐỊNH và test được:

    1. Công tắc dừng khẩn (kill switch) — chặn mọi lời gọi công cụ ngay lập tức
    2. Danh sách nguồn tin cậy — chỉ cho phép MCP server đã duyệt
    3. Quét chỉ thị đáng ngờ — trong mô tả công cụ VÀ trong kết quả trả về
    4. Vân tay công cụ — phát hiện MCP server đổi mô tả sau khi đã được duyệt

Không lớp nào dùng LLM. Đó là chủ ý: cơ chế phòng thủ không được phụ thuộc vào
chính thứ đang bị tấn công.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from rag_qa import config

__all__ = [
    "KillSwitchError",
    "NguonKhongTinCayError",
    "kill_switch_dang_bat",
    "kiem_tra_kill_switch",
    "bat_kill_switch",
    "tat_kill_switch",
    "nguon_duoc_phep",
    "kiem_tra_nguon_mcp",
    "quet_chi_thi_dang_ngo",
    "van_tay_cong_cu",
    "kiem_tra_van_tay",
    "ghi_nhat_ky",
    "PhatHien",
]

# ---------------------------------------------------------------------------
# Đường dẫn và cấu hình
# ---------------------------------------------------------------------------
SENTINEL_KILL = config.PROJECT_ROOT / ".agent-kill"
NHAT_KY = config.PROJECT_ROOT / "reports" / "nhat-ky-cong-cu.jsonl"
VAN_TAY = config.PROJECT_ROOT / "data" / "mcp-van-tay.json"

# MCP server được duyệt. Mặc định RỖNG — mọi nguồn đều bị từ chối cho tới khi
# có người thêm vào một cách có ý thức. Danh sách trắng, không phải danh sách đen.
NGUON_TIN_CAY_MAC_DINH: tuple[str, ...] = ()


class KillSwitchError(RuntimeError):
    """Công tắc dừng khẩn đang bật — mọi lời gọi công cụ bị chặn."""


class NguonKhongTinCayError(RuntimeError):
    """MCP server không nằm trong danh sách được duyệt."""


@dataclass
class PhatHien:
    """Một dấu hiệu đáng ngờ tìm thấy trong văn bản do bên ngoài kiểm soát."""

    muc_do: str          # "cao" | "trung_binh" | "thap"
    loai: str
    trich_doan: str
    vi_tri: int
    nguon: str = ""

    def __str__(self) -> str:
        return f"[{self.muc_do}] {self.loai} @ {self.vi_tri}: {self.trich_doan!r}"


# ---------------------------------------------------------------------------
# 1. Công tắc dừng khẩn
# ---------------------------------------------------------------------------
# Hai cách bật, cả hai đều KHÔNG cần khởi động lại dịch vụ:
#
#     touch .agent-kill                 <- tạo file sentinel
#     export RAG_QA_KILL_SWITCH=1       <- biến môi trường
#
# Vì sao cần cả hai? Biến môi trường tiện khi chạy test và CI. File sentinel tiện
# khi sự cố đang xảy ra trên production: người trực chỉ cần `touch` một file, không
# cần quyền deploy, không cần sửa cấu hình, không cần chờ dịch vụ khởi động lại.
def kill_switch_dang_bat() -> bool:
    """True nếu công tắc dừng khẩn đang bật. KHÔNG cache — phải đọc lại mỗi lần."""
    if os.getenv("RAG_QA_KILL_SWITCH", "").strip().lower() in {"1", "true", "yes"}:
        return True
    return SENTINEL_KILL.exists()


def kiem_tra_kill_switch(boi_canh: str = "") -> None:
    """Ném KillSwitchError nếu công tắc đang bật. Gọi TRƯỚC mỗi lời gọi công cụ."""
    if kill_switch_dang_bat():
        raise KillSwitchError(
            f"Công tắc dừng khẩn đang bật — đã chặn: {boi_canh or 'lời gọi công cụ'}. "
            f"Tắt bằng: rm {SENTINEL_KILL.name}  (hoặc unset RAG_QA_KILL_SWITCH)"
        )


def bat_kill_switch(ly_do: str = "") -> Path:
    """Bật công tắc bằng file sentinel. Trả về đường dẫn file."""
    SENTINEL_KILL.write_text(
        json.dumps(
            {"bat_luc": datetime.now(timezone.utc).isoformat(), "ly_do": ly_do},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return SENTINEL_KILL


def tat_kill_switch() -> None:
    SENTINEL_KILL.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 2. Danh sách nguồn tin cậy
# ---------------------------------------------------------------------------
def nguon_duoc_phep() -> tuple[str, ...]:
    """Danh sách MCP server được duyệt, đọc từ biến môi trường.

        export RAG_QA_MCP_TIN_CAY="https://mcp.noibo.congty.vn,stdio:./mcp-local"
    """
    tu_env = os.getenv("RAG_QA_MCP_TIN_CAY", "").strip()
    if not tu_env:
        return NGUON_TIN_CAY_MAC_DINH
    return tuple(x.strip() for x in tu_env.split(",") if x.strip())


def kiem_tra_nguon_mcp(url: str) -> None:
    """Ném NguonKhongTinCayError nếu `url` không nằm trong danh sách duyệt.

    So khớp theo TIỀN TỐ đầy đủ, không dùng `in`. Dùng `in` là lỗ hổng kinh
    điển: "https://mcp.congty.vn.kegian.com" chứa "mcp.congty.vn".
    """
    url = (url or "").strip()
    if not url:
        raise NguonKhongTinCayError("URL của MCP server bị bỏ trống.")
    duoc_phep = nguon_duoc_phep()
    if not duoc_phep:
        raise NguonKhongTinCayError(
            "Chưa có MCP server nào được duyệt. Đặt RAG_QA_MCP_TIN_CAY trước khi "
            "tích hợp. Mặc định là danh sách TRẮNG rỗng — từ chối tất cả."
        )
    for nguon in duoc_phep:
        if url == nguon or url.startswith(nguon.rstrip("/") + "/"):
            return
    raise NguonKhongTinCayError(
        f"MCP server '{url}' không nằm trong danh sách được duyệt: {duoc_phep}"
    )


# ---------------------------------------------------------------------------
# 3. Quét chỉ thị đáng ngờ
# ---------------------------------------------------------------------------
# Áp dụng cho MỌI văn bản do bên ngoài kiểm soát:
#   - mô tả công cụ MCP  (nguy hiểm nhất: đi vào system prompt)
#   - kết quả công cụ trả về
#   - nội dung tài liệu vừa nạp vào vector store
MAU_DANG_NGO: tuple[tuple[str, str, str], ...] = (
    # (mức độ, tên loại, regex)
    ("cao", "ghi_de_chi_thi",
     r"(?i)\b(bỏ qua|phớt lờ|ignore|disregard|forget)\b[^.\n]{0,40}"
     r"\b(chỉ thị|hướng dẫn|instruction|prompt|rule|quy tắc)"),
    ("cao", "gia_mao_vai_tro",
     r"(?i)^\s*(system|assistant|developer)\s*:|\[/?(INST|SYS)\]|<\|im_start\|>"),
    # Bắt CẢ HAI thứ tự. Bản đầu chỉ bắt "api_key ... gửi" và bỏ sót
    # "in ra api_key" — chính bộ test mẫu tấn công đã phát hiện thiếu sót đó.
    ("cao", "ro_ri_bi_mat",
     r"(?i)(?:"
     r"\b(api[_ -]?key|token|mật khẩu|password|secret|credential)\b"
     r"[^.\n]{0,40}?\b(gửi|send|reveal|in ra|print|xuất|leak|tiết lộ|hiển thị)\b"
     r"|"
     r"\b(gửi|send|reveal|in ra|print|xuất|leak|tiết lộ|hiển thị)\b"
     r"[^.\n]{0,40}?\b(api[_ -]?key|token|mật khẩu|password|secret|credential)\b"
     r")"),
    ("cao", "yeu_cau_goi_ra_ngoai",
     r"(?i)\b(gửi|post|upload|exfiltrat|đẩy)\b[^.\n]{0,40}(https?://|\bwebhook\b)"),
    ("trung_binh", "gia_mao_quyen",
     r"(?i)\b(tôi là|as the|with)\b[^.\n]{0,20}"
     r"\b(quản trị|admin|developer|nhà phát triển|anthropic|openai)\b"),
    ("trung_binh", "chi_thi_an",
     r"(?i)\b(không nói cho|đừng nói|do not tell|don'?t mention|silently|âm thầm)\b"),
    ("trung_binh", "yeu_cau_thuc_thi",
     r"(?i)\b(chạy lệnh|execute|eval\(|exec\(|subprocess|os\.system|rm -rf)\b"),
    ("thap", "van_ban_an",
     # Dùng escape thay vì ký tự nguyên bản. Chèn ký tự bidi thật vào mã
     # nguồn chính là thứ bandit B613 cảnh báo — và cũng là kỹ thuật giấu mã.
     "[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]"),
)


def quet_chi_thi_dang_ngo(
    van_ban: str, nguon: str = ""
) -> list[PhatHien]:
    """Tìm dấu hiệu prompt injection trong văn bản do bên ngoài kiểm soát.

    Đây là bộ lọc THÔ, cố ý. Nó không thay thế việc rà soát bằng người, và nó
    sẽ có báo động giả. Nhưng nó chạy trong micro-giây, tất định, và bắt được
    những mẫu tấn công phổ biến nhất trước khi chúng tới model.

    Nguyên tắc: KHÔNG dùng LLM để phát hiện tấn công vào LLM.
    """
    if not van_ban:
        return []
    ket_qua: list[PhatHien] = []
    for muc_do, loai, mau in MAU_DANG_NGO:
        for m in re.finditer(mau, van_ban, re.MULTILINE):
            trich = van_ban[max(0, m.start() - 20) : m.end() + 40].strip()
            ket_qua.append(
                PhatHien(muc_do=muc_do, loai=loai, trich_doan=trich[:120],
                         vi_tri=m.start(), nguon=nguon)
            )
    return ket_qua


# ---------------------------------------------------------------------------
# 4. Vân tay công cụ — chống "rug pull"
# ---------------------------------------------------------------------------
# Kịch bản tấn công: MCP server phục vụ mô tả công cụ hoàn toàn lành tính trong
# lúc bạn rà soát và duyệt. Một tuần sau, nó đổi mô tả thành chỉ thị độc hại.
# Không có dòng mã nào của bạn thay đổi, không có cảnh báo nào.
#
# Cách chặn: băm mô tả công cụ lúc duyệt, so lại mỗi lần khởi động.
def van_tay_cong_cu(cong_cu: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Băm (tên + mô tả + schema) của từng công cụ."""
    ra: dict[str, str] = {}
    for t in cong_cu:
        payload = json.dumps(
            {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "schema": t.get("inputSchema") or t.get("parameters") or {},
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        ra[t.get("name", "?")] = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return ra


def kiem_tra_van_tay(
    hien_tai: dict[str, str], duong_dan: Path | None = None
) -> list[str]:
    """So vân tay hiện tại với bản đã duyệt. Trả về danh sách cảnh báo.

    Lần đầu chạy (chưa có file) trả về danh sách rỗng và KHÔNG tự ghi —
    việc duyệt phải là hành động có ý thức của con người.
    """
    p = Path(duong_dan or VAN_TAY)
    if not p.exists():
        return []
    da_duyet: dict[str, str] = json.loads(p.read_text(encoding="utf-8"))

    canh_bao: list[str] = []
    for ten, bam in hien_tai.items():
        if ten not in da_duyet:
            canh_bao.append(f"Công cụ MỚI chưa được duyệt: '{ten}'")
        elif da_duyet[ten] != bam:
            canh_bao.append(
                f"Công cụ '{ten}' ĐÃ ĐỔI mô tả/schema sau khi được duyệt "
                f"({da_duyet[ten]} -> {bam}). Nghi vấn rug pull."
            )
    for ten in da_duyet:
        if ten not in hien_tai:
            canh_bao.append(f"Công cụ đã duyệt nay BIẾN MẤT: '{ten}'")
    return canh_bao


# ---------------------------------------------------------------------------
# Nhật ký — để sau khi dừng khẩn còn biết agent đã làm gì
# ---------------------------------------------------------------------------
def ghi_nhat_ky(su_kien: str, chi_tiet: dict[str, Any]) -> None:
    """Ghi một dòng JSONL. Không bao giờ ném lỗi ra ngoài."""
    try:
        NHAT_KY.parent.mkdir(parents=True, exist_ok=True)
        with NHAT_KY.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "luc": datetime.now(timezone.utc).isoformat(),
                        "su_kien": su_kien,
                        **chi_tiet,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except OSError:
        pass  # nhật ký hỏng không được làm sập ứng dụng
