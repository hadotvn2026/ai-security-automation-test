"""Backend HTTP của ứng dụng trợ lý pháp lý.

Kiến trúc:

    Trình duyệt  ──POST /api/hoi──►  FastAPI  ──►  agent_graph  ──►  Ollama
                                        │              │
                                        │              ├─► tra_cuu_van_ban ─► Chroma
                                        │              ├─► kiem_tra_cap_nhat ─► vault
                                        │              ├─► tinh_thoi_han
                                        │              └─► chuyen_chuyen_gia
                                        ▼
                            {cau_tra_loi, quy_dao, ngu_canh, canh_bao}

Điểm thiết kế quan trọng: API trả về **cả quỹ đạo và ngữ cảnh**, không chỉ câu
trả lời. Nhờ vậy:

  - Front-end hiển thị được agent đã gọi công cụ nào, người dùng thấy hệ thống
    "nghĩ" gì thay vì phải tin một hộp đen.
  - Test API kiểm tra được đường đi, không chỉ chuỗi ký tự trả về.

Chạy:
    uv run uvicorn rag_qa.api:app --reload --port 8000
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag_qa import agent_graph, config, rag_graph, tri_nho, vault

WEB_DIR = Path(__file__).resolve().parents[2] / "web"

app = FastAPI(
    title="Trợ lý pháp lý — An ninh mạng",
    description="Ứng dụng agentic có RAG và tool call, chạy hoàn toàn bằng model local.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Kiểu dữ liệu — khai báo rõ để front-end và test cùng dựa vào
# ---------------------------------------------------------------------------
class CauHoi(BaseModel):
    noi_dung: str = Field(min_length=1, max_length=2000)
    che_do: Literal["agent", "rag"] = "agent"


class BuocCongCu(BaseModel):
    tool: str
    args: dict[str, Any]
    ket_qua: str


class TraLoi(BaseModel):
    cau_tra_loi: str
    che_do: str
    quy_dao: list[str]
    buoc: list[BuocCongCu]
    ngu_canh: list[str]
    canh_bao: list[str] = []      # từ trí nhớ tự sửa
    so_buoc: int
    thoi_gian_ms: int


class TaiLieuVault(BaseModel):
    ma: str
    ten: str
    so_hieu: str
    ngay_ban_hanh: str
    trang_thai: str
    dang_dung_trong_rag: bool = False


# ---------------------------------------------------------------------------
# Điểm cuối
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def trang_chu() -> FileResponse:
    index = WEB_DIR / "index.html"
    if not index.exists():
        raise HTTPException(500, f"Không tìm thấy giao diện tại {index}")
    return FileResponse(index)


@app.get("/api/suc-khoe")
def suc_khoe() -> dict[str, Any]:
    """Kiểm tra nhanh: model nào đang dùng, vector store có bao nhiêu chunk."""
    from rag_qa import vector_store

    try:
        so_chunk = vector_store.count()
    except Exception as exc:  # Chroma chưa được nạp
        so_chunk = -1
        loi = str(exc)[:120]
    else:
        loi = None

    return {
        "model_ung_dung": config.APP_MODEL,
        "model_judge": config.JUDGE_MODEL,
        "model_embedding": config.EMBEDDING_MODEL,
        "so_chunk": so_chunk,
        "top_k": config.TOP_K,
        "loi": loi,
    }


@app.get("/api/tai-lieu", response_model=list[TaiLieuVault])
def danh_sach_tai_lieu() -> list[dict[str, Any]]:
    """Danh sách văn bản trong vault, kèm trạng thái hiệu lực."""
    try:
        return vault.danh_sach_tai_lieu()
    except vault.VaultError as exc:
        raise HTTPException(500, str(exc)) from exc


@app.get("/api/cap-nhat")
def kiem_tra_cap_nhat(ma: str | None = None) -> dict[str, Any]:
    """Văn bản đang dùng có bản mới hơn không."""
    try:
        return vault.tim_ban_cap_nhat(ma)
    except vault.VaultError as exc:
        raise HTTPException(500, str(exc)) from exc


@app.post("/api/hoi", response_model=TraLoi)
def hoi(cau_hoi: CauHoi) -> dict[str, Any]:
    """Hỏi trợ lý. `che_do="agent"` dùng agent có công cụ; `"rag"` dùng pipeline thẳng.

    Hai chế độ cùng tồn tại là chủ ý: người dùng (và học viên) so sánh được
    trực tiếp một pipeline cố định với một agent tự quyết định.
    """
    bat_dau = time.perf_counter()

    if cau_hoi.che_do == "rag":
        tra_loi, ngu_canh = rag_graph.answer_with_contexts(cau_hoi.noi_dung)
        ket_qua = {
            "cau_tra_loi": tra_loi,
            "quy_dao": ["retrieve", "generate"],
            "buoc": [],
            "ngu_canh": ngu_canh,
            "so_buoc": 2,
        }
    else:
        r = agent_graph.chay_agent(cau_hoi.noi_dung)
        ngu_canh = [
            b["ket_qua"] for b in r["scratchpad"] if b["tool"] == "tra_cuu_van_ban"
        ]
        ket_qua = {
            "cau_tra_loi": r["cau_tra_loi"],
            "quy_dao": r["quy_dao"],
            "buoc": r["scratchpad"],
            "ngu_canh": ngu_canh,
            "so_buoc": r["so_buoc"],
        }

    # Cảnh báo tại chỗ: câu trả lời này có dựa trên Điều nào đang nghi ngờ không?
    try:
        ket_qua["canh_bao"] = tri_nho.canh_bao_cho_ngu_canh(ket_qua["ngu_canh"])
    except Exception:
        ket_qua["canh_bao"] = []   # trí nhớ hỏng không được làm hỏng câu trả lời

    ket_qua["che_do"] = cau_hoi.che_do
    ket_qua["thoi_gian_ms"] = int((time.perf_counter() - bat_dau) * 1000)
    return ket_qua


@app.get("/api/tri-nho")
def trang_thai_tri_nho() -> dict[str, Any]:
    """Tình trạng kho khẳng định: cái nào đã xác thực, cái nào cần rà soát."""
    kho = tri_nho.doc_kho()
    lac_hau = tri_nho.phat_hien_lac_hau(kho)
    return {
        "tong": len(kho),
        "thong_ke": tri_nho.thong_ke(kho),
        "lac_hau": [
            {"ma": k.ma, "dieu": k.dieu, "ly_do": ly_do, "khang_dinh": k.noi_dung}
            for k, ly_do in lac_hau
        ],
    }
