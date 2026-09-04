"""PART 3a — test đường ống (plumbing), KHÔNG chấm điểm chất lượng.

Đây là loại test rẻ nhất và chạy nhiều nhất: mock hẳn LLM, chỉ kiểm tra dây
nối. Graph có đi đúng thứ tự node không? Retriever trả về gì khi câu hỏi rỗng?
Prompt có thực sự chứa ngữ cảnh không?

Không có metric nào ở đây, và đó là chủ ý. Chất lượng câu trả lời là việc của
test đánh giá (test_chatbot_deepeval.py / test_rag_ragas.py). Trộn hai loại
vào nhau là cách nhanh nhất để có một bộ test vừa chậm vừa không nói lên điều gì.

Chạy:
    uv run pytest tests/test_plumbing.py -v
"""

from __future__ import annotations

import pytest

from rag_qa import llm as llm_module
from rag_qa import prompts, rag_graph, retriever, vector_store

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 3a.1 — prompt: ngữ cảnh có thực sự đi vào prompt không?
# ---------------------------------------------------------------------------
def test_prompt_chua_toan_bo_ngu_canh():
    contexts = ["Điều 44: hiệu lực từ 01/7/2026.", "Điều 8: phân loại 5 cấp độ."]
    prompt = prompts.build_rag_prompt("Luật có hiệu lực khi nào?", contexts)
    for c in contexts:
        assert c in prompt


def test_prompt_chua_cau_hoi():
    prompt = prompts.build_rag_prompt("Luật có hiệu lực khi nào?", ["abc"])
    assert "Luật có hiệu lực khi nào?" in prompt


def test_prompt_danh_so_cac_doan():
    prompt = prompts.build_rag_prompt("q", ["A", "B", "C"])
    assert "[Đoạn 1]" in prompt and "[Đoạn 3]" in prompt


def test_prompt_khong_co_ngu_canh_thi_noi_ro():
    """Không tìm thấy tài liệu KHÔNG được im lặng gửi prompt rỗng cho model."""
    prompt = prompts.build_rag_prompt("q", [])
    assert "không tìm thấy" in prompt.lower()


# ---------------------------------------------------------------------------
# 3a.2 — retriever: các trường hợp biên
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cau_hoi", ["", "   ", "\n\t "])
def test_cau_hoi_rong_tra_ve_list_rong_khong_nem_loi(cau_hoi, monkeypatch):
    """Câu hỏi rỗng phải trả list rỗng, không được nổ và không được gọi vector store."""

    def khong_duoc_goi(*args, **kwargs):
        raise AssertionError("Không được truy vấn vector store với câu hỏi rỗng")

    monkeypatch.setattr(vector_store, "query", khong_duoc_goi)
    assert retriever.retrieve(cau_hoi) == []


def test_retriever_truyen_dung_top_k(monkeypatch):
    ghi_nhan = {}

    def gia_lap_query(question, top_k=None, persist_dir=None):
        ghi_nhan["top_k"] = top_k
        return ["chunk"]

    monkeypatch.setattr(vector_store, "query", gia_lap_query)
    retriever.retrieve("câu hỏi", top_k=7)
    assert ghi_nhan["top_k"] == 7


# ---------------------------------------------------------------------------
# 3a.3 — graph: thứ tự node và luồng dữ liệu
# ---------------------------------------------------------------------------
def test_rag_graph_chay_retrieve_truoc_generate(monkeypatch):
    """Kiểm chứng hình dạng START -> retrieve -> generate -> END."""
    thu_tu: list[str] = []

    def gia_lap_retrieve(question, top_k=None):
        thu_tu.append("retrieve")
        return ["ngữ cảnh giả"]

    def gia_lap_generate(prompt, system=None):
        thu_tu.append("generate")
        return "câu trả lời giả"

    monkeypatch.setattr(retriever, "retrieve", gia_lap_retrieve)
    monkeypatch.setattr(llm_module, "generate", gia_lap_generate)

    graph = rag_graph.build_rag_graph()
    ket_qua = graph.invoke({"question": "câu hỏi"})

    assert thu_tu == ["retrieve", "generate"]
    assert ket_qua["answer"] == "câu trả lời giả"
    assert ket_qua["contexts"] == ["ngữ cảnh giả"]


def test_ngu_canh_tu_retriever_di_vao_prompt_cua_llm(monkeypatch):
    """Test bắt được lỗi kinh điển: retrieve chạy nhưng kết quả không được dùng."""
    prompt_da_nhan: list[str] = []

    monkeypatch.setattr(retriever, "retrieve", lambda q, top_k=None: ["ĐOẠN ĐẶC BIỆT 12345"])
    monkeypatch.setattr(
        llm_module,
        "generate",
        lambda prompt, system=None: prompt_da_nhan.append(prompt) or "ok",
    )

    rag_graph.build_rag_graph().invoke({"question": "câu hỏi"})

    assert "ĐOẠN ĐẶC BIỆT 12345" in prompt_da_nhan[0]


def test_khong_tim_thay_tai_lieu_van_tra_loi_duoc(monkeypatch):
    """Retriever trả rỗng thì ứng dụng vẫn phải chạy, không được sập."""
    monkeypatch.setattr(retriever, "retrieve", lambda q, top_k=None: [])
    monkeypatch.setattr(llm_module, "generate", lambda prompt, system=None: "Không có thông tin.")

    ket_qua = rag_graph.build_rag_graph().invoke({"question": "câu hỏi lạ"})
    assert ket_qua["answer"] == "Không có thông tin."
    assert ket_qua["contexts"] == []


def test_rag_dung_dung_system_prompt(monkeypatch):
    """RAG phải dùng RAG_SYSTEM, không phải CHATBOT_SYSTEM."""
    system_da_nhan: list[str] = []

    monkeypatch.setattr(retriever, "retrieve", lambda q, top_k=None: ["ctx"])
    monkeypatch.setattr(
        llm_module,
        "generate",
        lambda prompt, system=None: system_da_nhan.append(system) or "ok",
    )

    rag_graph.build_rag_graph().invoke({"question": "q"})
    assert system_da_nhan[0] == prompts.RAG_SYSTEM
