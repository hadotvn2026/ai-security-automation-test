"""PART 5a — test đường ống của AGENT. Không model, không judge, không tiền.

Đây là bằng chứng cho luận điểm quan trọng nhất của Part 5:

    Agent khó XÂY hơn RAG, nhưng phần lớn việc TEST nó lại quay về đúng
    pytest cơ bản mà bạn đã học ở Part 1.

Toàn bộ file này chạy trong dưới một giây và bắt được những lỗi đắt nhất của
một agent: đi sai nhánh, gọi sai công cụ, lặp vô hạn, quên ghi nhớ.

    uv run pytest tests/test_agent_plumbing.py -v
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END

from rag_qa import agent_graph, llm as llm_module, retriever
from rag_qa.agent_graph import SO_BUOC_TOI_DA, dinh_tuyen, tools_node

pytestmark = pytest.mark.unit


def _ai_goi_tool(*goi: tuple[str, dict]) -> AIMessage:
    """Dựng tay một AIMessage có tool_calls — thay cho câu trả lời thật của model."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": ten, "args": args, "id": f"call-{i}"}
            for i, (ten, args) in enumerate(goi)
        ],
    )


# ---------------------------------------------------------------------------
# 5a.1 — RẼ NHÁNH: dòng code biến pipeline thành agent
# ---------------------------------------------------------------------------
def test_khong_co_tool_call_thi_ket_thuc():
    state = {"messages": [AIMessage(content="Xong rồi ạ.")], "so_buoc": 1}
    assert dinh_tuyen(state) == END


def test_co_tool_call_thi_di_sang_node_tools():
    state = {"messages": [_ai_goi_tool(("tinh_thoi_han", {"ngay_bat_dau": "01/07/2026", "so_thang": 12}))], "so_buoc": 1}
    assert dinh_tuyen(state) == "tools"


@pytest.mark.parametrize("so_buoc", [SO_BUOC_TOI_DA, SO_BUOC_TOI_DA + 1, 99])
def test_cham_tran_so_buoc_thi_dung_du_van_con_tool_call(so_buoc):
    """Test QUAN TRỌNG NHẤT của Part 5.

    Agent không có trần số bước là agent có thể chạy mãi: mỗi vòng một lời gọi
    model, mỗi lời gọi một khoản tiền. Đây là loại lỗi mà RAG tuyến tính không
    thể có, và không một metric chất lượng nào bắt được — câu trả lời cuối vẫn
    có thể hoàn hảo trong khi hoá đơn gấp mười lần.

    Đúng ba dòng `assert`, không cần model, không cần judge.
    """
    state = {
        "messages": [_ai_goi_tool(("tra_cuu_van_ban", {"cau_hoi": "x"}))],
        "so_buoc": so_buoc,
    }
    assert dinh_tuyen(state) == END


# ---------------------------------------------------------------------------
# 5a.2 — NODE TOOLS: thực thi đúng công cụ, đúng tham số
# ---------------------------------------------------------------------------
def test_goi_dung_cong_cu_va_tra_ve_ket_qua():
    state = {"messages": [_ai_goi_tool(("tinh_thoi_han", {"ngay_bat_dau": "01/07/2026", "so_thang": 12}))]}
    ket_qua = tools_node(state)
    assert ket_qua["messages"][0].content == "01/07/2027"
    assert isinstance(ket_qua["messages"][0], ToolMessage)


def test_ghi_lai_quy_dao_dung_thu_tu():
    """`quy_dao` là thứ Part 5b sẽ assert. Nó phải đúng thứ tự."""
    state = {
        "messages": [
            _ai_goi_tool(
                ("tra_cuu_van_ban", {"cau_hoi": "x"}),
                ("tinh_thoi_han", {"ngay_bat_dau": "01/07/2026", "so_thang": 12}),
            )
        ]
    }
    assert tools_node(state)["quy_dao"] == ["tra_cuu_van_ban", "tinh_thoi_han"]


def test_scratchpad_luu_ket_qua_de_buoc_sau_dung_lai():
    """MEMORY 2. Không có scratchpad thì không có chuỗi 2 tool nối tiếp."""
    state = {"messages": [_ai_goi_tool(("tinh_thoi_han", {"ngay_bat_dau": "01/07/2026", "so_thang": 6}))]}
    ghi_chu = tools_node(state)["scratchpad"]
    assert ghi_chu == [{"tool": "tinh_thoi_han",
                        "args": {"ngay_bat_dau": "01/07/2026", "so_thang": 6},
                        "ket_qua": "01/01/2027"}]


def test_goi_cong_cu_khong_ton_tai_thi_bao_loi_chu_khong_sap():
    """Model bịa ra tên công cụ là chuyện xảy ra thật. Agent không được nổ."""
    state = {"messages": [_ai_goi_tool(("cong_cu_ma", {}))]}
    ket_qua = tools_node(state)
    assert "Không có công cụ" in ket_qua["messages"][0].content
    assert ket_qua["quy_dao"] == ["cong_cu_ma"]


def test_tool_nhan_bieu_thuc_hong_thi_tra_thong_bao_khong_nem_loi():
    state = {"messages": [_ai_goi_tool(("tinh_thoi_han", {"ngay_bat_dau": "hôm nay", "so_thang": 3}))]}
    assert "Không đọc được ngày" in tools_node(state)["messages"][0].content


# ---------------------------------------------------------------------------
# 5a.3 — QUỸ ĐẠO đầy đủ, với model bị mock hoàn toàn
# ---------------------------------------------------------------------------
@pytest.fixture
def model_gia(monkeypatch):
    """Cho phép mỗi test tự viết kịch bản model sẽ 'quyết định' những gì.

    Đây chính là kỹ thuật `monkeypatch` của Part 3, áp lên `invoke_with_tools`
    thay vì `generate`. Vì agent chỉ chạm model qua đúng một hàm đó, một dòng
    patch là đủ để điều khiển toàn bộ quỹ đạo.
    """

    def dat_kich_ban(*tra_loi):
        hang_doi = list(tra_loi)
        monkeypatch.setattr(
            llm_module, "invoke_with_tools", lambda messages, tools: hang_doi.pop(0)
        )

    return dat_kich_ban


def test_quy_dao_hai_tool_noi_tiep(model_gia, monkeypatch):
    """Kịch bản đắt giá nhất: tra Điều 44 lấy ngày hiệu lực -> tính hạn 12 tháng."""
    monkeypatch.setattr(retriever, "retrieve", lambda q, top_k=None: ["Điều 44: có hiệu lực từ ngày 01/07/2026"])
    model_gia(
        _ai_goi_tool(("tra_cuu_van_ban", {"cau_hoi": "ngày hiệu lực"})),
        _ai_goi_tool(("tinh_thoi_han", {"ngay_bat_dau": "01/07/2026", "so_thang": 12})),
        AIMessage(content="Hạn chuyển tiếp là ngày 01/07/2027."),
    )

    ket_qua = agent_graph.chay_agent("Hạn chuyển tiếp 12 tháng rơi vào ngày nào?")

    assert ket_qua["quy_dao"] == ["tra_cuu_van_ban", "tinh_thoi_han"]
    assert ket_qua["cau_tra_loi"] == "Hạn chuyển tiếp là ngày 01/07/2027."
    assert ket_qua["scratchpad"][1]["ket_qua"] == "01/07/2027"


def test_cau_chao_hoi_khong_duoc_goi_tool_nao(model_gia):
    model_gia(AIMessage(content="Chào bạn! Tôi giúp gì được ạ?"))
    ket_qua = agent_graph.chay_agent("Chào bạn")
    assert ket_qua["quy_dao"] == []
    assert ket_qua["so_buoc"] == 1


def test_agent_dung_lai_khi_cham_tran_du_model_van_doi_goi_tool(model_gia, monkeypatch):
    """Model 'bướng' đòi gọi tool mãi — trần số bước phải chặn được nó."""
    monkeypatch.setattr(retriever, "retrieve", lambda q, top_k=None: ["..."])
    monkeypatch.setattr(
        llm_module,
        "invoke_with_tools",
        lambda messages, tools: _ai_goi_tool(("tra_cuu_van_ban", {"cau_hoi": "x"})),
    )

    ket_qua = agent_graph.chay_agent("câu hỏi khiến agent lặp")

    assert ket_qua["so_buoc"] == SO_BUOC_TOI_DA
    assert len(ket_qua["quy_dao"]) < SO_BUOC_TOI_DA + 1


def test_memory_giu_duoc_lich_su_hoi_thoai(model_gia):
    """MEMORY 1. Lịch sử truyền vào phải nằm trong messages gửi cho model."""
    model_gia(AIMessage(content="Vâng ạ."))
    lich_su = [HumanMessage(content="Tên tôi là Hà"), AIMessage(content="Chào anh Hà")]

    ket_qua = agent_graph.chay_agent("Tôi vừa nói tên gì?", lich_su=lich_su)

    noi_dung = [str(m.content) for m in ket_qua["messages"]]
    assert "Tên tôi là Hà" in noi_dung
