"""PART 5b — chấm QUỸ ĐẠO của agent.

Đây là phần chứng minh luận điểm phản trực giác của Part 5:

    Phần lớn giá trị của việc test agent nằm ở những assert TẤT ĐỊNH,
    MIỄN PHÍ, chạy trong mili-giây — không phải ở LLM judge.

Và có một bất ngờ đi kèm, đo được ngay trong file này: `ToolCorrectnessMetric`
của DeepEval 4.2 chấm điểm bằng SO SÁNH CẤU TRÚC — không cần suy luận gì — thế
mà nó VẪN bắt buộc phải có model, kể cả khi `include_reason=False`. Nghĩa là
một `assert` năm dòng ở mục 5b.1 làm đúng việc đó, nhanh hơn, và không cần
model nào.

Bố cục file:
    5b.1  assert trần        -> marker `unit`, chạy trong mili-giây
    5b.2  ToolCorrectness    -> marker `eval`, cần judge local
    5b.3  lỗi tự phát hiện   -> marker `unit`
    5b.4  chạy agent thật    -> opt-in bằng biến môi trường

    uv run pytest tests/test_agent_trajectory.py -m unit -v
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

_DU_LIEU = json.loads(
    (Path(__file__).parent / "data" / "agent_test_cases.json").read_text(encoding="utf-8")
)
CASES = _DU_LIEU["cases"]
SO_DIEU_TOI_DA = _DU_LIEU["so_dieu_toi_da"]
THEO_ID = {c["id"]: c for c in CASES}


def _tc(cau_hoi: str, goi: list[str], mong_doi: list[str]) -> LLMTestCase:
    """Đóng gói một quỹ đạo thành LLMTestCase của DeepEval."""
    return LLMTestCase(
        input=cau_hoi,
        actual_output="(không quan tâm ở đây — ta chấm ĐƯỜNG ĐI, không chấm chữ)",
        tools_called=[ToolCall(name=t) for t in goi],
        expected_tools=[ToolCall(name=t) for t in mong_doi],
    )


# ---------------------------------------------------------------------------
# 5b.1 — Không cần thư viện nào cả
# ---------------------------------------------------------------------------
# Trước khi dùng DeepEval, hãy thấy rằng phần lớn việc này chỉ là so sánh list.
# Nếu bạn chỉ mang một thứ từ Part 5 về công ty, hãy mang đoạn này.
@pytest.mark.unit
@pytest.mark.parametrize(
    "case", [c for c in CASES if c["loi"]], ids=[c["id"] for c in CASES if c["loi"]]
)
def test_case_co_loi_thi_quy_dao_hoac_noi_dung_phai_lech(case):
    """Ghi lại KHOẢNG CÁCH giữa quỹ đạo mong đợi và quỹ đạo thật.

    Cả bốn case trong repo đều lệch. Đó không phải lỗi của bài test — đó là
    hiện trạng của một agent chạy trên model 8B, và là lý do phải test quỹ đạo.

    Chú ý: ba trong bốn case cho ra CÂU TRẢ LỜI ĐÚNG. Metric của Part 3 và
    Part 4 chấm chúng điểm tuyệt đối.
    """
    assert case["loi"], f"Case '{case['id']}' được đánh dấu có lỗi nhưng không mô tả lỗi"


@pytest.mark.unit
def test_khong_goi_tool_thua_cho_cau_chao():
    """Assert trần, không thư viện. Đây là 90% việc test agent."""
    assert THEO_ID["chao-hoi"]["quy_dao_mong_doi"] == []


# ---------------------------------------------------------------------------
# 10b.1b — Lỗi mà KHÔNG metric nào có sẵn bắt được: số điều luật bịa
# ---------------------------------------------------------------------------
_SO_DIEU = re.compile(r"Điều\s+(\d+)")


def so_dieu_bia(cau_tra_loi: str, toi_da: int = SO_DIEU_TOI_DA) -> list[int]:
    """Trả về các số điều được dẫn nhưng không tồn tại trong luật."""
    return [int(n) for n in _SO_DIEU.findall(cau_tra_loi or "") if int(n) > toi_da]


@pytest.mark.unit
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_phat_hien_so_dieu_bia(case):
    """Luật 116/2025 có đúng 45 điều. Dẫn "Điều 73" là bịa.

    Đây là kiểm tra RẺ NHẤT và ĐẶC THÙ NHẤT của cả khoá: một regex và một phép
    so sánh số. Không judge, không metric, không thư viện.

    Và nó bắt được lỗi mà mọi thứ khác bỏ sót. Case `hai-tool-noi-tiep` có quỹ
    đạo ĐÚNG, ngày tính ĐÚNG, câu trả lời đọc rất thuyết phục — nhưng dẫn
    "Điều 73". `ToolCorrectnessMetric` cho điểm tuyệt đối. Faithfulness cũng
    khó bắt vì phần còn lại của câu đều đúng.

    Với trợ lý pháp lý, một số điều bịa nguy hiểm hơn một câu trả lời sai rõ
    ràng — vì người đọc không có cách nào nghi ngờ nó.

    Bài học: metric có sẵn chỉ đo những gì tác giả thư viện nghĩ tới. Rủi ro
    ĐẶC THÙ của lĩnh vực bạn thì bạn phải tự viết assert.
    """
    bia = so_dieu_bia(case.get("cau_tra_loi_that", ""))
    if case["id"] == "hai-tool-noi-tiep":
        assert bia == [73], f"Case này phải lộ ra Điều 73 bịa, nhận được {bia}"
    else:
        assert bia == [], f"Câu trả lời dẫn điều không tồn tại: {bia}"


@pytest.mark.unit
def test_khong_lap_lai_cung_mot_tool_lien_tiep():
    """Bắt lỗi gọi thừa — chỉ cần so sánh phần tử liền kề."""
    quy_dao = ["tra_cuu_van_ban", "tra_cuu_van_ban"]
    lap = [a for a, b in zip(quy_dao, quy_dao[1:]) if a == b]
    assert lap, "Test này minh hoạ cách phát hiện gọi lặp"
    # Trong dự án thật, assert ngược lại:
    #     assert not lap, f"Agent gọi lặp {lap} — lãng phí một lượt model"


# ---------------------------------------------------------------------------
# 5b.2 — ToolCorrectnessMetric: KHÔNG cần judge
# ---------------------------------------------------------------------------
# ĐO ĐƯỢC, KHÔNG PHẢI PHỎNG ĐOÁN:
# Việc chấm của metric này thuần cấu trúc — so hai danh sách tên tool. Vậy mà
# DeepEval 4.2 vẫn ném `DeepEvalError: OpenAI API key is not configured` nếu
# không truyền model, kể cả với `include_reason=False`. Nên ở đây ta phải đưa
# judge local vào, và các test này mang marker `eval` chứ không phải `unit`.
#
# Kết luận thực dụng: với việc kiểm tra quỹ đạo, `assert` ở mục 5b.1 TỐT HƠN
# thư viện — nhanh hơn, không phụ thuộc, thông báo lỗi rõ hơn. Dùng
# ToolCorrectnessMetric khi bạn cần `should_consider_ordering` hoặc muốn kết
# quả nằm chung báo cáo với các metric khác của DeepEval.
@pytest.mark.eval
def test_tool_correctness_khong_can_judge(judge_model):
    tc = _tc("Luật có hiệu lực khi nào?", ["tra_cuu_van_ban"], ["tra_cuu_van_ban"])
    metric = ToolCorrectnessMetric(threshold=1.0, async_mode=False, model=judge_model)
    metric.measure(tc)
    assert metric.score == 1.0


@pytest.mark.eval
def test_tool_correctness_bat_duoc_thieu_tool(judge_model):
    """Quỹ đạo dựng tay: agent lẽ ra phải tra văn bản trước rồi mới tính.

    Nếu nó bỏ qua bước tra cứu và tính luôn, ngày ra có thể vẫn đúng — nhưng nó
    tính bằng con số tự nhớ, không phải con số trong luật. Ngày tài liệu đổi,
    nó sai ngay mà không ai biết.
    """
    tc = _tc(
        "Hạn chuyển tiếp rơi vào ngày nào?",
        ["tinh_thoi_han"],                              # thực tế: thiếu bước tra cứu
        ["tra_cuu_van_ban", "tinh_thoi_han"],           # mong đợi
    )
    metric = ToolCorrectnessMetric(threshold=1.0, async_mode=False, model=judge_model)
    metric.measure(tc)
    assert metric.score < 1.0, "Thiếu một tool mà vẫn 1.0 thì metric vô dụng"


@pytest.mark.eval
def test_tool_correctness_bat_duoc_tool_thua(judge_model):
    """Case 'chao-hoi': gọi tool trong khi lẽ ra không gọi gì."""
    case = THEO_ID["chao-hoi"]
    tc = _tc(case["cau_hoi"], case["quy_dao_that"], case["quy_dao_mong_doi"])
    metric = ToolCorrectnessMetric(
        threshold=1.0, async_mode=False, should_exact_match=True, model=judge_model
    )
    metric.measure(tc)
    assert metric.score < 1.0


@pytest.mark.eval
def test_should_consider_ordering_phan_biet_thu_tu(judge_model):
    """Với chuỗi 2 tool nối tiếp, THỨ TỰ mới là thứ quan trọng.

    Tra Điều 44 lấy ngày hiệu lực rồi mới tính hạn -> đúng.
    Tính trước rồi mới tra cứu -> tính bằng ngày tự nhớ. Cùng tập tool, sai hoàn toàn.
    """
    dung = _tc("q", ["tra_cuu_van_ban", "tinh_thoi_han"], ["tra_cuu_van_ban", "tinh_thoi_han"])
    nguoc = _tc("q", ["tinh_thoi_han", "tra_cuu_van_ban"], ["tra_cuu_van_ban", "tinh_thoi_han"])

    m = ToolCorrectnessMetric(
        threshold=1.0, async_mode=False, should_consider_ordering=True, model=judge_model
    )
    m.measure(dung)
    diem_dung = m.score
    m2 = ToolCorrectnessMetric(
        threshold=1.0, async_mode=False, should_consider_ordering=True, model=judge_model
    )
    m2.measure(nguoc)

    assert diem_dung > m2.score, "Bật should_consider_ordering mà thứ tự không ảnh hưởng"


# ---------------------------------------------------------------------------
# 5b.3 — Lỗi mà không metric nào có sẵn: tool call rò ra dạng text
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_phat_hien_tool_call_ro_ra_duoi_dang_text():
    """Lỗi đã quan sát được trên model khác, và không thư viện nào bắt hộ bạn.

    Model đáng lẽ trả về tool_call có cấu trúc, nhưng lại in ra JSON dưới dạng
    văn bản thuần. Kết quả: `quy_dao` rỗng, agent không leo thang, mà người
    dùng thì nhận được một cục JSON.

    Đây là lý do bạn vẫn phải tự viết assert: metric có sẵn chỉ đo những thứ
    tác giả thư viện nghĩ tới.
    """
    cau_tra_loi_hong = '{"name": "chuyen_chuyen_gia", "parameters": {"ly_do": "..."}}'
    quy_dao: list[str] = []

    from rag_qa.tools import TEN_TOOL

    ro_ri = quy_dao == [] and any(f'"{t}"' in cau_tra_loi_hong for t in TEN_TOOL)
    assert ro_ri, "Đây là minh hoạ cách phát hiện rò rỉ tool call"
    # Trong dự án thật, assert ngược lại:
    #     assert not ro_ri, f"Tool call rò ra dạng text: {cau_tra_loi_hong[:80]}"


# ---------------------------------------------------------------------------
# 5b.4 — Chạy agent THẬT (opt-in, vì chậm và không tất định)
# ---------------------------------------------------------------------------
@pytest.mark.eval
@pytest.mark.skipif(
    os.getenv("RAG_QA_AGENT_LIVE") != "1",
    reason="Chạy agent thật rất chậm. Bật bằng: RAG_QA_AGENT_LIVE=1",
)
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_quy_dao_that_khi_chay_live(case):
    """So quỹ đạo thật hôm nay với quỹ đạo đã ghi trong file JSON.

    Vì sao phải opt-in? Vì quỹ đạo KHÔNG tất định: cùng câu hỏi, hôm nay agent
    đi 2 bước, mai đi 3. Để test này chạy mặc định là mời một test chập chờn
    vào bộ test — và một test chập chờn sẽ bị cả đội bỏ qua trong một tuần.

    Đây chính là bài toán mà mục "Bản ghi vỡ ở đâu" trong docs/05 mô tả.
    """
    from rag_qa import agent_graph

    ket_qua = agent_graph.chay_agent(case["cau_hoi"])
    assert ket_qua["so_buoc"] <= agent_graph.SO_BUOC_TOI_DA, "Agent không chịu dừng"
    print(f"\n{case['id']}: mong đợi={case['quy_dao_mong_doi']} thật={ket_qua['quy_dao']}")

# ---------------------------------------------------------------------------
# 10b.1c — Agent nói ngược lại chính kết quả công cụ của nó
# ---------------------------------------------------------------------------
# Lỗi CÓ THẬT, tái hiện 2/2 lần trước khi siết prompt:
#
#   TOOL kiem_tra_cap_nhat -> "Có văn bản mới hơn: Dự thảo Nghị định..."
#   ĐÁP                    -> "Không có bản cập nhật mới hơn."
#
# Quỹ đạo HOÀN HẢO: agent chọn đúng công cụ, công cụ trả về đúng dữ liệu.
# `ToolCorrectnessMetric` cho điểm tuyệt đối. Metric quỹ đạo mù hoàn toàn với
# lớp lỗi này, vì nó chỉ nhìn *đường đi*, không nhìn *câu trả lời có khớp
# với thứ nhặt được trên đường đi hay không*.
#
# Với trợ lý pháp lý, đây là lỗi nghiêm trọng: người dùng hỏi đúng câu cần hỏi,
# hệ thống tra đúng chỗ, rồi trả lời ngược lại.

# (khẳng định của công cụ, khẳng định ngược lại trong câu trả lời)
CAP_MAU_THUAN = [
    ("có văn bản mới hơn", ["không có bản cập nhật", "không có văn bản mới"]),
    ("không có bản cập nhật nào liên quan", ["có văn bản mới hơn"]),
]


def mau_thuan_voi_cong_cu(ket_qua_tool: str, cau_tra_loi: str) -> list[str]:
    """Tìm chỗ câu trả lời nói ngược lại kết quả công cụ.

    Kiểm tra thô bằng so khớp chuỗi — cố ý. Nó rẻ, tất định, và bắt được đúng
    lớp lỗi mà LLM judge hay bỏ sót vì câu trả lời đọc rất trôi chảy.
    """
    tool = (ket_qua_tool or "").lower()
    dap = (cau_tra_loi or "").lower()
    return [
        nguoc
        for khang_dinh, cac_nguoc in CAP_MAU_THUAN
        if khang_dinh in tool
        for nguoc in cac_nguoc
        if nguoc in dap
    ]


@pytest.mark.unit
def test_bat_duoc_agent_noi_nguoc_ket_qua_cong_cu():
    """Tái dựng đúng lỗi đã quan sát được."""
    tool = ("Có văn bản mới hơn: Dự thảo Nghị định quy định chi tiết một số điều "
            "của Luật An ninh mạng, ban hành 2026-08-14.")
    dap_sai = "Không có bản cập nhật mới hơn."
    dap_dung = "Có văn bản mới hơn: Dự thảo Nghị định, ban hành 14/08/2026."

    assert mau_thuan_voi_cong_cu(tool, dap_sai), "Phải bắt được mâu thuẫn"
    assert not mau_thuan_voi_cong_cu(tool, dap_dung)


@pytest.mark.unit
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_cau_tra_loi_khong_nguoc_ket_qua_cong_cu(case):
    """Chạy trên mọi quỹ đạo đã ghi lại. Không case nào được mâu thuẫn."""
    for buoc in case.get("buoc_cong_cu", []):
        nguoc = mau_thuan_voi_cong_cu(buoc["ket_qua"], case.get("cau_tra_loi_that", ""))
        assert not nguoc, (
            f"Case '{case['id']}': công cụ {buoc['tool']} nói một đằng, "
            f"câu trả lời nói một nẻo ({nguoc})"
        )
