"""PART 4 — chấm điểm hệ RAG bằng RAGAS.

Khác biệt cốt lõi so với Part 3: ở đây có NGỮ CẢNH. Nhờ đó ta tách được hai
loại lỗi mà Part 3 không phân biệt nổi:

    Faithfulness thấp        -> mô hình bịa. Lỗi ở prompt/Composer.
    Context Precision thấp   -> lấy về nhiều rác. Lỗi ở retriever.
    Context Recall thấp      -> lấy thiếu thông tin cần. Lỗi ở retriever/chunk.
    Response Relevancy thấp  -> trả lời lạc đề. Lỗi ở prompt.

Đó là lý do RAGAS đáng học: nó chỉ đúng vào bộ phận hỏng, thay vì chỉ nói
"câu trả lời tệ".

Bốn metric dưới đây tương ứng đúng bốn hộp cuối trong sơ đồ kiến trúc.

Lần đầu phải ghi lại câu trả lời của ứng dụng:
    uv run pytest tests/test_rag_ragas.py --record

Các lần sau:
    uv run pytest tests/test_rag_ragas.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ragas import SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from conftest import assert_metric, measure_with_retry
from rag_qa import rag_graph

pytestmark = pytest.mark.eval

CASES_PATH = Path(__file__).parent / "data" / "rag_test_cases.json"
CASES = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]

# ---------------------------------------------------------------------------
# NGƯỠNG — rút ra từ dữ liệu quan sát được trên chính corpus này
# ---------------------------------------------------------------------------
# Nguồn: `uv run python scripts/eval_report.py` (chế độ chỉ báo cáo).
# Xem docs/eval_report-mau.md.
#
# Điểm thấp nhất đo được (bỏ qua case cap-do-5, xem mục 9.4 của Session 9):
#     faithfulness        su-kien-don 0.500  |  tong-hop 0.400
#     response_relevancy  su-kien-don 0.630
#     context_precision   su-kien-don 1.000
#     context_recall      su-kien-don 0.500
#
# QUY TẮC: ngưỡng = (thấp nhất quan sát được) − (biên dao động của judge ≈ 0.10).
#
# Vì sao các ngưỡng này THẤP? Vì judge là model 8B chạy local, và nó chấm nhiễu
# trên văn bản pháp luật. Ví dụ `hieu-luc-thi-hanh` bị chấm faithfulness 0.500
# dù câu trả lời trích nguyên văn từ Điều 44.
#
# Đây là đánh đổi có ý thức: với judge yếu, một cổng chặn LỎNG MÀ ỔN ĐỊNH tốt
# hơn một cổng chặt mà chập chờn — vì cổng chập chờn sẽ bị cả đội bỏ qua trong
# một tuần. Có judge mạnh hơn thì siết lại. Đừng chép các con số này sang dự án
# khác: chạy eval_report.py trên dữ liệu của chính bạn.
NGUONG = {
    "faithfulness_su_kien_don": 0.40,   # thấp nhất 0.500
    "faithfulness_tong_hop": 0.30,      # thấp nhất 0.400
    "response_relevancy": 0.50,         # thấp nhất 0.630
    "context_precision": 0.80,          # thấp nhất 1.000
    "context_recall": 0.40,             # thấp nhất 0.500
}

@pytest.fixture(scope="module")
def ket_qua_ung_dung() -> dict[str, tuple[str, list[str]]]:
    """Chạy RAG một lần cho mỗi câu hỏi, dùng lại cho cả bốn metric.

    scope="module" là chi tiết quan trọng: không có nó, mỗi metric sẽ gọi lại
    ứng dụng một lần và thời gian chạy nhân bốn.
    """
    return {}


@pytest.fixture
def sample(request, ket_qua_ung_dung) -> SingleTurnSample:
    """Dựng SingleTurnSample — cấu trúc dữ liệu mà RAGAS chấm điểm.

    Bốn trường, bốn nguồn khác nhau:
        user_input         <- file rag_test_cases.json
        retrieved_contexts <- retriever THẬT của ứng dụng
        response           <- câu trả lời (phát lại từ bản ghi)
        reference          <- câu trả lời chuẩn do người viết
    """
    case = request.param
    if case["id"] not in ket_qua_ung_dung:
        ket_qua_ung_dung[case["id"]] = rag_graph.answer_with_contexts(case["user_input"])
    answer, contexts = ket_qua_ung_dung[case["id"]]

    return SingleTurnSample(
        user_input=case["user_input"],
        retrieved_contexts=contexts,
        response=answer,
        reference=case["reference"],
    )


def _params(nhom: str | None = None):
    cases = [c for c in CASES if nhom is None or c["nhom"] == nhom]
    return pytest.mark.parametrize(
        "sample", cases, indirect=True, ids=[c["id"] for c in cases]
    )


# ---------------------------------------------------------------------------
# 9.1 — Faithfulness: câu trả lời có bám vào ngữ cảnh không?
# ---------------------------------------------------------------------------
@_params("su-kien-don")
def test_faithfulness_su_kien_don(sample, ragas_llm):
    """Mọi khẳng định trong câu trả lời phải truy ngược được về ngữ cảnh.

    Điểm thấp = mô hình bịa. Đây là lỗi nghiêm trọng nhất của một hệ RAG.

    Vì sao nhóm `bay` không có mặt ở đây? Câu trả lời của nhóm đó là một lời
    TỪ CHỐI ("Tài liệu không đề cập..."). Nó không chứa mệnh đề nào để đối
    chiếu với ngữ cảnh, nên điểm faithfulness của nó vô nghĩa. Nhóm `bay` đã
    có test riêng ở mục 4.5, và test đó không cần judge.
    """
    metric = Faithfulness(llm=ragas_llm)
    score = measure_with_retry(
        lambda: metric.single_turn_score(sample), label="faithfulness"
    )
    assert_metric(
        score, NGUONG["faithfulness_su_kien_don"], f"faithfulness | {sample.user_input[:40]}"
    )


@_params("tong-hop")
def test_faithfulness_tong_hop(sample, ragas_llm):
    """Cùng metric, ngưỡng thấp hơn — xem phần bình luận ở NGUONG."""
    metric = Faithfulness(llm=ragas_llm)
    score = measure_with_retry(
        lambda: metric.single_turn_score(sample), label="faithfulness"
    )
    assert_metric(
        score, NGUONG["faithfulness_tong_hop"], f"faithfulness | {sample.user_input[:40]}"
    )


# ---------------------------------------------------------------------------
# 9.2 — Response Relevancy
# ---------------------------------------------------------------------------
# Hai case hỏng, HAI NGUYÊN NHÂN HOÀN TOÀN KHÁC NHAU. Phân biệt được chúng là
# kỹ năng chính của Session 9.
#
#   cap-do-5            -> MODEL ỨNG DỤNG không rút được câu trả lời khỏi ngữ
#                          cảnh dù nó nằm nguyên văn ở đó. Lỗi bên bị chấm.
#   so-cap-do-he-thong  -> MODEL JUDGE không sinh nổi JSON hợp lệ để chấm. Đo
#                          5 lần liên tiếp, 5 lần OutputParserException. Lỗi
#                          bên đi chấm. Ứng dụng trả lời hoàn toàn đúng.
#
# Cùng hiện ra là "test đỏ". Sửa ở hai chỗ khác nhau.
XFAIL_CO_LY_DO = {
    "hệ thống thông tin cấp độ 5": (
        "MODEL ỨNG DỤNG: llama3.1:8b không trích được câu trả lời dù ngữ cảnh "
        "có nguyên văn. Đã loại trừ retriever và prompt bằng 3 bước ở Session 9."
    ),
    "phân loại thành bao nhiêu cấp độ": (
        "MODEL JUDGE: ResponseRelevancy ném OutputParserException 5/5 lần trên "
        "chính input này. Ứng dụng trả lời ĐÚNG — lỗi nằm ở bên chấm điểm."
    ),
}


def _ly_do_xfail(cau_hoi: str) -> str | None:
    thap = cau_hoi.lower()
    for manh, ly_do in XFAIL_CO_LY_DO.items():
        if manh in thap:
            return ly_do
    return None


@_params("su-kien-don")
def test_response_relevancy(sample, ragas_llm, ragas_embeddings, request):
    """Câu trả lời có thực sự trả lời đúng câu được hỏi không?

    Hai case mang `xfail` — KHÔNG phải để giấu lỗi, mà vì đã khoanh vùng được
    nguyên nhân và cả hai đều nằm ngoài tầm sửa của bộ test. Đọc `XFAIL_CO_LY_DO`
    ở trên: một lỗi thuộc model ứng dụng, một lỗi thuộc model judge.

    Quy trình khoanh vùng ba bước (truy hồi -> ngữ cảnh tối thiểu -> prompt tối
    giản) nằm ở Session 9 mục 9.4.
    """
    ly_do = _ly_do_xfail(sample.user_input)
    if ly_do:
        request.node.add_marker(pytest.mark.xfail(reason=ly_do, strict=False))
    metric = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    score = measure_with_retry(
        lambda: metric.single_turn_score(sample), label="response_relevancy"
    )
    assert_metric(
        score, NGUONG["response_relevancy"], f"response_relevancy | {sample.user_input[:40]}"
    )


# ---------------------------------------------------------------------------
# 9.3 — Context Precision: trong những gì lấy về, bao nhiêu là hữu ích?
# ---------------------------------------------------------------------------
@_params("su-kien-don")
def test_context_precision(sample, ragas_llm):
    """Đây là metric chấm điểm RETRIEVER, không phải chấm mô hình sinh.

    Điểm thấp = retriever lấy về nhiều rác. Sửa ở retriever.py / top_k / cách cắt chunk,
    sửa prompt không giúp gì.
    """
    metric = LLMContextPrecisionWithReference(llm=ragas_llm)
    score = measure_with_retry(
        lambda: metric.single_turn_score(sample), label="context_precision"
    )
    assert_metric(
        score, NGUONG["context_precision"], f"context_precision | {sample.user_input[:40]}"
    )


# ---------------------------------------------------------------------------
# 9.4 — Context Recall
# ---------------------------------------------------------------------------
@_params("su-kien-don")
def test_context_recall(sample, ragas_llm):
    """Cặp đôi với Context Precision.

    Precision cao + Recall thấp = lấy ít mà đúng, nhưng thiếu.
    Precision thấp + Recall cao = lấy đủ nhưng lẫn nhiều rác.

    Trên corpus luật với cách cắt theo Điều, cả hai chỉ số đều tốt: precision
    đạt 1.000 ở mọi câu. Đó là kết quả trực tiếp của việc cắt theo cấu trúc
    thay vì theo số ký tự (Session 2).
    """
    metric = LLMContextRecall(llm=ragas_llm)
    score = measure_with_retry(
        lambda: metric.single_turn_score(sample), label="context_recall"
    )
    assert_metric(score, NGUONG["context_recall"], f"context_recall | {sample.user_input[:40]}")


# ---------------------------------------------------------------------------
# 9.5 — Nhóm câu hỏi bẫy: tài liệu KHÔNG chứa câu trả lời
# ---------------------------------------------------------------------------
@_params("bay")
def test_cau_hoi_bay_thi_phai_noi_khong_biet(sample):
    """Không cần LLM judge — chỉ cần kiểm tra ứng dụng có chịu im lặng không.

    Test rẻ nhất trong cả Part 4 và bắt được lỗi đắt nhất. Một hệ RAG bịa ra
    giá sản phẩm hay tên giám đốc là hỏng nặng hơn nhiều so với một hệ trả lời
    hơi lạc đề.
    """
    tu_choi = ["không đề cập", "không có thông tin", "không tìm thấy", "không biết"]
    assert any(t in sample.response.lower() for t in tu_choi), (
        f"Ứng dụng đã BỊA thay vì từ chối.\n"
        f"Câu hỏi: {sample.user_input}\n"
        f"Trả lời: {sample.response}"
    )
