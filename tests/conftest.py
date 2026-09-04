"""Hạ tầng dùng chung cho toàn bộ test.

Ba thứ được định nghĩa ở đây:

1. `recorded_llm` — cơ chế ghi/phát lại câu trả lời của LLM ỨNG DỤNG.
2. `judge_*`      — model chấm điểm cho DeepEval và RAGAS. LUÔN chạy thật.
3. `measure_with_retry` / `assert_metric` — xử lý trường hợp judge trả về NaN.

RANH GIỚI QUAN TRỌNG NHẤT CỦA KHOÁ HỌC
--------------------------------------
Mỗi test đánh giá có HAI lời gọi model, và chúng hoàn toàn khác nhau:

    LLM của ứng dụng  -> sinh ra câu trả lời ĐƯỢC CHẤM
    LLM của judge     -> sinh ra ĐIỂM SỐ

Mock cái thứ nhất = phát lại câu trả lời đã ghi. Hợp lệ: nhanh, tất định,
chạy offline, và metric vẫn làm việc thật trên câu trả lời đó.

Mock cái thứ hai = hardcode điểm số. Vô nghĩa: bộ test sẽ xanh kể cả khi bạn
xoá sạch ứng dụng. Trong repo này judge KHÔNG BAO GIỜ bị mock.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_qa import config  # noqa: E402

RESPONSES_DIR = Path(__file__).parent / "fixtures" / "responses"


# ---------------------------------------------------------------------------
# Tuỳ chọn dòng lệnh
# ---------------------------------------------------------------------------
def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--record",
        action="store_true",
        default=False,
        help="Gọi model thật và GHI LẠI câu trả lời vào tests/fixtures/responses/",
    )
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Gọi model thật nhưng KHÔNG ghi lại. Dùng để kiểm chứng bản ghi cũ.",
    )


# ---------------------------------------------------------------------------
# Ghi / phát lại câu trả lời của LLM ứng dụng
# ---------------------------------------------------------------------------
def _cache_key(prompt: str, system: str | None, model: str) -> str:
    """Khoá cache = băm của (model, system, prompt).

    Đổi prompt hoặc đổi model -> khoá đổi -> bản ghi cũ không còn được dùng.
    Đó là điều ta MUỐN: sửa prompt xong mà test vẫn xanh nhờ bản ghi cũ thì
    bộ test đang nói dối.
    """
    payload = json.dumps(
        {"model": model, "system": system or "", "prompt": prompt},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _load_recording(key: str) -> str | None:
    path = RESPONSES_DIR / f"{key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["response"]


def _save_recording(key: str, prompt: str, system: str | None, model: str, response: str) -> None:
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    (RESPONSES_DIR / f"{key}.json").write_text(
        json.dumps(
            {
                "model": model,
                "system": system,
                "prompt": prompt,
                "response": response,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


class MissingRecordingError(AssertionError):
    """Chưa có bản ghi cho prompt này và cũng không chạy ở chế độ --record."""


@pytest.fixture(autouse=True)
def recorded_llm(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """Thay `rag_qa.llm.generate` bằng bản ghi/phát lại.

    Vì `llm.generate` là điểm chạm model DUY NHẤT của ứng dụng, thay đúng một
    hàm này là đủ cho cả chatbot lẫn RAG — không phải sửa graph.

    Ba chế độ:
        (mặc định)  phát lại bản ghi. Không có bản ghi -> báo lỗi rõ ràng.
        --record    gọi thật rồi ghi lại.
        --live      gọi thật, không ghi.
    """
    from rag_qa import llm as llm_module

    record = request.config.getoption("--record")
    live = request.config.getoption("--live")
    real_generate = llm_module.generate

    calls: list[dict[str, Any]] = []

    def fake_generate(prompt: str, system: str | None = None) -> str:
        key = _cache_key(prompt, system, config.APP_MODEL)
        calls.append({"key": key, "prompt": prompt, "system": system})

        if live:
            return real_generate(prompt, system)

        cached = _load_recording(key)
        if cached is not None and not record:
            return cached

        if not record:
            raise MissingRecordingError(
                f"Chưa có bản ghi cho prompt này (key={key}).\n"
                f"Chạy lại kèm --record để ghi:\n"
                f"    uv run pytest {request.node.nodeid} --record\n"
                f"Prompt (200 ký tự đầu): {prompt[:200]!r}"
            )

        response = real_generate(prompt, system)
        _save_recording(key, prompt, system, config.APP_MODEL, response)
        return response

    monkeypatch.setattr(llm_module, "generate", fake_generate)
    yield calls


# ---------------------------------------------------------------------------
# Judge — LUÔN chạy thật, không bao giờ mock
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def judge_model():
    """Model chấm điểm cho DeepEval (Ollama local)."""
    from deepeval.models import OllamaModel

    return OllamaModel(
        model=config.JUDGE_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0,
    )


@pytest.fixture(scope="session")
def ragas_llm():
    """Model chấm điểm cho RAGAS, bọc qua LangChain."""
    from langchain_ollama import ChatOllama
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(
        ChatOllama(
            model=config.JUDGE_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,
            client_kwargs={"timeout": config.REQUEST_TIMEOUT},
        )
    )


@pytest.fixture(scope="session")
def ragas_embeddings():
    """Embedding cho các metric của RAGAS cần so sánh ngữ nghĩa."""
    from langchain_ollama import OllamaEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(
        OllamaEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )
    )


# ---------------------------------------------------------------------------
# Xử lý NaN — hệ quả trực tiếp của việc dùng judge 8B chạy local
# ---------------------------------------------------------------------------
class JudgeTimeoutError(AssertionError):
    """Judge không trả lời trong thời gian cho phép.

    Vì sao cần lớp lỗi riêng, và vì sao phải bọc timeout bằng thread?

    Trong lúc dựng repo này, một request kẹt phía Ollama đã treo bộ test HƠN 15
    PHÚT ở 0% CPU. Không có thông báo, không có gì để debug — cả đội chỉ biết
    bấm Ctrl-C. Tệ hơn nữa: Ollama chỉ tự phục hồi SAU KHI client thoát, nên
    càng chờ càng vô ích.

    `client_kwargs={"timeout": ...}` giải quyết được phần lớn, nhưng không phải
    mọi client đều tôn trọng nó (DeepEval `OllamaModel` không nhận tham số này).
    Bọc bằng thread là lớp phòng thủ cuối: dù client có treo, test vẫn ĐỎ sau
    N giây thay vì treo mãi.

    Nguyên tắc mang về công ty: MỌI lời gọi ra ngoài trong bộ test phải có trần
    thời gian. Một test đỏ thì sửa được; một test treo thì bị bỏ qua.
    """


def _goi_co_tran_thoi_gian(ham: Callable[[], float], giay: int, nhan: str) -> float:
    """Chạy `ham()` trong thread riêng, bỏ cuộc sau `giay` giây."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        tuong_lai = pool.submit(ham)
        try:
            return tuong_lai.result(timeout=giay)
        except concurrent.futures.TimeoutError as exc:
            raise JudgeTimeoutError(
                f"[{nhan}] judge không trả lời sau {giay} giây.\n"
                f"Thường là Ollama bị kẹt. Kiểm tra bằng:\n"
                f"    curl -m 10 http://localhost:11434/api/tags\n"
                f"Chỉnh trần bằng biến môi trường RAG_QA_TIMEOUT."
            ) from exc


class JudgeParseError(AssertionError):
    """Judge trả về NaN: nó không sinh nổi JSON hợp lệ cho metric.

    Đây là vấn đề CHẤT LƯỢNG JUDGE, không phải lỗi của ứng dụng.
    """


def measure_with_retry(
    measure: Callable[[], float],
    attempts: int = 3,
    label: str = "metric",
) -> float:
    """Gọi `measure()` tối đa `attempts` lần, bỏ qua kết quả NaN.

    Model local cỡ 8B thỉnh thoảng sinh JSON hỏng, và cả DeepEval lẫn RAGAS
    đều trả về NaN khi không parse được. NaN không phải điểm thấp — nó là
    "không đo được". Thử lại vài lần thường là đủ.
    """
    scores: list[float] = []
    for attempt in range(1, attempts + 1):
        try:
            score = float(
                _goi_co_tran_thoi_gian(measure, config.REQUEST_TIMEOUT, label)
            )
        except JudgeTimeoutError:
            raise  # treo là treo, thử lại cũng vô ích
        except Exception as exc:  # judge lỗi mạng / lỗi parse
            scores.append(float("nan"))
            if attempt == attempts:
                raise JudgeParseError(
                    f"[{label}] judge lỗi sau {attempts} lần thử: {exc}\n"
                    f"Đây là vấn đề chất lượng judge, không phải lỗi ứng dụng.\n"
                    f"Thử: RAG_QA_JUDGE_MODEL=<model lớn hơn> uv run pytest ..."
                ) from exc
            continue
        if not math.isnan(score):
            return score
        scores.append(score)

    raise JudgeParseError(
        f"[{label}] judge trả về NaN {attempts} lần liên tiếp.\n"
        f"NaN nghĩa là judge không sinh nổi JSON hợp lệ — KHÔNG phải điểm thấp.\n"
        f"Judge hiện tại: {config.JUDGE_MODEL}\n"
        f"Xem mục 'Khi judge yếu' trong docs/04-ragas-rag.md."
    )


def assert_metric(score: float, threshold: float, label: str) -> None:
    """Assert điểm số kèm thông điệp đọc được."""
    assert score >= threshold, (
        f"[{label}] {score:.3f} < ngưỡng {threshold:.2f}"
    )
