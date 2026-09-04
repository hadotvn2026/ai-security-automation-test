"""Ghi trước câu trả lời của LLM ứng dụng cho toàn bộ test case.

Vì sao cần script này thay vì chỉ chạy `pytest --record`?

`pytest --record` cũng ghi được, nhưng nó chạy CẢ phần chấm điểm — nghĩa là
bạn phải chờ judge chạy xong mới có bản ghi. Script này chỉ gọi model ứng dụng,
nhanh hơn nhiều, và là thứ nên chạy khi bạn vừa đổi prompt.

    uv run python scripts/record_responses.py

Sau khi chạy xong, toàn bộ bộ test chạy offline và không gọi model ứng dụng nữa.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import _cache_key, _load_recording, _save_recording  # noqa: E402
from rag_qa import chat_graph, config, llm, prompts, rag_graph, retriever  # noqa: E402

# Cùng bộ câu hỏi với tests/test_chatbot_deepeval.py
CAU_HOI_CHATBOT = [
    "An ninh mạng là gì và vì sao doanh nghiệp cần quan tâm?",
    "Công ty tôi nên bắt đầu từ đâu để chuẩn bị tuân thủ quy định về an ninh mạng?",
    "Ai trong doanh nghiệp chịu trách nhiệm chính về an ninh mạng?",
    "Sự cố an ninh mạng thì việc đầu tiên cần làm là gì?",
]

CASES_PATH = ROOT / "tests" / "data" / "rag_test_cases.json"


def _ghi(prompt: str, system: str | None, nhan: str) -> bool:
    """Ghi một cặp (prompt, câu trả lời). Trả về True nếu đã gọi model."""
    key = _cache_key(prompt, system, config.APP_MODEL)
    if _load_recording(key) is not None:
        print(f"  bỏ qua (đã có): {nhan}")
        return False
    started = time.perf_counter()
    response = llm.generate(prompt, system)
    _save_recording(key, prompt, system, config.APP_MODEL, response)
    print(f"  ghi {key} ({time.perf_counter() - started:.1f}s): {nhan}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Xoá các bản ghi không còn khớp prompt hiện tại",
    )
    args = parser.parse_args()

    print(f"Model ứng dụng: {config.APP_MODEL}")
    khoa_dang_dung: set[str] = set()

    print("\n[1/2] Chatbot (Part 3)")
    for cau_hoi in CAU_HOI_CHATBOT:
        # Phải dựng prompt GIỐNG HỆT chat_graph.generate_node, nếu không khoá
        # băm sẽ khác và bản ghi vô dụng.
        khoa_dang_dung.add(_cache_key(cau_hoi, prompts.CHATBOT_SYSTEM, config.APP_MODEL))
        _ghi(cau_hoi, prompts.CHATBOT_SYSTEM, cau_hoi[:45])

    print("\n[2/2] RAG (Part 4)")
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    for case in cases:
        contexts = retriever.retrieve(case["user_input"])
        prompt = prompts.build_rag_prompt(case["user_input"], contexts)
        khoa_dang_dung.add(_cache_key(prompt, prompts.RAG_SYSTEM, config.APP_MODEL))
        _ghi(prompt, prompts.RAG_SYSTEM, case["id"])

    thu_muc = ROOT / "tests" / "fixtures" / "responses"
    cu = [f for f in thu_muc.glob("*.json") if f.stem not in khoa_dang_dung]

    if cu and args.prune:
        for f in cu:
            f.unlink()
        print(f"\nĐã xoá {len(cu)} bản ghi cũ (prompt đã đổi).")
    elif cu:
        # Bản ghi thừa không làm test sai — khoá băm không khớp thì không ai đọc
        # tới. Nhưng để lâu sẽ không còn ai biết bản nào còn dùng.
        print(f"\n{len(cu)} bản ghi không còn khớp prompt hiện tại.")
        print("Xoá bằng: uv run python scripts/record_responses.py --prune")

    tong = len(list(thu_muc.glob("*.json")))
    print(f"\nTổng số bản ghi: {tong}")
    print("Giờ chạy được offline: uv run pytest -m eval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
