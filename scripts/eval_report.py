"""Chạy toàn bộ metric ở chế độ CHỈ BÁO CÁO — không assert, không ngưỡng.

Đây là bước ĐẦU TIÊN khi đặt ngưỡng cho một dự án mới. Bạn không thể chọn ngưỡng
hợp lý khi chưa biết hệ thống của mình thực sự đạt bao nhiêu.

    uv run python scripts/eval_report.py

Kết quả in ra bảng điểm + phân bố, và ghi vào reports/eval_report.md.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from langchain_ollama import ChatOllama, OllamaEmbeddings  # noqa: E402
from ragas import SingleTurnSample  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from conftest import _cache_key, _load_recording  # noqa: E402
from rag_qa import config, prompts, retriever  # noqa: E402

CASES = json.loads(
    (ROOT / "tests" / "data" / "rag_test_cases.json").read_text(encoding="utf-8")
)["cases"]


def _response_da_ghi(question: str, contexts: list[str]) -> str | None:
    prompt = prompts.build_rag_prompt(question, contexts)
    return _load_recording(_cache_key(prompt, prompts.RAG_SYSTEM, config.APP_MODEL))


def main() -> int:
    llm = LangchainLLMWrapper(
        ChatOllama(model=config.JUDGE_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=0)
    )
    emb = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL)
    )

    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "response_relevancy": ResponseRelevancy(llm=llm, embeddings=emb),
        "context_precision": LLMContextPrecisionWithReference(llm=llm),
        "context_recall": LLMContextRecall(llm=llm),
    }

    print(f"Judge: {config.JUDGE_MODEL}   App: {config.APP_MODEL}\n")
    ket_qua: dict[str, dict[str, float]] = {k: {} for k in metrics}

    for case in CASES:
        contexts = retriever.retrieve(case["user_input"])
        response = _response_da_ghi(case["user_input"], contexts)
        if response is None:
            print(f"! thiếu bản ghi cho {case['id']} — chạy scripts/record_responses.py")
            continue

        sample = SingleTurnSample(
            user_input=case["user_input"],
            retrieved_contexts=contexts,
            response=response,
            reference=case["reference"],
        )
        print(f"[{case['nhom']:12}] {case['id']}")
        for ten, metric in metrics.items():
            started = time.perf_counter()
            try:
                score = float(metric.single_turn_score(sample))
            except Exception as exc:
                score = float("nan")
                print(f"    {ten:20} LỖI {type(exc).__name__}")
                continue
            ket_qua[ten][case["id"]] = score
            print(f"    {ten:20} {score:.3f}   ({time.perf_counter() - started:.0f}s)")

    # --- Tổng hợp ---
    print("\n" + "=" * 62)
    print(f"{'metric':22}{'min':>8}{'p25':>8}{'trung vị':>10}{'max':>8}")
    print("=" * 62)
    dong_bang = []
    for ten, diem in ket_qua.items():
        vals = sorted(v for v in diem.values() if v == v)
        if not vals:
            continue
        p25 = vals[max(0, int(len(vals) * 0.25) - 1)]
        row = (ten, min(vals), p25, statistics.median(vals), max(vals))
        dong_bang.append(row)
        print(f"{ten:22}{row[1]:8.3f}{row[2]:8.3f}{row[3]:10.3f}{row[4]:8.3f}")

    out = ROOT / "reports" / "eval_report.md"
    out.parent.mkdir(exist_ok=True)
    lines = [
        "# Báo cáo đánh giá (chỉ báo cáo, không assert)",
        "",
        f"- Judge: `{config.JUDGE_MODEL}`",
        f"- App: `{config.APP_MODEL}`",
        f"- top_k: {config.TOP_K}, chunk_size: {config.CHUNK_SIZE}",
        "",
        "## Phân bố điểm",
        "",
        "| metric | min | p25 | trung vị | max |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {t} | {a:.3f} | {b:.3f} | {c:.3f} | {d:.3f} |" for t, a, b, c, d in dong_bang]
    lines += ["", "## Điểm từng câu hỏi", "", "| case | " + " | ".join(ket_qua) + " |",
              "|---" * (len(ket_qua) + 1) + "|"]
    for case in CASES:
        cells = [f"{ket_qua[t].get(case['id'], float('nan')):.3f}" for t in ket_qua]
        lines.append(f"| {case['id']} | " + " | ".join(cells) + " |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nĐã ghi: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
