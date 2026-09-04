"""Sinh thư mục snapshots/ — bản sao cứu hộ sau các session mốc.

Repo này lớn dần qua 10 session. Nếu đến Session 7 mà máy bạn hỏng, bạn không
cần làm lại từ đầu: copy đè thư mục snapshot tương ứng là bắt kịp lớp.

    cp -r snapshots/sau-bai-06/. .

Các file guides không bao giờ nhắc tới thư mục này — nó chỉ nằm đó phòng khi cần.

    uv run python scripts/make_snapshots.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "snapshots"

CHUNG = [
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    "README.md",
    "src/rag_qa/__init__.py",
    "src/rag_qa/config.py",
]

S02 = CHUNG + [
    "src/rag_qa/chunker.py",
    "tests/test_chunker.py",
]

# Xong Bài 6: toàn bộ ứng dụng agentic — RAG, 4 công cụ, vault, API, giao diện.
S06 = S02 + [
    "src/rag_qa/document_loader.py",
    "src/rag_qa/embeddings.py",
    "src/rag_qa/vector_store.py",
    "src/rag_qa/retriever.py",
    "src/rag_qa/llm.py",
    "src/rag_qa/prompts.py",
    "src/rag_qa/chat_graph.py",
    "src/rag_qa/rag_graph.py",
    "src/rag_qa/cli.py",
    "src/rag_qa/vault.py",
    "src/rag_qa/tools.py",
    "src/rag_qa/agent_graph.py",
    "src/rag_qa/api.py",
    "web/index.html",
    "data/vault/manifest.json",
    "data/luat-116-2025-an-ninh-mang.docx",
    "scripts/ocr_pdf.py",
    "scripts/build_corpus.py",
    "scripts/ingest.py",
]

# Xong Bài 8: test đường ống + ghi/phát lại + test vault/API (đều không gọi model).
S08 = S06 + [
    "tests/conftest.py",
    "tests/test_plumbing.py",
    "tests/test_vault.py",
    "tests/test_api.py",
    "scripts/record_responses.py",
    "src/rag_qa/an_toan.py",
    "tests/test_an_toan.py",
    "scripts/scan_bao_mat.py",
    "data/bao-mat-mien-tru.json",
    "src/rag_qa/tri_nho.py",
    "tests/test_tri_nho.py",
    "scripts/tri_nho.py",
]

# Xong Bài 11: đủ bộ test đánh giá chất lượng.
S11 = S08 + [
    "tests/test_chatbot_deepeval.py",
    "tests/test_rag_ragas.py",
    "tests/data/rag_test_cases.json",
    "scripts/eval_report.py",
]

# Xong Bài 12: thêm test agent. Bản đầy đủ.
S12 = S11 + [
    "tests/test_agent_plumbing.py",
    "tests/test_agent_trajectory.py",
    "tests/data/agent_test_cases.json",
]

GHI_CHU = {
    "sau-bai-02": "Xong Bài 2: document_loader + chunker (cắt theo Điều).",
    "sau-bai-06": "Xong Bài 6: ứng dụng agentic đầy đủ — RAG, 4 công cụ, vault, API, giao diện web. Chạy `uv run python scripts/ingest.py --reset` sau khi copy.",
    "sau-bai-08": "Xong Bài 8: test đường ống, cơ chế ghi/phát lại, test vault và API. Chạy `uv run python scripts/record_responses.py` sau khi copy.",
    "sau-bai-11": "Xong Bài 11: đủ bộ test DeepEval + RAGAS và bộ câu hỏi vàng.",
    "sau-bai-12": "Xong Bài 12: thêm test agent, backend và CI. Bản đầy đủ của khoá học.",
}


def _copy(files: list[str], dest: Path) -> int:
    n = 0
    for rel in files:
        src = ROOT / rel
        if not src.exists():
            print(f"  ! thiếu: {rel}")
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        n += 1
    return n


def main() -> int:
    for ten, files in [
        ("sau-bai-02", S02),
        ("sau-bai-06", S06),
        ("sau-bai-08", S08),
        ("sau-bai-11", S11),
        ("sau-bai-12", S12),
    ]:
        dest = SNAP / ten
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        n = _copy(files, dest)
        if ten in {"sau-bai-08", "sau-bai-11", "sau-bai-12"}:
            src_dir = ROOT / "tests" / "fixtures" / "responses"
            if src_dir.exists():
                shutil.copytree(src_dir, dest / "tests" / "fixtures" / "responses")
                n += len(list(src_dir.glob("*.json")))
        (dest / "GHI-CHU.md").write_text(
            f"# {ten}\n\n{GHI_CHU[ten]}\n\nCách dùng, đứng ở thư mục gốc repo:\n\n"
            f"```bash\ncp -r snapshots/{ten}/. .\n```\n",
            encoding="utf-8",
        )
        print(f"{ten}: {n} file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
