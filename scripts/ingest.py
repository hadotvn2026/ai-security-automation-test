"""Nạp corpus vào Chroma: docx -> text -> chunk -> vector -> Vault.

Chạy đúng chuỗi các hộp 2-3-4-5 trong sơ đồ kiến trúc.

    uv run python scripts/ingest.py            # nạp với cấu hình mặc định
    uv run python scripts/ingest.py --reset    # xoá sạch rồi nạp lại

Chroma DB sinh ra được commit vào repo, nên học viên KHÔNG bắt buộc phải chạy
script này. Chạy lại khi bạn muốn thử đổi CHUNK_SIZE (bài tập cuối Part 4).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_qa import config  # noqa: E402
from rag_qa.chunker import chunk_legal_document, chunk_text  # noqa: E402
from rag_qa.document_loader import load_document  # noqa: E402
from rag_qa.vector_store import count, index_chunks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Xoá Chroma DB cũ")
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument(
        "--cach-cat",
        choices=["dieu", "ky-tu"],
        default="dieu",
        help="dieu = cắt theo Điều (mặc định, dành cho văn bản luật); "
             "ky-tu = cắt theo số ký tự (để so sánh ở bài tập)",
    )
    args = parser.parse_args()

    if not config.CORPUS_PATH.exists():
        print(f"Chưa có corpus: {config.CORPUS_PATH}")
        print("Chạy trước: uv run python scripts/build_corpus.py")
        return 1

    if args.reset and config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
        print(f"Đã xoá {config.CHROMA_DIR}")

    print(f"[1/4] Đọc tài liệu: {config.CORPUS_PATH.name}")
    text = load_document(config.CORPUS_PATH)
    print(f"      {len(text):,} ký tự")

    print(f"[2/4] Cắt chunk ({args.cach_cat}, size={args.chunk_size}, overlap={args.overlap})")
    if args.cach_cat == "dieu":
        chunks = chunk_legal_document(text, args.chunk_size, args.overlap)
    else:
        chunks = chunk_text(text, args.chunk_size, args.overlap)
    print(f"      {len(chunks)} chunk")

    print(f"[3/4] Embedding + nạp vào Chroma ({config.EMBEDDING_MODEL})")
    started = time.perf_counter()
    n = index_chunks(chunks)
    elapsed = time.perf_counter() - started
    print(f"      Đã nạp {n} chunk trong {elapsed:.1f}s")

    print(f"[4/4] Kiểm tra: collection có {count()} vector")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
