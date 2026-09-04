"""Lưu vector + chunk vào ChromaDB (lưu trên đĩa, commit vào repo).

Ứng với hộp số 5 trong sơ đồ kiến trúc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from rag_qa import config
from rag_qa.embeddings import embed_query, embed_texts

__all__ = ["get_collection", "index_chunks", "query", "count"]


def get_collection(persist_dir: str | Path | None = None) -> Any:
    """Mở (hoặc tạo) collection Chroma nằm trên đĩa."""
    persist_dir = Path(persist_dir or config.CHROMA_DIR)
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    # embedding_function=None: ta tự embed bằng Ollama, không để Chroma tự làm.
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks: list[str], persist_dir: str | Path | None = None) -> int:
    """Embed và nạp toàn bộ chunk vào collection. Trả về số chunk đã nạp."""
    if not chunks:
        return 0
    collection = get_collection(persist_dir)
    vectors = embed_texts(chunks)
    collection.upsert(
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=vectors,
        metadatas=[{"index": i} for i in range(len(chunks))],
    )
    return len(chunks)


def query(
    question: str,
    top_k: int | None = None,
    persist_dir: str | Path | None = None,
) -> list[str]:
    """Tìm `top_k` chunk gần nghĩa nhất với câu hỏi."""
    top_k = top_k or config.TOP_K
    collection = get_collection(persist_dir)
    result = collection.query(
        query_embeddings=[embed_query(question)],
        n_results=top_k,
    )
    documents = result.get("documents") or [[]]
    return documents[0]


def count(persist_dir: str | Path | None = None) -> int:
    """Số chunk hiện có trong collection."""
    return get_collection(persist_dir).count()
