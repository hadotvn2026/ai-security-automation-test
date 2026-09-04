"""Biến text thành vector bằng model embedding chạy local trên Ollama.

Ứng với hộp số 4 trong sơ đồ kiến trúc.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_ollama import OllamaEmbeddings

from rag_qa import config

__all__ = ["get_embeddings", "embed_texts", "embed_query"]


@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    """Trả về client embedding (cache lại để không tạo kết nối lặp)."""
    return OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed nhiều đoạn văn bản (dùng khi nạp corpus vào vector store)."""
    if not texts:
        return []
    return get_embeddings().embed_documents(texts)


def embed_query(text: str) -> list[float]:
    """Embed một câu hỏi của người dùng."""
    return get_embeddings().embed_query(text)
