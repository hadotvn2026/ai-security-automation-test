"""Tìm các chunk liên quan tới câu hỏi (similarity search).

Ứng với hộp số 6 trong sơ đồ kiến trúc.

Đây là module quan trọng nhất đối với Part 4: hai metric Context Precision và
Context Recall của RAGAS chấm điểm chính xác đầu ra của hàm `retrieve()` dưới
đây. Nếu retriever sai, hai điểm đó tụt — kể cả khi câu trả lời cuối nghe hay.
"""

from __future__ import annotations

from rag_qa import config, vector_store

__all__ = ["retrieve"]


def retrieve(question: str, top_k: int | None = None) -> list[str]:
    """Trả về danh sách chunk liên quan nhất, sắp xếp theo độ gần nghĩa.

    Trả về list rỗng nếu câu hỏi rỗng — ứng dụng phải xử lý được trường hợp này
    thay vì ném lỗi (xem test plumbing trong Part 3).
    """
    question = (question or "").strip()
    if not question:
        return []
    return vector_store.query(question, top_k=top_k or config.TOP_K)
