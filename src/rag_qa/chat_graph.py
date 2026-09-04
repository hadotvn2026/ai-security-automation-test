"""Chatbot thuần — LangGraph MỘT node (Part 2a).

    START -> generate -> END

Cố ý làm đúng hình dạng của rag_graph.py nhưng thiếu node `retrieve`. Sang
Part 2b, việc duy nhất phải làm là chèn thêm một node vào phía trước.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from rag_qa import llm, prompts

__all__ = ["ChatState", "build_chat_graph", "chat"]


def _append(existing: list[dict], new: list[dict]) -> list[dict]:
    """Reducer: state mới được cộng dồn vào lịch sử thay vì ghi đè."""
    return (existing or []) + (new or [])


class ChatState(TypedDict, total=False):
    question: str
    answer: str
    history: Annotated[list[dict], _append]


def generate_node(state: ChatState) -> ChatState:
    """Node duy nhất: gọi model, trả lời."""
    question = state["question"]
    history = state.get("history") or []

    if history:
        transcript = "\n".join(
            f"{turn['role']}: {turn['content']}" for turn in history
        )
        prompt = f"Lịch sử hội thoại:\n{transcript}\n\nNgười dùng: {question}"
    else:
        prompt = question

    answer = llm.generate(prompt, system=prompts.CHATBOT_SYSTEM)
    return {
        "answer": answer,
        "history": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
    }


def build_chat_graph():
    graph = StateGraph(ChatState)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", END)
    return graph.compile()


_compiled = None


def chat(question: str, history: list[dict] | None = None) -> str:
    """Tiện ích gọi nhanh chatbot. Trả về câu trả lời."""
    global _compiled
    if _compiled is None:
        _compiled = build_chat_graph()
    result = _compiled.invoke({"question": question, "history": history or []})
    return result["answer"]
