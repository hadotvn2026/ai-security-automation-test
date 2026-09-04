"""RAG — LangGraph HAI node (Part 2b).

    START -> retrieve -> generate -> END

So với chat_graph.py, khác biệt duy nhất là node `retrieve` đứng trước.
Ứng với hộp số 7 trong sơ đồ kiến trúc.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from rag_qa import llm, prompts, retriever

__all__ = ["RagState", "build_rag_graph", "answer", "answer_with_contexts"]


class RagState(TypedDict, total=False):
    question: str
    contexts: list[str]
    answer: str


def retrieve_node(state: RagState) -> RagState:
    """Node 1: lấy các chunk liên quan từ vector store."""
    return {"contexts": retriever.retrieve(state["question"])}


def generate_node(state: RagState) -> RagState:
    """Node 2: nhét chunk vào prompt rồi gọi model."""
    contexts = state.get("contexts") or []
    prompt = prompts.build_rag_prompt(state["question"], contexts)
    return {"answer": llm.generate(prompt, system=prompts.RAG_SYSTEM)}


def build_rag_graph():
    graph = StateGraph(RagState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


_compiled = None


def _get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_rag_graph()
    return _compiled


def answer_with_contexts(question: str) -> tuple[str, list[str]]:
    """Trả về (câu trả lời, danh sách context đã dùng).

    Part 4 cần CẢ HAI: RAGAS chấm điểm câu trả lời dựa trên chính context mà
    ứng dụng thực sự đã lấy được, chứ không phải context lý tưởng.
    """
    result = _get_graph().invoke({"question": question})
    return result["answer"], result.get("contexts", [])


def answer(question: str) -> str:
    return answer_with_contexts(question)[0]
