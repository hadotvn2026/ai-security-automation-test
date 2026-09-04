"""CLI để nói chuyện thật với ứng dụng.

    uv run python -m rag_qa.cli chat   # chatbot thuần (Part 2a)
    uv run python -m rag_qa.cli rag    # hỏi đáp trên tài liệu (Part 2b)
"""

from __future__ import annotations

import sys

from rag_qa import chat_graph, config, rag_graph


def _repl_chat() -> None:
    print(f"Chatbot ({config.APP_MODEL}). Gõ 'exit' để thoát.\n")
    history: list[dict] = []
    while True:
        try:
            question = input("Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question.lower() in {"exit", "quit"}:
            return
        if not question:
            continue
        answer = chat_graph.chat(question, history)
        history += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        print(f"Bot: {answer}\n")


def _repl_rag() -> None:
    print(f"RAG ({config.APP_MODEL} + {config.EMBEDDING_MODEL}). Gõ 'exit' để thoát.\n")
    while True:
        try:
            question = input("Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question.lower() in {"exit", "quit"}:
            return
        if not question:
            continue
        answer, contexts = rag_graph.answer_with_contexts(question)
        print(f"Bot: {answer}")
        print(f"  ({len(contexts)} đoạn tài liệu đã dùng)\n")


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "rag"
    if mode == "chat":
        _repl_chat()
    elif mode == "rag":
        _repl_rag()
    else:
        print(f"Chế độ không hợp lệ: {mode}. Dùng 'chat' hoặc 'rag'.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
