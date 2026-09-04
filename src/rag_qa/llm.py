"""Điểm gọi LLM duy nhất của ứng dụng.

Ứng với hộp số 8 trong sơ đồ kiến trúc.

TOÀN BỘ ứng dụng gọi model qua đúng một hàm: `generate()`. Đó không phải tình
cờ — nó là ranh giới mock của Part 3. Vì chỉ có một chỗ chạm tới model, ta thay
được câu trả lời thật bằng câu trả lời đã ghi lại (recorded) mà không phải sửa
một dòng nào trong graph.

Quy tắc vàng của khoá học:
    Mock LLM CỦA ỨNG DỤNG  -> hợp lệ (phát lại câu trả lời đã ghi)
    Mock LLM CỦA JUDGE     -> vô nghĩa (hardcode điểm số, test không bao giờ đỏ)
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from rag_qa import config

__all__ = ["get_chat_model", "generate", "build_messages", "invoke_with_tools"]


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOllama:
    """Client chat trỏ tới Ollama local. Cache lại để tái dùng kết nối."""
    return ChatOllama(
        model=config.APP_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.TEMPERATURE,
        num_predict=config.MAX_TOKENS,
        client_kwargs={"timeout": config.REQUEST_TIMEOUT},
    )


def build_messages(prompt: str, system: str | None = None) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))
    return messages


def generate(prompt: str, system: str | None = None) -> str:
    """Sinh câu trả lời từ model ứng dụng. Trả về chuỗi text thuần.

    Đây là hàm sẽ bị patch trong test. Giữ chữ ký hàm đơn giản và ổn định.
    """
    response = get_chat_model().invoke(build_messages(prompt, system))
    content = response.content
    if isinstance(content, list):  # một số model trả về list block
        content = "".join(str(part) for part in content)
    return str(content).strip()


def invoke_with_tools(messages: list[BaseMessage], tools: list) -> BaseMessage:
    """Điểm chạm model DUY NHẤT của agent (Part 5).

    Song song với `generate()` ở trên, và cùng vai trò: mọi test của Part 5 cắm
    vào đúng hàm này.

    Khác biệt quan trọng so với `generate()`: hàm này trả về nguyên `AIMessage`,
    vì phần đáng quan tâm nhất không phải chữ mà là `.tool_calls` — quỹ đạo.
    Ghi lại một chuỗi AIMessage có tool_calls khó hơn ghi một chuỗi text rất
    nhiều; xem docs/05 mục "Bản ghi vỡ ở đâu".
    """
    return get_chat_model().bind_tools(tools).invoke(messages)
