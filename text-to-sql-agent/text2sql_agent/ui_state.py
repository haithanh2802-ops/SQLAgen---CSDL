from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

CHAT_SESSION_KEYS = (
    "conversation",
    "pending_question",
    "chat_question",
    "example_question",
    "response",
    "activity_steps",
    "activity_state",
    "question_input",
    "memory_session_id",
)


def clear_chat_session(state: MutableMapping[str, Any]) -> None:
    """Delete transient conversation data while preserving app-level settings."""
    for key in CHAT_SESSION_KEYS:
        state.pop(key, None)
    state["conversation"] = []


def recent_chat_messages(
    conversation: list[dict[str, Any]], *, max_turns: int = 10, max_chars: int = 8000
) -> list[dict[str, str]]:
    """Return a bounded, model-ready view of completed Streamlit chat turns."""
    messages: list[dict[str, str]] = []
    for turn in conversation[-max_turns:]:
        question = str(turn.get("question", "")).strip()
        response = turn.get("response")
        if question:
            messages.append({"role": "user", "content": question})
        if response is None:
            continue
        if response.summary is not None:
            answer = response.summary.answer
        elif response.write_preview is not None:
            answer = response.write_preview.explanation
        else:
            answer = response.error or ""
        if answer:
            messages.append({"role": "assistant", "content": answer})

    bounded: list[dict[str, str]] = []
    used = 0
    for message in reversed(messages):
        remaining = max_chars - used
        if remaining <= 0:
            break
        content = message["content"][-remaining:]
        bounded.append({**message, "content": content})
        used += len(content)
    return list(reversed(bounded))
