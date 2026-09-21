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
)


def clear_chat_session(state: MutableMapping[str, Any]) -> None:
    """Delete transient conversation data while preserving app-level settings."""
    for key in CHAT_SESSION_KEYS:
        state.pop(key, None)
    state["conversation"] = []
