from text2sql_agent.ui_state import clear_chat_session


def test_clear_chat_session_removes_conversation_but_preserves_settings():
    state = {
        "conversation": [{"question": "How many orders?"}],
        "pending_question": "Retry this",
        "chat_question": "Draft",
        "selected_chat_model": "qwen3-coder:30b",
    }

    clear_chat_session(state)

    assert state["conversation"] == []
    assert "pending_question" not in state
    assert "chat_question" not in state
    assert state["selected_chat_model"] == "qwen3-coder:30b"
