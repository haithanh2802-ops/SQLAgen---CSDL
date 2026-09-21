from text2sql_agent.contracts import AgentResponse, BusinessSummary
from text2sql_agent.ui_state import clear_chat_session, recent_chat_messages


def test_clear_chat_session_removes_conversation_but_preserves_settings():
    state = {
        "conversation": [{"question": "How many orders?"}],
        "pending_question": "Retry this",
        "chat_question": "Draft",
        "selected_chat_model": "qwen3-coder:30b",
        "memory_session_id": "session-1",
    }

    clear_chat_session(state)

    assert state["conversation"] == []
    assert "pending_question" not in state
    assert "chat_question" not in state
    assert "memory_session_id" not in state
    assert state["selected_chat_model"] == "qwen3-coder:30b"


def test_recent_chat_messages_returns_completed_bounded_turns():
    conversation = [
        {
            "question": "Show delivered orders.",
            "response": AgentResponse(
                kind="analysis",
                question="Show delivered orders.",
                summary=BusinessSummary(answer="There were ten delivered orders."),
            ),
        },
        {"question": "What about canceled orders?", "response": None},
    ]

    messages = recent_chat_messages(conversation, max_chars=200)

    assert messages == [
        {"role": "user", "content": "Show delivered orders."},
        {"role": "assistant", "content": "There were ten delivered orders."},
        {"role": "user", "content": "What about canceled orders?"},
    ]
