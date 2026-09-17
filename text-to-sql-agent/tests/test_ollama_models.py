import io
import json

from text2sql_agent import ollama_models


def test_discover_chat_models_keeps_default_and_excludes_embeddings(monkeypatch):
    payload = {
        "models": [
            {"name": "qwen3-coder:30b"},
            {"name": "qwen3:8b"},
            {"name": "qwen3-embedding:latest"},
        ]
    }
    monkeypatch.setattr(
        ollama_models,
        "urlopen",
        lambda request, timeout: io.BytesIO(json.dumps(payload).encode("utf-8")),
    )

    models, error = ollama_models.discover_chat_models(
        "http://localhost:11434",
        configured_model="qwen3:8b",
        embedding_model="qwen3-embedding:latest",
    )

    assert models == ("qwen3:8b", "qwen3-coder:30b")
    assert error is None


def test_discover_chat_models_falls_back_when_ollama_is_unavailable(monkeypatch):
    def unavailable(request, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(ollama_models, "urlopen", unavailable)

    models, error = ollama_models.discover_chat_models(
        "http://localhost:11434",
        configured_model="qwen3:8b",
        embedding_model="qwen3-embedding:latest",
    )

    assert models == ("qwen3:8b",)
    assert "connection refused" in error
