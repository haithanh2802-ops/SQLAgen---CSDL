from pathlib import Path

import pytest

from text2sql_agent.config import Settings


def test_requested_ollama_models_are_defaults(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("OLLAMA_CHAT_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_EMBEDDING_MODEL", raising=False)
    settings = Settings.from_env(tmp_path)
    assert settings.chat_model == "qwen3-coder:30b"
    assert settings.embedding_model == "qwen3-embedding:latest"


def test_chat_model_can_be_overridden(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "qwen3-coder:30b")
    settings = Settings.from_env(tmp_path)
    assert settings.chat_model == "qwen3-coder:30b"


def test_gpu_offload_can_be_disabled(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OLLAMA_NUM_GPU", "0")
    settings = Settings.from_env(tmp_path)
    assert settings.ollama_num_gpu == 0


def test_gpu_all_layers_override_is_allowed(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OLLAMA_NUM_GPU", "-1")
    settings = Settings.from_env(tmp_path)
    assert settings.ollama_num_gpu == -1


def test_read_and_write_identities_must_be_separate(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MYSQL_READ_USER", "same_user")
    monkeypatch.setenv("MYSQL_WRITE_USER", "same_user")
    settings = Settings.from_env(tmp_path)
    with pytest.raises(RuntimeError, match="must be different"):
        settings.assert_separate_database_identities()


def test_database_url_hides_password_in_string(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MYSQL_READ_USER", "reader")
    monkeypatch.setenv("MYSQL_READ_PASSWORD", "secret value")
    settings = Settings.from_env(tmp_path)
    rendered = str(settings.database_url(read_only=True))
    assert "secret value" not in rendered
    assert "reader" in rendered
