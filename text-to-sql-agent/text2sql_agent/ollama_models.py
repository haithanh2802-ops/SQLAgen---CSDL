from __future__ import annotations

import json
from urllib.request import Request, urlopen


def discover_chat_models(
    base_url: str,
    *,
    configured_model: str,
    embedding_model: str,
    timeout_seconds: float = 2.0,
) -> tuple[tuple[str, ...], str | None]:
    """Return locally installed chat models, keeping the configured model first."""
    request = Request(
        f"{base_url.rstrip('/')}/api/tags",
        headers={"Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.load(response)
        if not isinstance(payload, dict):
            raise TypeError("Ollama returned an unexpected model-list response.")
        installed = {
            str(item.get("name") or item.get("model", "")).strip()
            for item in payload.get("models", [])
            if isinstance(item, dict)
        }
        chat_models = {
            name
            for name in installed
            if name and name != embedding_model and "embed" not in name.casefold()
        }
        ordered = (configured_model, *sorted(chat_models - {configured_model}))
        return ordered, None
    except (OSError, TimeoutError, ValueError, TypeError) as exc:
        return (configured_model,), f"Could not load Ollama's installed models: {exc}"
