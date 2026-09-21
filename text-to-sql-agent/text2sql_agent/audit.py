from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class AuditLogger:
    """Append compact, non-result-bearing audit events to a local JSONL file."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def append(self, event: str, **details: Any) -> None:
        entry = {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "event": event,
            **details,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def try_append(self, event: str, **details: Any) -> bool:
        try:
            self.append(event, **details)
        except OSError:
            return False
        return True
