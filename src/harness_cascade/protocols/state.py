"""Persistent references for opaque provider reasoning blocks."""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any


class OpaqueStateStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._values: dict[str, dict[str, Any]] = {}
        if path and path.is_file():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._values = {
                        str(key): value
                        for key, value in raw.items()
                        if isinstance(value, dict)
                    }
            except (OSError, ValueError):
                self._values = {}

    def put(self, value: dict[str, Any]) -> str:
        token = uuid.uuid4().hex
        with self._lock:
            self._values[token] = value
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(
                    json.dumps(self._values, ensure_ascii=False),
                    encoding="utf-8",
                )
        return token

    def get(self, token: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._values.get(token)
            return dict(value) if value else None
