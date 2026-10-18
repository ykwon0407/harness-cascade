"""Shared execution events, independent of harness output formats."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class EventKind(StrEnum):
    RUN_STARTED = "run.started"
    RUN_FINISHED = "run.finished"
    RUN_FAILED = "run.failed"
    STAGE_STARTED = "stage.started"
    STAGE_FINISHED = "stage.finished"
    MODEL_CALL_STARTED = "model_call.started"
    MODEL_CALL_FINISHED = "model_call.finished"
    TURN_STARTED = "turn.started"
    ASSISTANT_TEXT = "assistant.text"
    THINKING_SUMMARY = "assistant.thinking_summary"
    TOOL_CALL_STARTED = "tool.started"
    TOOL_CALL_FINISHED = "tool.finished"
    SWITCH_TRIGGERED = "switch.triggered"
    HANDOFF_STARTED = "handoff.started"
    HANDOFF_SUMMARY = "handoff.summary"
    HANDOFF_SKIPPED = "handoff.skipped"
    WORKSPACE_SNAPSHOT = "workspace.snapshot"
    CONTEXT_LIMITS = "context.limits"
    COMPACTION_REQUESTED = "context.compaction_requested"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class CascadeEvent:
    kind: EventKind
    run_id: str
    sequence: int
    timestamp: str
    stage: str | None = None
    harness: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


Subscriber = Callable[[CascadeEvent], None]


class EventBus:
    """Thread-safe sequence assignment and synchronous fan-out."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._sequence = 0
        self._subscribers: list[Subscriber] = []
        self._lock = threading.Lock()
        self._secret = ""

    def subscribe(self, subscriber: Subscriber) -> None:
        self._subscribers.append(subscriber)

    def redact(self, secret: str) -> None:
        """Keep `secret` out of every event, whichever upstream echoed it."""
        # A placeholder key as short as a word would redact ordinary text.
        self._secret = secret if len(secret) >= 8 else ""

    def _scrub(self, value: Any) -> Any:
        if isinstance(value, str):
            return value.replace(self._secret, "[REDACTED]")
        if isinstance(value, dict):
            return {key: self._scrub(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._scrub(item) for item in value]
        return value

    def emit(
        self,
        kind: EventKind,
        *,
        stage: str | None = None,
        harness: str | None = None,
        **data: Any,
    ) -> CascadeEvent:
        with self._lock:
            sequence = self._sequence
            self._sequence += 1
        event = CascadeEvent(
            kind=kind,
            run_id=self.run_id,
            sequence=sequence,
            timestamp=datetime.now(UTC).isoformat(),
            stage=stage,
            harness=harness,
            data=self._scrub(data) if self._secret else data,
        )
        for subscriber in tuple(self._subscribers):
            subscriber(event)
        return event
