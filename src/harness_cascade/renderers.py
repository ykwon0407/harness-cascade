"""Terminal and machine-readable event renderers."""

from __future__ import annotations

import json
import os
import shutil
import sys
import textwrap
import threading
from dataclasses import replace
from pathlib import Path
from typing import ClassVar, TextIO

from .events import CascadeEvent, EventKind


class EventRenderer:
    def __call__(self, event: CascadeEvent) -> None:
        raise NotImplementedError


class PrettyRenderer(EventRenderer):
    def __init__(self, stream: TextIO | None = None, *, color: str = "auto") -> None:
        self.stream = stream or sys.stderr
        self._open_stage: str | None = None
        self._has_handoff = False
        self._pending_tools: dict[str, list[str]] = {}
        self._interactive = bool(getattr(self.stream, "isatty", lambda: False)())
        self._write_lock = threading.RLock()
        self._activity: tuple[str, str | None] | None = None
        self._activity_width = 0
        self._activity_tick = 0
        self._activity_wakeup = threading.Event()
        self._activity_closed = threading.Event()
        self._activity_thread: threading.Thread | None = None
        if color == "always":
            self._color_enabled = True
        elif color == "never":
            self._color_enabled = False
        else:
            self._color_enabled = bool(
                self._interactive and "NO_COLOR" not in os.environ
            )

    _RESET = "\033[0m"
    _DIM = "\033[2m"
    _BOLD = "\033[1m"
    _WHITE = "\033[97m"
    _COLORS: ClassVar[dict[str, str]] = {
        "first": "\033[96m",
        "handoff": "\033[93m",
        "second": "\033[92m",
        "checkpoint": "\033[95m",
        "warning": "\033[91m",
    }

    def _styled(self, text: str, color: str | None = None, *, dim: bool = False) -> str:
        if self._color_enabled and text:
            prefix = self._COLORS.get(color or "", "")
            if dim:
                prefix += self._DIM
            return f"{prefix}{text}{self._RESET}"
        return text

    def _tool_label(self, text: str, stage: str | None) -> str:
        if not self._color_enabled:
            return text
        stage_color = self._COLORS.get(stage or "", "")
        return f"{self._BOLD}{self._WHITE}{text}{self._RESET}{stage_color}"

    @staticmethod
    def _tool_detail_lines(detail: str, *, limit: int = 1_000) -> list[str]:
        value = detail.strip()
        if not value:
            return []
        truncated = len(value) > limit
        value = value[:limit].rstrip()
        if truncated:
            value += " … [truncated]"
        columns = shutil.get_terminal_size(fallback=(100, 24)).columns
        width = max(24, min(columns - 8, 120))
        lines: list[str] = []
        for logical_line in value.splitlines():
            lines.extend(
                textwrap.wrap(
                    logical_line,
                    width=width,
                    replace_whitespace=False,
                    drop_whitespace=True,
                )
                or [""]
            )
        return lines

    def _clear_activity_locked(self) -> None:
        if self._activity_width:
            self.stream.write("\r" + (" " * self._activity_width) + "\r")
            self._activity_width = 0

    def _line(self, text: str = "", *, color: str | None = None, dim: bool = False) -> None:
        with self._write_lock:
            self._clear_activity_locked()
            self.stream.write(self._styled(text, color, dim=dim) + "\n")
            self.stream.flush()

    def _box(self, title: str, rows: list[str], *, color: str) -> None:
        columns = shutil.get_terminal_size(fallback=(100, 24)).columns
        content_width = max(24, min(columns - 4, 106))
        title_text = f" {title} "
        content_width = max(content_width, len(title_text))
        top_fill = max(1, content_width + 2 - len(title_text))
        self._line(
            f"╔{title_text}{'═' * top_fill}╗",
            color=color,
        )
        for row in rows:
            logical_lines = row.splitlines() or [""]
            for logical_line in logical_lines:
                wrapped = textwrap.wrap(
                    logical_line,
                    width=content_width,
                    subsequent_indent="  ",
                    drop_whitespace=True,
                ) or [""]
                for line in wrapped:
                    self._line(
                        f"║ {line}{' ' * (content_width - len(line))} ║",
                        color=color,
                    )
        self._line(f"╚{'═' * (content_width + 2)}╝", color=color)

    @staticmethod
    def _token_summary(usage: dict | None) -> str:
        usage = usage or {}
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        reasoning_tokens = int(usage.get("reasoning_tokens") or 0)
        total_tokens = int(
            usage.get("total_tokens") or input_tokens + output_tokens
        )
        return (
            f"{total_tokens:,} total "
            f"(in {input_tokens:,} · out {output_tokens:,} · "
            f"reasoning {reasoning_tokens:,})"
        )

    def _activity_loop(self) -> None:
        while not self._activity_closed.is_set():
            self._activity_wakeup.wait(0.35)
            self._activity_wakeup.clear()
            if self._activity_closed.is_set():
                break
            with self._write_lock:
                if self._activity is None:
                    continue
                label, color = self._activity
                self._activity_tick = (self._activity_tick % 5) + 1
                plain = f"│ {label}{'.' * self._activity_tick}"
                self._clear_activity_locked()
                self.stream.write(self._styled(plain, color, dim=True))
                self.stream.flush()
                self._activity_width = len(plain)

    def _start_activity(self, label: str, color: str | None) -> None:
        if not self._interactive:
            return
        with self._write_lock:
            self._activity = (label, color)
            self._activity_tick = 0
            if self._activity_thread is None:
                self._activity_thread = threading.Thread(
                    target=self._activity_loop,
                    name="cascade-pretty-activity",
                    daemon=True,
                )
                self._activity_thread.start()
            self._activity_wakeup.set()

    def _stop_activity(self, *, close: bool = False) -> None:
        with self._write_lock:
            self._activity = None
            self._clear_activity_locked()
            self.stream.flush()
        if close:
            self._activity_closed.set()
            self._activity_wakeup.set()

    @staticmethod
    def _stage_title(stage: str | None) -> str:
        return {
            "first": "FIRST HARNESS",
            "handoff": "HANDOFF SUMMARY",
            "second": "SECOND HARNESS",
        }.get(stage or "", (stage or "STAGE").upper())

    def __call__(self, event: CascadeEvent) -> None:
        data = event.data
        if event.kind == EventKind.RUN_STARTED:
            first = data.get("first") or {}
            second = data.get("second") or {}
            self._has_handoff = bool(second)
            route = f"{first.get('harness')}:{first.get('effort')}"
            if second:
                route += f"  →  {second.get('harness')}:{second.get('effort')}"
            switch = first.get("switch_after")
            self._line()
            self._box(
                "CASCADE START",
                [
                    f"Session ID  {event.run_id}",
                    f"Model      {data.get('provider')} / {data.get('model')}",
                    f"Route      {route}",
                    f"Switch     {switch if switch else 'natural completion'}",
                ],
                color="checkpoint",
            )
        elif event.kind == EventKind.STAGE_STARTED:
            self._open_stage = event.stage
            stage_key = event.stage or ""
            self._pending_tools[stage_key] = []
            self._line()
            self._line(
                f"╭─ {self._stage_title(event.stage)} · "
                f"{event.harness} · {data.get('effort', '')}",
                color=event.stage,
            )
            activity = "summarizing" if event.stage == "handoff" else "thinking"
            self._start_activity(activity, event.stage)
        elif event.kind == EventKind.MODEL_CALL_STARTED:
            activity = "summarizing" if event.stage == "handoff" else "thinking"
            self._start_activity(activity, event.stage)
        elif event.kind == EventKind.ASSISTANT_TEXT:
            self._stop_activity()
            text = str(data.get("text") or "").strip()
            if text:
                for line in text.splitlines():
                    self._line(f"│ {line}", color=event.stage)
        elif event.kind == EventKind.THINKING_SUMMARY:
            self._stop_activity()
            text = str(data.get("text") or "").strip()
            if text:
                for index, line in enumerate(text.splitlines()):
                    label = "thinking: " if index == 0 else "          "
                    self._line(
                        f"│ {label}{line}",
                        color=event.stage,
                        dim=True,
                    )
        elif event.kind == EventKind.TURN_STARTED:
            self._stop_activity()
            limit = data.get("limit")
            number = f"{data.get('turn')}/{limit}" if limit else str(data.get("turn"))
            tools = ", ".join(str(name) for name in data.get("tools") or [])
            self._line(
                self._tool_label(f"│ ▸ turn {number} · {tools}", event.stage),
                color=event.stage,
            )
        elif event.kind == EventKind.TOOL_CALL_STARTED:
            name = data.get("name") or "tool"
            detail = str(data.get("detail") or "").replace("\n", " ").strip()
            stage_key = event.stage or ""
            label_text = str(name)
            self._pending_tools.setdefault(stage_key, []).append(label_text)
            label = self._tool_label(label_text, event.stage)
            self._line(
                f"│ {label}: {detail[:180]}",
                color=event.stage,
            )
            self._start_activity(f"executing {name}", event.stage)
        elif event.kind == EventKind.TOOL_CALL_FINISHED:
            self._stop_activity()
            stage_key = event.stage or ""
            pending = self._pending_tools.get(stage_key) or []
            label_text = pending.pop(0) if pending else str(data.get("name") or "tool")
            label = self._tool_label(label_text, event.stage)
            self._line(
                f"│ ✗ {label} failed" if data.get("failed") else f"│ ✓ {label} finished",
                color=event.stage,
            )
            for line in self._tool_detail_lines(str(data.get("detail") or "")):
                self._line(f"│   ↳ {line}", color=event.stage, dim=True)
            self._start_activity(
                "summarizing" if event.stage == "handoff" else "thinking",
                event.stage,
            )
        elif event.kind == EventKind.SWITCH_TRIGGERED:
            self._stop_activity()
            turns = data.get("turns")
            self._line(
                "╰─ configured handoff boundary reached"
                + (f" after {turns} turn{'s' if turns != 1 else ''}" if turns else ""),
                color=event.stage,
            )
            self._open_stage = None
        elif event.kind == EventKind.STAGE_FINISHED:
            self._stop_activity()
            if self._open_stage == event.stage and data.get("status") == "switch_limit":
                # Claude Code stops itself there, so the gate never fires.
                self(replace(event, kind=EventKind.SWITCH_TRIGGERED))
            elif self._open_stage == event.stage:
                turns = data.get("turns")
                if data.get("status") == "completed" and event.stage != "handoff" and turns is not None:
                    self._line(
                        self._tool_label(
                            f"│ finished at {turns} turn{'s' if turns != 1 else ''}", event.stage
                        ),
                        color=event.stage,
                    )
                self._line(
                    f"╰─ {data.get('status', 'finished')} · "
                    f"{float(data.get('duration_seconds', 0)):.1f}s",
                    color=event.stage,
                )
                self._open_stage = None
        elif event.kind == EventKind.COMPACTION_REQUESTED:
            self._line(
                f"│ context at {int(data.get('context_tokens') or 0):,} tokens, past "
                f"{int(data.get('threshold') or 0):,}; compacting",
                color=event.stage,
                dim=True,
            )
        elif event.kind == EventKind.HANDOFF_SKIPPED:
            self._stop_activity()
            self._has_handoff = False
            self._line()
            self._line(
                "first harness finished before the switch point; "
                "second harness skipped",
                color="checkpoint",
            )
        elif event.kind in {
            EventKind.HANDOFF_STARTED,
            EventKind.HANDOFF_SUMMARY,
            EventKind.WORKSPACE_SNAPSHOT,
        }:
            return
        elif event.kind in {EventKind.WARNING, EventKind.ERROR, EventKind.RUN_FAILED}:
            self._stop_activity(close=event.kind == EventKind.RUN_FAILED)
            message = str(data.get("message", event.kind.value))
            if data.get("attempt"):
                message = (
                    f"model call attempt {data['attempt']} of {data.get('of')} failed "
                    f"after {float(data.get('seconds') or 0):.1f}s: {data.get('reason')}"
                    + ("; retrying" if data.get("retrying") else "")
                )
            self._line(f"cascade ! {message}", color="warning")
        elif event.kind == EventKind.RUN_FINISHED:
            self._stop_activity(close=True)
            stage_usage = data.get("stage_usage") or {}
            rows = [
                f"Duration   {float(data.get('duration_seconds', 0)):.1f}s",
                f"First      {self._token_summary(stage_usage.get('first'))}",
            ]
            if self._has_handoff:
                rows.extend(
                    [
                        (
                            "Handoff    "
                            f"{self._token_summary(stage_usage.get('handoff'))}"
                        ),
                        (
                            "Second     "
                            f"{self._token_summary(stage_usage.get('second'))}"
                        ),
                    ]
                )
            rows.extend(
                [
                    f"All stages {self._token_summary(data.get('usage'))}",
                    f"Artifacts  {data.get('artifacts')}",
                ]
            )
            self._line()
            self._box(
                "CASCADE COMPLETED",
                rows,
                color="checkpoint",
            )


class JsonlRenderer(EventRenderer):
    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stdout

    def __call__(self, event: CascadeEvent) -> None:
        self.stream.write(event.to_json() + "\n")
        self.stream.flush()


class TrajectoryWriter(EventRenderer):
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")

    def __call__(self, event: CascadeEvent) -> None:
        self._file.write(event.to_json() + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def print_final_result(result: dict, *, output_format: str) -> None:
    if output_format in {"pretty", "jsonl"}:
        return
    final_text = str(result.get("final_text") or "").strip()
    if output_format == "text":
        sys.stdout.write(final_text + ("\n" if final_text else ""))
        return
    payload = {
        "run_id": result.get("run_id"),
        "status": result.get("status"),
        "first": result.get("first"),
        "handoff": result.get("handoff"),
        "second": result.get("second"),
        "usage": result.get("usage"),
        "final_text": final_text,
        "trajectory": result.get("trajectory"),
    }
    sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
