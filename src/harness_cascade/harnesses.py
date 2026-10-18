"""Headless process adapters for Claude Code, Pi, and Codex."""

from __future__ import annotations

import errno
import json
import os
import queue
import re
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import StagePlan
from .credentials import CredentialSet, redact, scrub_credential_env
from .events import EventBus, EventKind
from .gateway import (
    COMPACTION_POINT_MARKER,
    EmbeddedGateway,
    codex_builtin_entry,
    codex_entry_servable,
)


# Between the switch point and the kill. The harness is blocked on the held
# request by then, so this only lets it finish writing the session file the
# handoff summary resumes from.
SWITCH_GRACE_SECONDS = 1.0
# How long to keep reading once the harness has exited. A tool process that
# outlived it can hold the pipe open indefinitely.
DRAIN_SECONDS = 2.0
KILL_GRACE_SECONDS = 2.0
ERROR_CHARS = 2000
MAX_ARG_STRLEN = 128 * 1024
# Room under MAX_ARG_STRLEN for Pi's "Task:" prefix and the terminating NUL.
PROMPT_ARG_BYTES = MAX_ARG_STRLEN - 1024


class HarnessError(RuntimeError):
    pass


@dataclass(slots=True)
class HarnessResult:
    harness: str
    status: str
    returncode: int
    session_id: str
    # Model replies that asked for a tool; the unit `switch_after` counts.
    turns: int = 0
    # Model replies of any kind.
    model_turns: int = 0
    final_text: str = ""
    final_reasoning: str = ""
    commands: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    log_path: str = ""
    # Why the stage failed, in the harness's or the provider's words.
    error: str = ""
    pending_tools: list[str] = field(default_factory=list)
    raw_lines: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class HarnessContext:
    stage: str
    plan: StagePlan
    model: str
    prompt: str
    workspace: Path
    state_dir: Path
    log_path: Path
    credentials: CredentialSet
    events: EventBus
    gateway: EmbeddedGateway | None
    session_id: str
    timeout_seconds: int
    resume: bool = False
    summary: bool = False

    @property
    def prompt_on_stdin(self) -> bool:
        """Whether the prompt is too long for one argv entry.

        Linux caps a single argument at 128 KiB (MAX_ARG_STRLEN). A prompt that
        fits stays on the command line, as it always has; a longer one is
        written to the harness's stdin, which all three read as the prompt
        when none is given as an argument.
        """
        return len(self.prompt.encode("utf-8")) > PROMPT_ARG_BYTES


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    """Stop a harness that is still running, and every process it started.

    Codex runs each tool command in a session of its own, outside the
    harness's process group; those are found by parentage before the harness
    dies and orphans them. The grace is short because a benchmark's SIGTERM
    is followed by SIGKILL a few seconds later.
    """
    if proc.poll() is not None:
        return
    descendants = _descendants(proc.pid)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    _signal_all(descendants, signal.SIGTERM)
    try:
        proc.wait(timeout=KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=KILL_GRACE_SECONDS)
    _signal_all(descendants, signal.SIGKILL)


def _descendants(root: int) -> list[int]:
    """Every live process below `root`, read from /proc; empty elsewhere."""
    children: dict[int, list[int]] = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return []
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", encoding="utf-8", errors="replace") as handle:
                stat = handle.read()
        except OSError:
            continue
        # The command name in parentheses may itself contain spaces or ')'.
        fields = stat[stat.rfind(")") + 2 :].split()
        if len(fields) > 1 and fields[1].isdigit():
            children.setdefault(int(fields[1]), []).append(int(entry))
    found: list[int] = []
    pending = [root]
    while pending:
        for child in children.get(pending.pop(), ()):
            found.append(child)
            pending.append(child)
    return found


def _signal_all(pids: list[int], signum: int) -> None:
    for pid in pids:
        try:
            os.kill(pid, signum)
        except (ProcessLookupError, PermissionError):
            continue


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            text = block.get("text")
            if text:
                parts.append(str(text))
    return "\n".join(parts)


def _reasoning_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(block.get("thinking") or block.get("text") or "")
        for block in content
        if isinstance(block, dict)
        and block.get("type") in {"thinking", "reasoning", "reasoning_summary"}
        and (block.get("thinking") or block.get("text"))
    )


def _set_usage(result: HarnessResult, raw: Any) -> None:
    if not isinstance(raw, dict):
        return
    aliases = {
        "input_tokens": ("input_tokens", "input"),
        "output_tokens": ("output_tokens", "output"),
        "cache_read_tokens": ("cache_read_input_tokens", "cacheRead"),
        "cache_write_tokens": ("cache_creation_input_tokens", "cacheWrite"),
        "reasoning_tokens": ("reasoning_tokens", "reasoning"),
    }
    for target, sources in aliases.items():
        value = next((raw.get(source) for source in sources if raw.get(source) is not None), None)
        if value is not None:
            result.usage[target] = result.usage.get(target, 0) + int(value)


def _agent_env() -> dict[str, str]:
    """Environment for Claude / Pi / Codex: local token only, no upstream keys."""
    return scrub_credential_env()


def _codex_catalog_enabled(model: str) -> bool:
    """Whether Codex is handed a GPT-derived catalog entry for `model`.

    Normally not. A model Codex does not know -- every open model -- runs on
    its fallback metadata: the general Codex CLI prompt, no dedicated
    apply_patch tool (the prompt has the model run it through the shell) and no
    parallel tool calls. A GPT model Codex ships an entry for runs on that
    entry, its own prompt included. The exception is an entry this gateway
    cannot serve (the gpt-5.6 tier's code mode and lite wire), which would
    break the run; those get the GPT-derived entry instead. Under a compaction
    policy a model without an entry gets a copy of the fallback metadata, and
    one with an entry gets that entry, since only an entry can carry its
    window; see EmbeddedGateway.codex_fallback_model and
    codex_builtin_catalog. CASCADE_CODEX_CATALOG=1 or 0 forces
    the GPT-derived entry on or off.
    """
    value = os.environ.get("CASCADE_CODEX_CATALOG", "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    builtin = codex_builtin_entry(model)
    return builtin is not None and not codex_entry_servable(builtin)


def _signal_death(returncode: int) -> str:
    """A kill cascade did not send: the stage was neither switched nor timed
    out. In the TB 2.1 runs these were the agent's own `pkill -f` matching its
    argv, or the kernel's OOM killer."""
    number = -returncode if returncode < 0 else 0
    if returncode in {128 + signal.SIGKILL, 128 + signal.SIGTERM}:
        number = returncode - 128
    if not number:
        return ""
    try:
        name = signal.Signals(number).name
    except ValueError:
        name = f"signal {number}"
    return f"killed by {name}, which cascade did not send"


CLAUDE_CODE_DEFAULT_WINDOW = 200_000
CLAUDE_CODE_1M_WINDOW = 1_000_000


def _claude_code_model(context: HarnessContext) -> str:
    """The model name to give Claude Code.

    Claude Code trusts a Claude model's 1M window only when it talks to
    api.anthropic.com itself; behind the gateway it plans every Claude model
    against 200,000 and ignores CLAUDE_CODE_MAX_CONTEXT_TOKENS for them. The
    `[1m]` suffix is its own opt-in to 1M. The gateway sends its configured
    model upstream whatever name the harness asks for.
    """
    model = context.model
    config = context.gateway.config if context.gateway is not None else None
    if (
        config is not None
        and config.compact_at_percent
        and model.lower().startswith("claude-")
        and "[1m]" not in model.lower()
        and config.context_window > CLAUDE_CODE_DEFAULT_WINDOW
    ):
        return f"{model}[1m]"
    return model


def _reply_number(context: HarnessContext) -> int:
    """Which of the stage's tool replies the tool calls now being reported
    belong to. The gateway counts a reply before handing it over."""
    return context.gateway.gate.tool_replies if context.gateway is not None else 0


def _feed_prompt(proc: subprocess.Popen[str], prompt: str) -> None:
    assert proc.stdin is not None
    try:
        proc.stdin.write(prompt)
        proc.stdin.close()
    except (OSError, ValueError):
        # The harness exited before reading it; its own exit says why.
        pass


class HarnessAdapter:
    name: str

    def build_command(self, context: HarnessContext) -> tuple[list[str], dict[str, str]]:
        raise NotImplementedError

    def consume_event(
        self,
        event: dict[str, Any],
        result: HarnessResult,
        context: HarnessContext,
    ) -> None:
        raise NotImplementedError

    def consume_plain(
        self, line: str, result: HarnessResult, context: HarnessContext
    ) -> None:
        """A line of output that is not a JSON event."""

    def run(self, context: HarnessContext) -> HarnessResult:
        command, env = self.build_command(context)
        result = HarnessResult(
            harness=self.name,
            status="running",
            returncode=1,
            session_id=context.session_id,
        )
        on_stdin = context.prompt_on_stdin
        try:
            proc = subprocess.Popen(
                command,
                cwd=context.workspace,
                env=env,
                stdin=subprocess.PIPE if on_stdin else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise HarnessError(
                f"{self.name} executable is not installed: {command[0]}"
            ) from exc
        except OSError as exc:
            if exc.errno != errno.E2BIG:
                raise
            longest = max(len(part.encode("utf-8")) for part in command)
            raise HarnessError(
                f"{self.name} command line is too long: its longest argument is "
                f"{longest} bytes and Linux allows {MAX_ARG_STRLEN}"
            ) from exc
        if on_stdin:
            threading.Thread(
                target=_feed_prompt,
                args=(proc, context.prompt),
                name=f"{self.name}-stdin",
                daemon=True,
            ).start()
        assert proc.stdout is not None
        stdout = proc.stdout
        gate = context.gateway.gate if context.gateway is not None else None
        timed_out = threading.Event()
        done = threading.Event()
        context.log_path.parent.mkdir(parents=True, exist_ok=True)

        def expire() -> None:
            timed_out.set()
            _kill_process_group(proc)

        def end_at_switch_point() -> None:
            assert gate is not None
            while not gate.reached.wait(0.2):
                if done.is_set():
                    return
            if not done.wait(SWITCH_GRACE_SECONDS):
                _kill_process_group(proc)

        lines: queue.Queue[str | None] = queue.Queue()

        def pump() -> None:
            try:
                for raw in stdout:
                    lines.put(raw)
            except (OSError, ValueError):
                pass
            finally:
                lines.put(None)

        timer = threading.Timer(context.timeout_seconds, expire)
        timer.daemon = True
        timer.start()
        threading.Thread(target=pump, name=f"{self.name}-stdout", daemon=True).start()
        if gate is not None:
            threading.Thread(
                target=end_at_switch_point, name=f"{self.name}-switch", daemon=True
            ).start()
        last_plain = ""
        try:
            with context.log_path.open("w", encoding="utf-8") as log_file:
                result.log_path = str(context.log_path)
                exited_at: float | None = None
                while True:
                    try:
                        raw = lines.get(timeout=0.2)
                    except queue.Empty:
                        if proc.poll() is None:
                            continue
                        exited_at = exited_at or time.monotonic()
                        if time.monotonic() - exited_at > DRAIN_SECONDS:
                            break
                        continue
                    if raw is None:
                        break
                    line = redact(raw.rstrip("\n"), context.credentials)
                    result.raw_lines.append(line)
                    log_file.write(line + "\n")
                    log_file.flush()
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        last_plain = line.strip()
                        self.consume_plain(line, result, context)
                        continue
                    if not isinstance(event, dict):
                        continue
                    self.consume_event(event, result, context)
            if proc.poll() is None:
                proc.wait()
        finally:
            done.set()
            timer.cancel()
            if proc.poll() is None:
                _kill_process_group(proc)
        result.returncode = int(proc.returncode or 0)
        limit = None if context.summary else context.plan.switch_after
        if gate is not None:
            result.turns = gate.tool_replies
            result.model_turns = gate.replies
        # Claude Code stops itself there, under `--max-turns`; the others are
        # stopped by the gate.
        if (
            (gate is not None and gate.reached.is_set())
            or (limit and result.turns >= limit)
            or result.status == "switch_limit"
        ):
            result.status = "switch_limit"
        elif timed_out.is_set():
            result.status = "timeout"
            result.error = (
                f"no result within the {context.timeout_seconds}s stage timeout"
                + (f"; last error: {result.error}" if result.error else "")
            )
        elif result.status == "failed":
            pass
        elif result.returncode == 0:
            result.status = "completed"
        else:
            result.status = "failed"
        if result.status == "failed" and not result.error:
            result.error = (
                _signal_death(result.returncode)
                or (context.gateway.last_error if context.gateway is not None else "")
                or last_plain
            )
        result.final_text = result.final_text.strip()
        # A retry that later succeeded leaves its message behind. A stage that
        # finished without an answer keeps it, as the reason there is none.
        if result.status == "switch_limit" or (
            result.status == "completed" and result.final_text
        ):
            result.error = ""
        result.error = result.error[:ERROR_CHARS]
        return result


class ClaudeAdapter(HarnessAdapter):
    name = "claude-code"

    def build_command(self, context: HarnessContext) -> tuple[list[str], dict[str, str]]:
        if context.gateway is None:
            raise HarnessError("Claude Code requires the embedded Cascade gateway")
        env = _agent_env()
        # The gateway answers only once a reply is complete and re-sends a lost
        # upstream call itself, so Claude Code has to wait as long as the
        # gateway does. Its own default gives up after five minutes, which a
        # long reply from a slow model exceeds.
        patience_ms = str(int(context.gateway.config.request_timeout_seconds * 1000))
        model = _claude_code_model(context)
        env.update(
            {
                "ANTHROPIC_BASE_URL": context.gateway.base_url,
                "ANTHROPIC_AUTH_TOKEN": context.gateway.config.local_token,
                "ANTHROPIC_MODEL": model,
                "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1",
                # Claude Code's own default is 32,000 whatever the model allows,
                # half of what Pi and Codex get from the same gateway.
                "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(
                    context.gateway.config.max_output_tokens
                ),
                "API_TIMEOUT_MS": patience_ms,
                "CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS": patience_ms,
                # The env form of `--bare`. Without it Claude Code declares 20
                # tools -- CronCreate, Workflow, ScheduleWakeup, EnterWorktree,
                # SendMessage and the rest -- none of which can do anything in
                # a benchmark container, and which cost 11.6k of the 12.5k
                # token preamble re-sent on every call. Bare mode leaves bash,
                # edit and read, for a 842 token preamble. The published
                # Terminal-Bench runs set exactly this, so matching them is
                # also what makes our numbers comparable to theirs.
                "CLAUDE_CODE_SIMPLE": "1",
                "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY": "1",
                "DISABLE_FEEDBACK_COMMAND": "1",
                "CLAUDE_CONFIG_DIR": str(context.state_dir / "claude"),
            }
        )
        if context.gateway.config.compact_at_percent:
            # Claude Code ignores the window the gateway's model list reports
            # for a model it does not recognise and plans against 200,000.
            # Claude models are covered by `_claude_code_model` instead.
            env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(
                context.gateway.config.context_window
            )
            env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = str(
                context.gateway.config.compact_at_percent
            )
            if model != context.model:
                # Under `[1m]` alone Claude Code compacts only once the API
                # rejects a prompt; a window it is given is one it enforces.
                env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(
                    min(context.gateway.config.context_window, CLAUDE_CODE_1M_WINDOW)
                )
        (context.state_dir / "claude").mkdir(parents=True, exist_ok=True)
        cmd = ["claude"]
        if context.resume:
            cmd.extend(["--resume", context.session_id])
        else:
            cmd.extend(["--session-id", context.session_id])
        cli_effort = (
            "low"
            if context.plan.effort in {"none", "off", "minimal"}
            else context.plan.effort
        )
        cmd.extend(
            [
                "--model",
                model,
                "--effort",
                cli_effort,
                "--permission-mode",
                "bypassPermissions",
                "--exclude-dynamic-system-prompt-sections",
            ]
        )
        # Only a turn budget of our own is passed: one for a summary, the
        # handoff point otherwise. A solo stage runs to Claude Code's default,
        # which sets no limit -- as Pi and Codex run. Claude Code counts one
        # turn per model call and stops once the tools of the last one have
        # run, which is where the gateway would hold it anyway.
        turn_budget = 1 if context.summary else context.plan.switch_after
        if turn_budget:
            cmd.extend(["--max-turns", str(turn_budget)])
        if context.summary:
            cmd.extend(
                [
                    "--tools",
                    "",
                    "--disallowedTools",
                    "mcp__*",
                    "--output-format",
                    "json",
                ]
            )
        else:
            cmd.extend(["--output-format", "stream-json", "--verbose"])
        cmd.extend(
            [
                "--settings",
                json.dumps(
                    {
                        "skipWebFetchPreflight": True,
                        "showThinkingSummaries": True,
                        "attribution": {"commit": "", "pr": ""},
                    },
                    separators=(",", ":"),
                ),
                "-p",
            ]
        )
        if not context.prompt_on_stdin:
            cmd.extend(["--", context.prompt])
        return cmd, env

    def consume_event(
        self,
        event: dict[str, Any],
        result: HarnessResult,
        context: HarnessContext,
    ) -> None:
        kind = event.get("type")
        if kind == "assistant":
            message = event.get("message") or {}
            text = _message_text(message)
            reasoning = _reasoning_text(message)
            if reasoning:
                result.final_reasoning = reasoning
                context.events.emit(
                    EventKind.THINKING_SUMMARY,
                    stage=context.stage,
                    harness=self.name,
                    text=reasoning,
                )
            if text:
                result.final_text = text
                context.events.emit(
                    EventKind.ASSISTANT_TEXT,
                    stage=context.stage,
                    harness=self.name,
                    text=text,
                )
            for block in message.get("content") or []:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                name = str(block.get("name") or "tool")
                detail = json.dumps(block.get("input") or {}, ensure_ascii=False)
                if name.lower() in {"bash", "shell"}:
                    command = str((block.get("input") or {}).get("command") or "")
                    if command:
                        result.commands.append(command)
                context.events.emit(
                    EventKind.TOOL_CALL_STARTED,
                    stage=context.stage,
                    harness=self.name,
                    turn=_reply_number(context),
                    name=name,
                    detail=detail,
                )
        elif kind == "user":
            message = event.get("message") or {}
            for block in message.get("content") or []:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                context.events.emit(
                    EventKind.TOOL_CALL_FINISHED,
                    stage=context.stage,
                    harness=self.name,
                    name="tool",
                    detail=_message_text({"content": block.get("content")}),
                    failed=bool(block.get("is_error")),
                )
        elif kind == "result":
            _set_usage(result, event.get("usage"))
            if event.get("subtype") == "error_max_turns" and not context.summary:
                result.status = "switch_limit"
            elif event.get("is_error"):
                detail = str(event.get("result") or "").strip()
                errors = event.get("errors")
                if not detail and isinstance(errors, list):
                    detail = "; ".join(str(item) for item in errors if item)
                # An API error arrives as `subtype: success` with `is_error`.
                subtype = str(event.get("subtype") or "")
                if subtype == "success":
                    subtype = ""
                result.error = (
                    ": ".join(part for part in (subtype, detail) if part)
                    or "Claude Code reported an error"
                )
            if event.get("result"):
                result.final_text = str(event["result"])
                if context.summary and not event.get("is_error"):
                    context.events.emit(
                        EventKind.ASSISTANT_TEXT,
                        stage=context.stage,
                        harness=self.name,
                        text=result.final_text,
                    )


class PiAdapter(HarnessAdapter):
    name = "pi"

    def build_command(self, context: HarnessContext) -> tuple[list[str], dict[str, str]]:
        if context.gateway is None:
            raise HarnessError("Pi requires the embedded Cascade gateway")
        env = _agent_env()
        pi_config_dir = context.state_dir / "pi-config"
        pi_config_dir.mkdir(parents=True, exist_ok=True)
        (pi_config_dir / "models.json").write_text(
            json.dumps(
                {
                    "providers": {
                        "cascade": {
                            "baseUrl": f"{context.gateway.base_url}/v1",
                            "api": "openai-completions",
                            "apiKey": "$CASCADE_GATEWAY_TOKEN",
                            "authHeader": True,
                            "compat": {
                                "supportsDeveloperRole": False,
                                "supportsReasoningEffort": False,
                                "maxTokensField": "max_tokens",
                            },
                            "models": [
                                {
                                    "id": context.model,
                                    "name": context.model,
                                    "reasoning": context.plan.effort != "off",
                                    "input": ["text"],
                                    # Pi decides when to compact from these, so
                                    # a made-up number changes how the agent
                                    # behaves. Both are resolved against the
                                    # provider at gateway start; see
                                    # limits.discover_model_limits.
                                    "contextWindow": context.gateway.config.context_window,
                                    "maxTokens": context.gateway.config.max_output_tokens,
                                    "cost": {
                                        "input": 0,
                                        "output": 0,
                                        "cacheRead": 0,
                                        "cacheWrite": 0,
                                    },
                                }
                            ],
                        }
                    }
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        # Pi abandons a call after five idle minutes and starts it over. The
        # gateway already re-sends a lost upstream call itself, so a second
        # clock in Pi only throws away a reply the gateway is still producing.
        (pi_config_dir / "settings.json").write_text(
            json.dumps(
                {
                    "httpIdleTimeoutMs": int(
                        context.gateway.config.request_timeout_seconds * 1000
                    )
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        env.update(
            {
                "CASCADE_GATEWAY_TOKEN": context.gateway.config.local_token,
                "PI_CODING_AGENT_DIR": str(pi_config_dir),
            }
        )
        session_dir = context.state_dir / "pi"
        session_dir.mkdir(parents=True, exist_ok=True)
        env["PI_CODING_AGENT_SESSION_DIR"] = str(session_dir)
        cmd = [
            "pi",
            "--provider",
            "cascade",
            "--model",
            context.model,
            "--thinking",
            context.plan.effort,
            "--mode",
            "json",
            "--print",
            "--no-extensions",
        ]
        if context.resume:
            cmd.extend(["--session", context.session_id])
        else:
            cmd.extend(["--session-id", context.session_id])
        if context.summary:
            cmd.append("--no-tools")
        if context.prompt_on_stdin:
            return cmd, env
        prompt = context.prompt
        if prompt.lstrip().startswith("-"):
            prompt = f"Task:\n{prompt}"
        cmd.append(prompt)
        return cmd, env

    def consume_event(
        self,
        event: dict[str, Any],
        result: HarnessResult,
        context: HarnessContext,
    ) -> None:
        kind = event.get("type")
        message = event.get("message") or {}
        if kind == "message_end":
            if message.get("role") != "assistant":
                return
            text = _message_text(message)
            reasoning = _reasoning_text(message)
            if reasoning:
                result.final_reasoning = reasoning
                context.events.emit(
                    EventKind.THINKING_SUMMARY,
                    stage=context.stage,
                    harness=self.name,
                    text=reasoning,
                )
            if text:
                result.final_text = text
                context.events.emit(
                    EventKind.ASSISTANT_TEXT,
                    stage=context.stage,
                    harness=self.name,
                    text=text,
                )
            for block in message.get("content") or []:
                if not isinstance(block, dict) or block.get("type") != "toolCall":
                    continue
                name = str(block.get("name") or "tool")
                arguments = block.get("arguments") or {}
                detail = json.dumps(arguments, ensure_ascii=False)
                if name.lower() == "bash":
                    command = str(arguments.get("command") or arguments.get("cmd") or "")
                    if command:
                        result.commands.append(command)
                result.pending_tools.append(name)
                context.events.emit(
                    EventKind.TOOL_CALL_STARTED,
                    stage=context.stage,
                    harness=self.name,
                    turn=_reply_number(context),
                    name=name,
                    detail=detail,
                )
            _set_usage(result, message.get("usage"))
            error = str(message.get("errorMessage") or "").strip()
            if COMPACTION_POINT_MARKER in error:
                # The gateway's cue to compact. Pi compacts and resends, which
                # clears this; it ends the run instead when it finds nothing
                # old enough to summarise.
                result.error = f"Pi did not compact when asked: {error}"
                return
            if not text and message.get("stopReason") == "length":
                result.error = "the reply reached the output token limit before any text"
            elif not text and not message.get("content"):
                result.error = "the model returned an empty reply"
            if message.get("stopReason") == "error" or error:
                # Pi still exits 0 after this, so it is the only record of
                # why the stage failed.
                result.status = "failed"
                result.error = error or "Pi model turn failed"
                context.events.emit(
                    EventKind.ERROR,
                    stage=context.stage,
                    harness=self.name,
                    message=result.error,
                )
        elif kind == "turn_end":
            error = str(message.get("errorMessage") or "").strip()
            if (
                message.get("stopReason") == "toolUse"
                and event.get("toolResults")
                and not error
            ):
                for index, tool_result in enumerate(event.get("toolResults") or []):
                    name = (
                        result.pending_tools[index]
                        if index < len(result.pending_tools)
                        else "tool"
                    )
                    detail = _message_text(
                        {"content": (tool_result or {}).get("content")}
                    )
                    context.events.emit(
                        EventKind.TOOL_CALL_FINISHED,
                        stage=context.stage,
                        harness=self.name,
                        name=name,
                        detail=detail,
                        failed=bool((tool_result or {}).get("isError")),
                    )
                result.pending_tools.clear()
        elif kind == "auto_retry_end" and event.get("success"):
            # A failed attempt emits an error message before Pi's built-in
            # retry. Clear that provisional failure when the retry succeeds.
            result.status = "running"
            result.error = ""
        elif kind == "compaction_end":
            if event.get("errorMessage"):
                result.error = f"compaction failed: {event['errorMessage']}"
            elif event.get("result") is not None:
                # After a context overflow Pi compacts and resends; the
                # overflow's error message no longer ends the stage.
                result.error = ""
                if result.status == "failed":
                    result.status = "running"


# Codex features whose tools no task here needs, turned off so they are not
# offered. Open models call the subagent and goal tools by names that do not
# exist, and the upstream drops such a call along with the turn. An image
# view_image returns never reaches the upstream, yet Codex re-sends it on
# every later request until the body outgrows the gateway's limit (413).
CODEX_DISABLED_FEATURES = ("multi_agent", "goals", "view_image")


class CodexAdapter(HarnessAdapter):
    name = "codex"

    def build_command(self, context: HarnessContext) -> tuple[list[str], dict[str, str]]:
        if context.gateway is None:
            raise HarnessError("Codex requires the embedded Cascade gateway")
        env = _agent_env()
        token_env = "CASCADE_GATEWAY_TOKEN"
        env[token_env] = context.gateway.config.local_token
        codex_home = context.state_dir / "codex"
        codex_home.mkdir(parents=True, exist_ok=True)
        env["CODEX_HOME"] = str(codex_home)
        gateway = context.gateway
        threshold = gateway.config.compact_at_tokens
        models: list[dict[str, Any]] | None = None
        if _codex_catalog_enabled(context.model):
            models = [gateway.codex_model()]
        elif threshold is not None and codex_builtin_entry(context.model) is not None:
            models = gateway.codex_builtin_catalog()
        elif threshold is not None:
            models = gateway.codex_fallback_catalog()
            if models is None:
                context.events.emit(
                    EventKind.WARNING,
                    stage=context.stage,
                    harness=self.name,
                    message=(
                        "Codex's fallback prompt was not found in its binary, so "
                        "Codex keeps its own 272,000-token window"
                    ),
                )
        catalog: list[str] = []
        if models is not None:
            catalog_path = codex_home / f"models-{context.session_id}.json"
            catalog_path.write_text(
                json.dumps({"models": models}, indent=2),
                encoding="utf-8",
            )
            catalog = ["-c", f'model_catalog_json="{catalog_path}"']
        if threshold is not None:
            # Codex clamps the window to the entry's max_context_window and
            # compacts at the lower of this limit and 90% of the window.
            catalog += [
                "-c",
                f"model_context_window={gateway.config.context_window}",
                "-c",
                f"model_auto_compact_token_limit={threshold}",
            ]

        cmd = ["codex", "exec"]
        if context.resume:
            cmd.append("resume")
        cmd.extend(
            ["--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check", "--json"]
        )
        if not context.resume:
            cmd.extend(["-C", str(context.workspace)])
        cmd.extend(
            [
                "-c",
                f'model="{context.model}"',
                "-c",
                'model_provider="cascade"',
                *catalog,
                "-c",
                'model_providers.cascade.name="Cascade"',
                "-c",
                f'model_providers.cascade.base_url="{context.gateway.base_url}/v1"',
                "-c",
                'model_providers.cascade.wire_api="responses"',
                "-c",
                f'model_providers.cascade.env_key="{token_env}"',
                "-c",
                f'model_reasoning_effort="{context.plan.effort}"',
            ]
        )
        for feature in CODEX_DISABLED_FEATURES:
            cmd.extend(["-c", f"features.{feature}=false"])
        if context.resume:
            cmd.append(context.session_id)
        cmd.extend(["-"] if context.prompt_on_stdin else ["--", context.prompt])
        return cmd, env

    def consume_event(
        self,
        event: dict[str, Any],
        result: HarnessResult,
        context: HarnessContext,
    ) -> None:
        kind = str(event.get("type") or "")
        item = event.get("item") if isinstance(event.get("item"), dict) else {}
        item_type = str(item.get("type") or item.get("item_type") or "")
        if kind == "thread.started":
            result.session_id = str(event.get("thread_id") or result.session_id)
        elif kind == "item.started" and item_type in {
            "command_execution",
            "file_change",
            "mcp_tool_call",
            "web_search",
        }:
            detail = str(item.get("command") or item.get("text") or item_type)
            context.events.emit(
                EventKind.TOOL_CALL_STARTED,
                stage=context.stage,
                harness=self.name,
                turn=_reply_number(context),
                name=item_type,
                detail=detail,
            )
        elif kind == "turn.completed":
            _set_usage(result, event.get("usage"))
        elif kind == "turn.failed":
            error = event.get("error")
            message = error.get("message") if isinstance(error, dict) else error
            result.error = str(message or "Codex turn failed")
        elif kind == "error":
            result.error = str(event.get("message") or "Codex reported an error")
        elif kind == "item.completed":
            if item_type in {
                "command_execution",
                "file_change",
                "mcp_tool_call",
                "web_search",
            }:
                detail = str(item.get("command") or item.get("text") or item_type)
                if item_type == "command_execution" and detail:
                    result.commands.append(detail)
                context.events.emit(
                    EventKind.TOOL_CALL_FINISHED,
                    stage=context.stage,
                    harness=self.name,
                    turn=_reply_number(context),
                    name=item_type,
                    detail=detail,
                    failed=item.get("status") == "failed",
                )
            elif item_type in {"agent_message", "message"}:
                text = str(item.get("text") or _message_text(item))
                if text:
                    result.final_text = text
                    context.events.emit(
                        EventKind.ASSISTANT_TEXT,
                        stage=context.stage,
                        harness=self.name,
                        text=text,
                    )
            elif item_type in {"reasoning", "reasoning_summary"}:
                text = str(item.get("text") or item.get("summary") or "")
                if text:
                    result.final_reasoning = text
                    context.events.emit(
                        EventKind.THINKING_SUMMARY,
                        stage=context.stage,
                        harness=self.name,
                        text=text,
                    )


    def consume_plain(
        self, line: str, result: HarnessResult, context: HarnessContext
    ) -> None:
        # A command Codex's exec policy refuses never becomes an item; the
        # only trace is the tool router's error log, which ends with
        # "`<command>` rejected: <justification>\")" }".
        match = _CODEX_REJECTED.search(line)
        if match:
            context.events.emit(
                EventKind.TOOL_CALL_FINISHED,
                stage=context.stage,
                harness=self.name,
                turn=_reply_number(context),
                name="exec_command",
                detail=f"Codex refused to run it: {match.group(1)}",
                failed=True,
            )


_CODEX_REJECTED = re.compile(r'` rejected: (.*?)\\"\)" \}\s*$')


ADAPTERS: dict[str, type[HarnessAdapter]] = {
    "claude-code": ClaudeAdapter,
    "pi": PiAdapter,
    "codex": CodexAdapter,
}


def adapter_for(name: str) -> HarnessAdapter:
    adapter = ADAPTERS.get(name)
    if adapter is None:
        raise HarnessError(f"unsupported harness {name!r}")
    return adapter()


def new_session_id() -> str:
    return str(uuid.uuid4())
