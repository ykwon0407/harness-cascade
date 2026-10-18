"""Embedded protocol gateway for Claude Code, Codex, and OpenAI-compatible chat."""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import mmap
import os
import re
import shutil
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, replace
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Self

import httpx

from .events import EventBus, EventKind
from .limits import (
    FALLBACK_CONTEXT_WINDOW,
    FALLBACK_MAX_OUTPUT_TOKENS,
    discover_model_limits,
    output_ceiling,
)
from .protocols import anthropic_messages, openai_chat, openai_responses
from .protocols.canonical import (
    Block,
    CanonicalRequest,
    CanonicalResponse,
    Message,
    ProtocolCompatibilityError,
    Tool,
)
from .protocols.state import OpaqueStateStore
from .providers import (
    BACKOFF_CEILING_SECONDS,
    ProviderClient,
    ProviderError,
    TransportConfig,
    retry_delay,
)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _retry_budget() -> int:
    """How many times to re-send a rejected request.

    Three is enough for a dedicated endpoint, which rate limits nobody. A
    shared serverless endpoint under a campaign's worth of parallel agents
    answers 429 often enough that three attempts inside eight seconds lose
    whole trials, so campaigns against one raise this.
    """
    try:
        budget = int(os.environ.get("CASCADE_MAX_RETRIES", ""))
    except ValueError:
        return 3
    return budget if budget >= 0 else 3


def _codex_native_binary() -> Path | None:
    """The vendored Rust executable behind the `codex` shim on PATH."""
    shim = shutil.which("codex")
    if not shim:
        return None
    here = Path(shim).resolve()
    for parent in [here.parent, *here.parents][:6]:
        matches = sorted(parent.glob("**/vendor/*/bin/codex"))
        if matches:
            return matches[0]
    return here if here.suffix != ".js" else None


_OPEN, _CLOSE = ord("{"), ord("}")


@contextlib.contextmanager
def _mapped(binary: Path) -> Iterator[bytes | mmap.mmap]:
    with binary.open("rb") as handle:
        if os.fstat(handle.fileno()).st_size == 0:
            yield b""
            return
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as blob:
            yield blob


def _catalog_entries(blob: bytes | mmap.mmap) -> list[dict[str, Any]]:
    """Every model object in Codex's embedded catalog.

    The catalog is a JSON literal compiled into the binary, so the entries are
    recovered by finding each `"slug"` and brace-matching outwards. Entries that
    do not parse are skipped rather than raising: the goal is to find one usable
    reference, not to model the whole file. The blob stays bytes: the binary
    is a quarter of a gigabyte, and as text it took a gigabyte of memory.
    """
    found: dict[str, dict[str, Any]] = {}
    for match in re.finditer(rb'"slug":\s*"([A-Za-z0-9._-]+)"', blob):
        depth = 0
        start = match.start()
        while start > 0:
            if blob[start] == _CLOSE:
                depth += 1
            elif blob[start] == _OPEN:
                if depth == 0:
                    break
                depth -= 1
            start -= 1
        depth = 0
        stop = start
        while stop < len(blob):
            if blob[stop] == _OPEN:
                depth += 1
            elif blob[stop] == _CLOSE:
                depth -= 1
                if depth == 0:
                    break
            stop += 1
        try:
            entry = json.loads(blob[start : stop + 1])
        except ValueError:
            continue
        if isinstance(entry, dict) and "context_window" in entry:
            found.setdefault(str(entry.get("slug")), entry)
    return list(found.values())


_HOSTED_PARALLEL_SENTENCE = re.compile(r"[ \t]*[^.\n`]*`multi_tool_use\.parallel`[^.\n]*\.")


def _without_hosted_parallel_tool(text: str) -> str:
    """Codex's prompt minus its reference to OpenAI's hosted parallel wrapper.

    `multi_tool_use.parallel` is not a Codex tool: OpenAI's server declares it
    to its own models and fans each call out into ordinary parallel calls.
    Nothing declares it to a model behind this gateway, and GLM told to use
    it invents wrapper calls (`exec_tool_join`, `multi_tool_use.parallel:
    true`) that the upstream's tool parser drops, leaving a reply with no
    call, which Codex takes as the end of the task. Measured on replayed
    requests: 26 of 96 samples lost their call with the sentence, 0 of 96
    without. Parallel calls themselves stay available through the native
    `parallel_tool_calls` the request already carries.
    """
    stripped = _HOSTED_PARALLEL_SENTENCE.sub("", text)
    if "multi_tool_use" in stripped:
        raise GatewayError(
            500,
            "Codex's prompt references `multi_tool_use` in a form this gateway "
            "does not recognise; the wrapper cannot be served here, so update "
            "_HOSTED_PARALLEL_SENTENCE rather than send it to the model",
        )
    return stripped


@lru_cache(maxsize=1)
def _builtin_catalog() -> tuple[dict[str, Any], ...]:
    binary = _codex_native_binary()
    if binary is None:
        raise GatewayError(
            500, "codex is not installed, so its model catalog cannot be read"
        )
    with _mapped(binary) as blob:
        return tuple(_catalog_entries(blob))


def codex_entry_servable(entry: dict[str, Any]) -> bool:
    """Whether this gateway can speak the wire the entry asks Codex to use.

    The gpt-5.6 tier cannot: those run in code mode and over the lite
    Responses wire, neither of which this gateway implements.
    """
    return (
        not entry.get("tool_mode")
        and not entry.get("use_responses_lite")
        and bool((entry.get("model_messages") or {}).get("instructions_template"))
    )


def codex_builtin_entry(model: str) -> dict[str, Any] | None:
    """Codex's own catalog entry for `model`, if it ships one."""
    if _codex_native_binary() is None:
        return None
    slug = model.rsplit("/", 1)[-1]
    return next((e for e in _builtin_catalog() if e.get("slug") == slug), None)


@lru_cache(maxsize=1)
def codex_reference_entry() -> dict[str, Any]:
    """A real Codex catalog entry to model ours on.

    Picks the highest-priority entry the gateway can actually serve, since
    copying one it cannot would hand Codex a contract we cannot honour.
    """
    usable = [entry for entry in _builtin_catalog() if codex_entry_servable(entry)]
    if not usable:
        raise GatewayError(
            500,
            f"no usable model entry found in {_codex_native_binary()}; refusing "
            "to invent a system prompt, which would silently change how the "
            "agent behaves",
        )
    return min(usable, key=lambda entry: entry.get("priority") or 99)


# First and last line of the prompt Codex gives a model it has no metadata
# for (its `prompt.md`, compiled in with `include_str!`).
_CODEX_FALLBACK_PROMPT_START = b"You are a coding agent running in the Codex CLI"
_CODEX_FALLBACK_PROMPT_END = (
    b"If all steps are complete, ensure you call `update_plan` to mark all "
    b"steps as `completed`.\n"
)


@lru_cache(maxsize=1)
def codex_fallback_prompt() -> str | None:
    """Codex's prompt for a model it does not know, read from its binary.

    None when it cannot be found, which a Codex release that rewrites the
    prompt's first or last line would cause; the caller then leaves Codex on
    its own fallback rather than guess at the text.
    """
    binary = _codex_native_binary()
    if binary is None:
        return None
    with _mapped(binary) as blob:
        start = blob.find(_CODEX_FALLBACK_PROMPT_START)
        if start < 0:
            return None
        end = blob.find(_CODEX_FALLBACK_PROMPT_END, start)
        if end < 0:
            return None
        try:
            prompt = blob[start : end + len(_CODEX_FALLBACK_PROMPT_END)].decode("utf-8")
        except UnicodeDecodeError:
            return None
    return prompt if 5_000 <= len(prompt) <= 100_000 else None


def _dump_inbound(body: dict[str, Any], config: Any, label: str) -> Path | None:
    """Mirror of providers._dump_request for the harness side of the bridge.

    Seeing both sides matters: a cache miss upstream is usually caused by
    something the harness varies downstream, and only the pair shows that.
    """
    directory = os.environ.get("CASCADE_DUMP_REQUESTS")
    if not directory:
        return None
    try:
        target = Path(directory) / f"inbound-{label}"
        target.mkdir(parents=True, exist_ok=True)
        seq = len(list(target.glob("[0-9][0-9][0-9][0-9].json")))
        path = target / f"{seq:04d}.json"
        path.write_text(
            json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        return path
    except OSError:
        return None


def _dump_outbound(request_path: Path | None, payload: dict[str, Any]) -> None:
    """What the harness was handed back, beside the request that asked for it."""
    if request_path is None:
        return
    try:
        request_path.with_suffix(".response.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    except OSError:
        pass


MAX_REQUEST_BYTES = 32 * 1024 * 1024


class CascadeHTTPServer(ThreadingHTTPServer):
    def handle_error(self, request: Any, client_address: Any) -> None:
        error = sys.exc_info()[1]
        if isinstance(error, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


class GatewayError(RuntimeError):
    def __init__(self, status: int, message: str, *, code: str = "api_error") -> None:
        super().__init__(message)
        self.status = status
        self.code = code


class SwitchHold(GatewayError):
    """The answer to a request held at the switch point; not a model failure."""


# Pi reads this as a context overflow (it matches Pi's `context_length_exceeded`
# pattern), compacts, and resends; the marker lets the adapter tell it apart
# from a real one.
COMPACTION_POINT_MARKER = "cascade compaction point"


class CompactionPoint(GatewayError):
    """Pi's cue to compact now; Pi checks its own threshold only between prompts."""

    def __init__(self, context_tokens: int, threshold: int) -> None:
        super().__init__(
            400,
            f"context_length_exceeded: {COMPACTION_POINT_MARKER}: the context "
            f"reached {context_tokens:,} tokens, past {threshold:,}; compact and retry",
            code="context_length_exceeded",
        )


def _as_gateway_error(exc: BaseException) -> GatewayError:
    if isinstance(exc, GatewayError):
        return exc
    if isinstance(exc, ProviderError):
        return GatewayError(exc.status, str(exc), code=exc.code)
    if isinstance(exc, ProtocolCompatibilityError):
        return GatewayError(400, str(exc), code="compatibility_error")
    return GatewayError(500, f"gateway error: {exc}")


class SwitchGate:
    """Counts a stage's work and holds back the request past its switch point.

    The unit is one model reply that asked for at least one tool, counted here
    because every harness's model traffic passes through this one place. Each
    harness reports progress differently on stdout -- Claude Code counts model
    calls, Pi its turns, Codex every finished tool item -- so a limit read off
    those streams meant a different amount of work for each, and could only
    act after the harness had already sent its next request, which the
    provider then billed after the stage's usage had been taken.

    Once `limit` tool replies have been answered, the next request is not
    forwarded: it waits here until the gateway stops, and `reached` tells the
    adapter to end the harness. Every tool the last reply asked for has run by
    then, because a harness only asks again once it has all of their results.
    """

    def __init__(
        self,
        limit: int | None,
        on_reached: Callable[[int], None] | None = None,
    ) -> None:
        self.limit = limit
        self.replies = 0
        self.tool_replies = 0
        self.reached = threading.Event()
        self._on_reached = on_reached
        self._released = threading.Event()
        self._lock = threading.Lock()

    def admit(self) -> None:
        with self._lock:
            hold = bool(self.limit) and self.tool_replies >= int(self.limit or 0)
            first = hold and not self.reached.is_set()
            if hold:
                self.reached.set()
        if not hold:
            return
        if first and self._on_reached is not None:
            self._on_reached(self.tool_replies)
        self._released.wait()
        raise SwitchHold(
            503,
            "stage ended at its configured switch point",
            code="overloaded_error",
        )

    def record(self, response: CanonicalResponse) -> int:
        """Count a reply; return its number among tool replies, or 0."""
        with self._lock:
            self.replies += 1
            if any(block.type == "tool_call" for block in response.blocks):
                self.tool_replies += 1
                return self.tool_replies
            return 0

    def release(self) -> None:
        self._released.set()


# Codex plans every model it ships against 272,000 tokens, the input above
# which OpenAI bills a whole request at its long-context rate.
CODEX_COMPACT_AT_MOST = 272_000

# Room left free beyond a request's estimated size, as Pi leaves, since a
# provider that counts input and output against one window rejects a request
# whose output limit would not fit in what remains.
CONTEXT_SAFETY_TOKENS = 4096
CHARS_PER_TOKEN = 4
IMAGE_TOKENS = 1200


def estimate_tokens(messages: Iterable[Message], tools: Iterable[Tool] = ()) -> int:
    chars = 0
    images = 0
    for message in messages:
        for block in message.blocks:
            if block.type in {"image", "document"}:
                images += 1
            chars += len(block.text) + len(block.name)
            if block.arguments:
                chars += len(json.dumps(block.arguments, ensure_ascii=False))
    for tool in tools:
        chars += len(tool.name) + len(tool.description)
        chars += len(json.dumps(tool.parameters, ensure_ascii=False))
    return -(-chars // CHARS_PER_TOKEN) + images * IMAGE_TOKENS


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    model: str
    effort: str
    api_key: str
    local_token: str
    base_url: str = "https://api.together.ai/v1"
    provider: str = "openai-compatible"
    protocol: str = "openai-chat"
    supported_efforts: tuple[str, ...] = ()
    # The limits in force, valid once start() has resolved them against the
    # provider. Until then they hold the values this code used before it
    # learned to ask, so a gateway that is never started still behaves.
    max_output_tokens: int = FALLBACK_MAX_OUTPUT_TOKENS
    context_window: int = FALLBACK_CONTEXT_WINDOW
    # What the model profile recorded, if anything. Only consulted when the
    # provider will not say; see limits.discover_model_limits.
    profile_context_window: int | None = None
    profile_max_output_tokens: int | None = None
    request_timeout_seconds: float = 1800
    # How long a streamed chat reply may go without a byte on the wire. The
    # upstream is heard out to the end before anything is relayed, so a long
    # generation would otherwise look to the agent like a dead connection.
    stream_heartbeat_seconds: float = 10
    max_retries: int = field(default_factory=_retry_budget)
    disable_tools: bool = False
    stage: str = ""
    harness: str = ""
    state_path: str = ""
    # Tool replies after which the stage hands off; see SwitchGate.
    switch_after: int | None = None
    # Share of context_window at which the agent compacts; None leaves each
    # harness its own window and threshold.
    compact_at_percent: int | None = None

    @property
    def compact_at_tokens(self) -> int | None:
        if not self.compact_at_percent:
            return None
        threshold = self.context_window * self.compact_at_percent // 100
        if self.harness == "codex":
            return min(threshold, CODEX_COMPACT_AT_MOST)
        return threshold


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            text = item.get("text") or item.get("content")
            if text is not None:
                parts.append(str(text))
    return "\n".join(part for part in parts if part)


def _tool_result_text(content: Any) -> str:
    text = _text_content(content)
    if text:
        return text
    if isinstance(content, list):
        placeholders: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("type") or "")
            if kind == "image":
                media_type = str(
                    (item.get("source") or {}).get("media_type") or "image"
                )
                placeholders.append(f"[{media_type} tool result omitted]")
            elif kind:
                placeholders.append(f"[{kind} tool result omitted]")
        if placeholders:
            return "\n".join(placeholders)
    return "[tool completed with no textual output]"


def _usage(payload: dict[str, Any]) -> dict[str, int]:
    raw = payload.get("usage") or {}
    details = (
        raw.get("completion_tokens_details") or raw.get("output_tokens_details") or {}
    )
    reasoning = int(details.get("reasoning_tokens") or raw.get("reasoning_tokens") or 0)
    input_tokens = int(raw.get("prompt_tokens") or raw.get("input_tokens") or 0)
    output_tokens = int(raw.get("completion_tokens") or raw.get("output_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning,
        "total_tokens": input_tokens + output_tokens,
    }


def _error_code(status: int) -> str:
    if status == 400:
        return "invalid_request_error"
    if status == 401:
        return "authentication_error"
    if status == 403:
        return "permission_error"
    if status == 404:
        return "not_found_error"
    if status == 408 or status == 504:
        return "timeout_error"
    if status == 429:
        return "rate_limit_error"
    if status in {500, 502, 503}:
        return "overloaded_error"
    return "api_error"


def chat_completions_url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


class ChatCompletionsClient:
    def __init__(self, config: GatewayConfig, events: EventBus | None = None) -> None:
        self.config = config
        self.events = events

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        outgoing = dict(payload)
        outgoing.update({"model": self.config.model, "stream": False})
        outgoing.pop("stream_options", None)
        if self.config.effort != "off":
            outgoing["reasoning_effort"] = self.config.effort
        else:
            outgoing.pop("reasoning", None)
            outgoing.pop("reasoning_effort", None)
        if self.config.provider == "together":
            outgoing.update(
                {
                    "reasoning": {"enabled": self.config.effort != "off"},
                    "temperature": 1.0,
                    "top_p": 0.95,
                }
            )
        outgoing["max_tokens"] = min(
            max(1, int(outgoing.get("max_tokens") or self.config.max_output_tokens)),
            self.config.max_output_tokens,
        )
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        url = chat_completions_url(self.config.base_url)

        started = time.perf_counter()
        if self.events:
            self.events.emit(
                EventKind.MODEL_CALL_STARTED,
                stage=self.config.stage,
                harness=self.config.harness,
                model=self.config.model,
                effort=self.config.effort,
            )
        last_error: Exception | None = None
        try:
            for attempt in range(self.config.max_retries + 1):
                try:
                    response = httpx.post(
                        url,
                        headers=headers,
                        json=outgoing,
                        timeout=httpx.Timeout(
                            self.config.request_timeout_seconds,
                            connect=30.0,
                        ),
                    )
                except httpx.HTTPError as exc:
                    last_error = exc
                    if attempt >= self.config.max_retries:
                        raise GatewayError(503, f"LLM request failed: {exc}") from exc
                    time.sleep(min(2**attempt, BACKOFF_CEILING_SECONDS))
                    continue

                if response.status_code < 400:
                    try:
                        result = response.json()
                    except ValueError as exc:
                        raise GatewayError(
                            502, "LLM endpoint returned invalid JSON"
                        ) from exc
                    if not isinstance(result, dict) or not result.get("choices"):
                        raise GatewayError(
                            502, "LLM endpoint returned no completion choices"
                        )
                    if self.events:
                        self.events.emit(
                            EventKind.MODEL_CALL_FINISHED,
                            stage=self.config.stage,
                            harness=self.config.harness,
                            model=self.config.model,
                            duration_seconds=round(time.perf_counter() - started, 3),
                            usage=_usage(result),
                        )
                    return result

                detail = response.text[:1000]
                try:
                    error_payload = response.json()
                    detail = str(
                        (error_payload.get("error") or {}).get("message")
                        or error_payload.get("message")
                        or detail
                    )
                except ValueError:
                    pass
                if len(self.config.api_key) >= 8:
                    detail = detail.replace(self.config.api_key, "[REDACTED]")
                if (
                    response.status_code not in RETRYABLE_STATUS
                    or attempt >= self.config.max_retries
                ):
                    raise GatewayError(
                        response.status_code,
                        f"LLM endpoint returned {response.status_code}: {detail}",
                        code=_error_code(response.status_code),
                    )
                time.sleep(retry_delay(response.headers.get("retry-after"), attempt))
        finally:
            if last_error and self.events:
                self.events.emit(
                    EventKind.ERROR,
                    stage=self.config.stage,
                    harness=self.config.harness,
                    message=str(last_error),
                )
        raise GatewayError(503, "LLM request failed without a response")


# Backward-compatible import name for early adopters of the package.
TogetherClient = ChatCompletionsClient


def anthropic_to_chat(body: dict[str, Any], config: GatewayConfig) -> dict[str, Any]:
    system_parts = [
        "You are a coding model running through Harness Cascade. "
        f"Selected backend: {config.model}."
    ]
    anthropic_system = _text_content(body.get("system"))
    if anthropic_system:
        system_parts.append(anthropic_system)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "\n\n".join(system_parts)}
    ]

    for raw_message in body.get("messages") or []:
        if not isinstance(raw_message, dict):
            continue
        role = str(raw_message.get("role") or "user")
        content = raw_message.get("content")
        if role == "system":
            extra = (
                _text_content(content)
                if not isinstance(content, list)
                else _text_content(content)
            )
            if extra:
                messages[0]["content"] = f"{messages[0]['content']}\n\n{extra}"
            continue
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            messages.append({"role": role, "content": _text_content(content)})
            continue

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        multimodal_parts: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                text_parts.append(str(block))
                continue
            kind = block.get("type")
            if kind == "text":
                text = str(block.get("text") or "")
                text_parts.append(text)
                multimodal_parts.append({"type": "text", "text": text})
            elif kind in {"thinking", "redacted_thinking"}:
                reasoning_parts.append(
                    str(block.get("thinking") or block.get("data") or "")
                )
            elif kind in {"tool_use", "server_tool_use"}:
                call_id = str(block.get("id") or _new_id("call"))
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": str(block.get("name") or "tool"),
                            "arguments": json.dumps(
                                block.get("input") or {}, ensure_ascii=False
                            ),
                        },
                    }
                )
            elif kind == "tool_result":
                result_text = _tool_result_text(block.get("content"))
                if block.get("is_error"):
                    result_text = f"[tool_result error]\n{result_text}"
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(block.get("tool_use_id") or "unknown"),
                        "content": result_text,
                    }
                )
            elif kind == "image":
                source = block.get("source") or {}
                if source.get("type") == "base64":
                    multimodal_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    f"data:{source.get('media_type', 'image/png')};base64,"
                                    f"{source.get('data', '')}"
                                )
                            },
                        }
                    )

        messages.extend(tool_results)
        if text_parts or reasoning_parts or tool_calls or multimodal_parts:
            message: dict[str, Any] = {"role": role}
            if any(item.get("type") == "image_url" for item in multimodal_parts):
                message["content"] = multimodal_parts
            else:
                message["content"] = "\n".join(text_parts) or None
            if reasoning_parts and role == "assistant":
                message["reasoning_content"] = "\n".join(reasoning_parts)
            if tool_calls:
                message["tool_calls"] = tool_calls
            messages.append(message)

    tools: list[dict[str, Any]] = []
    for tool in [] if config.disable_tools else body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": str(tool.get("name") or "tool"),
                    "description": str(tool.get("description") or ""),
                    "parameters": tool.get("input_schema")
                    or {"type": "object", "properties": {}},
                },
            }
        )

    payload: dict[str, Any] = {
        "messages": messages,
        "max_tokens": min(
            int(body.get("max_tokens") or config.max_output_tokens),
            config.max_output_tokens,
        ),
    }
    if tools:
        payload["tools"] = tools
    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        if choice.get("type") == "auto":
            payload["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            payload["tool_choice"] = "required"
        elif choice.get("type") == "tool":
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": choice.get("name")},
            }
    if body.get("stop_sequences"):
        payload["stop"] = body["stop_sequences"]
    if body.get("temperature") is not None:
        payload["temperature"] = body["temperature"]
    return payload


def _chat_message(payload: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    choice = payload["choices"][0]
    message = choice.get("message") or {}
    return message, choice.get("finish_reason")


def _anthropic_stop(finish_reason: str | None, has_tools: bool) -> str:
    if has_tools or finish_reason == "tool_calls":
        return "tool_use"
    if finish_reason == "length":
        return "max_tokens"
    return "end_turn"


def anthropic_response(
    request: dict[str, Any],
    chat: dict[str, Any],
    config: GatewayConfig,
) -> dict[str, Any]:
    message, finish_reason = _chat_message(chat)
    blocks: list[dict[str, Any]] = []
    reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
    if reasoning:
        signature = hashlib.sha256(reasoning.encode()).hexdigest()[:16]
        blocks.append(
            {
                "type": "thinking",
                "thinking": reasoning,
                "signature": f"cascade:{signature}",
            }
        )
    content = message.get("content")
    if content:
        blocks.append({"type": "text", "text": _text_content(content)})
    tool_calls = message.get("tool_calls") or []
    for call in tool_calls:
        function = call.get("function") or {}
        raw_arguments = function.get("arguments") or "{}"
        try:
            arguments = json.loads(raw_arguments)
        except (TypeError, ValueError):
            arguments = {}
        blocks.append(
            {
                "type": "tool_use",
                "id": str(call.get("id") or _new_id("toolu")),
                "name": str(function.get("name") or "tool"),
                "input": arguments,
            }
        )
    usage = _usage(chat)
    return {
        "id": str(chat.get("id") or _new_id("msg")),
        "type": "message",
        "role": "assistant",
        "model": str(request.get("model") or config.model),
        "content": blocks,
        "stop_reason": _anthropic_stop(finish_reason, bool(tool_calls)),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
        },
    }


def _sse(event: str, payload: dict[str, Any]) -> bytes:
    return (
        f"event: {event}\ndata: "
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"
    ).encode()


def openai_chat_sse(response: dict[str, Any]) -> Iterable[bytes]:
    message, finish_reason = _chat_message(response)
    delta: dict[str, Any] = {"role": "assistant"}
    for key in ("content", "reasoning", "reasoning_content"):
        if message.get(key) is not None:
            delta[key] = message[key]
    if message.get("tool_calls"):
        delta["tool_calls"] = [
            {"index": index, **call} for index, call in enumerate(message["tool_calls"])
        ]
    response_id = str(response.get("id") or _new_id("chatcmpl"))
    created = int(response.get("created") or time.time())
    model = str(response.get("model") or "")

    def chunk(delta_value: dict[str, Any], finish: str | None) -> bytes:
        payload = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta_value,
                    "finish_reason": finish,
                }
            ],
        }
        if finish is not None:
            payload["usage"] = response.get("usage")
        return (
            "data: "
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            + "\n\n"
        ).encode()

    yield chunk(delta, None)
    yield chunk({}, finish_reason or "stop")
    yield b"data: [DONE]\n\n"


def anthropic_sse(response: dict[str, Any]) -> Iterable[bytes]:
    shell = dict(response)
    blocks = shell.pop("content")
    shell["content"] = []
    shell["stop_reason"] = None
    shell["usage"] = {"input_tokens": 0, "output_tokens": 0}
    yield _sse("message_start", {"type": "message_start", "message": shell})
    for index, block in enumerate(blocks):
        kind = block["type"]
        if kind == "thinking":
            start = {"type": "thinking", "thinking": "", "signature": ""}
        elif kind == "tool_use":
            start = {
                "type": "tool_use",
                "id": block["id"],
                "name": block["name"],
                "input": {},
            }
        else:
            start = {"type": "text", "text": ""}
        yield _sse(
            "content_block_start",
            {"type": "content_block_start", "index": index, "content_block": start},
        )
        if kind == "thinking":
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "thinking_delta",
                        "thinking": block["thinking"],
                    },
                },
            )
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "signature_delta",
                        "signature": block["signature"],
                    },
                },
            )
        elif kind == "tool_use":
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(block["input"], ensure_ascii=False),
                    },
                },
            )
        else:
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": block["text"]},
                },
            )
        yield _sse(
            "content_block_stop",
            {"type": "content_block_stop", "index": index},
        )
    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": response["stop_reason"],
                "stop_sequence": None,
            },
            "usage": response["usage"],
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})


def _codex_item_text(item: dict[str, Any]) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict):
            text = (
                part.get("text")
                or part.get("input_text")
                or part.get("output_text")
                or part.get("content")
            )
            if text:
                parts.append(str(text))
    return "\n".join(parts)


def codex_to_chat(body: dict[str, Any], config: GatewayConfig) -> dict[str, Any]:
    system_parts = [
        "You are a coding model running inside Codex through Harness Cascade. "
        f"Selected backend: {config.model}."
    ]
    if body.get("instructions"):
        system_parts.append(str(body["instructions"]))
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "\n\n".join(system_parts)}
    ]
    input_value = body.get("input")
    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
    elif isinstance(input_value, list):
        for item in input_value:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            role = str(item.get("role") or "user")
            if kind in {"message", "input_text", "output_text"} or "role" in item:
                text = _codex_item_text(item) or str(item.get("text") or "")
                if text:
                    messages.append({"role": role, "content": text})
            elif kind in {"function_call", "custom_tool_call", "local_shell_call"}:
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": str(
                                    item.get("call_id")
                                    or item.get("id")
                                    or _new_id("call")
                                ),
                                "type": "function",
                                "function": {
                                    "name": str(item.get("name") or kind),
                                    "arguments": str(
                                        item.get("arguments")
                                        or json.dumps({"input": item.get("input", "")})
                                    ),
                                },
                            }
                        ],
                    }
                )
            elif kind in {
                "function_call_output",
                "custom_tool_call_output",
                "local_shell_call_output",
            }:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(item.get("call_id") or "unknown"),
                        "content": _text_content(item.get("output")),
                    }
                )
            elif kind == "reasoning":
                summary = _text_content(item.get("summary"))
                if summary:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": summary,
                        }
                    )
            elif kind == "compaction":
                encoded = str(item.get("encrypted_content") or "")
                if encoded.startswith("cascade1:"):
                    messages.append(
                        {"role": "user", "content": encoded.removeprefix("cascade1:")}
                    )

    tools: list[dict[str, Any]] = []
    for tool in [] if config.disable_tools else body.get("tools") or []:
        if not isinstance(tool, dict) or tool.get("defer_loading"):
            continue
        kind = tool.get("type")
        if kind == "namespace":
            namespace = str(tool.get("name") or "namespace")
            for child in tool.get("tools") or []:
                if not isinstance(child, dict):
                    continue
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": f"{namespace}__{child.get('name', 'tool')}",
                            "description": str(child.get("description") or ""),
                            "parameters": child.get("parameters")
                            or {"type": "object", "properties": {}},
                        },
                    }
                )
        elif kind == "custom":
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": str(tool.get("name") or "custom"),
                        "description": str(tool.get("description") or ""),
                        "parameters": {
                            "type": "object",
                            "properties": {"input": {"type": "string"}},
                            "required": ["input"],
                        },
                    },
                }
            )
        elif kind in {"function", None}:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": str(tool.get("name") or "tool"),
                        "description": str(tool.get("description") or ""),
                        "parameters": tool.get("parameters")
                        or {"type": "object", "properties": {}},
                    },
                }
            )
    payload: dict[str, Any] = {
        "messages": messages,
        "max_tokens": min(
            int(body.get("max_output_tokens") or config.max_output_tokens),
            config.max_output_tokens,
        ),
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = body.get("tool_choice") or "auto"
    return payload


def codex_response(
    request: dict[str, Any],
    chat: dict[str, Any],
    config: GatewayConfig,
) -> dict[str, Any]:
    message, finish_reason = _chat_message(chat)
    output: list[dict[str, Any]] = []
    reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
    if reasoning:
        output.append(
            {
                "id": _new_id("rs"),
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": reasoning}],
                "content": [],
            }
        )
    text = _text_content(message.get("content"))
    if text:
        output.append(
            {
                "id": _new_id("msg"),
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                    }
                ],
            }
        )
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        output.append(
            {
                "id": _new_id("fc"),
                "type": "function_call",
                "status": "completed",
                "call_id": str(call.get("id") or _new_id("call")),
                "name": str(function.get("name") or "tool"),
                "arguments": str(function.get("arguments") or "{}"),
            }
        )
    usage = _usage(chat)
    requested = int(request.get("max_output_tokens") or config.max_output_tokens)
    real_truncation = finish_reason == "length" and (
        usage["output_tokens"] >= requested * 0.9
        or requested - usage["output_tokens"] <= 1024
    )
    status = "incomplete" if real_truncation else "completed"
    return {
        "id": str(chat.get("id") or _new_id("resp")),
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "incomplete_details": (
            {"reason": "max_output_tokens"} if real_truncation else None
        ),
        "model": str(request.get("model") or config.model),
        "output": output,
        "usage": {
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "output_tokens_details": {"reasoning_tokens": usage["reasoning_tokens"]},
        },
    }


def codex_sse(response: dict[str, Any]) -> Iterable[bytes]:
    sequence = 0

    def event(name: str, payload: dict[str, Any]) -> bytes:
        nonlocal sequence
        payload = {"type": name, **payload, "sequence_number": sequence}
        sequence += 1
        return _sse(name, payload)

    shell = {
        "id": response["id"],
        "object": "response",
        "created_at": response["created_at"],
        "status": "in_progress",
        "model": response["model"],
        "output": [],
    }
    yield event("response.created", {"response": shell})
    yield event("response.in_progress", {"response": shell})
    for output_index, item in enumerate(response["output"]):
        pending = dict(item)
        pending["status"] = "in_progress"
        if item["type"] == "reasoning":
            pending["summary"] = []
        elif item["type"] == "message":
            pending["content"] = []
        yield event(
            "response.output_item.added",
            {"output_index": output_index, "item": pending},
        )
        if item["type"] == "reasoning":
            text = _text_content(item.get("summary"))
            yield event(
                "response.reasoning_text.delta",
                {
                    "item_id": item["id"],
                    "output_index": output_index,
                    "content_index": 0,
                    "delta": text,
                },
            )
            yield event(
                "response.reasoning_text.done",
                {
                    "item_id": item["id"],
                    "output_index": output_index,
                    "content_index": 0,
                    "text": text,
                },
            )
        elif item["type"] == "message":
            part = {"type": "output_text", "text": "", "annotations": []}
            text = _text_content(item.get("content"))
            yield event(
                "response.content_part.added",
                {
                    "item_id": item["id"],
                    "output_index": output_index,
                    "content_index": 0,
                    "part": part,
                },
            )
            yield event(
                "response.output_text.delta",
                {
                    "item_id": item["id"],
                    "output_index": output_index,
                    "content_index": 0,
                    "delta": text,
                },
            )
            done_part = {"type": "output_text", "text": text, "annotations": []}
            yield event(
                "response.output_text.done",
                {
                    "item_id": item["id"],
                    "output_index": output_index,
                    "content_index": 0,
                    "text": text,
                },
            )
            yield event(
                "response.content_part.done",
                {
                    "item_id": item["id"],
                    "output_index": output_index,
                    "content_index": 0,
                    "part": done_part,
                },
            )
        yield event(
            "response.output_item.done",
            {"output_index": output_index, "item": item},
        )
    terminal = (
        "response.incomplete"
        if response["status"] == "incomplete"
        else "response.completed"
    )
    yield event(terminal, {"response": response})


class EmbeddedGateway:
    def __init__(self, config: GatewayConfig, events: EventBus | None = None) -> None:
        self.config = config
        self.events = events
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._state = OpaqueStateStore(
            Path(config.state_path) if config.state_path else None
        )
        self.gate = SwitchGate(
            None if config.disable_tools else config.switch_after,
            self._switch_reached,
        )
        # The last error an agent was handed. Agents report it in their own
        # words if at all -- Pi exits 0 after one -- so the engine reads the
        # cause of a failed stage from here when the harness gave none.
        self.last_error = ""
        # Upstream calls not yet answered. One still open when its stage ends
        # was cut off by a kill; the provider bills it, but its usage may
        # never be seen.
        self.calls_in_flight = 0
        self._calls_lock = threading.Lock()
        # Prompt plus reply of the last reply that called a tool: the size the
        # agent's next request starts from.
        self.context_tokens = 0
        # Messages in the last request forwarded, and the tokens its prompt and
        # reply came to: a measured start for estimating the next request.
        self._last_prompt = (0, 0)

    def _switch_reached(self, tool_replies: int) -> None:
        if self.events is not None:
            self.events.emit(
                EventKind.SWITCH_TRIGGERED,
                stage=self.config.stage or None,
                harness=self.config.harness or None,
                turns=tool_replies,
            )

    def _failed(self, exc: BaseException) -> GatewayError:
        error = _as_gateway_error(exc)
        if isinstance(error, (SwitchHold, CompactionPoint)):
            return error
        message = str(error)
        if self.config.api_key:
            message = message.replace(self.config.api_key, "[REDACTED]")
        self.last_error = f"gateway returned {error.status} to the agent: {message}"
        if self.events is not None:
            self.events.emit(
                EventKind.WARNING,
                stage=self.config.stage or None,
                harness=self.config.harness or None,
                message=self.last_error[:2000],
                status=error.status,
            )
        return error

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("gateway is not started")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def _resolve_limits(self) -> None:
        """Replace the placeholder limits with what the provider reports.

        Done before the socket opens so that every answer the gateway gives --
        the model catalogs the agents read at startup, the output cap sent
        upstream -- is the same number. Cached per provider/model/endpoint, so
        the lookup costs one request per process however many stages run.
        """
        resolved = discover_model_limits(
            provider=self.config.provider,
            model=self.config.model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            profile_context_window=self.config.profile_context_window,
            profile_max_output_tokens=self.config.profile_max_output_tokens,
        )
        max_output = resolved.max_output_tokens
        ceiling = output_ceiling()
        if ceiling is not None:
            max_output = min(max_output, ceiling)
        self.config = replace(
            self.config,
            context_window=resolved.context_window,
            max_output_tokens=max_output,
        )
        if self.events is not None:
            for note in resolved.notes:
                self.events.emit(
                    EventKind.WARNING,
                    stage=self.config.stage or None,
                    message=note,
                )
            self.events.emit(
                EventKind.CONTEXT_LIMITS,
                stage=self.config.stage or None,
                harness=self.config.harness or None,
                context_window=self.config.context_window,
                max_output_tokens=self.config.max_output_tokens,
                source=resolved.source,
                compact_at_tokens=self.config.compact_at_tokens,
            )

    def start(self) -> Self:
        self._resolve_limits()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: object) -> None:
                return

            def _authorized(self) -> bool:
                auth = self.headers.get("Authorization", "")
                bearer = auth[7:] if auth.startswith("Bearer ") else ""
                api_key = self.headers.get("x-api-key", "")
                supplied = bearer or api_key
                return bool(supplied) and hmac.compare_digest(
                    supplied,
                    owner.config.local_token,
                )

            def _json(self, status: int, payload: dict[str, Any]) -> None:
                raw = json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(raw)

            def _gateway_error(self, exc: GatewayError, *, anthropic: bool) -> None:
                if anthropic:
                    payload = {
                        "type": "error",
                        "error": {"type": exc.code, "message": str(exc)},
                    }
                else:
                    payload = {
                        "error": {
                            "type": exc.code,
                            "code": exc.code,
                            "message": str(exc),
                        }
                    }
                self._json(exc.status, payload)

            def _body(self) -> dict[str, Any]:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError as exc:
                    raise GatewayError(400, "invalid Content-Length") from exc
                if length < 0 or length > MAX_REQUEST_BYTES:
                    raise GatewayError(413, "request body too large")
                raw = self.rfile.read(length)
                try:
                    payload = json.loads(raw or b"{}")
                except json.JSONDecodeError as exc:
                    raise GatewayError(400, "invalid JSON body") from exc
                if not isinstance(payload, dict):
                    raise GatewayError(400, "JSON object required")
                return payload

            def _stream(self, chunks: Iterable[bytes]) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                for chunk in chunks:
                    self.wfile.write(chunk)
                    self.wfile.flush()

            def _stream_when_ready(
                self,
                work: Callable[[], dict[str, Any]],
                render: Callable[[dict[str, Any]], Iterable[bytes]],
                failure: Callable[[GatewayError], Iterable[bytes]],
            ) -> None:
                # A reply that is ready within one heartbeat goes out exactly as
                # an unhurried one would, errors included as HTTP statuses. A
                # slower one commits to a 200 stream at once and keeps the socket
                # busy with SSE comments, which clients skip, so the agent's idle
                # timeout measures a stalled gateway rather than a long answer.
                # Past that point a failure can only be reported in the stream.
                heartbeat = owner.config.stream_heartbeat_seconds
                outcome: dict[str, Any] = {}
                done = threading.Event()

                def run() -> None:
                    try:
                        outcome["response"] = work()
                    except BaseException as exc:  # noqa: BLE001 - handed back to the request thread.
                        outcome["error"] = exc
                    finally:
                        done.set()

                threading.Thread(target=run, name="cascade-gateway-reply", daemon=True).start()
                if done.wait(heartbeat):
                    if "error" in outcome:
                        raise outcome["error"]
                    self._stream(render(outcome["response"]))
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.flush()
                while not done.wait(heartbeat):
                    self.wfile.write(b": waiting on the model\n\n")
                    self.wfile.flush()
                if "error" in outcome:
                    chunks = failure(owner._failed(outcome["error"]))
                else:
                    chunks = render(outcome["response"])
                for chunk in chunks:
                    self.wfile.write(chunk)
                    self.wfile.flush()

            def do_HEAD(self) -> None:
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path in {"/", "/health", "/healthz"}:
                    self._json(200, {"ok": True, "model": owner.config.model})
                    return
                if not self._authorized():
                    self._json(
                        401,
                        {
                            "type": "error",
                            "error": {
                                "type": "authentication_error",
                                "message": "Unauthorized local gateway request.",
                            },
                        },
                    )
                    return
                if path in {"/models", "/v1/models"}:
                    self._json(200, owner.model_catalog())
                    return
                if path.startswith("/v1/models/"):
                    self._json(200, owner.anthropic_model())
                    return
                self._json(404, {"error": {"message": "not found"}})

            def do_POST(self) -> None:
                path = self.path.split("?", 1)[0]
                anthropic = path.startswith("/v1/messages")
                if not self._authorized():
                    self._gateway_error(
                        GatewayError(
                            401,
                            "Unauthorized local gateway request.",
                            code="authentication_error",
                        ),
                        anthropic=anthropic,
                    )
                    return
                try:
                    body = self._body()
                    if path == "/v1/messages/count_tokens":
                        estimate = max(
                            1,
                            len(json.dumps(body, ensure_ascii=False).encode()) // 4,
                        )
                        self._json(200, {"input_tokens": estimate})
                        return
                    if path == "/v1/messages":
                        response = owner.handle_anthropic(body)
                        if body.get("stream"):
                            self._stream(anthropic_messages.sse(response))
                        else:
                            self._json(200, response)
                        return
                    if path in {"/responses", "/v1/responses"}:
                        response = owner.handle_codex(body)
                        if body.get("stream", True):
                            self._stream(openai_responses.sse(response))
                        else:
                            self._json(200, response)
                        return
                    if path == "/v1/responses/compact":
                        response = owner.handle_codex_compaction(body)
                        self._json(200, response)
                        return
                    if path == "/v1/chat/completions":
                        if body.get("stream"):
                            self._stream_when_ready(
                                lambda: owner.handle_chat(body),
                                openai_chat.sse,
                                lambda exc: openai_chat.sse_error(
                                    exc.status, exc.code, str(exc)
                                ),
                            )
                        else:
                            self._json(200, owner.handle_chat(body))
                        return
                    self._json(404, {"error": {"message": "not found"}})
                except (BrokenPipeError, ConnectionResetError):
                    return
                except Exception as exc:  # noqa: BLE001 - HTTP boundary returns JSON errors.
                    self._gateway_error(owner._failed(exc), anthropic=anthropic)

        self._server = CascadeHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="cascade-gateway",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self.gate.release()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *args: object) -> None:
        self.stop()

    def anthropic_model(self) -> dict[str, Any]:
        return {
            "id": self.config.model,
            "type": "model",
            "object": "model",
            "display_name": self.config.model.split("/")[-1],
            "max_input_tokens": self.config.context_window,
            "max_tokens": self.config.max_output_tokens,
            "created_at": "2026-01-01T00:00:00Z",
        }

    def codex_model(self) -> dict[str, Any]:
        """A catalog entry for our model, built from one of Codex's own.

        Codex refuses a catalog entry that carries neither `base_instructions`
        nor `model_messages.instructions_template`, so the prompt has to come
        from somewhere. Hand-writing one is what this code used to do, and the
        stand-in measured 211 characters on the wire against the 11k-21k that
        Codex ships. That prompt is not decoration: it is where Codex is told
        how persistently to pursue a task, what counts as verified, and when a
        turn may end. Without it the agent stopped after a handful of tool
        calls. So start from a real entry and override only what identifies our
        model, which also keeps the other three dozen fields honest.
        """
        base = dict(codex_reference_entry())
        base.update(
            {
                "slug": self.config.model,
                "display_name": self.config.model.split("/")[-1],
                "description": f"Model {self.config.model} via Cascade",
                # Codex keeps two windows: `max_context_window` is the model's
                # ceiling, while `context_window` is the smaller budget it
                # actually plans against -- 272,000 on every entry it ships,
                # which is the size above which OpenAI re-prices the whole
                # request. Without a compaction policy keep that budget, since
                # it is the harness's own choice, but never let either exceed
                # what our model has. Under one the threshold carries the cap.
                "context_window": (
                    self.config.context_window
                    if self.config.compact_at_percent
                    else min(
                        base.get("context_window") or self.config.context_window,
                        self.config.context_window,
                    )
                ),
                "max_context_window": self.config.context_window,
                "default_reasoning_level": self.config.effort,
                "supported_reasoning_levels": [
                    {
                        "effort": effort,
                        "description": f"{effort} reasoning effort through Cascade",
                    }
                    for effort in (
                        self.config.supported_efforts
                        or ("low", "medium", "high", "max")
                    )
                ],
                # The gateway speaks plain HTTP, so refuse the websocket path.
                "prefer_websockets": False,
                # Codex asks for a server-side `tool_search` when this is on,
                # and upstream answers `hosted tool type 'tool_search' is not
                # translatable`. It is a retrieval aid, not part of the agent
                # loop, so declining it costs nothing here.
                "supports_search_tool": False,
                "visibility": "list",
                "supported_in_api": True,
                "priority": 1,
                "minimal_client_version": None,
                # Copied wholesale this announces the model we borrowed from,
                # which is not the model anyone is running.
                "availability_nux": None,
            }
        )
        messages = dict(base.get("model_messages") or {})
        if messages.get("instructions_template"):
            messages["instructions_template"] = _without_hosted_parallel_tool(
                str(messages["instructions_template"])
            )
            base["model_messages"] = messages
        if base.get("base_instructions"):
            base["base_instructions"] = _without_hosted_parallel_tool(
                str(base["base_instructions"])
            )
        return base

    def codex_fallback_model(self) -> dict[str, Any] | None:
        """Codex's own fallback metadata, but with our model's window.

        Codex runs a model it does not know on fixed fallback metadata whose
        `max_context_window` of 272,000 caps any `model_context_window` it is
        given, so the window can only be raised through a catalog entry. This
        one copies `model_info_from_slug` (codex-rs/models-manager) field for
        field -- the same prompt, tools and truncation -- so the requests
        Codex sends are unchanged; only the window and the effort list, which
        `spawn_agent` validates against once the entry is not a fallback,
        differ. None if the prompt cannot be read from the binary.
        """
        prompt = codex_fallback_prompt()
        if prompt is None:
            return None
        return {
            "slug": self.config.model,
            "display_name": self.config.model,
            "description": None,
            "supported_reasoning_levels": [
                {"effort": effort, "description": f"{effort} reasoning effort through Cascade"}
                for effort in self.config.supported_efforts
            ],
            "shell_type": "default",
            "visibility": "none",
            "supported_in_api": True,
            "priority": 99,
            "additional_speed_tiers": [],
            "service_tiers": [],
            "availability_nux": None,
            "upgrade": None,
            "model_messages": {
                "instructions_template": prompt,
                "instructions_variables": None,
                "approvals": None,
                "collaboration_modes": None,
                "auto_review": None,
                "permissions": None,
            },
            "include_skills_usage_instructions": False,
            "include_plugin_usage_instructions": False,
            "include_apps_usage_instructions": False,
            "default_reasoning_summary": "auto",
            "support_verbosity": False,
            "default_verbosity": None,
            "apply_patch_tool_type": None,
            "web_search_tool_type": "text",
            "truncation_policy": {"mode": "bytes", "limit": 10_000},
            "supports_parallel_tool_calls": False,
            "supports_image_detail_original": False,
            "context_window": self.config.context_window,
            "max_context_window": self.config.context_window,
            "effective_context_window_percent": 95,
            "experimental_supported_tools": [],
            "input_modalities": ["text", "image"],
            "supports_search_tool": False,
            "use_responses_lite": False,
        }

    def codex_fallback_catalog(self) -> list[dict[str, Any]] | None:
        """The fallback copy plus every entry Codex ships.

        `model_catalog_json` replaces Codex's catalog rather than adding to it,
        and the catalog is also the list of models `spawn_agent` offers as
        overrides, so without Codex's own entries that tool's description
        would change.
        """
        entry = self.codex_fallback_model()
        if entry is None:
            return None
        return [entry, *(e for e in _builtin_catalog() if e.get("slug") != entry["slug"])]

    def codex_builtin_catalog(self) -> list[dict[str, Any]]:
        """Every entry Codex ships, with our model's window raised to its own.

        Codex clamps `model_context_window` to the entry's
        `max_context_window`, 272,000 on most, and compacts at no more than
        90% of the result, which would pull the threshold under the cap.
        """
        slug = self.config.model.rsplit("/", 1)[-1]
        return [
            {
                **entry,
                "context_window": self.config.context_window,
                "max_context_window": self.config.context_window,
            }
            if entry.get("slug") == slug
            else entry
            for entry in _builtin_catalog()
        ]

    def model_catalog(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [self.anthropic_model()],
            "models": [self.codex_model()],
        }

    def _complete(self, request: CanonicalRequest):
        if self.config.disable_tools:
            request.tools = []
            request.tool_choice = None
        self.gate.admit()
        self._compaction_point(request)
        with self._calls_lock:
            self.calls_in_flight += 1
        try:
            response = self._forward(request)
        finally:
            with self._calls_lock:
                self.calls_in_flight -= 1
        if not request.tools:
            self._drop_unoffered_tool_calls(response)
        reply = self.gate.record(response)
        if reply and self.events is not None:
            # Harnesses report the tools they ran, not the replies: Codex
            # prints nothing for a poll of a running command or for a command
            # its policy refuses. This is the count the switch point uses.
            self.events.emit(
                EventKind.TURN_STARTED,
                stage=self.config.stage or None,
                harness=self.config.harness or None,
                turn=reply,
                limit=self.gate.limit,
                tools=[b.name for b in response.blocks if b.type == "tool_call"],
            )
        if request.tools:
            self.context_tokens = (
                response.usage.input_tokens + response.usage.output_tokens
            )
        return response

    def _drop_unoffered_tool_calls(self, response: CanonicalResponse) -> None:
        """Remove tool calls from a reply to a request that offered no tools.

        The handoff summary and Codex's compaction offer none, so the harness
        must not act on the reply. Some upstreams call tools regardless, and a
        harness that ran one would go on working after the switch.
        """
        dropped = [b.name for b in response.blocks if b.type == "tool_call"]
        if not dropped:
            return
        response.blocks = [b for b in response.blocks if b.type != "tool_call"]
        if self.events is not None:
            self.events.emit(
                EventKind.WARNING,
                stage=self.config.stage or None,
                harness=self.config.harness or None,
                message="dropped tool calls from a reply to a request without tools",
                tools=dropped,
            )

    def _compaction_point(self, request: CanonicalRequest) -> None:
        """Turn Pi's next tool request into an overflow once past the threshold.

        Claude Code and Codex compact mid-task on their own once told the
        window and threshold. Pi checks its threshold only after a whole
        prompt has run, which in a stage is never, so on its own it runs on
        until the provider rejects a request. A request without tools is a
        summary and always passes.
        """
        threshold = self.config.compact_at_tokens
        if (
            self.config.harness != "pi"
            or threshold is None
            or not request.tools
            or self.context_tokens < threshold
        ):
            return
        reached, self.context_tokens = self.context_tokens, 0
        if self.events is not None:
            self.events.emit(
                EventKind.COMPACTION_REQUESTED,
                stage=self.config.stage or None,
                harness=self.config.harness or None,
                context_tokens=reached,
                threshold=threshold,
            )
        raise CompactionPoint(reached, threshold)

    def _output_limit(self, request: CanonicalRequest) -> int:
        """The model's output limit, less whatever the window cannot hold."""
        count, tokens = self._last_prompt
        if 0 < count <= len(request.messages):
            size = tokens + estimate_tokens(request.messages[count:])
        else:
            size = estimate_tokens(request.messages, request.tools)
        room = self.config.context_window - size - CONTEXT_SAFETY_TOKENS
        return max(1, min(self.config.max_output_tokens, room))

    def _forward(self, request: CanonicalRequest) -> CanonicalResponse:
        response = self._send(request, self._output_limit(request))
        if response.usage.input_tokens:
            self._last_prompt = (
                len(request.messages),
                response.usage.input_tokens + response.usage.output_tokens,
            )
        return response

    def _send(self, request: CanonicalRequest, max_output_tokens: int) -> CanonicalResponse:
        return ProviderClient(
            TransportConfig(
                provider=self.config.provider,
                protocol=self.config.protocol,
                model=self.config.model,
                effort=self.config.effort,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                supported_efforts=self.config.supported_efforts,
                max_output_tokens=max_output_tokens,
                timeout_seconds=self.config.request_timeout_seconds,
                max_retries=self.config.max_retries,
                stage=self.config.stage,
                harness=self.config.harness,
            ),
            self.events,
        ).complete(request)

    def handle_anthropic(self, body: dict[str, Any]) -> dict[str, Any]:
        _dump_inbound(body, self.config, "anthropic")
        request = anthropic_messages.decode_request(body, self._state)
        response = self._complete(request)
        return anthropic_messages.response_payload(
            response,
            model=self.config.model,
            state=self._state,
        )

    def handle_codex(self, body: dict[str, Any]) -> dict[str, Any]:
        if any(
            isinstance(item, dict) and item.get("type") == "compaction_trigger"
            for item in (body.get("input") or [])
            if isinstance(body.get("input"), list)
        ):
            return self._codex_compaction_response(body)
        dumped = _dump_inbound(body, self.config, "codex")
        request = openai_responses.decode_request(body, self._state)
        tools = list(request.tools)
        response = self._complete(request)
        payload = openai_responses.response_payload(
            response,
            model=self.config.model,
            state=self._state,
            tools=tools,
        )
        _dump_outbound(dumped, payload)
        return payload

    def handle_chat(self, body: dict[str, Any]) -> dict[str, Any]:
        request = openai_chat.decode_request(body)
        response = self._complete(request)
        return openai_chat.response_payload(response, model=self.config.model)

    def _codex_compaction_response(self, body: dict[str, Any]) -> dict[str, Any]:
        compact_body = dict(body)
        compact_body["input"] = [
            item
            for item in body.get("input") or []
            if not (isinstance(item, dict) and item.get("type") == "compaction_trigger")
        ]
        compact_body["tools"] = []
        request = openai_responses.decode_request(compact_body, self._state)
        request.tools = []
        request.messages.append(
            Message(
                "user",
                [
                    Block(
                        "text",
                        text=(
                            "Summarize the work so another coding agent can continue. "
                            "Include progress, decisions, files changed, tests, blockers, and next steps."
                        ),
                    )
                ],
            )
        )
        completed = self._complete(request)
        summary = "\n".join(
            block.text
            for block in completed.blocks
            if block.type in {"text", "reasoning"} and block.text
        )
        return {
            "id": completed.response_id or _new_id("resp"),
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "incomplete_details": None,
            "model": str(body.get("model") or self.config.model),
            "output": [
                {
                    "id": _new_id("cmp"),
                    "type": "compaction",
                    "encrypted_content": f"cascade1:{summary}",
                }
            ],
            "usage": {
                "input_tokens": completed.usage.input_tokens,
                "output_tokens": completed.usage.output_tokens,
                "total_tokens": completed.usage.total_tokens,
            },
        }

    def handle_codex_compaction(self, body: dict[str, Any]) -> dict[str, Any]:
        response = self._codex_compaction_response(body)
        return {"output": response["output"], "usage": response["usage"]}
