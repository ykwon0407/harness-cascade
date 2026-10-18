"""Native upstream provider transports."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import httpx

from .events import EventBus, EventKind
from .protocols import anthropic_messages, openai_chat, openai_responses
from .protocols.canonical import CanonicalRequest, CanonicalResponse

RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Longest gap between two attempts. Exponential backoff reaches it on the
# fifth attempt and stays there, so a caller that allows n retries waits
# roughly n * this in the worst case. A rate limit imposed by a shared
# endpoint outlasts single-digit seconds, which is what this used to cap at.
BACKOFF_CEILING_SECONDS = 30

# Longest a server's Retry-After is honoured. Past it the request goes out
# anyway: a header asking for an hour would otherwise hold the gateway thread,
# and the agent, past any stage timeout.
RETRY_AFTER_CEILING_SECONDS = 60.0


def retry_delay(retry_after: str | None, attempt: int) -> float:
    """Seconds to wait before re-sending attempt `attempt` (0-based)."""
    backoff = float(min(2**attempt, BACKOFF_CEILING_SECONDS))
    if not retry_after:
        return backoff
    try:
        asked = float(retry_after)
    except ValueError:
        return backoff
    try:
        ceiling = float(os.environ.get("CASCADE_MAX_RETRY_AFTER", ""))
    except ValueError:
        ceiling = RETRY_AFTER_CEILING_SECONDS
    return max(0.1, min(asked, ceiling))


# Longest one attempt may wait before it is re-sent. The shared endpoint
# occasionally drops a request while holding the connection open: its router
# stops counting the request in flight, no byte ever comes back, and before
# this cap the client waited out the whole trial budget. A non-streamed answer
# arrives in one piece, so waiting cannot tell a lost request from a slow one;
# the cap is set well clear of the slow ones instead. The longest of ~310k
# successful calls across four campaigns took 551s.
ATTEMPT_CEILING_SECONDS = 900.0


def _attempt_ceiling() -> float:
    try:
        value = float(os.environ.get("CASCADE_ATTEMPT_TIMEOUT", ""))
    except ValueError:
        return ATTEMPT_CEILING_SECONDS
    return value if value > 0 else ATTEMPT_CEILING_SECONDS


# Chat Completions calls are streamed upstream and reassembled here, so what
# the agent receives is unchanged. A stream shows progress, which is what lets
# a dropped request be told from a slow one: on the dedicated endpoint the
# first chunk arrives within 3s and later chunks at most 0.2s apart, so a
# silence this long means the request is gone and is re-sent at once rather
# than after the attempt ceiling.
STREAM_IDLE_SECONDS = 120.0


def _stream_upstream() -> bool:
    value = os.environ.get("CASCADE_UPSTREAM_STREAM", "").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _stream_idle() -> float:
    try:
        value = float(os.environ.get("CASCADE_STREAM_IDLE_TIMEOUT", ""))
    except ValueError:
        return STREAM_IDLE_SECONDS
    return value if value > 0 else STREAM_IDLE_SECONDS


# A reply can be billed for output it never delivers. vLLM's tool parser drops
# a call to a tool the request did not offer, along with the tokens that wrote
# it, and finishes the reply as an ordinary stop: GLM-5.3-Flash did this to
# 3-9% of Codex's calls, and every harness reads a reply with no call as the
# agent being done. Sampling again recovers it; a third loss in a row is
# passed on as it came.
UNDELIVERED_REPLY_RETRIES = 2

# Output tokens a reply may leave unaccounted for before it counts as lost.
# The lost calls measured 57-163 tokens; a reply that delivers what it was
# billed for misses by a handful at most.
UNDELIVERED_TOKENS = 40

# Bytes per token when estimating what a reply's text cost. Three
# undercounts English (about four) and roughly matches CJK, so the estimate
# errs toward calling a reply delivered.
BYTES_PER_TOKEN = 3

_TRUNCATED_STOPS = {"length", "max_tokens", "max_output_tokens", "incomplete"}


def undelivered_tokens(request: CanonicalRequest, response: CanonicalResponse) -> int:
    """Output tokens the reply was billed for but carries neither as text nor as a call.

    0 unless the loss is large enough, and large relative to the reply, to
    mean a tool call went missing upstream.
    """
    if not request.tools or response.stop_reason in _TRUNCATED_STOPS:
        return 0
    if any(block.type == "tool_call" for block in response.blocks):
        return 0
    usage = response.usage
    # Reasoning has its own count when the provider reports one; without it,
    # the reasoning text stands in for the tokens it took.
    spent = usage.output_tokens - usage.reasoning_tokens
    shown = sum(
        len(block.text.encode("utf-8"))
        for block in response.blocks
        if block.type == "text" or (block.type == "reasoning" and not usage.reasoning_tokens)
    )
    missing = spent - shown // BYTES_PER_TOKEN
    if missing < UNDELIVERED_TOKENS or 2 * missing <= spent:
        return 0
    return missing


class StreamBroken(RuntimeError):
    """The stream stopped before the reply was complete; worth sending again."""


def assemble_chat_stream(lines: Iterable[str]) -> dict[str, Any]:
    """Rebuild the non-streamed Chat Completions reply from its chunks."""
    payload: dict[str, Any] = {"object": "chat.completion"}
    content: list[str] = []
    reasoning: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    finish: str | None = None
    usage: dict[str, Any] | None = None
    done = False
    for line in lines:
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        try:
            event = json.loads(data)
        except ValueError as exc:
            raise StreamBroken(f"unparseable stream chunk: {data[:200]}") from exc
        if not isinstance(event, dict):
            continue
        if event.get("error"):
            error = event["error"]
            message = error.get("message") if isinstance(error, dict) else error
            raise StreamBroken(f"endpoint reported an error mid-stream: {message}")
        for key in ("id", "created", "model", "system_fingerprint"):
            if event.get(key) is not None:
                payload.setdefault(key, event[key])
        usage = event.get("usage") or usage
        for choice in event.get("choices") or []:
            if int(choice.get("index") or 0) != 0:
                continue
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            # One of the two per chunk: servers that fill both fill them with
            # the same text, and joining both would say everything twice.
            piece = delta.get("reasoning_content") or delta.get("reasoning")
            if piece:
                reasoning.append(piece)
            for call in delta.get("tool_calls") or []:
                slot = calls.setdefault(
                    int(call.get("index") or 0),
                    {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if call.get("id"):
                    slot["id"] = call["id"]
                function = call.get("function") or {}
                slot["function"]["name"] += function.get("name") or ""
                slot["function"]["arguments"] += function.get("arguments") or ""
            finish = choice.get("finish_reason") or finish
    if not done and finish is None:
        raise StreamBroken("stream closed before the reply was complete")
    message: dict[str, Any] = {"role": "assistant", "content": "".join(content) or None}
    if reasoning:
        message["reasoning"] = "".join(reasoning)
    if calls:
        message["tool_calls"] = [calls[index] for index in sorted(calls)]
    payload["choices"] = [{"index": 0, "message": message, "finish_reason": finish}]
    payload["usage"] = usage or {}
    return payload


def _dump_request(body: dict[str, Any], config: Any) -> Path | None:
    """Write the outbound body to CASCADE_DUMP_REQUESTS, if that is set.

    Prompt caching is a property of the byte prefix we send upstream, so the
    only way to diagnose a cache miss is to compare consecutive requests. Off
    unless the directory is named, because these files carry the full prompt.
    """
    directory = os.environ.get("CASCADE_DUMP_REQUESTS")
    if not directory:
        return None
    try:
        target = Path(directory) / f"{config.harness}-{config.stage}"
        target.mkdir(parents=True, exist_ok=True)
        seq = len(list(target.glob("[0-9][0-9][0-9][0-9].json")))
        path = target / f"{seq:04d}.json"
        path.write_text(
            json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        return path
    except OSError:
        return None


def _dump_response(
    request_path: Path | None, payload: dict[str, Any], attempt: int = 0
) -> None:
    """The upstream's answer to a dumped request, beside it.

    A tool call the upstream returned but the harness never ran is invisible
    from the harness side, so the raw answer is the only place it shows up.
    A reply sampled again keeps the one it replaced, as `.response.json`,
    next to `.response.retry<n>.json`.
    """
    if request_path is None:
        return
    suffix = f".response.retry{attempt}.json" if attempt else ".response.json"
    try:
        request_path.with_suffix(suffix).write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    except OSError:
        pass


class ProviderError(RuntimeError):
    def __init__(self, status: int, message: str, *, code: str = "api_error") -> None:
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(frozen=True, slots=True)
class TransportConfig:
    provider: str
    protocol: str
    model: str
    effort: str
    api_key: str
    base_url: str
    supported_efforts: tuple[str, ...] = ()
    max_output_tokens: int = 65_536
    timeout_seconds: float = 1800
    max_retries: int = 3
    stage: str = ""
    harness: str = ""


def _url(base_url: str, suffix: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith(suffix):
        return base
    if base.endswith("/v1"):
        return f"{base}{suffix.removeprefix('/v1')}"
    return f"{base}{suffix}"


def _error_code(status: int) -> str:
    if status == 400:
        return "invalid_request_error"
    if status == 401:
        return "authentication_error"
    if status == 403:
        return "permission_error"
    if status == 404:
        return "not_found_error"
    if status in {408, 504}:
        return "timeout_error"
    if status == 429:
        return "rate_limit_error"
    if status in {500, 502, 503}:
        return "overloaded_error"
    return "api_error"


class ProviderClient:
    def __init__(
        self,
        config: TransportConfig,
        events: EventBus | None = None,
    ) -> None:
        self.config = config
        self.events = events
        self._stream_timing: dict[str, float] = {}

    def _request(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.config.protocol == "openai-responses":
            url = _url(self.config.base_url, "/v1/responses")
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }
        elif self.config.protocol == "anthropic-messages":
            url = _url(self.config.base_url, "/v1/messages")
            headers = {
                "x-api-key": self.config.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
        else:
            url = _url(self.config.base_url, "/v1/chat/completions")
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }

        last_error: Exception | None = None
        per_attempt = min(self.config.timeout_seconds, _attempt_ceiling())
        for attempt in range(self.config.max_retries + 1):
            sent = time.perf_counter()
            assembled: dict[str, Any] | None = None
            try:
                if body.get("stream"):
                    response, assembled = self._post_streamed(url, headers, body)
                else:
                    response = httpx.post(
                        url,
                        headers=headers,
                        json=body,
                        timeout=httpx.Timeout(per_attempt, connect=30.0),
                    )
            except (httpx.HTTPError, StreamBroken) as exc:
                last_error = exc
                self._attempt_failed(
                    attempt,
                    f"{type(exc).__name__}: {exc}",
                    sent,
                    retrying=attempt < self.config.max_retries,
                )
                if attempt >= self.config.max_retries:
                    raise ProviderError(503, f"LLM request failed: {exc}") from exc
                time.sleep(min(2**attempt, BACKOFF_CEILING_SECONDS))
                continue
            if assembled is not None:
                return assembled
            if response.status_code < 400:
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise ProviderError(
                        502, "LLM endpoint returned invalid JSON"
                    ) from exc
                if not isinstance(payload, dict):
                    raise ProviderError(
                        502, "LLM endpoint returned a non-object response"
                    )
                return payload
            detail = response.text[:1000]
            try:
                error_payload = response.json()
                error = error_payload.get("error") or {}
                detail = str(
                    error.get("message")
                    if isinstance(error, dict)
                    else error or error_payload.get("message") or detail
                )
            except ValueError:
                pass
            detail = self._redacted(detail)
            self._attempt_failed(
                attempt,
                f"HTTP {response.status_code}: {detail[:300]}",
                sent,
                retrying=response.status_code in RETRYABLE_STATUS
                and attempt < self.config.max_retries,
            )
            if (
                response.status_code not in RETRYABLE_STATUS
                or attempt >= self.config.max_retries
            ):
                raise ProviderError(
                    response.status_code,
                    f"LLM endpoint returned {response.status_code}: {detail}",
                    code=_error_code(response.status_code),
                )
            time.sleep(retry_delay(response.headers.get("retry-after"), attempt))
        raise ProviderError(503, f"LLM request failed: {last_error or 'no response'}")

    def _post_streamed(
        self, url: str, headers: dict[str, str], body: dict[str, Any]
    ) -> tuple[httpx.Response, dict[str, Any] | None]:
        """One streamed attempt: the assembled reply, or the error response."""
        idle = _stream_idle()
        deadline = time.perf_counter() + self.config.timeout_seconds
        with httpx.stream(
            "POST",
            url,
            headers=headers,
            json=body,
            timeout=httpx.Timeout(idle, connect=30.0),
        ) as response:
            if response.status_code >= 400:
                response.read()
                return response, None
            return response, assemble_chat_stream(
                self._watched(response.iter_lines(), idle, deadline)
            )

    def _watched(self, lines: Iterator[str], idle: float, deadline: float) -> Iterator[str]:
        """Pass the lines through, turning silence into a named failure.

        Also records when the first data arrived and the longest gap after it,
        which is the evidence for whether `idle` is set where it should be.
        """
        started = last = time.perf_counter()
        chunks = 0
        self._stream_timing = {}
        try:
            for line in lines:
                now = time.perf_counter()
                if line.startswith("data:"):
                    if chunks == 0:
                        self._stream_timing["first_chunk_seconds"] = round(now - started, 3)
                    else:
                        gap = self._stream_timing.get("max_gap_seconds", 0.0)
                        self._stream_timing["max_gap_seconds"] = round(max(gap, now - last), 3)
                    chunks += 1
                    last = now
                if now > deadline:
                    raise StreamBroken(
                        f"stream still running at the {self.config.timeout_seconds:.0f}s budget"
                    )
                yield line
        except httpx.ReadTimeout as exc:
            raise StreamBroken(
                f"no data for {idle:.0f}s after {chunks} chunks"
            ) from exc

    def _redacted(self, text: str) -> str:
        """An error body can echo the key back; it goes on to the agent."""
        # A placeholder key as short as a word would redact ordinary text.
        if len(self.config.api_key) < 8:
            return text
        return text.replace(self.config.api_key, "[REDACTED]")

    def _attempt_failed(
        self, attempt: int, reason: str, sent: float, *, retrying: bool
    ) -> None:
        """Record a failed attempt; retries are otherwise invisible in the trace.

        Without this, a call that times out and is re-sent reads in the event
        log as one call that simply never finished.
        """
        if not self.events:
            return
        reason = self._redacted(reason)
        self.events.emit(
            EventKind.WARNING,
            stage=self.config.stage,
            harness=self.config.harness,
            message="model call attempt failed",
            attempt=attempt + 1,
            of=self.config.max_retries + 1,
            reason=reason,
            seconds=round(time.perf_counter() - sent, 3),
            retrying=retrying,
        )

    def complete(self, request: CanonicalRequest) -> CanonicalResponse:
        config = self.config
        if config.protocol == "openai-responses":
            body = openai_responses.encode_request(
                request,
                model=config.model,
                effort=config.effort,
                max_tokens=config.max_output_tokens,
            )
        elif config.protocol == "anthropic-messages":
            body = anthropic_messages.encode_request(
                request,
                model=config.model,
                effort=config.effort,
                max_tokens=config.max_output_tokens,
            )
        else:
            body = openai_chat.encode_request(
                request,
                model=config.model,
                effort=config.effort,
                max_tokens=config.max_output_tokens,
            )
            if config.effort in {"none", "off"}:
                # Dropping the field is not the same as asking for no thinking:
                # every measured deployment treats an absent reasoning_effort as
                # its default, which is the *largest* budget. Send the explicit
                # value whenever the model profile says the model accepts it.
                if "none" in config.supported_efforts:
                    body["reasoning_effort"] = "none"
                else:
                    body.pop("reasoning_effort", None)
            if config.provider == "together":
                body["reasoning"] = {"enabled": config.effort not in {"none", "off"}}
                body.setdefault("temperature", 1.0)
                body.setdefault("top_p", 0.95)
            if _stream_upstream():
                body["stream"] = True
                body["stream_options"] = {"include_usage": True}
        dumped = _dump_request(body, config)
        for attempt in range(UNDELIVERED_REPLY_RETRIES + 1):
            result = self._call(body, dumped, attempt)
            missing = undelivered_tokens(request, result)
            if not missing:
                return result
            retrying = attempt < UNDELIVERED_REPLY_RETRIES
            if self.events:
                self.events.emit(
                    EventKind.WARNING,
                    stage=config.stage,
                    harness=config.harness,
                    message="upstream billed a reply it did not deliver",
                    attempt=attempt + 1,
                    of=UNDELIVERED_REPLY_RETRIES + 1,
                    output_tokens=result.usage.output_tokens,
                    undelivered_tokens=missing,
                    finish_reason=result.stop_reason,
                    retrying=retrying,
                )
            if not retrying:
                return result
        raise AssertionError("unreachable")

    def _call(self, body: dict[str, Any], dumped: Path | None, attempt: int) -> CanonicalResponse:
        """One model call as the event log counts it: sent, answered, decoded."""
        config = self.config
        if self.events:
            self.events.emit(
                EventKind.MODEL_CALL_STARTED,
                stage=config.stage,
                harness=config.harness,
                model=config.model,
                effort=config.effort,
                protocol=config.protocol,
            )
        started = time.perf_counter()
        payload = self._request(body)
        _dump_response(dumped, payload, attempt)
        try:
            if config.protocol == "openai-responses":
                result = openai_responses.decode_response(payload)
            elif config.protocol == "anthropic-messages":
                result = anthropic_messages.decode_response(payload)
            else:
                result = openai_chat.decode_response(payload)
        except (TypeError, ValueError, KeyError) as exc:
            raise ProviderError(
                502,
                f"{config.protocol} endpoint returned an invalid response: {exc}",
            ) from exc
        if self.events:
            self.events.emit(
                EventKind.MODEL_CALL_FINISHED,
                stage=config.stage,
                harness=config.harness,
                model=config.model,
                protocol=config.protocol,
                duration_seconds=round(time.perf_counter() - started, 3),
                usage=result.usage.to_dict(),
                **self._stream_timing,
            )
        return result
