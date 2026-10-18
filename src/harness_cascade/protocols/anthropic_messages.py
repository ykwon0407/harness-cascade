"""Anthropic Messages request, response, and SSE codec."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from typing import Any

from .canonical import (
    Block,
    CanonicalRequest,
    CanonicalResponse,
    Message,
    ProtocolCompatibilityError,
    Tool,
    Usage,
    arguments_dict,
    split_preamble,
    text_content,
)
from .state import OpaqueStateStore


def _restore_thinking(
    block: dict[str, Any], state: OpaqueStateStore | None
) -> dict[str, Any]:
    signature = str(block.get("signature") or "")
    if signature.startswith("cascade:") and state:
        stored = state.get(signature.removeprefix("cascade:"))
        if stored:
            return {
                **block,
                "_cascade_data": stored,
            }
    return block


def decode_request(
    body: dict[str, Any],
    state: OpaqueStateStore | None = None,
) -> CanonicalRequest:
    messages: list[Message] = []
    system = text_content(body.get("system"))
    if system:
        messages.append(Message("system", [Block("text", text=system)]))
    for raw in body.get("messages") or []:
        if not isinstance(raw, dict):
            continue
        role = str(raw.get("role") or "user")
        content = raw.get("content")
        if isinstance(content, str):
            messages.append(Message(role, [Block("text", text=content)]))
            continue
        blocks: list[Block] = []
        for part in content or []:
            if not isinstance(part, dict):
                blocks.append(Block("text", text=str(part)))
                continue
            kind = str(part.get("type") or "")
            if kind == "text":
                blocks.append(Block("text", text=str(part.get("text") or "")))
            elif kind in {"thinking", "redacted_thinking"}:
                restored = _restore_thinking(part, state)
                cascade_data = restored.get("_cascade_data")
                blocks.append(
                    Block(
                        "reasoning",
                        text=str(
                            restored.get("thinking")
                            or restored.get("data")
                            or ""
                        ),
                        data=(
                            cascade_data
                            if isinstance(cascade_data, dict)
                            else {"anthropic": restored}
                        ),
                    )
                )
            elif kind in {"tool_use", "server_tool_use"}:
                blocks.append(
                    Block(
                        "tool_call",
                        call_id=str(part.get("id") or uuid.uuid4().hex),
                        name=str(part.get("name") or "tool"),
                        arguments=arguments_dict(part.get("input")),
                        data={"anthropic": part},
                    )
                )
            elif kind == "tool_result":
                blocks.append(
                    Block(
                        "tool_result",
                        call_id=str(part.get("tool_use_id") or "unknown"),
                        text=text_content(part.get("content"))
                        or "[tool completed with no textual output]",
                        data={"is_error": bool(part.get("is_error"))},
                    )
                )
            elif kind in {"image", "document"}:
                blocks.append(Block(kind, data=dict(part)))
        # One Anthropic user turn can carry tool results and new text together;
        # Claude Code sends that when it follows a result with a message of its
        # own, such as its notice after an output-limit cut. The results have
        # to leave as their own tool turn, ahead of the text: kept inside a user
        # message, a chat upstream never sees them.
        results = [block for block in blocks if block.type == "tool_result"]
        rest = [block for block in blocks if block.type != "tool_result"]
        if results:
            messages.append(Message("tool", results))
        if rest:
            messages.append(Message(role, rest))
    tools: list[Tool] = []
    for raw in body.get("tools") or []:
        if not isinstance(raw, dict):
            continue
        if raw.get("type") not in {None, "custom"}:
            raise ProtocolCompatibilityError(
                f"Anthropic server tool type {raw.get('type')!r} is not translatable"
            )
        tools.append(
            Tool(
                name=str(raw.get("name") or "tool"),
                description=str(raw.get("description") or ""),
                parameters=raw.get("input_schema")
                or {"type": "object", "properties": {}},
            )
        )
    return CanonicalRequest(
        messages=messages,
        tools=tools,
        max_tokens=int(body.get("max_tokens") or 0),
        tool_choice=body.get("tool_choice"),
        temperature=body.get("temperature"),
        stop=body.get("stop_sequences"),
    )


_EPHEMERAL = {"type": "ephemeral"}
# Anthropic accepts at most four. Two go on the static prefix and two roll along
# the tail of the conversation.
_MAX_BREAKPOINTS = 4


def _mark_cache_breakpoints(body: dict[str, Any]) -> None:
    """Opt the request in to prompt caching.

    Anthropic caching is opt-in per request: without an explicit `cache_control`
    marker there is no cache at all, and an agent loop re-sends its entire
    transcript every turn, so the whole prompt is re-billed at the full input
    rate each time. Claude Code marks its own requests, but those markers are
    dropped crossing the canonical IR, so the breakpoints have to be re-applied
    here on the way out.

    Anthropic caches the prefix up to and including each marker, in the wire
    order tools -> system -> messages. One marker on the last tool and one on
    the system block pin the part that never changes; two rolling markers on the
    tail of the message list extend the cached prefix as the transcript grows,
    so each turn re-reads what the previous turn wrote instead of re-sending it.
    A prefix shorter than the model's minimum is ignored rather than rejected,
    so this is safe on the first, tiny request of a run.
    """

    def mark(blocks: Any) -> bool:
        # Replace rather than mutate: reasoning and image blocks are appended to
        # `content` by reference from the canonical Block they were decoded
        # into, and those Blocks outlive the request. Writing through the
        # reference would leave a stale marker on a block that gets re-sent
        # every later turn, and five markers is a 400 from Anthropic.
        if not isinstance(blocks, list) or not blocks or not isinstance(blocks[-1], dict):
            return False
        blocks[-1] = {**blocks[-1], "cache_control": dict(_EPHEMERAL)}
        return True

    budget = _MAX_BREAKPOINTS
    for static in (body.get("tools"), body.get("system")):
        if budget and mark(static):
            budget -= 1
    for message in reversed(body.get("messages") or []):
        if not budget:
            break
        if mark(message.get("content")):
            budget -= 1


def encode_request(
    request: CanonicalRequest,
    *,
    model: str,
    effort: str,
    max_tokens: int,
) -> dict[str, Any]:
    raw_messages: list[dict[str, Any]] = []

    def append_message(role: str, content: list[dict[str, Any]]) -> None:
        if not content:
            return
        if raw_messages and raw_messages[-1]["role"] == role:
            raw_messages[-1]["content"].extend(content)
        else:
            raw_messages.append({"role": role, "content": content})

    system_parts, rest = split_preamble(request.messages)
    for message in rest:
        # Anthropic has no system role inside `messages`, so a positional system
        # message rides along as user text rather than being hoisted.
        role = "user" if message.role in {"user", "tool", "system"} else "assistant"
        content: list[dict[str, Any]] = []
        for block in message.blocks:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "reasoning" and role == "assistant":
                original = block.data.get("anthropic")
                if isinstance(original, dict):
                    content.append(original)
            elif block.type == "tool_call" and role == "assistant":
                content.append(
                    {
                        "type": "tool_use",
                        "id": block.call_id,
                        "name": block.name,
                        "input": block.arguments,
                    }
                )
            elif block.type == "tool_result":
                content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.call_id,
                        "content": block.text,
                        "is_error": bool(block.data.get("is_error")),
                    }
                )
            elif block.type in {"image", "document"}:
                if block.data.get("type") not in {"image", "document"}:
                    raise ProtocolCompatibilityError(
                        f"{block.type} block cannot be represented in Anthropic Messages"
                    )
                content.append(block.data)
        append_message(role, content)
    body: dict[str, Any] = {
        "model": model,
        "messages": raw_messages,
        "max_tokens": min(max(1, request.max_tokens or max_tokens), max_tokens),
        "stream": False,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": effort},
    }
    if system_parts:
        body["system"] = [{"type": "text", "text": "\n".join(system_parts)}]
    if request.tools:
        body["tools"] = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in request.tools
        ]
    _mark_cache_breakpoints(body)
    if request.tools and isinstance(request.tool_choice, dict):
        body["tool_choice"] = request.tool_choice
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.stop is not None:
        body["stop_sequences"] = request.stop
    return body


def decode_response(payload: dict[str, Any]) -> CanonicalResponse:
    blocks: list[Block] = []
    for part in payload.get("content") or []:
        if not isinstance(part, dict):
            continue
        kind = str(part.get("type") or "")
        if kind == "text":
            blocks.append(Block("text", text=str(part.get("text") or "")))
        elif kind in {"thinking", "redacted_thinking"}:
            blocks.append(
                Block(
                    "reasoning",
                    text=str(part.get("thinking") or part.get("data") or ""),
                    data={"anthropic": part},
                )
            )
        elif kind in {"tool_use", "server_tool_use"}:
            blocks.append(
                Block(
                    "tool_call",
                    call_id=str(part.get("id") or uuid.uuid4().hex),
                    name=str(part.get("name") or "tool"),
                    arguments=arguments_dict(part.get("input")),
                    data={"anthropic": part},
                )
            )
    usage = payload.get("usage") or {}
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_write = int(usage.get("cache_creation_input_tokens") or 0)
    return CanonicalResponse(
        blocks=blocks,
        usage=Usage(
            # Anthropic reports `input_tokens` NET of the cached parts. Add them
            # back so the canonical total means the same thing on every provider.
            input_tokens=int(usage.get("input_tokens") or 0) + cache_read + cache_write,
            output_tokens=int(usage.get("output_tokens") or 0),
            reasoning_tokens=int(usage.get("reasoning_tokens") or 0),
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        ),
        stop_reason=str(payload.get("stop_reason") or "end_turn"),
        response_id=str(payload.get("id") or f"msg_{uuid.uuid4().hex}"),
        model=str(payload.get("model") or ""),
        raw=payload,
    )


def response_payload(
    response: CanonicalResponse,
    *,
    model: str,
    state: OpaqueStateStore | None = None,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    has_tools = False
    for block in response.blocks:
        if block.type == "reasoning":
            original = block.data.get("anthropic")
            if isinstance(original, dict):
                content.append(original)
            else:
                token = state.put(block.data) if state else uuid.uuid4().hex
                content.append(
                    {
                        "type": "thinking",
                        "thinking": block.text,
                        "signature": f"cascade:{token}",
                    }
                )
        elif block.type == "text":
            content.append({"type": "text", "text": block.text})
        elif block.type == "tool_call":
            has_tools = True
            content.append(
                {
                    "type": "tool_use",
                    "id": block.call_id,
                    "name": block.name,
                    "input": block.arguments,
                }
            )
    stop_reason = "tool_use" if has_tools else (
        "max_tokens"
        if response.stop_reason in {"length", "max_tokens", "incomplete"}
        else "end_turn"
    )
    return {
        "id": response.response_id or f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }


def _event(name: str, payload: dict[str, Any]) -> bytes:
    return (
        f"event: {name}\ndata: "
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"
    ).encode()


def sse(response: dict[str, Any]) -> Iterable[bytes]:
    shell = dict(response)
    blocks = shell.pop("content")
    shell["content"] = []
    shell["stop_reason"] = None
    shell["usage"] = {"input_tokens": 0, "output_tokens": 0}
    yield _event("message_start", {"type": "message_start", "message": shell})
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
        yield _event(
            "content_block_start",
            {"type": "content_block_start", "index": index, "content_block": start},
        )
        if kind == "thinking":
            deltas = [
                {"type": "thinking_delta", "thinking": block["thinking"]},
                {"type": "signature_delta", "signature": block["signature"]},
            ]
        elif kind == "tool_use":
            deltas = [
                {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(block["input"], ensure_ascii=False),
                }
            ]
        else:
            deltas = [{"type": "text_delta", "text": block["text"]}]
        for delta in deltas:
            yield _event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": delta,
                },
            )
        yield _event(
            "content_block_stop",
            {"type": "content_block_stop", "index": index},
        )
    yield _event(
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
    yield _event("message_stop", {"type": "message_stop"})
