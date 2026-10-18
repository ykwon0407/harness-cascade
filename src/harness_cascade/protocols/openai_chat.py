"""OpenAI-compatible Chat Completions codec."""

from __future__ import annotations

import json
import time
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


def decode_request(body: dict[str, Any]) -> CanonicalRequest:
    messages: list[Message] = []
    for raw in body.get("messages") or []:
        if not isinstance(raw, dict):
            continue
        role = str(raw.get("role") or "user")
        blocks: list[Block] = []
        content = raw.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    blocks.append(Block("text", text=str(part)))
                elif part.get("type") in {"text", "input_text", "output_text"}:
                    blocks.append(Block("text", text=str(part.get("text") or "")))
                elif part.get("type") in {"image_url", "image"}:
                    blocks.append(Block("image", data=dict(part)))
        else:
            text = text_content(content)
            if text:
                blocks.append(Block("text", text=text))
        reasoning = raw.get("reasoning_content") or raw.get("reasoning")
        if reasoning:
            blocks.insert(0, Block("reasoning", text=text_content(reasoning)))
        for call in raw.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            blocks.append(
                Block(
                    "tool_call",
                    call_id=str(call.get("id") or uuid.uuid4().hex),
                    name=str(function.get("name") or "tool"),
                    arguments=arguments_dict(function.get("arguments")),
                    data={"chat": call},
                )
            )
        if role == "tool":
            blocks = [
                Block(
                    "tool_result",
                    call_id=str(raw.get("tool_call_id") or "unknown"),
                    text=text_content(content),
                )
            ]
        messages.append(Message(role, blocks))
    tools: list[Tool] = []
    for raw in body.get("tools") or []:
        if not isinstance(raw, dict):
            continue
        if raw.get("type") not in {None, "function"}:
            raise ProtocolCompatibilityError(
                f"Chat Completions tool type {raw.get('type')!r} is not translatable"
            )
        function = raw.get("function") or raw
        tools.append(
            Tool(
                name=str(function.get("name") or "tool"),
                description=str(function.get("description") or ""),
                parameters=function.get("parameters")
                or {"type": "object", "properties": {}},
            )
        )
    return CanonicalRequest(
        messages=messages,
        tools=tools,
        max_tokens=int(body.get("max_tokens") or 0),
        tool_choice=body.get("tool_choice"),
        temperature=body.get("temperature"),
        stop=body.get("stop"),
    )


def _join_content(left: Any, right: Any) -> Any:
    if left is None or right is None:
        return right if left is None else left
    if isinstance(left, list) or isinstance(right, list):
        def parts(value: Any) -> list[Any]:
            if isinstance(value, list):
                return value
            return [{"type": "text", "text": value}]

        return [*parts(left), *parts(right)]
    return f"{left}\n{right}"


def _merge_assistant_runs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse consecutive assistant turns into one message.

    Chat Completions requires every tool_call on an assistant message to be
    answered before the next assistant message appears. Codex issues a parallel
    group as one `function_call` item per call, and each of those arrives here
    as its own assistant message, so a three-way group becomes three assistant
    messages in a row and the upstream rejects the second one for leaving the
    first unanswered. A parallel group is one turn, so it has to go out as one
    message carrying all of its tool_calls.

    Nothing else is affected: a sequential turn puts its tool result between the
    two assistant messages, which ends the run, and ordinary conversation
    separates assistant turns with a user message.
    """
    merged: list[dict[str, Any]] = []
    for message in messages:
        previous = merged[-1] if merged else None
        if (
            previous is None
            or previous.get("role") != "assistant"
            or message.get("role") != "assistant"
        ):
            merged.append(message)
            continue
        previous["content"] = _join_content(
            previous.get("content"), message.get("content")
        )
        reasoning = _join_content(
            previous.get("reasoning_content"), message.get("reasoning_content")
        )
        if reasoning is not None:
            previous["reasoning_content"] = reasoning
        calls = [*(previous.get("tool_calls") or []), *(message.get("tool_calls") or [])]
        if calls:
            previous["tool_calls"] = calls
    return merged


def _drop_unanswered_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove tool calls the transcript never answers.

    A handoff cuts the first harness off after a fixed number of turns, and the
    cut can land inside a parallel group: two commands go out, one result comes
    back, the turn budget runs out. The summary request then replays that
    history and the upstream rejects it, because a tool call without its result
    is not a conversation it can continue from.

    Dropping the call is truthful here in a way that inventing a result would
    not be. The command was interrupted, so there is nothing to report about
    it, and the summary is being asked what happened -- not to pretend the
    interrupted command produced something.
    """
    answered = {
        message.get("tool_call_id")
        for message in messages
        if message.get("role") == "tool"
    }
    kept: list[dict[str, Any]] = []
    for message in messages:
        calls = message.get("tool_calls")
        if not calls:
            kept.append(message)
            continue
        live = [call for call in calls if call.get("id") in answered]
        if len(live) == len(calls):
            kept.append(message)
            continue
        message = {key: value for key, value in message.items() if key != "tool_calls"}
        if live:
            message["tool_calls"] = live
        elif not message.get("content"):
            # Nothing survives: no text, and every call it made was cut off.
            continue
        kept.append(message)
    return kept


def encode_request(
    request: CanonicalRequest,
    *,
    model: str,
    effort: str,
    max_tokens: int,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    # A mid-conversation system message falls through to the generic branch
    # below, which Chat Completions accepts as a system-role turn in place.
    system_parts, rest = split_preamble(request.messages)
    for message in rest:
        if message.role == "tool":
            for block in message.blocks:
                if block.type == "tool_result":
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.call_id,
                            "content": block.text,
                        }
                    )
            continue
        raw: dict[str, Any] = {"role": message.role}
        texts = [block.text for block in message.blocks if block.type == "text"]
        images = [block.data for block in message.blocks if block.type == "image"]
        if images:
            if any(image.get("type") != "image_url" for image in images):
                raise ProtocolCompatibilityError(
                    "image block cannot be represented as Chat Completions image_url"
                )
            raw["content"] = [
                *({"type": "text", "text": text} for text in texts),
                *images,
            ]
        else:
            raw["content"] = "\n".join(texts) or None
        reasoning = [
            block.text for block in message.blocks if block.type == "reasoning"
        ]
        if reasoning and message.role == "assistant":
            raw["reasoning_content"] = "\n".join(reasoning)
        calls = [block for block in message.blocks if block.type == "tool_call"]
        if calls:
            raw["tool_calls"] = [
                {
                    "id": block.call_id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.arguments, ensure_ascii=False),
                    },
                }
                for block in calls
            ]
        messages.append(raw)
    messages = _drop_unanswered_calls(_merge_assistant_runs(messages))
    if system_parts:
        messages.insert(
            0,
            {"role": "system", "content": "\n\n".join(system_parts)},
        )
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": min(max(1, request.max_tokens or max_tokens), max_tokens),
        "reasoning_effort": effort,
    }
    if request.tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in request.tools
        ]
    if request.tools and request.tool_choice is not None:
        body["tool_choice"] = request.tool_choice
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.stop is not None:
        body["stop"] = request.stop
    return body


def decode_response(payload: dict[str, Any]) -> CanonicalResponse:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("Chat Completions response contains no choices")
    choice = choices[0]
    message = choice.get("message") or {}
    blocks: list[Block] = []
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        blocks.append(Block("reasoning", text=text_content(reasoning)))
    content = text_content(message.get("content"))
    if content:
        blocks.append(Block("text", text=content))
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        blocks.append(
            Block(
                "tool_call",
                call_id=str(call.get("id") or uuid.uuid4().hex),
                name=str(function.get("name") or "tool"),
                arguments=arguments_dict(function.get("arguments")),
                data={"chat": call},
            )
        )
    usage = payload.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    return CanonicalResponse(
        blocks=blocks,
        usage=Usage(
            # `prompt_tokens` is gross of cache here too. Together caches
            # automatically and reports the hit in `prompt_tokens_details`, so
            # recording it costs nothing and is the only way a finished run can
            # be priced against a cached-input rate.
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            reasoning_tokens=int(
                details.get("reasoning_tokens") or usage.get("reasoning_tokens") or 0
            ),
            cache_read_tokens=int(prompt_details.get("cached_tokens") or 0),
        ),
        stop_reason=str(choice.get("finish_reason") or "stop"),
        response_id=str(payload.get("id") or f"chatcmpl_{uuid.uuid4().hex}"),
        model=str(payload.get("model") or ""),
        raw=payload,
    )


def response_payload(response: CanonicalResponse, *, model: str) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": None}
    texts = [block.text for block in response.blocks if block.type == "text"]
    if texts:
        message["content"] = "\n".join(texts)
    reasoning = [
        block.text for block in response.blocks if block.type == "reasoning"
    ]
    if reasoning:
        message["reasoning_content"] = "\n".join(reasoning)
    calls = [block for block in response.blocks if block.type == "tool_call"]
    if calls:
        message["tool_calls"] = [
            {
                "id": block.call_id,
                "type": "function",
                "function": {
                    "name": block.name,
                    "arguments": json.dumps(block.arguments, ensure_ascii=False),
                },
            }
            for block in calls
        ]
    finish = "tool_calls" if calls else (
        "length" if response.stop_reason in {"length", "max_tokens"} else "stop"
    )
    return {
        "id": response.response_id or f"chatcmpl_{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
            "completion_tokens_details": {
                "reasoning_tokens": response.usage.reasoning_tokens
            },
        },
    }


def sse(payload: dict[str, Any]) -> Iterable[bytes]:
    choice = payload["choices"][0]
    message = choice["message"]
    delta = {key: value for key, value in message.items() if value is not None}
    delta["role"] = "assistant"

    def chunk(value: dict[str, Any], finish: str | None) -> bytes:
        body = {
            "id": payload["id"],
            "object": "chat.completion.chunk",
            "created": payload["created"],
            "model": payload["model"],
            "choices": [{"index": 0, "delta": value, "finish_reason": finish}],
        }
        if finish is not None:
            body["usage"] = payload["usage"]
        return f"data: {json.dumps(body, separators=(',', ':'))}\n\n".encode()

    yield chunk(delta, None)
    yield chunk({}, choice["finish_reason"])
    yield b"data: [DONE]\n\n"


def sse_error(status: int, code: str, message: str) -> Iterable[bytes]:
    # An error that arrives after the 200 has gone out. Clients raise it with no
    # status of their own, and Pi decides whether to retry by matching the text,
    # so the status that would have been on the response goes into the message.
    body = {"error": {"type": code, "code": code, "message": f"{status} {message}"}}
    yield f"data: {json.dumps(body, ensure_ascii=False, separators=(',', ':'))}\n\n".encode()
    yield b"data: [DONE]\n\n"
