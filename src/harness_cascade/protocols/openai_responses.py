"""OpenAI Responses request, response, and SSE codec."""

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
from .state import OpaqueStateStore


def _restore_item(
    item: dict[str, Any], state: OpaqueStateStore | None
) -> dict[str, Any]:
    encoded = str(item.get("encrypted_content") or "")
    if encoded.startswith("cascade:") and state:
        stored = state.get(encoded.removeprefix("cascade:"))
        if stored:
            return {**item, "_cascade_data": stored}
    return item


def _namespaced(namespace: str, name: str) -> str:
    """The flat name a namespace member is offered under upstream.

    Responses scopes a tool inside a namespace and names the call by both
    parts; a function-calling upstream has one flat namespace, so the two
    are joined here and split again in `response_payload`.
    """
    return f"{namespace}__{name}"


def _custom_description(raw: dict[str, Any]) -> str:
    """A freeform tool's description, carrying its input grammar.

    OpenAI constrains a custom tool's input with the declared grammar, so the
    model never has to be told the syntax. A function-calling upstream cannot
    constrain anything, and the grammar is the only statement of the format
    the harness will accept, so it has to travel in the description.
    """
    description = str(raw.get("description") or "")
    fmt = raw.get("format")
    if not isinstance(fmt, dict) or fmt.get("type") != "grammar":
        return description
    definition = str(fmt.get("definition") or "").strip()
    if not definition:
        return description
    syntax = str(fmt.get("syntax") or "grammar")
    return (
        f"{description}\n\nThe `input` argument must match this {syntax} "
        f"grammar:\n```{syntax}\n{definition}\n```"
    ).lstrip()


def _custom_input(arguments: dict[str, Any]) -> str:
    """The freeform text a model produced for a custom tool.

    The upstream was offered a single `input` string. A model that names the
    field differently still meant its one string as the input; anything else
    is passed through as JSON so the harness can report the malformed input
    back to the model instead of rejecting the call outright.
    """
    value = arguments.get("input")
    if isinstance(value, str):
        return value
    strings = [v for v in arguments.values() if isinstance(v, str)]
    if len(arguments) == 1 and len(strings) == 1:
        return strings[0]
    return json.dumps(arguments, ensure_ascii=False)


def decode_request(
    body: dict[str, Any],
    state: OpaqueStateStore | None = None,
) -> CanonicalRequest:
    messages: list[Message] = []
    instructions = str(body.get("instructions") or "")
    if instructions:
        messages.append(Message("system", [Block("text", text=instructions)]))
    value = body.get("input")
    if isinstance(value, str):
        messages.append(Message("user", [Block("text", text=value)]))
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        item = _restore_item(item, state)
        kind = str(item.get("type") or "")
        role = str(item.get("role") or "user")
        if role == "developer":
            # OpenAI's name for system-level instructions. Chat templates are
            # free to drop a role they do not know, and DeepSeek's does,
            # silently: Codex's permissions and skills never reached it.
            role = "system"
        if kind in {"message", "input_text", "output_text"} or "role" in item:
            blocks: list[Block] = []
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        blocks.append(Block("text", text=str(part)))
                    elif part.get("type") in {"input_text", "output_text", "text"}:
                        blocks.append(Block("text", text=str(part.get("text") or "")))
                    elif part.get("type") in {"input_image", "image_url"}:
                        blocks.append(Block("image", data=dict(part)))
            else:
                text = text_content(content) or str(item.get("text") or "")
                if text:
                    blocks.append(Block("text", text=text))
            if blocks:
                messages.append(Message(role, blocks))
        elif kind in {"function_call", "custom_tool_call", "local_shell_call"}:
            name = str(item.get("name") or kind)
            if item.get("namespace"):
                name = _namespaced(str(item["namespace"]), name)
            messages.append(
                Message(
                    "assistant",
                    [
                        Block(
                            "tool_call",
                            call_id=str(
                                item.get("call_id") or item.get("id") or uuid.uuid4().hex
                            ),
                            name=name,
                            arguments=arguments_dict(
                                item.get("arguments")
                                or json.dumps({"input": item.get("input", "")})
                            ),
                            data={"openai": item},
                        )
                    ],
                )
            )
        elif kind in {
            "function_call_output",
            "custom_tool_call_output",
            "local_shell_call_output",
        }:
            messages.append(
                Message(
                    "tool",
                    [
                        Block(
                            "tool_result",
                            call_id=str(item.get("call_id") or "unknown"),
                            text=text_content(item.get("output")),
                            data={"openai": item},
                        )
                    ],
                )
            )
        elif kind == "reasoning":
            cascade_data = item.get("_cascade_data")
            messages.append(
                Message(
                    "assistant",
                    [
                        Block(
                            "reasoning",
                            text=text_content(item.get("summary"))
                            or text_content(item.get("content")),
                            data=(
                                cascade_data
                                if isinstance(cascade_data, dict)
                                else {"openai": item}
                            ),
                        )
                    ],
                )
            )
        elif kind == "compaction":
            encoded = str(item.get("encrypted_content") or "")
            if encoded.startswith("cascade1:"):
                messages.append(
                    Message(
                        "user",
                        [
                            Block(
                                "text",
                                text=encoded.removeprefix("cascade1:"),
                            )
                        ],
                    )
                )
    tools: list[Tool] = []
    for raw in body.get("tools") or []:
        if not isinstance(raw, dict) or raw.get("defer_loading"):
            continue
        kind = raw.get("type")
        if kind == "namespace":
            namespace = str(raw.get("name") or "namespace")
            for child in raw.get("tools") or []:
                if isinstance(child, dict):
                    child_name = str(child.get("name") or "tool")
                    tools.append(
                        Tool(
                            name=_namespaced(namespace, child_name),
                            description=str(child.get("description") or ""),
                            parameters=child.get("parameters")
                            or {"type": "object", "properties": {}},
                            data={
                                "openai": {
                                    "type": "function",
                                    "namespace": namespace,
                                    "name": child_name,
                                }
                            },
                        )
                    )
        elif kind == "custom":
            tools.append(
                Tool(
                    name=str(raw.get("name") or "custom"),
                    description=_custom_description(raw),
                    parameters={
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": "The raw tool input, verbatim.",
                            }
                        },
                        "required": ["input"],
                    },
                    data={"openai": {"type": "custom"}},
                )
            )
        elif kind in {"function", None}:
            tools.append(
                Tool(
                    name=str(raw.get("name") or "tool"),
                    description=str(raw.get("description") or ""),
                    parameters=raw.get("parameters")
                    or {"type": "object", "properties": {}},
                )
            )
        elif kind in {"web_search", "web_search_preview", "tool_search"}:
            # Codex advertises OpenAI-hosted search by default, and the GPT
            # entries it ships also offer hosted tool search (deferred tool
            # loading). No upstream behind this gateway runs hosted tools, but
            # local function tools remain usable, so omit them in translation.
            continue
        else:
            raise ProtocolCompatibilityError(
                f"OpenAI hosted tool type {kind!r} is not translatable"
            )
    return CanonicalRequest(
        messages=messages,
        tools=tools,
        max_tokens=int(body.get("max_output_tokens") or 0),
        tool_choice=body.get("tool_choice"),
    )


def encode_request(
    request: CanonicalRequest,
    *,
    model: str,
    effort: str,
    max_tokens: int,
) -> dict[str, Any]:
    instructions, rest = split_preamble(request.messages)
    items: list[dict[str, Any]] = []
    for message in rest:
        if message.role == "system":
            # Responses accepts a system-role input item, so a mid-conversation
            # system message keeps its place instead of being hoisted.
            text = "\n".join(
                block.text
                for block in message.blocks
                if block.type == "text" and block.text
            )
            if text:
                items.append(
                    {
                        "type": "message",
                        "role": "system",
                        "content": [{"type": "input_text", "text": text}],
                    }
                )
            continue
        for block in message.blocks:
            if block.type == "text":
                role = "assistant" if message.role == "assistant" else "user"
                content_type = "output_text" if role == "assistant" else "input_text"
                items.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": [{"type": content_type, "text": block.text}],
                    }
                )
            elif block.type == "reasoning":
                original = block.data.get("openai")
                if isinstance(original, dict):
                    items.append(original)
            elif block.type == "tool_call":
                original = block.data.get("openai")
                if isinstance(original, dict):
                    items.append(original)
                else:
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": block.call_id,
                            "name": block.name,
                            "arguments": json.dumps(
                                block.arguments, ensure_ascii=False
                            ),
                        }
                    )
            elif block.type == "tool_result":
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": block.call_id,
                        "output": block.text,
                    }
                )
            elif block.type == "image":
                if block.data.get("type") not in {"input_image", "image_url"}:
                    raise ProtocolCompatibilityError(
                        "image block cannot be represented as a Responses input image"
                    )
                items.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": [block.data],
                    }
                )
    body: dict[str, Any] = {
        "model": model,
        "input": items,
        "stream": False,
        "max_output_tokens": min(max(1, request.max_tokens or max_tokens), max_tokens),
        "reasoning": {"effort": effort},
    }
    if instructions:
        body["instructions"] = "\n".join(instructions)
    if request.tools:
        body["tools"] = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in request.tools
        ]
    if request.tools and request.tool_choice is not None:
        body["tool_choice"] = request.tool_choice
    return body


def decode_response(payload: dict[str, Any]) -> CanonicalResponse:
    blocks: list[Block] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or "")
        if kind == "message":
            text = text_content(item.get("content"))
            if text:
                blocks.append(Block("text", text=text, data={"openai": item}))
        elif kind == "reasoning":
            blocks.append(
                Block(
                    "reasoning",
                    text=text_content(item.get("summary"))
                    or text_content(item.get("content")),
                    data={"openai": item},
                )
            )
        elif kind in {"function_call", "custom_tool_call", "local_shell_call"}:
            blocks.append(
                Block(
                    "tool_call",
                    call_id=str(
                        item.get("call_id") or item.get("id") or uuid.uuid4().hex
                    ),
                    name=str(item.get("name") or kind),
                    arguments=arguments_dict(
                        item.get("arguments")
                        or json.dumps({"input": item.get("input", "")})
                    ),
                    data={"openai": item},
                )
            )
    usage = payload.get("usage") or {}
    details = usage.get("output_tokens_details") or {}
    input_details = usage.get("input_tokens_details") or {}
    return CanonicalResponse(
        blocks=blocks,
        usage=Usage(
            # Already gross of cache on this API, so it needs no adjustment.
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            reasoning_tokens=int(details.get("reasoning_tokens") or 0),
            cache_read_tokens=int(input_details.get("cached_tokens") or 0),
        ),
        stop_reason=(
            "incomplete"
            if payload.get("status") == "incomplete"
            else "completed"
        ),
        response_id=str(payload.get("id") or f"resp_{uuid.uuid4().hex}"),
        model=str(payload.get("model") or ""),
        raw=payload,
    )


def response_payload(
    response: CanonicalResponse,
    *,
    model: str,
    state: OpaqueStateStore | None = None,
    tools: Iterable[Tool] = (),
) -> dict[str, Any]:
    """A Responses payload; `tools` are the request's, decoded by this module.

    Codex dispatches on the item shape: a freeform tool answered with a
    `function_call` is a fatal "incompatible payload" that the model only
    ever sees as "aborted", and a namespace member answered under its flat
    upstream name is an "unsupported call".
    """
    declared = {
        tool.name: tool.data["openai"]
        for tool in tools
        if isinstance(tool.data.get("openai"), dict)
    }
    output: list[dict[str, Any]] = []
    for block in response.blocks:
        if block.type == "reasoning":
            original = block.data.get("openai")
            if isinstance(original, dict):
                output.append(original)
            else:
                token = state.put(block.data) if state else uuid.uuid4().hex
                output.append(
                    {
                        "id": f"rs_{uuid.uuid4().hex}",
                        "type": "reasoning",
                        "status": "completed",
                        "summary": (
                            [{"type": "summary_text", "text": block.text}]
                            if block.text
                            else []
                        ),
                        "content": [],
                        "encrypted_content": f"cascade:{token}",
                    }
                )
        elif block.type == "text":
            output.append(
                {
                    "id": f"msg_{uuid.uuid4().hex}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": block.text,
                            "annotations": [],
                        }
                    ],
                }
            )
        elif block.type == "tool_call":
            origin = declared.get(block.name) or {}
            if origin.get("type") == "custom":
                output.append(
                    {
                        "id": f"ctc_{uuid.uuid4().hex}",
                        "type": "custom_tool_call",
                        "status": "completed",
                        "call_id": block.call_id,
                        "name": block.name,
                        "input": _custom_input(block.arguments),
                    }
                )
                continue
            item = {
                "id": f"fc_{uuid.uuid4().hex}",
                "type": "function_call",
                "status": "completed",
                "call_id": block.call_id,
                "name": block.name,
                "arguments": json.dumps(block.arguments, ensure_ascii=False),
            }
            if origin.get("namespace"):
                item["name"] = str(origin["name"])
                item["namespace"] = str(origin["namespace"])
            output.append(item)
    status = (
        "incomplete"
        if response.stop_reason in {"length", "max_tokens", "incomplete"}
        else "completed"
    )
    return {
        "id": response.response_id or f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "incomplete_details": (
            {"reason": "max_output_tokens"} if status == "incomplete" else None
        ),
        "model": model,
        "output": output,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
            "output_tokens_details": {
                "reasoning_tokens": response.usage.reasoning_tokens
            },
        },
    }


def _event(name: str, payload: dict[str, Any], sequence: int) -> bytes:
    data = {"type": name, **payload, "sequence_number": sequence}
    return (
        f"event: {name}\ndata: "
        f"{json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"
    ).encode()


def sse(response: dict[str, Any]) -> Iterable[bytes]:
    sequence = 0

    def emit(name: str, payload: dict[str, Any]) -> bytes:
        nonlocal sequence
        value = _event(name, payload, sequence)
        sequence += 1
        return value

    shell = {
        "id": response["id"],
        "object": "response",
        "created_at": response["created_at"],
        "status": "in_progress",
        "model": response["model"],
        "output": [],
    }
    yield emit("response.created", {"response": shell})
    yield emit("response.in_progress", {"response": shell})
    for output_index, item in enumerate(response["output"]):
        pending = dict(item)
        pending["status"] = "in_progress"
        if item["type"] == "reasoning":
            pending["summary"] = []
        elif item["type"] == "message":
            pending["content"] = []
        elif item["type"] == "custom_tool_call":
            pending["input"] = ""
        yield emit(
            "response.output_item.added",
            {"output_index": output_index, "item": pending},
        )
        if item["type"] == "reasoning":
            text = text_content(item.get("summary"))
            if text:
                yield emit(
                    "response.reasoning_summary_text.delta",
                    {
                        "item_id": item["id"],
                        "output_index": output_index,
                        "summary_index": 0,
                        "delta": text,
                    },
                )
        elif item["type"] == "message":
            text = text_content(item.get("content"))
            yield emit(
                "response.output_text.delta",
                {
                    "item_id": item["id"],
                    "output_index": output_index,
                    "content_index": 0,
                    "delta": text,
                },
            )
            yield emit(
                "response.output_text.done",
                {
                    "item_id": item["id"],
                    "output_index": output_index,
                    "content_index": 0,
                    "text": text,
                },
            )
        elif item["type"] == "function_call":
            yield emit(
                "response.function_call_arguments.delta",
                {
                    "item_id": item["id"],
                    "output_index": output_index,
                    "delta": item["arguments"],
                },
            )
            yield emit(
                "response.function_call_arguments.done",
                {
                    "item_id": item["id"],
                    "output_index": output_index,
                    "arguments": item["arguments"],
                },
            )
        elif item["type"] == "custom_tool_call":
            yield emit(
                "response.custom_tool_call_input.delta",
                {
                    "item_id": item["id"],
                    "output_index": output_index,
                    "delta": item["input"],
                },
            )
            yield emit(
                "response.custom_tool_call_input.done",
                {
                    "item_id": item["id"],
                    "output_index": output_index,
                    "input": item["input"],
                },
            )
        yield emit(
            "response.output_item.done",
            {"output_index": output_index, "item": item},
        )
    terminal = (
        "response.incomplete"
        if response["status"] == "incomplete"
        else "response.completed"
    )
    yield emit(terminal, {"response": response})
