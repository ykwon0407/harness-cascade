"""Small loss-aware intermediate representation for model API messages."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


class ProtocolCompatibilityError(ValueError):
    pass


@dataclass(slots=True)
class Block:
    type: str
    text: str = ""
    call_id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Message:
    role: str
    blocks: list[Block] = field(default_factory=list)


@dataclass(slots=True)
class Tool:
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    # How the tool was declared on its source wire, keyed by protocol, for
    # when a call must be answered in that shape rather than as a function.
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Usage:
    """Token counts for one model call.

    `input_tokens` is the TOTAL prompt, inclusive of whatever part of it the
    provider served from cache — the LiteLLM and harbor convention, and the one
    every run already on disk was recorded under. The cache fields are a
    breakdown OF that total, not additions to it, so a cost calculation must
    subtract them before applying the uncached input rate. Providers disagree
    about this on the wire (Anthropic reports `input_tokens` net of cache,
    OpenAI reports it gross), which is why each decoder normalises rather than
    passing the number through.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(slots=True)
class CanonicalRequest:
    messages: list[Message] = field(default_factory=list)
    tools: list[Tool] = field(default_factory=list)
    # 0 when the harness set no limit, which leaves the model's own.
    max_tokens: int = 0
    tool_choice: Any = None
    temperature: float | None = None
    stop: list[str] | str | None = None


@dataclass(slots=True)
class CanonicalResponse:
    blocks: list[Block] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = "end"
    response_id: str = ""
    model: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return str(value or "")
    parts: list[str] = []
    for item in value:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            text = (
                item.get("text")
                or item.get("input_text")
                or item.get("output_text")
                or item.get("content")
            )
            if text is not None:
                parts.append(str(text))
    return "\n".join(parts)


def split_preamble(messages: list[Message]) -> tuple[list[str], list[Message]]:
    """Separate the opening system messages from the rest of the transcript.

    Only a system message at the very front is a preamble. One that arrives
    partway through is positional: Claude Code emits a fresh system message
    before each of its turns carrying a live `<total_tokens> N left` budget.
    Folding those into the preamble moves them out of order and, because the
    preamble sits at offset zero of the prompt, rewrites the prefix on every
    single turn -- which is precisely what OpenAI's automatic prefix caching
    keys on, so the whole prompt is re-billed each turn instead of being read
    from cache. Leave them where the harness put them.
    """
    index = 0
    while index < len(messages) and messages[index].role == "system":
        index += 1
    preamble = [
        block.text
        for message in messages[:index]
        for block in message.blocks
        if block.type == "text" and block.text
    ]
    return preamble, messages[index:]


def arguments_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
