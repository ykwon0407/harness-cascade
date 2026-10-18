from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from harness_cascade.events import EventKind
from harness_cascade.gateway import EmbeddedGateway, GatewayConfig
from harness_cascade.models import (
    ModelProfileError,
    load_model_registry,
    resolve_model_profile,
)
from harness_cascade.protocols import (
    anthropic_messages,
    openai_chat,
    openai_responses,
)
from harness_cascade.protocols.canonical import (
    Block,
    CanonicalRequest,
    CanonicalResponse,
    Message,
    Tool,
    Usage,
)
from harness_cascade.protocols.state import OpaqueStateStore
from harness_cascade.providers import (
    ProviderClient,
    TransportConfig,
    retry_delay,
    undelivered_tokens,
)


def upstream_payload(protocol: str) -> dict[str, Any]:
    if protocol == "openai-responses":
        return {
            "id": "resp_upstream",
            "object": "response",
            "status": "completed",
            "model": "provider/model",
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "matrix ok"}],
                }
            ],
            "usage": {
                "input_tokens": 3,
                "output_tokens": 2,
                "output_tokens_details": {"reasoning_tokens": 1},
            },
        }
    if protocol == "anthropic-messages":
        return {
            "id": "msg_upstream",
            "type": "message",
            "role": "assistant",
            "model": "provider/model",
            "content": [{"type": "text", "text": "matrix ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }
    return {
        "id": "chat_upstream",
        "model": "provider/model",
        "choices": [
            {
                "message": {"role": "assistant", "content": "matrix ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }


def local_request(protocol: str) -> tuple[str, dict[str, Any]]:
    if protocol == "openai-responses":
        return "/v1/responses", {
            "model": "local",
            "input": "test",
            "stream": False,
        }
    if protocol == "anthropic-messages":
        return "/v1/messages", {
            "model": "local",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 100,
            "stream": False,
        }
    return "/v1/chat/completions", {
        "model": "local",
        "messages": [{"role": "user", "content": "test"}],
        "stream": False,
    }


def local_text(protocol: str, payload: dict[str, Any]) -> str:
    if protocol == "openai-responses":
        return payload["output"][0]["content"][0]["text"]
    if protocol == "anthropic-messages":
        return payload["content"][0]["text"]
    return payload["choices"][0]["message"]["content"]


@pytest.mark.parametrize(
    "inbound_protocol",
    ["openai-responses", "anthropic-messages", "openai-chat"],
)
@pytest.mark.parametrize(
    "upstream_protocol",
    ["openai-responses", "anthropic-messages", "openai-chat"],
)
def test_all_nine_protocol_provider_combinations(
    monkeypatch: pytest.MonkeyPatch,
    inbound_protocol: str,
    upstream_protocol: str,
) -> None:
    seen: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        seen["url"] = url
        seen["body"] = kwargs["json"]
        seen["headers"] = kwargs["headers"]
        return httpx.Response(
            200,
            json=upstream_payload(upstream_protocol),
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    config = GatewayConfig(
        model="provider/model",
        effort="high",
        api_key="upstream-secret",
        local_token="local-secret",
        base_url="https://provider.test",
        provider=(
            "openai"
            if upstream_protocol == "openai-responses"
            else "anthropic"
            if upstream_protocol == "anthropic-messages"
            else "together"
        ),
        protocol=upstream_protocol,
        max_retries=0,
    )
    path, body = local_request(inbound_protocol)
    with EmbeddedGateway(config) as gateway:
        with httpx.Client() as client:
            response = client.post(
                gateway.base_url + path,
                headers={"Authorization": "Bearer local-secret"},
                json=body,
            )
    assert response.status_code == 200
    assert local_text(inbound_protocol, response.json()) == "matrix ok"
    expected_suffix = {
        "openai-responses": "/v1/responses",
        "anthropic-messages": "/v1/messages",
        "openai-chat": "/v1/chat/completions",
    }[upstream_protocol]
    assert seen["url"].endswith(expected_suffix)
    assert seen["body"]["model"] == "provider/model"
    if upstream_protocol == "anthropic-messages":
        assert seen["headers"]["x-api-key"] == "upstream-secret"
        assert "Authorization" not in seen["headers"]
    else:
        assert seen["headers"]["Authorization"] == "Bearer upstream-secret"


def test_cross_protocol_tool_and_opaque_reasoning_round_trip(tmp_path) -> None:
    state = OpaqueStateStore(tmp_path / "opaque.json")
    upstream = CanonicalResponse(
        blocks=[
            Block(
                "reasoning",
                text="summary",
                data={
                    "openai": {
                        "id": "rs_1",
                        "type": "reasoning",
                        "encrypted_content": "opaque-provider-value",
                        "summary": [{"type": "summary_text", "text": "summary"}],
                    }
                },
            ),
            Block(
                "tool_call",
                call_id="call_1",
                name="read_file",
                arguments={"path": "auth.py"},
            ),
        ],
        usage=Usage(10, 4, 3),
    )
    anthropic_payload = anthropic_messages.response_payload(
        upstream,
        model="model",
        state=state,
    )
    history = {
        "messages": [
            {"role": "assistant", "content": anthropic_payload["content"]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": "contents",
                    }
                ],
            },
        ]
    }
    canonical = anthropic_messages.decode_request(history, state)
    encoded = openai_responses.encode_request(
        canonical,
        model="model",
        effort="high",
        max_tokens=1000,
    )
    assert any(
        item.get("encrypted_content") == "opaque-provider-value"
        for item in encoded["input"]
    )
    assert encoded["input"][-1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "contents",
    }


def test_packaged_registry_and_exact_effort_validation() -> None:
    registry = load_model_registry()
    profile = registry.profile("openai", "gpt-5.6")
    assert profile.protocol == "openai-responses"
    assert "max" in profile.efforts
    with pytest.raises(ModelProfileError, match="available: low, medium, high"):
        resolve_model_profile(
            "together",
            "openai/gpt-oss-120b",
            "xhigh",
        )


def sent_body(model: str, effort: str, supported: tuple[str, ...]) -> dict[str, Any]:
    client = ProviderClient(
        TransportConfig(
            provider="together",
            protocol="openai-chat",
            model=model,
            effort=effort,
            api_key="k",
            base_url="http://endpoint/v1",
            supported_efforts=supported,
        )
    )
    captured: dict[str, Any] = {}

    def fake_request(body: dict[str, Any]) -> dict[str, Any]:
        captured.update(body)
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    client._request = fake_request  # type: ignore[method-assign]
    client.complete(
        CanonicalRequest(messages=[Message("user", [Block("text", text="hi")])])
    )
    return captured


def test_failed_attempts_are_visible_in_the_event_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A re-sent call must not read as one call that never finished."""
    from harness_cascade.events import EventBus, EventKind

    replies: list[Any] = [
        httpx.ReadTimeout("read timed out"),
        httpx.Response(
            503,
            json={"error": {"message": "no capacity"}},
            request=httpx.Request("POST", "http://endpoint/v1/chat/completions"),
        ),
        httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
            request=httpx.Request("POST", "http://endpoint/v1/chat/completions"),
        ),
    ]

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("harness_cascade.providers.time.sleep", lambda _: None)
    bus = EventBus("run")
    seen: list[Any] = []
    bus.subscribe(seen.append)
    client = ProviderClient(
        TransportConfig(
            provider="together",
            protocol="openai-chat",
            model="m",
            effort="low",
            api_key="k",
            base_url="http://endpoint/v1",
            max_retries=2,
            harness="codex",
            stage="first",
        ),
        bus,
    )
    client.complete(CanonicalRequest(messages=[Message("user", [Block("text", text="hi")])]))
    warnings = [e for e in seen if e.kind == EventKind.WARNING]
    assert [(w.data["attempt"], w.data["of"]) for w in warnings] == [(1, 3), (2, 3)]
    assert warnings[0].data["reason"].startswith("ReadTimeout")
    assert warnings[1].data["reason"].startswith("HTTP 503: no capacity")
    assert all(w.data["retrying"] for w in warnings)
    assert all(w.harness == "codex" and w.stage == "first" for w in warnings)
    assert [e.kind for e in seen].count(EventKind.MODEL_CALL_FINISHED) == 1


@pytest.mark.parametrize(
    ("budget", "env", "expected"),
    [(3600, None, 900.0), (300, None, 300.0), (3600, "120", 120.0), (3600, "junk", 900.0)],
)
def test_one_attempt_never_waits_out_the_whole_trial(
    monkeypatch: pytest.MonkeyPatch, budget: float, env: str | None, expected: float
) -> None:
    """A request the endpoint silently dropped used to hold the trial to its end."""
    seen: list[httpx.Timeout] = []

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        seen.append(kwargs["timeout"])
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    if env is None:
        monkeypatch.delenv("CASCADE_ATTEMPT_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("CASCADE_ATTEMPT_TIMEOUT", env)
    ProviderClient(
        TransportConfig(
            provider="together",
            protocol="openai-chat",
            model="m",
            effort="low",
            api_key="k",
            base_url="http://endpoint/v1",
            timeout_seconds=budget,
        )
    ).complete(CanonicalRequest(messages=[Message("user", [Block("text", text="hi")])]))
    assert seen[0].read == expected
    assert seen[0].connect == 30.0


def sse(*events: dict[str, Any], done: bool = True) -> list[bytes]:
    frames = [f"data: {json.dumps(event)}\n\n".encode() for event in events]
    return frames + ([b"data: [DONE]\n\n"] if done else [])


def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


STREAMED_TOOL_CALL = sse(
    chunk({"role": "assistant", "content": ""}),
    chunk({"reasoning": "Look at "}),
    chunk({"reasoning": "the files."}),
    chunk({"content": "Listing."}),
    chunk({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                           "function": {"name": "exec_command"}}]}),
    chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"cmd": "'}}]}),
    chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'ls"}'}}]}),
    chunk({}, finish="tool_calls"),
    {"id": "chatcmpl-1", "choices": [], "usage": {
        "prompt_tokens": 10, "completion_tokens": 7,
        "completion_tokens_details": {"reasoning_tokens": 3}}},
)


def streaming_client(
    monkeypatch: pytest.MonkeyPatch, replies: list[Any], **overrides: Any
) -> tuple[ProviderClient, list[dict[str, Any]], list[Any]]:
    """A client whose endpoint streams `replies` in turn, one per attempt."""
    import contextlib

    from harness_cascade.events import EventBus

    monkeypatch.setenv("CASCADE_UPSTREAM_STREAM", "1")
    monkeypatch.setattr("harness_cascade.providers.time.sleep", lambda _: None)
    sent: list[dict[str, Any]] = []

    @contextlib.contextmanager
    def fake_stream(method: str, url: str, **kwargs: Any):
        sent.append(kwargs)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, int):
            yield httpx.Response(reply, json={"error": {"message": "no capacity"}},
                                 request=httpx.Request(method, url))
            return
        yield httpx.Response(200, content=iter(reply), request=httpx.Request(method, url))

    monkeypatch.setattr(httpx, "stream", fake_stream)
    bus = EventBus("run")
    events: list[Any] = []
    bus.subscribe(events.append)
    values: dict[str, Any] = {
        "provider": "together", "protocol": "openai-chat", "model": "m", "effort": "low",
        "api_key": "k", "base_url": "http://endpoint/v1", "max_retries": 2,
    }
    values.update(overrides)
    return ProviderClient(TransportConfig(**values), bus), sent, events


def ask(client: ProviderClient) -> CanonicalResponse:
    return client.complete(CanonicalRequest(messages=[Message("user", [Block("text", text="hi")])]))


def test_a_streamed_reply_reads_the_same_as_an_unstreamed_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Everything after the transport sees the reply it saw before streaming."""
    client, sent, events = streaming_client(monkeypatch, [STREAMED_TOOL_CALL])
    streamed = ask(client)
    unstreamed = openai_chat.decode_response({
        "id": "chatcmpl-1",
        "model": "m",
        "choices": [{"message": {
            "role": "assistant",
            "content": "Listing.",
            "reasoning": "Look at the files.",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "exec_command", "arguments": '{"cmd": "ls"}'}}],
        }, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 7,
                  "completion_tokens_details": {"reasoning_tokens": 3}},
    })
    assert streamed.blocks == unstreamed.blocks
    assert streamed.usage == unstreamed.usage
    assert streamed.stop_reason == "tool_calls"
    assert sent[0]["json"]["stream"] is True
    assert sent[0]["json"]["stream_options"] == {"include_usage": True}
    finished = [e for e in events if e.kind.value == "model_call.finished"]
    assert "first_chunk_seconds" in finished[0].data


def stalls_after(frames: list[bytes]):
    yield from frames
    raise httpx.ReadTimeout("timed out")


@pytest.mark.parametrize(
    ("broken", "reason"),
    [
        (stalls_after(STREAMED_TOOL_CALL[:3]), "no data for 120s after 3 chunks"),
        (iter(STREAMED_TOOL_CALL[:3]), "stream closed before the reply was complete"),
        (iter(sse({"error": {"message": "engine died"}})), "engine died"),
        (httpx.ConnectTimeout("connect timed out"), "ConnectTimeout"),
        (503, "HTTP 503: no capacity"),
    ],
)
def test_a_broken_stream_is_sent_again(
    monkeypatch: pytest.MonkeyPatch, broken: Any, reason: str
) -> None:
    """A dropped request now shows as silence and is re-sent, not waited out."""
    monkeypatch.delenv("CASCADE_STREAM_IDLE_TIMEOUT", raising=False)
    client, sent, events = streaming_client(monkeypatch, [broken, STREAMED_TOOL_CALL])
    assert ask(client).stop_reason == "tool_calls"
    assert len(sent) == 2
    assert sent[0]["timeout"].read == 120.0
    warnings = [e for e in events if e.kind.value == "warning"]
    assert len(warnings) == 1 and reason in warnings[0].data["reason"]


# What vLLM's GLM parser returns after dropping a call to a tool it was not
# offered: billed for the call, carrying nothing.
STREAMED_LOST_CALL = sse(
    chunk({"role": "assistant", "content": ""}),
    chunk({}, finish="stop"),
    {"id": "chatcmpl-1", "choices": [], "usage": {
        "prompt_tokens": 10, "completion_tokens": 120,
        "completion_tokens_details": {"reasoning_tokens": 0}}},
)


def ask_with_tools(client: ProviderClient) -> CanonicalResponse:
    return client.complete(CanonicalRequest(
        messages=[Message("user", [Block("text", text="hi")])],
        tools=[Tool("exec_command")],
    ))


def test_a_reply_billed_for_a_call_it_does_not_carry_is_sampled_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passed on, the empty reply reads to the harness as the agent being done."""
    client, sent, events = streaming_client(monkeypatch, [STREAMED_LOST_CALL, STREAMED_TOOL_CALL])
    result = ask_with_tools(client)
    assert [b.name for b in result.blocks if b.type == "tool_call"] == ["exec_command"]
    assert len(sent) == 2
    warnings = [e for e in events if e.kind == EventKind.WARNING]
    assert len(warnings) == 1
    assert warnings[0].data["message"] == "upstream billed a reply it did not deliver"
    assert warnings[0].data["undelivered_tokens"] == 120
    assert warnings[0].data["retrying"] is True
    # Both samples were billed, so both are counted.
    kinds = [e.kind for e in events]
    assert kinds.count(EventKind.MODEL_CALL_STARTED) == 2
    assert kinds.count(EventKind.MODEL_CALL_FINISHED) == 2


def test_a_reply_lost_three_times_is_passed_on(monkeypatch: pytest.MonkeyPatch) -> None:
    client, sent, events = streaming_client(monkeypatch, [STREAMED_LOST_CALL] * 3)
    result = ask_with_tools(client)
    assert not result.blocks
    assert len(sent) == 3
    warnings = [e for e in events if e.kind == EventKind.WARNING]
    assert [(w.data["attempt"], w.data["of"], w.data["retrying"]) for w in warnings] == [
        (1, 3, True), (2, 3, True), (3, 3, False),
    ]


def test_a_reply_without_tools_on_offer_is_not_sampled_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A summary turn offers no tools; nothing it returns can be a lost call."""
    client, sent, events = streaming_client(monkeypatch, [STREAMED_LOST_CALL])
    ask(client)
    assert len(sent) == 1
    assert not [e for e in events if e.kind == EventKind.WARNING]


@pytest.mark.parametrize(
    ("blocks", "output", "reasoning", "stop", "expected"),
    [
        ([], 120, 0, "stop", 120),
        # Reasoning reported separately is accounted for.
        ([Block("reasoning", text="x" * 30)], 280, 260, "stop", 0),
        ([Block("reasoning", text="x" * 30)], 400, 260, "stop", 140),
        # Without a reasoning count, its text stands in for it.
        ([Block("reasoning", text="x" * 900)], 300, 0, "stop", 0),
        # A plain final answer carries what it cost, in English or in CJK.
        ([Block("text", text="Done. " * 80)], 120, 0, "stop", 0),
        ([Block("text", text="完成了。" * 30)], 120, 0, "stop", 0),
        # Too few tokens, or too small a share, to be a call.
        ([], 30, 0, "stop", 0),
        ([Block("text", text="y" * 450)], 250, 0, "stop", 0),
        # A truncated reply is the model's own doing.
        ([], 120, 0, "length", 0),
        ([Block("tool_call", name="exec_command")], 120, 0, "tool_calls", 0),
    ],
)
def test_undelivered_tokens(
    blocks: list[Block], output: int, reasoning: int, stop: str, expected: int
) -> None:
    request = CanonicalRequest(tools=[Tool("exec_command")])
    response = CanonicalResponse(
        blocks=blocks,
        usage=Usage(output_tokens=output, reasoning_tokens=reasoning),
        stop_reason=stop,
    )
    assert undelivered_tokens(request, response) == expected


def test_the_stall_limit_is_campaign_settable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CASCADE_STREAM_IDLE_TIMEOUT", "45")
    client, sent, _ = streaming_client(monkeypatch, [STREAMED_TOOL_CALL])
    ask(client)
    assert sent[0]["timeout"].read == 45.0
    assert sent[0]["timeout"].connect == 30.0


def test_streaming_can_be_turned_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CASCADE_UPSTREAM_STREAM", "0")
    body = sent_body("m", "low", ("low",))
    assert body["stream"] is False
    assert "stream_options" not in body


def test_effort_none_is_sent_explicitly_when_the_model_accepts_it() -> None:
    # An absent reasoning_effort means "endpoint default", which every measured
    # deployment serves at its largest budget -- the opposite of what was asked.
    body = sent_body(
        "Qwen/Qwen3.8-27B",
        "none",
        load_model_registry().profile("together", "Qwen/Qwen3.8-27B").efforts,
    )
    assert body["reasoning_effort"] == "none"
    assert body["reasoning"] == {"enabled": False}


def test_effort_none_is_dropped_when_the_model_does_not_accept_it() -> None:
    body = sent_body("zainhas/dsv41", "none", ("low", "high", "max"))
    assert "reasoning_effort" not in body
    assert body["reasoning"] == {"enabled": False}


def test_named_effort_is_passed_through_untouched() -> None:
    body = sent_body("zai-org/GLM-5.3-Flash", "max", ("none", "low", "high", "max"))
    assert body["reasoning_effort"] == "max"
    assert body["reasoning"] == {"enabled": True}


def test_each_codec_preserves_tool_call() -> None:
    canonical = CanonicalResponse(
        blocks=[Block("tool_call", call_id="call_1", name="run", arguments={"x": 1})]
    )
    chat = openai_chat.response_payload(canonical, model="model")
    anthropic = anthropic_messages.response_payload(canonical, model="model")
    responses = openai_responses.response_payload(canonical, model="model")
    assert chat["choices"][0]["message"]["tool_calls"][0]["id"] == "call_1"
    assert anthropic["content"][0]["id"] == "call_1"
    assert responses["output"][0]["call_id"] == "call_1"


def test_openai_chat_coalesces_system_messages_at_start() -> None:
    request = CanonicalRequest(
        messages=[
            Message("system", [Block("text", text="Harness system.")]),
            Message("user", [Block("text", text="Do the task.")]),
        ]
    )
    body = openai_chat.encode_request(
        request,
        model="Qwen/Qwen3.8-27B",
        effort="low",
        max_tokens=1000,
    )
    assert body["messages"][0] == {
        "role": "system",
        "content": "Harness system.",
    }
    assert sum(1 for msg in body["messages"] if msg.get("role") == "system") == 1


def _parallel_group() -> CanonicalRequest:
    """One Codex turn that issued three tool calls at once.

    Codex emits a parallel group as one `function_call` item per call, and each
    arrives as its own canonical assistant message.
    """
    return CanonicalRequest(
        messages=[
            Message("user", [Block("text", text="Inspect the environment.")]),
            Message("assistant", [Block("reasoning", text="Check three things.")]),
            *(
                Message(
                    "assistant",
                    [Block("tool_call", call_id=f"call_{n}", name="bash", arguments={})],
                )
                for n in "abc"
            ),
            *(
                Message("tool", [Block("tool_result", call_id=f"call_{n}", text="out")])
                for n in "abc"
            ),
        ]
    )


def test_openai_chat_sends_a_parallel_tool_group_as_one_assistant_turn() -> None:
    """Chat Completions rejects an assistant turn that leaves a call unanswered.

    Sent one message per call, the second assistant message arrives before the
    first call has a result and the upstream answers 400. Codex only started
    issuing parallel groups once it was given its real system prompt, so this
    surfaced as every Together-backed Codex trial dying on its second model
    call.
    """
    body = openai_chat.encode_request(
        _parallel_group(), model="deepseek-ai/DeepSeek-V4.1-Flash", effort="low",
        max_tokens=1000,
    )
    assistants = [m for m in body["messages"] if m["role"] == "assistant"]
    assert len(assistants) == 1
    assert [c["id"] for c in assistants[0]["tool_calls"]] == ["call_a", "call_b", "call_c"]
    assert assistants[0]["reasoning_content"] == "Check three things."
    # Every call is answered, and after the assistant turn rather than before.
    roles = [m["role"] for m in body["messages"]]
    assert roles[roles.index("assistant") + 1 :] == ["tool", "tool", "tool"]


def test_openai_chat_keeps_sequential_tool_turns_apart() -> None:
    """A result between two calls means two turns, not one parallel group."""
    request = CanonicalRequest(
        messages=[
            Message("user", [Block("text", text="Go.")]),
            Message("assistant", [Block("tool_call", call_id="call_a", name="bash", arguments={})]),
            Message("tool", [Block("tool_result", call_id="call_a", text="out")]),
            Message("assistant", [Block("tool_call", call_id="call_b", name="bash", arguments={})]),
            Message("tool", [Block("tool_result", call_id="call_b", text="out")]),
        ]
    )
    body = openai_chat.encode_request(
        request, model="model", effort="low", max_tokens=1000
    )
    assert [m["role"] for m in body["messages"]] == [
        "user", "assistant", "tool", "assistant", "tool",
    ]


@pytest.mark.parametrize(
    "codec",
    [openai_chat, openai_responses, anthropic_messages],
)
def test_codec_omits_tool_choice_when_no_tools(codec: Any) -> None:
    request = CanonicalRequest(tool_choice="auto")
    body = codec.encode_request(
        request,
        model="model",
        effort="low",
        max_tokens=1000,
    )
    assert "tools" not in body
    assert "tool_choice" not in body


@pytest.mark.parametrize(
    ("codec", "field", "body"),
    [
        (openai_chat, "max_tokens", {"messages": [{"role": "user", "content": "go"}]}),
        (openai_responses, "max_output_tokens", {"input": "go"}),
        (anthropic_messages, "max_tokens", {"messages": [{"role": "user", "content": "go"}]}),
    ],
)
def test_a_request_without_an_output_limit_gets_the_model_s(
    codec: Any, field: str, body: dict[str, Any]
) -> None:
    """Codex never sends one; it used to be held to 65,536 whatever the model."""
    decoded = codec.decode_request(body)
    sent = codec.encode_request(decoded, model="model", effort="low", max_tokens=262_144)
    assert sent[field] == 262_144
    limited = codec.decode_request({**body, field: 1_000})
    sent = codec.encode_request(limited, model="model", effort="low", max_tokens=262_144)
    assert sent[field] == 1_000


def test_untranslatable_server_tool_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="server tool"):
        anthropic_messages.decode_request(
            {
                "messages": [{"role": "user", "content": "test"}],
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            }
        )


def test_a_tool_result_sharing_a_turn_with_text_still_reaches_a_chat_upstream() -> None:
    """Claude Code's notice after an output-limit cut rides in the result's turn.

    Dropping the result there also drops the call it answers, so the model
    loses the file it just read and reads it again, which is how a stage spent
    its whole budget re-reading one file.
    """
    notice = "Output token limit hit. Resume directly."
    canonical = anthropic_messages.decode_request(
        {
            "messages": [
                {"role": "user", "content": "compress data.txt"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_read",
                            "name": "Read",
                            "input": {"file_path": "/app/decomp.c"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_read",
                            "content": "int radix = 255;",
                        },
                        {"type": "text", "text": notice},
                    ],
                },
            ]
        }
    )
    body = openai_chat.encode_request(
        canonical, model="model", effort="max", max_tokens=1000
    )
    assert body["messages"][-3]["tool_calls"][0]["id"] == "call_read"
    assert body["messages"][-2] == {
        "role": "tool",
        "tool_call_id": "call_read",
        "content": "int radix = 255;",
    }
    assert body["messages"][-1] == {"role": "user", "content": notice}


def test_openai_hosted_search_is_omitted_for_cross_protocol_translation() -> None:
    canonical = openai_responses.decode_request(
        {
            "input": [{"role": "user", "content": "test"}],
            "tools": [
                {"type": "web_search"},
                {
                    "type": "function",
                    "name": "shell",
                    "parameters": {"type": "object", "properties": {}},
                },
            ],
        }
    )
    assert [tool.name for tool in canonical.tools] == ["shell"]


def _agent_turn(history: int) -> CanonicalRequest:
    """A request shaped like turn N of an agent loop: system, tools, transcript."""
    messages = [Message("system", [Block("text", text="you are a terminal agent")])]
    for index in range(history):
        messages.append(
            Message("assistant", [Block("tool_call", call_id=f"c{index}", name="bash")])
        )
        messages.append(
            Message("tool", [Block("tool_result", call_id=f"c{index}", text="ok")])
        )
    return CanonicalRequest(
        messages=messages,
        tools=[Tool(name="bash", description="run a command")],
    )


def test_anthropic_requests_opt_in_to_prompt_caching() -> None:
    """Without a marker there is no cache, and the transcript is re-billed every turn."""
    body = anthropic_messages.encode_request(
        _agent_turn(3), model="claude-opus-5-5", effort="high", max_tokens=4096
    )
    assert body["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert body["system"][-1]["cache_control"] == {"type": "ephemeral"}
    rolling = [
        message
        for message in body["messages"]
        if any("cache_control" in block for block in message["content"])
    ]
    assert rolling == body["messages"][-2:]


def test_cache_breakpoints_stay_within_the_anthropic_limit() -> None:
    """Five markers is a 400, so the budget has to hold as the transcript grows."""
    for history in (0, 1, 8, 40):
        body = anthropic_messages.encode_request(
            _agent_turn(history), model="claude-opus-5-5", effort="high", max_tokens=4096
        )
        markers = sum(
            "cache_control" in block
            for group in (body["tools"], body["system"])
            for block in group
        ) + sum(
            "cache_control" in block
            for message in body["messages"]
            for block in message["content"]
        )
        assert 0 < markers <= 4, f"{markers} markers at history={history}"


def test_marking_does_not_write_through_to_a_reused_block() -> None:
    """Reasoning blocks are re-sent every turn; a stale marker would accumulate."""
    original = {"type": "thinking", "thinking": "...", "signature": "sig"}
    request = CanonicalRequest(
        messages=[
            Message("user", [Block("text", text="go")]),
            Message("assistant", [Block("reasoning", data={"anthropic": original})]),
        ]
    )
    anthropic_messages.encode_request(
        request, model="claude-opus-5-5", effort="high", max_tokens=4096
    )
    assert "cache_control" not in original


def test_cache_counts_are_recorded_per_protocol() -> None:
    anthropic = anthropic_messages.decode_response(
        {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 2,
                "cache_read_input_tokens": 800,
                "cache_creation_input_tokens": 190,
            },
        }
    )
    # Anthropic reports input net of cache; the canonical total is gross.
    assert anthropic.usage.input_tokens == 1000
    assert anthropic.usage.cache_read_tokens == 800
    assert anthropic.usage.cache_write_tokens == 190

    responses = openai_responses.decode_response(
        {
            "output": [],
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 2,
                "input_tokens_details": {"cached_tokens": 800},
            },
        }
    )
    # OpenAI already reports it gross, so it must not be adjusted again.
    assert responses.usage.input_tokens == 1000
    assert responses.usage.cache_read_tokens == 800

    chat = openai_chat.decode_response(
        {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 800},
            },
        }
    )
    assert chat.usage.input_tokens == 1000
    assert chat.usage.cache_read_tokens == 800


def _cc_style_turns(n: int) -> CanonicalRequest:
    """A transcript shaped like Claude Code's: a fresh system message per turn.

    Claude Code sends `<total_tokens> N left</total_tokens>` as a system message
    before each of its turns, so the text differs every time.
    """
    messages = [
        Message("system", [Block("text", text="You are Claude Code.")]),
        Message("user", [Block("text", text="do the task")]),
    ]
    for turn in range(n):
        messages.append(
            Message(
                "system",
                [Block("text", text=f"<total_tokens>{15_000_000 - turn} left</total_tokens>")],
            )
        )
        messages.append(
            Message("assistant", [Block("tool_call", call_id=f"c{turn}", name="Bash")])
        )
        messages.append(
            Message("tool", [Block("tool_result", call_id=f"c{turn}", text="ok")])
        )
    return CanonicalRequest(messages=messages)


def test_a_later_system_message_does_not_reach_the_preamble() -> None:
    """Hoisting it would put per-turn text at offset zero of the prompt."""
    body = openai_responses.encode_request(
        _cc_style_turns(3), model="gpt-6-astra", effort="low", max_tokens=4096
    )
    assert body["instructions"] == "You are Claude Code."
    assert "total_tokens" not in body["instructions"]
    # It is kept, just in its own place in the transcript.
    carried = [item for item in body["input"] if item.get("role") == "system"]
    assert len(carried) == 3


def _without_markers(value: Any) -> Any:
    """Drop cache_control, which is meant to move from one turn to the next."""
    if isinstance(value, dict):
        return {
            key: _without_markers(item)
            for key, item in value.items()
            if key != "cache_control"
        }
    if isinstance(value, list):
        return [_without_markers(item) for item in value]
    return value


def test_the_cached_prefix_survives_another_turn() -> None:
    """The whole point: turn N+1 must extend turn N rather than rewrite it.

    Only the two OpenAI protocols are held to byte extension, because only they
    cache by matching the longest identical prefix. Anthropic caches at explicit
    breakpoints and requires user and assistant turns to alternate, so a system
    message has to be merged into the neighbouring user turn there -- which
    rewrites that turn by design.
    """
    for protocol, prefix_key, body_key in (
        (openai_responses, "instructions", "input"),
        (openai_chat, None, "messages"),
    ):
        first, second = (
            protocol.encode_request(
                _cc_style_turns(n), model="m", effort="low", max_tokens=4096
            )
            for n in (2, 3)
        )
        if prefix_key:
            assert first[prefix_key] == second[prefix_key], protocol.__name__
        head = first[body_key]
        assert second[body_key][: len(head)] == head, protocol.__name__


def test_anthropic_keeps_per_turn_text_out_of_the_system_prompt() -> None:
    """The cached prefix on Anthropic starts at `system`, so it must not move."""
    first, second = (
        anthropic_messages.encode_request(
            _cc_style_turns(n), model="claude-opus-5-5", effort="low", max_tokens=4096
        )
        for n in (2, 3)
    )
    assert _without_markers(first["system"]) == _without_markers(second["system"])
    assert "total_tokens" not in json.dumps(first["system"])


def test_an_opening_run_of_system_messages_is_still_a_preamble() -> None:
    request = CanonicalRequest(
        messages=[
            Message("system", [Block("text", text="one")]),
            Message("system", [Block("text", text="two")]),
            Message("user", [Block("text", text="go")]),
        ]
    )
    body = openai_responses.encode_request(
        request, model="gpt-6-astra", effort="low", max_tokens=4096
    )
    assert body["instructions"] == "one\ntwo"
    assert not [item for item in body["input"] if item.get("role") == "system"]


def _cut_parallel_group() -> CanonicalRequest:
    """A handoff that landed inside a parallel group.

    Two commands went out together, only the first came back, and the turn
    budget ran out before the second did. This is what the summary stage is
    handed, and what Together rejected with "incomplete parallel tool-call
    group" across fourteen trials of the 2026-09-25 sweep.
    """
    return CanonicalRequest(
        messages=[
            Message("user", [Block("text", text="Compile SQLite with gcov.")]),
            *(
                Message(
                    "assistant",
                    [Block("tool_call", call_id=f"call_{n}", name="bash", arguments={})],
                )
                for n in "ab"
            ),
            Message("tool", [Block("tool_result", call_id="call_a", text="out")]),
            Message("user", [Block("text", text="Summarise what you did.")]),
        ],
        max_tokens=1000,
    )


def test_openai_chat_drops_a_tool_call_the_handoff_cut_off() -> None:
    body = openai_chat.encode_request(
        _cut_parallel_group(),
        model="deepseek-ai/DeepSeek-V4.1-Flash",
        effort="low",
        max_tokens=1000,
    )
    (assistant,) = [m for m in body["messages"] if m["role"] == "assistant"]
    # call_b never came back, so it cannot be sent as part of the history.
    assert [c["id"] for c in assistant["tool_calls"]] == ["call_a"]
    answered = {m["tool_call_id"] for m in body["messages"] if m["role"] == "tool"}
    assert {c["id"] for c in assistant["tool_calls"]} == answered


def test_openai_chat_drops_an_assistant_turn_that_lost_every_call() -> None:
    """With nothing answered and nothing said, the turn carries no information."""
    request = CanonicalRequest(
        messages=[
            Message("user", [Block("text", text="Compile SQLite with gcov.")]),
            Message(
                "assistant",
                [Block("tool_call", call_id="call_a", name="bash", arguments={})],
            ),
            Message("user", [Block("text", text="Summarise what you did.")]),
        ],
        max_tokens=1000,
    )
    body = openai_chat.encode_request(
        request,
        model="deepseek-ai/DeepSeek-V4.1-Flash",
        effort="low",
        max_tokens=1000,
    )
    assert [m["role"] for m in body["messages"]] == ["user", "user"]


def test_openai_chat_keeps_an_answered_group_whole() -> None:
    """The repair must not touch a conversation that was never cut."""
    body = openai_chat.encode_request(
        _parallel_group(),
        model="deepseek-ai/DeepSeek-V4.1-Flash",
        effort="low",
        max_tokens=1000,
    )
    (assistant,) = [m for m in body["messages"] if m["role"] == "assistant"]
    assert [c["id"] for c in assistant["tool_calls"]] == ["call_a", "call_b", "call_c"]


def test_an_error_body_that_echoes_the_key_reaches_no_one_with_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """X-19: the error goes on to the agent, its session files and the trace."""
    import contextlib

    from harness_cascade.providers import ProviderError

    key = "sk-test-echoed-0123456789"
    client, _, events = streaming_client(monkeypatch, [], api_key=key)

    @contextlib.contextmanager
    def fake_stream(method: str, url: str, **kwargs: Any):
        yield httpx.Response(
            400,
            json={"error": {"message": f"invalid key Bearer {key}"}},
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr(httpx, "stream", fake_stream)
    with pytest.raises(ProviderError) as raised:
        ask(client)
    assert key not in str(raised.value) and "[REDACTED]" in str(raised.value)
    assert all(key not in event.to_json() for event in events)


@pytest.mark.parametrize(
    ("retry_after", "env", "expected"),
    [
        ("3600", None, 60.0),
        ("5", None, 5.0),
        ("0", None, 0.1),
        ("3600", "10", 10.0),
        ("3600", "junk", 60.0),
        ("Wed, 21 Oct 2026 07:28:00 GMT", None, 4.0),
        (None, None, 4.0),
    ],
)
def test_retry_after_is_honoured_up_to_a_ceiling(
    monkeypatch: pytest.MonkeyPatch,
    retry_after: str | None,
    env: str | None,
    expected: float,
) -> None:
    if env is None:
        monkeypatch.delenv("CASCADE_MAX_RETRY_AFTER", raising=False)
    else:
        monkeypatch.setenv("CASCADE_MAX_RETRY_AFTER", env)
    assert retry_delay(retry_after, attempt=2) == expected


def test_a_429_asking_for_an_hour_waits_at_most_the_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replies = [
        httpx.Response(
            429,
            headers={"retry-after": "3600"},
            json={"error": {"message": "slow down"}},
            request=httpx.Request("POST", "http://endpoint/v1/chat/completions"),
        ),
        httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
            request=httpx.Request("POST", "http://endpoint/v1/chat/completions"),
        ),
    ]
    slept: list[float] = []
    monkeypatch.delenv("CASCADE_MAX_RETRY_AFTER", raising=False)
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: replies.pop(0))
    monkeypatch.setattr("harness_cascade.providers.time.sleep", slept.append)
    client = ProviderClient(
        TransportConfig(
            provider="together",
            protocol="openai-chat",
            model="m",
            effort="low",
            api_key="k",
            base_url="http://endpoint/v1",
            max_retries=2,
        )
    )
    client.complete(CanonicalRequest(messages=[Message("user", [Block("text", text="hi")])]))
    assert slept == [60.0]


def test_codex_hosted_tool_search_is_omitted_like_web_search() -> None:
    canonical = openai_responses.decode_request(
        {
            "input": [{"role": "user", "content": "test"}],
            "tools": [
                {"type": "tool_search"},
                {"type": "web_search"},
                {
                    "type": "function",
                    "name": "exec_command",
                    "parameters": {"type": "object", "properties": {}},
                },
            ],
        }
    )
    assert [tool.name for tool in canonical.tools] == ["exec_command"]
