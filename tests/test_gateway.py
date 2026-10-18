import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler
from typing import Any

import httpx
import pytest

import harness_cascade.gateway as gateway_module
from harness_cascade.gateway import (
    COMPACTION_POINT_MARKER,
    CascadeHTTPServer,
    EmbeddedGateway,
    GatewayConfig,
    GatewayError,
    SwitchGate,
    SwitchHold,
    codex_reference_entry,
    TogetherClient,
    anthropic_response,
    anthropic_sse,
    anthropic_to_chat,
    codex_response,
    codex_sse,
    codex_to_chat,
)
from harness_cascade.events import EventBus, EventKind
from harness_cascade.protocols.canonical import (
    Block,
    CanonicalRequest,
    CanonicalResponse,
    Message,
    Tool,
    Usage,
)


def test_expected_client_disconnect_does_not_print_server_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    server = CascadeHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    try:
        try:
            raise ConnectionResetError(104, "Connection reset by peer")
        except ConnectionResetError:
            server.handle_error(None, ("127.0.0.1", 12345))
    finally:
        server.server_close()
    assert capsys.readouterr().err == ""


def config(**overrides: Any) -> GatewayConfig:
    values = {
        "model": "zai-org/GLM-5.3",
        "effort": "low",
        "api_key": "secret-upstream",
        "local_token": "secret-local",
        "base_url": "https://example.test/v1",
        "max_retries": 0,
    }
    values.update(overrides)
    return GatewayConfig(**values)


def chat_payload(*, tools: bool = False) -> dict[str, Any]:
    message: dict[str, Any] = {
        "content": "Fixed the bug.",
        "reasoning_content": "I inspected auth.py.",
    }
    finish_reason = "stop"
    if tools:
        message["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "Bash",
                    "arguments": '{"command":"pytest"}',
                },
            }
        ]
        finish_reason = "tool_calls"
    return {
        "id": "chatcmpl_1",
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 25,
            "completion_tokens_details": {"reasoning_tokens": 10},
        },
    }


def test_anthropic_request_translates_tools_and_history() -> None:
    body = {
        "system": [{"type": "text", "text": "Be careful."}],
        "max_tokens": 5000,
        "messages": [
            {"role": "user", "content": "Read auth.py"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "Read",
                        "input": {"file_path": "auth.py"},
                    }
                ],
            },
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
        ],
        "tools": [
            {
                "name": "Read",
                "description": "Read file",
                "input_schema": {"type": "object"},
            }
        ],
    }
    payload = anthropic_to_chat(body, config())
    assert payload["messages"][0]["role"] == "system"
    assert sum(1 for msg in payload["messages"] if msg.get("role") == "system") == 1
    assert "Be careful." in str(payload["messages"][0]["content"])
    assert payload["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "contents",
    }
    assert payload["tools"][0]["function"]["name"] == "Read"


def test_anthropic_image_tool_result_remains_a_nonempty_tool_response() -> None:
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_image",
                        "name": "Read",
                        "input": {"file_path": "plot.png"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_image",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "AAAA",
                                },
                            }
                        ],
                    }
                ],
            },
        ]
    }
    payload = anthropic_to_chat(body, config())
    assert payload["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_image",
        "content": "[image/png tool result omitted]",
    }


def test_anthropic_mixed_tool_result_precedes_new_user_prompt() -> None:
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_last",
                        "name": "Write",
                        "input": {"file_path": "gallery.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_last",
                        "content": "File created successfully",
                    },
                    {
                        "type": "text",
                        "text": "Now produce the handoff summary.",
                    },
                ],
            },
        ]
    }
    payload = anthropic_to_chat(body, config())
    assert payload["messages"][-2] == {
        "role": "tool",
        "tool_call_id": "call_last",
        "content": "File created successfully",
    }
    assert payload["messages"][-1] == {
        "role": "user",
        "content": "Now produce the handoff summary.",
    }


def test_anthropic_sse_is_terminal_and_has_signature() -> None:
    response = anthropic_response(
        {"model": "cascade-model"},
        chat_payload(tools=True),
        config(),
    )
    wire = b"".join(anthropic_sse(response)).decode()
    assert "event: message_start" in wire
    assert "event: message_stop" in wire
    assert '"stop_reason":"tool_use"' in wire
    assert "cascade:" in wire


def test_codex_request_and_sse_contract() -> None:
    body = {
        "model": "cascade-model",
        "instructions": "Fix bugs.",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Inspect auth.py"}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "shell",
                "arguments": '{"command":"ls"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "auth.py",
            },
        ],
        "tools": [
            {
                "type": "function",
                "name": "shell",
                "parameters": {"type": "object"},
            }
        ],
    }
    outgoing = codex_to_chat(body, config())
    assert outgoing["messages"][-1]["role"] == "tool"
    assert outgoing["tools"][0]["function"]["name"] == "shell"
    response = codex_response(body, chat_payload(tools=True), config())
    wire = b"".join(codex_sse(response)).decode()
    assert "event: response.created" in wire
    assert "event: response.completed" in wire
    assert "function_call_arguments.delta" not in wire
    sequence_numbers = [
        json.loads(line.removeprefix("data: "))["sequence_number"]
        for line in wire.splitlines()
        if line.startswith("data: ")
    ]
    assert sequence_numbers == list(range(len(sequence_numbers)))


def test_upstream_401_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            401,
            json={"error": {"message": "bad key"}},
            request=httpx.Request("POST", "https://example.test"),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(GatewayError) as captured:
        TogetherClient(config(max_retries=5)).chat({"messages": []})
    assert captured.value.status == 401
    assert calls == 1


def test_non_streaming_upstream_omits_stream_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_body: dict[str, Any] = {}

    def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
        request_body.update(kwargs["json"])
        return httpx.Response(
            200,
            json=chat_payload(),
            request=httpx.Request("POST", "https://example.test"),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    TogetherClient(config()).chat(
        {
            "messages": [],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    )
    assert request_body["stream"] is False
    assert "stream_options" not in request_body


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_status_is_retried_once_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    calls = 0

    def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                status,
                json={"error": {"message": "temporary"}},
                request=httpx.Request("POST", "https://example.test"),
            )
        return httpx.Response(
            200,
            json=chat_payload(),
            request=httpx.Request("POST", "https://example.test"),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("harness_cascade.gateway.time.sleep", lambda _: None)
    result = TogetherClient(config(max_retries=3)).chat({"messages": []})
    assert result["id"] == "chatcmpl_1"
    assert calls == 2


def test_embedded_gateway_preserves_auth_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            200,
            json=chat_payload(),
            request=httpx.Request("POST", "https://example.test"),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    with EmbeddedGateway(config()) as gateway:
        with httpx.Client() as client:
            unauthorized = client.post(
                f"{gateway.base_url}/v1/messages",
                json={"messages": []},
            )
            assert unauthorized.status_code == 401
            response = client.post(
                f"{gateway.base_url}/v1/messages",
                headers={"Authorization": "Bearer secret-local"},
                json={"model": "cascade-model", "messages": [], "stream": True},
            )
        assert response.status_code == 200
        assert "event: message_stop" in response.text


def _stream_chat(
    monkeypatch: pytest.MonkeyPatch, upstream: Any
) -> tuple[httpx.Response, list[bytes]]:
    monkeypatch.setattr(httpx, "post", upstream)
    with EmbeddedGateway(config(stream_heartbeat_seconds=0.05)) as gateway:
        with httpx.Client() as client:
            with client.stream(
                "POST",
                f"{gateway.base_url}/v1/chat/completions",
                headers={"Authorization": "Bearer secret-local"},
                json={"model": "cascade-model", "messages": [], "stream": True},
            ) as response:
                return response, list(response.iter_raw())


def test_a_slow_chat_reply_keeps_the_stream_alive_until_it_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pi gives up on a connection that is silent for five minutes.

    A model that thinks for longer than that is not a dead connection, so the
    gateway has to answer at once and keep talking while it waits.
    """

    def slow_post(*args: Any, **kwargs: Any) -> httpx.Response:
        time.sleep(0.3)
        return httpx.Response(
            200,
            json=chat_payload(),
            request=httpx.Request("POST", "https://example.test"),
        )

    response, chunks = _stream_chat(monkeypatch, slow_post)
    body = b"".join(chunks).decode()
    assert response.status_code == 200
    assert body.startswith(": waiting on the model\n\n")
    events = [line for line in body.splitlines() if line.startswith("data: ")]
    assert json.loads(events[0][6:])["choices"][0]["delta"]["content"] == "Fixed the bug."
    assert events[-1] == "data: [DONE]"


def test_a_fast_chat_reply_streams_without_keepalives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fast_post(*args: Any, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            200,
            json=chat_payload(),
            request=httpx.Request("POST", "https://example.test"),
        )

    response, chunks = _stream_chat(monkeypatch, fast_post)
    assert response.status_code == 200
    assert not b"".join(chunks).startswith(b":")


def test_a_failure_after_the_stream_opened_still_carries_its_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pi retries by matching the error text, not a status it no longer has."""

    def slow_failure(*args: Any, **kwargs: Any) -> httpx.Response:
        time.sleep(0.3)
        return httpx.Response(
            503,
            json={"error": {"message": "no capacity"}},
            request=httpx.Request("POST", "https://example.test"),
        )

    response, chunks = _stream_chat(monkeypatch, slow_failure)
    events = [
        line for line in b"".join(chunks).decode().splitlines() if line.startswith("data: ")
    ]
    assert response.status_code == 200
    error = json.loads(events[0][6:])["error"]
    assert error["message"].startswith("503 ")


def test_a_fast_chat_failure_keeps_its_http_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fast_failure(*args: Any, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "bad key"}},
            request=httpx.Request("POST", "https://example.test"),
        )

    response, _ = _stream_chat(monkeypatch, fast_failure)
    assert response.status_code == 401


def test_codex_catalog_contains_current_required_schema() -> None:
    model = EmbeddedGateway(config()).codex_model()
    required = {
        "slug",
        "shell_type",
        "model_messages",
        "truncation_policy",
        "context_window",
        "supported_reasoning_levels",
    }
    assert required <= model.keys()
    # Codex takes the prompt from whichever of the two is present.
    assert model.get("base_instructions") or model["model_messages"].get(
        "instructions_template"
    )


def test_codex_gets_a_real_system_prompt_not_a_stand_in() -> None:
    """The prompt has to be Codex's own, not something we wrote.

    This entry used to carry a 152 character stand-in, which measured 211
    characters on the wire and cost roughly half the token spend and a large
    part of the score, because the agent stopped early without the section
    telling it how long to persist. The size floor is what catches a silent
    regression to another stand-in; the section name is what catches a prompt
    that is merely long.
    """
    model = EmbeddedGateway(config()).codex_model()
    prompt = model["model_messages"]["instructions_template"]
    assert len(prompt) > 10_000
    assert "Autonomy and persistence" in prompt


def test_codex_prompt_does_not_name_openais_hosted_parallel_wrapper() -> None:
    """Only OpenAI's server can serve `multi_tool_use.parallel`.

    Told to use it, GLM wraps its calls in inventions the upstream parser
    drops, and a reply without a call ends the Codex task. The advice to
    parallelize stays; only the sentence naming the wrapper goes.
    """
    prompt = EmbeddedGateway(config()).codex_model()["model_messages"][
        "instructions_template"
    ]
    assert "multi_tool_use" not in prompt
    assert "You parallelize tool calls whenever you can" in prompt
    assert "multi_tool_use" in codex_reference_entry()["model_messages"][
        "instructions_template"
    ], "the cached reference entry must not be edited in place"


APPLY_PATCH_GRAMMAR = 'start: begin_patch hunk+ end_patch\nbegin_patch: "*** Begin Patch" LF\n'
PATCH = "*** Begin Patch\n*** Add File: hello.txt\n+hello\n*** End Patch\n"


def codex_tools() -> list[dict[str, Any]]:
    """The tool kinds Codex 0.147 declares, trimmed to one of each."""
    return [
        {
            "type": "function",
            "name": "exec_command",
            "description": "Runs a command.",
            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
        },
        {
            "type": "custom",
            "name": "apply_patch",
            "description": "The `apply_patch` tool can be used to edit files.",
            "format": {"type": "grammar", "syntax": "lark", "definition": APPLY_PATCH_GRAMMAR},
        },
        {
            "type": "namespace",
            "name": "multi_agent_v1",
            "description": "Sub-agents.",
            "tools": [
                {
                    "type": "function",
                    "name": "close_agent",
                    "description": "Closes an agent.",
                    "parameters": {
                        "type": "object",
                        "properties": {"target": {"type": "string"}},
                    },
                }
            ],
        },
    ]


def codex_round_trip(
    monkeypatch: pytest.MonkeyPatch,
    body: dict[str, Any],
    upstream_message: dict[str, Any],
) -> tuple[dict[str, Any], httpx.Response]:
    seen: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        seen["body"] = kwargs["json"]
        return httpx.Response(
            200,
            json={
                "id": "chat_1",
                "choices": [{"message": upstream_message, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    with EmbeddedGateway(config(protocol="openai-chat")) as gateway:
        with httpx.Client() as client:
            response = client.post(
                f"{gateway.base_url}/v1/responses",
                headers={"Authorization": "Bearer secret-local"},
                json=body,
            )
    assert response.status_code == 200
    return seen["body"], response


def chat_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def test_codex_freeform_and_namespaced_calls_come_back_in_codex_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex dispatches on item shape, verified against the 0.147 binary.

    A `function_call` to apply_patch is fatal ("incompatible payload") and
    reaches the model only as "aborted"; a flat `multi_agent_v1__close_agent`
    is an "unsupported call". The shapes asserted here are the ones Codex
    executed when a scripted server returned them.
    """
    upstream, response = codex_round_trip(
        monkeypatch,
        {"model": "m", "input": "edit", "tools": codex_tools(), "stream": False},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                chat_call("call_patch", "apply_patch", {"input": PATCH}),
                chat_call("call_close", "multi_agent_v1__close_agent", {"target": "a1"}),
                chat_call("call_exec", "exec_command", {"cmd": "ls"}),
            ],
        },
    )
    patch, close, run = response.json()["output"]
    assert patch["type"] == "custom_tool_call"
    assert (patch["call_id"], patch["name"], patch["input"]) == ("call_patch", "apply_patch", PATCH)
    assert "arguments" not in patch
    assert close["type"] == "function_call"
    assert (close["name"], close["namespace"]) == ("close_agent", "multi_agent_v1")
    assert json.loads(close["arguments"]) == {"target": "a1"}
    assert run["type"] == "function_call" and "namespace" not in run

    offered = {tool["function"]["name"]: tool["function"] for tool in upstream["tools"]}
    assert set(offered) == {"exec_command", "apply_patch", "multi_agent_v1__close_agent"}
    assert APPLY_PATCH_GRAMMAR.strip() in offered["apply_patch"]["description"]
    assert offered["apply_patch"]["parameters"]["required"] == ["input"]


def test_codex_custom_call_streams_its_input_events(monkeypatch: pytest.MonkeyPatch) -> None:
    _, response = codex_round_trip(
        monkeypatch,
        {"model": "m", "input": "edit", "tools": codex_tools(), "stream": True},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [chat_call("call_patch", "apply_patch", {"input": PATCH})],
        },
    )
    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    kinds = [event["type"] for event in events]
    assert "response.custom_tool_call_input.done" in kinds
    assert "response.function_call_arguments.done" not in kinds
    done = next(e for e in events if e["type"] == "response.output_item.done")
    assert done["item"]["type"] == "custom_tool_call"
    assert done["item"]["input"] == PATCH


def test_codex_custom_input_under_another_key_is_still_the_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model that names the one string field differently still meant it."""
    _, response = codex_round_trip(
        monkeypatch,
        {"model": "m", "input": "edit", "tools": codex_tools(), "stream": False},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [chat_call("call_patch", "apply_patch", {"patch": PATCH})],
        },
    )
    assert response.json()["output"][0]["input"] == PATCH


def test_codex_history_of_freeform_and_namespaced_calls_reaches_upstream_intact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The items Codex echoes back, as captured from the 0.147 binary."""
    history = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "edit"}]},
        {
            "type": "custom_tool_call",
            "id": "ctc_1",
            "status": "completed",
            "call_id": "call_patch",
            "name": "apply_patch",
            "input": PATCH,
        },
        {
            "type": "custom_tool_call_output",
            "call_id": "call_patch",
            "output": "Success. Updated the following files:\nA hello.txt\n",
        },
        {
            "type": "function_call",
            "id": "fc_1",
            "name": "close_agent",
            "namespace": "multi_agent_v1",
            "arguments": '{"target": "a1"}',
            "call_id": "call_close",
        },
        {"type": "function_call_output", "call_id": "call_close", "output": "closed"},
    ]
    upstream, _ = codex_round_trip(
        monkeypatch,
        {"model": "m", "input": history, "tools": codex_tools(), "stream": False},
        {"role": "assistant", "content": "done"},
    )
    calls = [
        call
        for message in upstream["messages"]
        if message["role"] == "assistant"
        for call in message.get("tool_calls") or []
    ]
    assert [(c["id"], c["function"]["name"]) for c in calls] == [
        ("call_patch", "apply_patch"),
        ("call_close", "multi_agent_v1__close_agent"),
    ]
    assert json.loads(calls[0]["function"]["arguments"]) == {"input": PATCH}
    results = {m["tool_call_id"]: m["content"] for m in upstream["messages"] if m["role"] == "tool"}
    assert results["call_patch"].startswith("Success.")
    assert results["call_close"] == "closed"


def test_codex_developer_message_reaches_upstream_as_system_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DeepSeek's chat template drops a `developer` turn without a trace.

    Measured: a secret word in a developer message came back "NONE" (58
    prompt tokens), the same word in a system message came back correctly
    (80). Codex puts its permissions and skills there.
    """
    upstream, _ = codex_round_trip(
        monkeypatch,
        {
            "model": "m",
            "instructions": "Codex base prompt.",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Approval policy: never."}],
                },
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]},
            ],
            "stream": False,
        },
        {"role": "assistant", "content": "done"},
    )
    roles = [message["role"] for message in upstream["messages"]]
    assert "developer" not in roles
    assert roles == ["system", "user"]
    assert "Codex base prompt." in upstream["messages"][0]["content"]
    assert "Approval policy: never." in upstream["messages"][0]["content"]


def test_codex_is_never_told_of_more_room_than_the_model_has() -> None:
    """The entry is copied from another model, and its window comes along.

    Codex sizes its context budget from these two fields, so a borrowed ceiling
    is one it will fill and the upstream will then reject.
    """
    model = EmbeddedGateway(config(context_window=100_000)).codex_model()
    assert model["context_window"] == 100_000
    assert model["max_context_window"] == 100_000


def test_codex_keeps_its_own_planning_budget_below_that_ceiling() -> None:
    """Under the ceiling, the harness's own choice of budget stands.

    Codex ships every entry with a 272,000 budget against much larger ceilings,
    because that is where OpenAI re-prices the request. Replacing it with the
    full window would be us overriding the harness.
    """
    reference = codex_reference_entry()
    model = EmbeddedGateway(config(context_window=1_050_000)).codex_model()
    assert model["context_window"] == reference["context_window"]
    assert model["max_context_window"] == 1_050_000
    # Everything the copy exists to provide is still inherited untouched.
    assert model["auto_compact_token_limit"] == reference.get("auto_compact_token_limit")


def config_without_retry_override() -> GatewayConfig:
    """Everything `config` pins except max_retries, so the default can be seen."""
    return GatewayConfig(
        model="zai-org/GLM-5.3",
        effort="low",
        api_key="secret-upstream",
        local_token="secret-local",
        base_url="https://example.test/v1",
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [("8", 8), ("0", 0), ("", 3), ("many", 3), ("-1", 3)],
)
def test_retry_budget_is_campaign_settable(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: int
) -> None:
    """A campaign against a rate-limiting endpoint needs more than three tries.

    Nonsense and negative settings fall back rather than raise, because this is
    read while a trial is starting inside a container where nobody sees a
    traceback. Zero is honoured: it means send once and report the failure.
    """
    monkeypatch.setenv("CASCADE_MAX_RETRIES", value)
    assert config_without_retry_override().max_retries == expected


def test_retry_budget_defaults_without_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CASCADE_MAX_RETRIES", raising=False)
    assert config_without_retry_override().max_retries == 3


@pytest.mark.parametrize(
    ("value", "expected"),
    [("32000", 32_000), ("500000", 131_072), ("", 131_072), ("lots", 131_072), ("0", 131_072)],
)
def test_output_cap_is_campaign_settable_below_the_model_limit(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: int
) -> None:
    """The ceiling can only lower the cap every agent is told about.

    It is applied where the limits are resolved, so the number Pi writes into
    its model entry, the one Claude Code reads from the catalog and the clamp
    on Codex's upstream calls all move together.
    """
    monkeypatch.setenv("CASCADE_MAX_OUTPUT_TOKENS", value)
    with EmbeddedGateway(config(profile_max_output_tokens=131_072)) as gateway:
        assert gateway.config.max_output_tokens == expected
        assert gateway.anthropic_model()["max_tokens"] == expected


def _reply(*kinds: str) -> CanonicalResponse:
    return CanonicalResponse(blocks=[Block(kind, text="x") for kind in kinds])


def test_the_switch_gate_counts_only_replies_that_call_a_tool() -> None:
    gate = SwitchGate(2)
    gate.record(_reply("text"))
    gate.record(_reply("reasoning", "tool_call", "tool_call"))
    gate.admit()
    assert (gate.replies, gate.tool_replies) == (2, 1)
    gate.record(_reply("tool_call"))
    assert (gate.replies, gate.tool_replies) == (3, 2)


def test_each_reply_that_calls_a_tool_is_announced_with_its_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = EventBus("run")
    seen: list = []
    bus.subscribe(seen.append)
    gateway = EmbeddedGateway(
        config(switch_after=5, stage="first", harness="codex"), events=bus
    )
    replies = iter(
        [
            CanonicalResponse(blocks=[Block("tool_call", name="exec_command")]),
            CanonicalResponse(blocks=[Block("text", text="thinking aloud")]),
            CanonicalResponse(
                blocks=[Block("tool_call", name="write_stdin"), Block("tool_call", name="exec_command")]
            ),
        ]
    )
    monkeypatch.setattr(gateway, "_forward", lambda request: next(replies))
    for _ in range(3):
        gateway._complete(CanonicalRequest(tools=[Tool(name="exec_command")]))

    announced = [event for event in seen if event.kind == EventKind.TURN_STARTED]
    assert [event.data for event in announced] == [
        {"turn": 1, "limit": 5, "tools": ["exec_command"]},
        {"turn": 2, "limit": 5, "tools": ["write_stdin", "exec_command"]},
    ]
    assert {(event.stage, event.harness) for event in announced} == {("first", "codex")}


def test_the_switch_gate_holds_every_request_past_the_limit_until_released() -> None:
    reached: list[int] = []
    gate = SwitchGate(1, reached.append)
    gate.record(_reply("tool_call"))
    outcomes: list[BaseException] = []

    def ask() -> None:
        try:
            gate.admit()
        except BaseException as exc:  # noqa: BLE001
            outcomes.append(exc)

    waiting = [threading.Thread(target=ask) for _ in range(2)]
    for thread in waiting:
        thread.start()
    assert gate.reached.wait(5)
    time.sleep(0.2)
    assert outcomes == []
    gate.release()
    for thread in waiting:
        thread.join(5)
    assert len(outcomes) == 2
    assert all(isinstance(exc, SwitchHold) for exc in outcomes)
    assert reached == [1]


def test_a_summary_gateway_has_no_switch_point() -> None:
    gateway = EmbeddedGateway(config(switch_after=3, disable_tools=True))
    assert gateway.gate.limit is None
    assert EmbeddedGateway(config(switch_after=3)).gate.limit == 3


def test_the_gateway_keeps_the_last_error_it_handed_an_agent() -> None:
    events = EventBus("errors")
    seen: list = []
    events.subscribe(seen.append)
    gateway = EmbeddedGateway(config(api_key="sk-upstream", stage="first"), events)
    gateway._failed(GatewayError(400, "rejected key sk-upstream: flagged"))
    assert gateway.last_error == "gateway returned 400 to the agent: rejected key [REDACTED]: flagged"
    assert seen[-1].kind == EventKind.WARNING and seen[-1].stage == "first"
    gateway._failed(SwitchHold(503, "held"))
    assert gateway.last_error.endswith("flagged")


def _compacting_gateway(events: EventBus, harness: str) -> EmbeddedGateway:
    gateway = EmbeddedGateway(
        config(
            stage="first",
            harness=harness,
            profile_context_window=1_000,
            compact_at_percent=50,
        ),
        events,
    )
    gateway._forward = lambda request: CanonicalResponse(  # type: ignore[method-assign]
        blocks=[Block("tool_call", call_id="c1", name="bash", arguments={"command": "ls"})]
        if request.tools
        else [Block("text", text="compacted")],
        usage=Usage(input_tokens=600, output_tokens=10),
    )
    return gateway


def _ask(gateway: EmbeddedGateway, *, tools: bool = True) -> httpx.Response:
    body: dict[str, Any] = {
        "model": "zai-org/GLM-5.3",
        "messages": [{"role": "user", "content": "go"}],
    }
    if tools:
        body["tools"] = [
            {"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}
        ]
    return httpx.post(
        f"{gateway.base_url}/v1/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {gateway.config.local_token}"},
        timeout=10,
    )


def test_pi_is_told_to_compact_once_its_context_passes_the_threshold() -> None:
    """Pi checks its own threshold only between prompts, never mid-stage.

    So the gateway answers its next tool request with an overflow Pi
    recognises; Pi compacts, which needs a request without tools, and resends.
    """
    events = EventBus("compact")
    seen: list = []
    events.subscribe(seen.append)
    with _compacting_gateway(events, "pi") as gateway:
        assert gateway.config.compact_at_tokens == 500
        assert _ask(gateway).status_code == 200
        refused = _ask(gateway)
        assert refused.status_code == 400
        message = refused.json()["error"]["message"]
        # Pi's own pattern for a context overflow.
        assert re.search(r"context[_ ]length[_ ]exceeded", message, re.IGNORECASE)
        assert COMPACTION_POINT_MARKER in message
        assert _ask(gateway, tools=False).status_code == 200
        assert _ask(gateway).status_code == 200
    assert gateway.last_error == ""
    assert not [event for event in seen if event.kind == EventKind.WARNING]
    requested = [event.data for event in seen if event.kind == EventKind.COMPACTION_REQUESTED]
    assert requested == [{"context_tokens": 610, "threshold": 500}]
    limits = next(event.data for event in seen if event.kind == EventKind.CONTEXT_LIMITS)
    assert (limits["context_window"], limits["compact_at_tokens"]) == (1_000, 500)


@pytest.mark.parametrize("harness", ["claude-code", "codex"])
def test_harnesses_that_compact_mid_task_are_never_refused(harness: str) -> None:
    with _compacting_gateway(EventBus("compact"), harness) as gateway:
        assert [_ask(gateway).status_code for _ in range(3)] == [200, 200, 200]


def test_no_compaction_policy_leaves_pi_to_its_own() -> None:
    gateway = EmbeddedGateway(config(harness="pi", profile_context_window=1_000))
    assert gateway.config.compact_at_tokens is None


def test_codex_plans_against_the_full_window_under_a_compaction_policy() -> None:
    model = EmbeddedGateway(
        config(context_window=1_050_000, compact_at_percent=85)
    ).codex_model()
    assert model["context_window"] == 1_050_000
    assert model["max_context_window"] == 1_050_000


@pytest.mark.parametrize(
    ("harness", "window", "threshold"),
    [
        ("codex", 1_050_000, 272_000),
        ("codex", 262_144, 222_822),
        ("claude-code", 1_050_000, 892_500),
        ("pi", 1_050_000, 892_500),
    ],
)
def test_compaction_is_85_percent_of_the_window_and_at_most_272k_for_codex(
    harness: str, window: int, threshold: int
) -> None:
    planned = config(harness=harness, context_window=window, compact_at_percent=85)
    assert planned.compact_at_tokens == threshold


def _sent_limits(gateway: EmbeddedGateway) -> list[int]:
    sent: list[int] = []

    def send(request: Any, max_output_tokens: int) -> CanonicalResponse:
        sent.append(max_output_tokens)
        return CanonicalResponse(
            blocks=[Block("text", text="ok")],
            usage=Usage(input_tokens=90_000, output_tokens=2_000),
        )

    gateway._send = send  # type: ignore[method-assign]
    return sent


def _request(*texts: str) -> CanonicalRequest:
    return CanonicalRequest(
        messages=[Message("user", [Block("text", text=text)]) for text in texts]
    )


def test_the_output_limit_shrinks_to_what_the_window_has_left() -> None:
    """Together rejects a request whose input and output limit overflow the
    window, and no harness but Pi lowers its limit to fit."""
    gateway = EmbeddedGateway(config(context_window=131_072, max_output_tokens=131_072))
    sent = _sent_limits(gateway)
    gateway._forward(_request("x" * 40_000))
    assert sent == [131_072 - 10_000 - 4_096]
    # The next request is measured from the last reply's usage, plus the
    # messages added since.
    gateway._forward(_request("x" * 40_000, "y" * 4_000))
    assert sent[1] == 131_072 - (92_000 + 1_000) - 4_096
    gateway._forward(_request("z" * 4 * 131_072))
    assert sent[2] == 1


def test_a_small_request_gets_the_model_s_whole_output_limit() -> None:
    gateway = EmbeddedGateway(config(context_window=1_048_576, max_output_tokens=262_144))
    sent = _sent_limits(gateway)
    gateway._forward(_request("hello"))
    assert sent == [262_144]


def test_codex_fallback_entry_copies_codex_and_carries_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Field for field `model_info_from_slug` in codex-rs/models-manager 0.147,
    whose max_context_window of 272,000 is what caps the window otherwise."""
    monkeypatch.setattr(gateway_module, "codex_fallback_prompt", lambda: "Codex prompt.")
    entry = EmbeddedGateway(
        config(context_window=1_048_576, supported_efforts=("low", "high"))
    ).codex_fallback_model()
    assert entry is not None
    assert entry["model_messages"]["instructions_template"] == "Codex prompt."
    assert (entry["context_window"], entry["max_context_window"]) == (1_048_576, 1_048_576)
    assert [level["effort"] for level in entry["supported_reasoning_levels"]] == ["low", "high"]
    assert entry["apply_patch_tool_type"] is None
    assert entry["supports_parallel_tool_calls"] is False
    assert entry["truncation_policy"] == {"mode": "bytes", "limit": 10_000}
    assert entry["shell_type"] == "default"
    assert entry["effective_context_window_percent"] == 95
    assert "default_reasoning_level" not in entry


def test_no_fallback_entry_without_codex_s_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway_module, "codex_fallback_prompt", lambda: None)
    assert EmbeddedGateway(config()).codex_fallback_model() is None


def test_codex_fallback_prompt_is_read_whole_from_the_binary(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt = (
        "You are a coding agent running in the Codex CLI, a terminal-based assistant.\n"
        + "Plan your work.\n" * 400
        + "If all steps are complete, ensure you call `update_plan` to mark all "
        "steps as `completed`.\n"
    )
    binary = tmp_path / "codex"
    binary.write_bytes(b"\x00\x01rust" + prompt.encode() + b"You are Codex, a coding agent")
    monkeypatch.setattr(gateway_module, "_codex_native_binary", lambda: binary)
    gateway_module.codex_fallback_prompt.cache_clear()
    try:
        assert gateway_module.codex_fallback_prompt() == prompt
        binary.write_bytes(b"You are a coding agent running in the Codex CLI, but rewritten")
        gateway_module.codex_fallback_prompt.cache_clear()
        assert gateway_module.codex_fallback_prompt() is None
    finally:
        gateway_module.codex_fallback_prompt.cache_clear()


def test_tool_calls_in_a_reply_to_a_request_without_tools_are_dropped() -> None:
    events = EventBus("summary")
    seen: list = []
    events.subscribe(seen.append)
    gateway = EmbeddedGateway(
        config(stage="handoff", harness="codex", disable_tools=True), events
    )
    gateway._forward = lambda request: CanonicalResponse(  # type: ignore[method-assign]
        blocks=[
            Block("text", text="Summary: tests pass."),
            Block("tool_call", call_id="c1", name="exec_command", arguments={"cmd": "touch x"}),
        ],
    )
    request = CanonicalRequest(
        messages=[Message("user", [Block("text", text="summarize")])],
        tools=[Tool(name="exec_command", parameters={"type": "object"})],
    )
    response = gateway._complete(request)
    assert [block.type for block in response.blocks] == ["text"]
    assert seen[-1].kind == EventKind.WARNING
    assert seen[-1].data["tools"] == ["exec_command"]


def test_tool_calls_survive_when_the_request_offered_tools() -> None:
    gateway = EmbeddedGateway(config(stage="first", harness="codex"))
    call = Block("tool_call", call_id="c1", name="exec_command", arguments={})
    gateway._forward = lambda request: CanonicalResponse(blocks=[call])  # type: ignore[method-assign]
    request = CanonicalRequest(
        messages=[Message("user", [Block("text", text="go")])],
        tools=[Tool(name="exec_command", parameters={"type": "object"})],
    )
    assert gateway._complete(request).blocks == [call]


def test_the_codex_catalog_is_scanned_as_bytes(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = [
        {"slug": "gpt-5.5", "context_window": 272000, "nested": {"a": {"b": 1}}},
        {"slug": "gpt-5.4", "context_window": 1, "notes": "caf\u00e9 \u2014 ok"},
    ]
    binary = tmp_path / "codex"
    binary.write_bytes(
        b"\x7fELF\xff\xfe\x00"
        + json.dumps({"models": entries}, ensure_ascii=False).encode()
        + b"\x00\xc3\x28 tail"
    )
    monkeypatch.setattr(gateway_module, "_codex_native_binary", lambda: binary)
    gateway_module._builtin_catalog.cache_clear()
    try:
        assert list(gateway_module._builtin_catalog()) == entries
        binary.write_bytes(b"")
        gateway_module._builtin_catalog.cache_clear()
        assert gateway_module._builtin_catalog() == ()
    finally:
        gateway_module._builtin_catalog.cache_clear()
