import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

import harness_cascade.harnesses as harnesses_module
from harness_cascade.config import StagePlan
from harness_cascade.credentials import CredentialSet
from harness_cascade.events import EventBus, EventKind
from harness_cascade.gateway import EmbeddedGateway, GatewayConfig
from harness_cascade.protocols.canonical import Block, CanonicalResponse, Usage
from harness_cascade.harnesses import (
    CodexAdapter,
    ClaudeAdapter,
    HarnessContext,
    HarnessError,
    HarnessResult,
    PiAdapter,
    adapter_for,
)


@pytest.mark.parametrize("first", ["pi", "claude-code", "codex"])
@pytest.mark.parametrize("second", ["pi", "claude-code", "codex"])
def test_every_ordered_harness_pair_is_registered(first: str, second: str) -> None:
    assert adapter_for(first).name == first
    assert adapter_for(second).name == second


def context(
    tmp_path: Path,
    *,
    harness: str,
    gateway: EmbeddedGateway | None = None,
    prompt: str = "Fix auth.py",
) -> HarnessContext:
    return HarnessContext(
        stage="first",
        plan=StagePlan(harness=harness, effort="low", switch_after=3),
        model="zai-org/GLM-5.3",
        prompt=prompt,
        workspace=tmp_path,
        state_dir=tmp_path / "state",
        log_path=tmp_path / "stage.log",
        credentials=CredentialSet("together", "upstream-secret", "test"),
        events=EventBus("test-run"),
        gateway=gateway,
        session_id="00000000-0000-4000-8000-000000000001",
        timeout_seconds=60,
    )


def test_pi_does_not_pass_double_dash_and_protects_flag_prompt(tmp_path: Path) -> None:
    gateway = EmbeddedGateway(
        GatewayConfig(
            model="zai-org/GLM-5.3",
            effort="low",
            api_key="upstream-secret",
            local_token="local-secret",
            base_url="https://example.test/v1",
        )
    )
    with gateway:
        command, env = PiAdapter().build_command(
            context(
                tmp_path,
                harness="pi",
                gateway=gateway,
                prompt="-- inspect auth.py",
            )
        )
    assert "--" not in command
    assert command[-1].startswith("Task:\n--")
    assert "cascade" in command
    assert env["CASCADE_GATEWAY_TOKEN"] == "local-secret"
    assert env["PI_CODING_AGENT_DIR"].endswith("pi-config")
    assert "upstream-secret" not in command
    assert "TOGETHER_API_KEY" not in env
    assert "CASCADE_API_KEY_FILE" not in env


def test_claude_code_runs_in_bare_mode(tmp_path: Path) -> None:
    """Bare mode is what the published Terminal-Bench runs use.

    Off, Claude Code offers itself cron jobs, workflows, worktrees and
    agent-to-user messaging inside a sealed benchmark container. They cannot
    fire, but their schemas are 11.6k of a 12.5k token preamble that goes out
    on every single call.
    """
    gateway = EmbeddedGateway(
        GatewayConfig(
            model="claude-opus-5-5",
            effort="low",
            api_key="upstream-secret",
            local_token="local-secret",
            base_url="https://example.test/v1",
        )
    )
    with gateway:
        _, env = ClaudeAdapter().build_command(
            context(tmp_path, harness="claude-code", gateway=gateway, prompt="go")
        )
    assert env["CLAUDE_CODE_SIMPLE"] == "1"


def _pi_model_entry(tmp_path: Path, window: int | None) -> dict:
    gateway = EmbeddedGateway(
        GatewayConfig(
            model="zai-org/GLM-5.3",
            effort="low",
            api_key="upstream-secret",
            local_token="local-secret",
            base_url="https://example.test/v1",
            profile_context_window=window,
        )
    )
    with gateway:
        _, env = PiAdapter().build_command(
            context(tmp_path, harness="pi", gateway=gateway, prompt="go")
        )
    written = Path(env["PI_CODING_AGENT_DIR"]) / "models.json"
    return json.loads(written.read_text())["providers"]["cascade"]["models"][0]


def test_pi_is_told_the_context_window_from_the_model_profile(tmp_path: Path) -> None:
    """Pi compacts based on this number, so it has to be the real one.

    It used to be hard-coded to 1,000,000 against models whose real windows are
    200k-272k, which would have let Pi keep filling a prompt the upstream would
    then reject. Nothing failed that way in either campaign, but only because
    the runs stayed far below both figures.
    """
    assert _pi_model_entry(tmp_path, 272_000)["contextWindow"] == 272_000


def test_pi_falls_back_to_the_smallest_window_we_serve(tmp_path: Path) -> None:
    """With no profile figure, understating is the safe direction.

    Compacting early costs context; claiming a window the model does not have
    fails the trial outright partway through.
    """
    assert _pi_model_entry(tmp_path, None)["contextWindow"] == 200_000


def test_pi_waits_as_long_as_the_gateway_does(tmp_path: Path) -> None:
    """Pi's own five-minute limit threw away replies the gateway was still producing."""
    gateway = EmbeddedGateway(
        GatewayConfig(
            model="zai-org/GLM-5.3",
            effort="high",
            api_key="upstream-secret",
            local_token="local-secret",
            base_url="https://example.test/v1",
            request_timeout_seconds=5400,
        )
    )
    with gateway:
        _, env = PiAdapter().build_command(
            context(tmp_path, harness="pi", gateway=gateway, prompt="go")
        )
    settings = json.loads((Path(env["PI_CODING_AGENT_DIR"]) / "settings.json").read_text())
    assert settings["httpIdleTimeoutMs"] == 5_400_000


def test_pi_resume_uses_exact_session_without_conflicting_continue(
    tmp_path: Path,
) -> None:
    gateway = EmbeddedGateway(
        GatewayConfig(
            model="zai-org/GLM-5.3",
            effort="low",
            api_key="upstream-secret",
            local_token="local-secret",
            base_url="https://example.test/v1",
        )
    )
    with gateway:
        resumed = replace(
            context(tmp_path, harness="pi", gateway=gateway),
            resume=True,
            summary=True,
        )
        command, _ = PiAdapter().build_command(resumed)
    assert "--session" in command
    assert "--session-id" not in command
    assert "--continue" not in command
    assert "--no-tools" in command


def test_pi_successful_auto_retry_clears_provisional_failure(tmp_path: Path) -> None:
    harness_context = context(tmp_path, harness="pi")
    result = HarnessResult(
        harness="pi",
        status="running",
        returncode=0,
        session_id=harness_context.session_id,
    )
    adapter = PiAdapter()
    adapter.consume_event(
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "stopReason": "error",
                "errorMessage": "429 rate limited",
            },
        },
        result,
        harness_context,
    )
    assert result.status == "failed"

    adapter.consume_event(
        {"type": "auto_retry_end", "success": True},
        result,
        harness_context,
    )
    assert result.status == "running"


def test_codex_uses_local_responses_gateway(tmp_path: Path) -> None:
    gateway = EmbeddedGateway(
        GatewayConfig(
            model="zai-org/GLM-5.3",
            effort="low",
            api_key="upstream-secret",
            local_token="local-secret",
        )
    )
    with gateway:
        command, env = CodexAdapter().build_command(
            context(tmp_path, harness="codex", gateway=gateway)
        )
    joined = " ".join(command)
    assert 'model_provider="cascade"' in joined
    assert 'wire_api="responses"' in joined
    assert gateway.config.api_key not in joined
    assert env["CASCADE_GATEWAY_TOKEN"] == "local-secret"
    assert "TOGETHER_API_KEY" not in env
    assert "OPENAI_API_KEY" not in env
    assert "CASCADE_API_KEY_FILE" not in env


@pytest.mark.parametrize("resume", [False, True])
def test_codex_does_not_offer_subagents_goals_or_images(tmp_path: Path, resume: bool) -> None:
    """GLM calls these tools by names that do not exist, and the call is lost upstream."""
    gateway = EmbeddedGateway(
        GatewayConfig(
            model="zai-org/GLM-5.3",
            effort="low",
            api_key="upstream-secret",
            local_token="local-secret",
        )
    )
    with gateway:
        command, _ = CodexAdapter().build_command(
            replace(context(tmp_path, harness="codex", gateway=gateway), resume=resume)
        )
    positional = command.index("--") - (1 if resume else 0)
    for feature in ("multi_agent", "goals", "view_image"):
        flag = command.index(f"features.{feature}=false")
        assert command[flag - 1] == "-c"
        assert flag < positional
    if resume:
        assert command[positional] == "00000000-0000-4000-8000-000000000001"


def test_agents_do_not_inherit_upstream_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TOGETHER_API_KEY", "together-secret")
    monkeypatch.setenv("LLM_API_KEY", "llm-secret")
    monkeypatch.setenv("API_KEY", "generic-secret")
    monkeypatch.setenv("CASCADE_API_KEY_FILE", "/run/secrets/cascade-api-key")
    gateway = EmbeddedGateway(
        GatewayConfig(
            model="zai-org/GLM-5.3",
            effort="low",
            api_key="upstream-secret",
            local_token="local-secret",
            base_url="https://example.test/v1",
        )
    )
    with gateway:
        for adapter, token_key in (
            (PiAdapter(), "CASCADE_GATEWAY_TOKEN"),
            (CodexAdapter(), "CASCADE_GATEWAY_TOKEN"),
            (ClaudeAdapter(), "ANTHROPIC_AUTH_TOKEN"),
        ):
            harness = "pi" if adapter.name == "pi" else adapter.name
            _, env = adapter.build_command(
                context(tmp_path, harness=harness, gateway=gateway)
            )
            assert env[token_key] == "local-secret"
            assert env.get("TOGETHER_API_KEY") is None
            assert env.get("LLM_API_KEY") is None
            assert env.get("API_KEY") is None
            assert env.get("CASCADE_API_KEY_FILE") is None
            assert "upstream-secret" not in env.values()
            assert "together-secret" not in env.values()


def test_claude_summary_result_is_emitted_as_assistant_text(tmp_path: Path) -> None:
    harness_context = replace(
        context(tmp_path, harness="claude-code"),
        stage="handoff",
        summary=True,
    )
    events = []
    harness_context.events.subscribe(events.append)
    result = HarnessResult(
        harness="claude-code",
        status="running",
        returncode=0,
        session_id=harness_context.session_id,
    )

    ClaudeAdapter().consume_event(
        {
            "type": "result",
            "result": "Completed work and verified tests.",
            "num_turns": 1,
            "is_error": False,
        },
        result,
        harness_context,
    )

    assistant_events = [
        event for event in events if event.kind == EventKind.ASSISTANT_TEXT
    ]
    assert result.final_text == "Completed work and verified tests."
    assert assistant_events[0].data["text"] == result.final_text


def test_claude_terminal_result_is_allowed_to_exit_gracefully(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = iter(
                [
                    '{"type":"result","subtype":"error_max_turns","is_error":true,'
                    '"num_turns":3,"result":"Reached maximum turns"}\n'
                ]
            )
            self.returncode: int | None = None
            self.pid = 12345

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            self.returncode = 1
            return self.returncode

    process = FakeProcess()
    forced_terminations: list[int] = []
    monkeypatch.setattr(
        harnesses_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(
        harnesses_module,
        "_kill_process_group",
        lambda proc: forced_terminations.append(proc.pid),
    )
    gateway = EmbeddedGateway(
        GatewayConfig(
            model="zai-org/GLM-5.3",
            effort="low",
            api_key="upstream-secret",
            local_token="local-secret",
            base_url="https://example.test/v1",
        )
    )

    with gateway:
        for _ in range(3):
            gateway.gate.record(_tool_reply())
        result = ClaudeAdapter().run(
            context(tmp_path, harness="claude-code", gateway=gateway)
        )

    assert result.status == "switch_limit"
    assert result.turns == 3
    assert result.error == ""
    assert result.returncode == 1
    assert forced_terminations == []


def _gateway() -> EmbeddedGateway:
    return EmbeddedGateway(
        GatewayConfig(
            model="zai-org/GLM-5.3",
            effort="low",
            api_key="upstream-secret",
            local_token="local-secret",
            base_url="https://example.test/v1",
        )
    )


@pytest.mark.parametrize(
    ("switch_after", "summary", "expected"),
    [(0, False, None), (3, False, "3"), (0, True, "1")],
)
def test_claude_code_turn_budget_is_only_ever_ours(
    tmp_path: Path, switch_after: int, summary: bool, expected: str | None
) -> None:
    """A solo stage runs unbounded, as Pi and Codex do.

    It used to stop at 80 turns, a number no harness default sets, which ended
    one DeepSeek solo stage in six before the work was done.
    """
    ctx = context(tmp_path, harness="claude-code", prompt="go")
    ctx = replace(ctx, plan=replace(ctx.plan, switch_after=switch_after), summary=summary)
    with _gateway() as gateway:
        command, _ = ClaudeAdapter().build_command(replace(ctx, gateway=gateway))
    if expected is None:
        assert "--max-turns" not in command
    else:
        assert command[command.index("--max-turns") + 1] == expected


def test_claude_code_gets_the_output_cap_and_patience_the_gateway_has(
    tmp_path: Path,
) -> None:
    """Its defaults are a 32,000 token cap and a five minute wait.

    At GLM's speed a 32,000 token reply takes about 200 seconds, so the cap is
    what a hard task hits, and a longer reply would outlast the wait.
    """
    ctx = context(tmp_path, harness="claude-code", prompt="go")
    with _gateway() as gateway:
        _, env = ClaudeAdapter().build_command(replace(ctx, gateway=gateway))
        cap = gateway.config.max_output_tokens
        patience = int(gateway.config.request_timeout_seconds * 1000)
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == str(cap)
    assert cap > 32_000
    assert env["API_TIMEOUT_MS"] == str(patience)
    assert env["CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS"] == str(patience)


@pytest.mark.parametrize(
    ("model", "builtin", "catalog"),
    [
        ("zai-org/GLM-5.3-Flash", None, False),
        ("gpt-5.5", {"model_messages": {"instructions_template": "GPT"}}, False),
        ("gpt-5.6-sol", {"tool_mode": "code_mode_only", "use_responses_lite": True,
                         "model_messages": {"instructions_template": "GPT"}}, True),
    ],
)
def test_codex_uses_its_own_entry_for_a_model_unless_we_cannot_serve_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    builtin: dict | None,
    catalog: bool,
) -> None:
    """Open models get Codex's general prompt, GPT models their own.

    Only an entry whose wire this gateway does not speak is replaced.
    """
    monkeypatch.delenv("CASCADE_CODEX_CATALOG", raising=False)
    monkeypatch.setattr(harnesses_module, "codex_builtin_entry", lambda _: builtin)
    ctx = replace(context(tmp_path, harness="codex", prompt="go"), model=model)
    with _gateway() as gateway:
        command, _ = CodexAdapter().build_command(replace(ctx, gateway=gateway))
    assert any(arg.startswith("model_catalog_json=") for arg in command) is catalog


@pytest.mark.parametrize(("value", "catalog"), [("", False), ("0", False), ("1", True), ("on", True)])
def test_codex_runs_our_model_as_an_unknown_one_unless_told_otherwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str, catalog: bool
) -> None:
    monkeypatch.setenv("CASCADE_CODEX_CATALOG", value)
    with _gateway() as gateway:
        command, env = CodexAdapter().build_command(
            context(tmp_path, harness="codex", gateway=gateway, prompt="go")
        )
    passed = [arg for arg in command if arg.startswith("model_catalog_json=")]
    written = list((Path(env["CODEX_HOME"])).glob("models-*.json"))
    assert bool(passed) is catalog
    assert bool(written) is catalog
    assert 'model_provider="cascade"' in command


def _policy_gateway(
    percent: int | None,
    *,
    model: str = "zai-org/GLM-5.3",
    window: int = 1_048_576,
    harness: str = "",
) -> EmbeddedGateway:
    return EmbeddedGateway(
        GatewayConfig(
            model=model,
            effort="low",
            api_key="upstream-secret",
            local_token="local-secret",
            base_url="https://example.test/v1",
            supported_efforts=("low", "high"),
            profile_context_window=window,
            compact_at_percent=percent,
            harness=harness,
        )
    )


@pytest.mark.parametrize(
    ("builtin", "prompt", "catalog"),
    [
        (None, "Codex prompt.", True),
        ({"model_messages": {"instructions_template": "GPT"}}, "Codex prompt.", True),
        (None, None, False),
    ],
)
def test_codex_is_given_the_window_and_threshold_of_the_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    builtin: dict | None,
    prompt: str | None,
    catalog: bool,
) -> None:
    """A model reaches its window only through a catalog entry: a copy of the
    fallback, or the entry Codex ships for it; with no prompt to copy, Codex
    keeps its 272,000 and says so. Codex compacts at no more than 272,000."""
    import harness_cascade.gateway as gateway_module

    monkeypatch.delenv("CASCADE_CODEX_CATALOG", raising=False)
    monkeypatch.setattr(harnesses_module, "codex_builtin_entry", lambda _: builtin)
    monkeypatch.setattr(gateway_module, "codex_fallback_prompt", lambda: prompt)
    monkeypatch.setattr(gateway_module, "_builtin_catalog", lambda: ({"slug": "gpt-5.5"},))
    ctx = context(tmp_path, harness="codex", prompt="go")
    seen: list = []
    ctx.events.subscribe(seen.append)
    with _policy_gateway(85, harness="codex") as gateway:
        command, env = CodexAdapter().build_command(replace(ctx, gateway=gateway))
    assert "model_context_window=1048576" in command
    assert "model_auto_compact_token_limit=272000" in command
    written = list(Path(env["CODEX_HOME"]).glob("models-*.json"))
    assert bool(written) is catalog
    if catalog and builtin is None:
        entry, *shipped = json.loads(written[0].read_text())["models"]
        assert entry["max_context_window"] == 1_048_576
        assert entry["model_messages"]["instructions_template"] == "Codex prompt."
        # Codex's own entries stay, or spawn_agent would lose its model list.
        assert shipped == [{"slug": "gpt-5.5"}]
    warned = [event for event in seen if event.kind == EventKind.WARNING]
    assert bool(warned) is (builtin is None and prompt is None)


def test_codex_without_a_policy_keeps_its_own_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CASCADE_CODEX_CATALOG", raising=False)
    monkeypatch.setattr(harnesses_module, "codex_builtin_entry", lambda _: None)
    ctx = context(tmp_path, harness="codex", prompt="go")
    with _policy_gateway(None) as gateway:
        command, env = CodexAdapter().build_command(replace(ctx, gateway=gateway))
    assert not [arg for arg in command if arg.startswith("model_context_window=")]
    assert not [arg for arg in command if arg.startswith("model_auto_compact_token_limit=")]
    assert not list(Path(env["CODEX_HOME"]).glob("models-*.json"))


@pytest.mark.parametrize("percent", [85, None])
def test_claude_code_is_told_the_window_and_threshold_of_the_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, percent: int | None
) -> None:
    """Measured on 2.1.282: without these it compacts near 170,000 tokens
    whatever the window; with them at 85% of the window."""
    monkeypatch.delenv("CLAUDE_CODE_MAX_CONTEXT_TOKENS", raising=False)
    monkeypatch.delenv("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE", raising=False)
    ctx = context(tmp_path, harness="claude-code", prompt="go")
    with _policy_gateway(percent) as gateway:
        _, env = ClaudeAdapter().build_command(replace(ctx, gateway=gateway))
    if percent is None:
        assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in env
        assert "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE" not in env
    else:
        assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "1048576"
        assert env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] == "85"
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW" not in env


@pytest.mark.parametrize(
    ("percent", "window", "model_arg", "compact_window"),
    [
        (85, 1_000_000, "claude-opus-4-8[1m]", "1000000"),
        (85, 500_000, "claude-opus-4-8[1m]", "500000"),
        (85, 200_000, "claude-opus-4-8", None),
        (None, 1_000_000, "claude-opus-4-8", None),
    ],
)
def test_claude_code_plans_a_claude_model_against_its_real_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    percent: int | None,
    window: int,
    model_arg: str,
    compact_window: str | None,
) -> None:
    """Measured on 2.1.282 with claude-opus-4-8 behind the gateway: 179k
    without the policy, 163k with only the percent (85% of 200k), 85.0% of
    1M with `[1m]` and the window; `[1m]` alone never compacted."""
    monkeypatch.delenv("CLAUDE_CODE_AUTO_COMPACT_WINDOW", raising=False)
    ctx = replace(context(tmp_path, harness="claude-code", prompt="go"), model="claude-opus-4-8")
    with _policy_gateway(percent, model="claude-opus-4-8", window=window) as gateway:
        command, env = ClaudeAdapter().build_command(replace(ctx, gateway=gateway))
    assert command[command.index("--model") + 1] == model_arg
    assert env["ANTHROPIC_MODEL"] == model_arg
    assert env.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW") == compact_window


def test_a_gpt_model_compacts_at_272k_in_codex_and_at_85_percent_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import harness_cascade.gateway as gateway_module

    monkeypatch.delenv("CASCADE_CODEX_CATALOG", raising=False)
    shipped = {
        "slug": "gpt-5.5",
        "model_messages": {"instructions_template": "GPT"},
        "context_window": 272_000,
        "max_context_window": 272_000,
    }
    monkeypatch.setattr(harnesses_module, "codex_builtin_entry", lambda _: shipped)
    monkeypatch.setattr(gateway_module, "_builtin_catalog", lambda: (shipped, {"slug": "gpt-5.4"}))
    gpt = {"model": "gpt-5.5", "window": 1_050_000}
    ctx = replace(context(tmp_path, harness="codex", prompt="go"), model="gpt-5.5")
    with _policy_gateway(85, harness="codex", **gpt) as gateway:
        command, env = CodexAdapter().build_command(replace(ctx, gateway=gateway))
    assert "model_context_window=1050000" in command
    assert "model_auto_compact_token_limit=272000" in command
    # Left at 272,000 the entry would hold Codex to 90% of that, 244,800.
    written = next(Path(env["CODEX_HOME"]).glob("models-*.json"))
    assert json.loads(written.read_text())["models"] == [
        {**shipped, "context_window": 1_050_000, "max_context_window": 1_050_000},
        {"slug": "gpt-5.4"},
    ]
    ctx = replace(context(tmp_path, harness="claude-code", prompt="go"), model="gpt-5.5")
    with _policy_gateway(85, harness="claude-code", **gpt) as gateway:
        _, env = ClaudeAdapter().build_command(replace(ctx, gateway=gateway))
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "1050000"
    assert env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] == "85"


def test_pi_takes_the_gateway_s_compaction_cue_in_its_stride(tmp_path: Path) -> None:
    from harness_cascade.gateway import COMPACTION_POINT_MARKER

    ctx = context(tmp_path, harness="pi")
    seen: list = []
    ctx.events.subscribe(seen.append)
    result = HarnessResult(harness="pi", status="running", returncode=0, session_id="s")
    PiAdapter().consume_event(
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [],
                "stopReason": "error",
                "errorMessage": f"400 context_length_exceeded: {COMPACTION_POINT_MARKER}: ...",
            },
        },
        result,
        ctx,
    )
    assert result.status == "running"
    assert result.error.startswith("Pi did not compact when asked")
    assert not [event for event in seen if event.kind == EventKind.ERROR]
    PiAdapter().consume_event(
        {"type": "compaction_end", "reason": "overflow", "result": {"summary": "s"}},
        result,
        ctx,
    )
    assert result.error == ""
    PiAdapter().consume_event(
        {"type": "compaction_end", "reason": "overflow", "errorMessage": "summary call failed"},
        result,
        ctx,
    )
    assert result.error == "compaction failed: summary call failed"


def test_a_command_codex_refuses_shows_as_a_failed_tool(tmp_path: Path) -> None:
    ctx = context(tmp_path, harness="codex")
    seen: list = []
    ctx.events.subscribe(seen.append)
    result = HarnessResult(harness="codex", status="running", returncode=0, session_id="s")
    log = [
        "2026-10-01T17:20:00.281023Z ERROR codex_core::tools::router: error=exec_command failed"
        " for `/usr/bin/bash -lc \"cd /tmp/g && apply_patch <<'PATCH'",
        "rm -rf images",
        "python build.py",
        '"`: CreateProcess { message: "Rejected(\\"`/usr/bin/bash -lc \\\\\\"rm -rf images\\\\\\"`'
        ' rejected: rm -f style commands are not permitted. Use a safer approach\\")" }',
    ]
    for line in log:
        CodexAdapter().consume_plain(line, result, ctx)

    finished = [event for event in seen if event.kind == EventKind.TOOL_CALL_FINISHED]
    assert len(finished) == 1
    assert finished[0].data["failed"] is True
    assert finished[0].data["detail"] == (
        "Codex refused to run it: rm -f style commands are not permitted. Use a safer approach"
    )
    assert result.error == ""


def _tool_reply() -> CanonicalResponse:
    return CanonicalResponse(
        blocks=[Block("tool_call", call_id="call_1", name="bash", arguments={"command": "ls"})],
        usage=Usage(input_tokens=100, output_tokens=10),
    )


def _text_reply() -> CanonicalResponse:
    return CanonicalResponse(
        blocks=[Block("text", text="all done")],
        usage=Usage(input_tokens=100, output_tokens=10),
    )


# An agent loop reduced to its wire behaviour: ask, and ask again for as long
# as the reply calls a tool.
FAKE_AGENT = r"""
import json, sys, urllib.request
base, token = sys.argv[1], sys.argv[2]
while True:
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps({"model": "m", "messages": [{"role": "user", "content": "go"}],
                         "tools": [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}]}).encode(),
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        reply = json.loads(response.read())
    message = reply["choices"][0]["message"]
    print(json.dumps({"type": "reply", "tools": len(message.get("tool_calls") or [])}), flush=True)
    if not message.get("tool_calls"):
        print(json.dumps({"type": "message_end", "message": {"role": "assistant", "content": message.get("content") or ""}}), flush=True)
        break
"""


class FakeAgentAdapter(PiAdapter):
    def build_command(self, context: HarnessContext) -> tuple[list[str], dict[str, str]]:
        assert context.gateway is not None
        return (
            [sys.executable, "-c", FAKE_AGENT, context.gateway.base_url,
             context.gateway.config.local_token],
            dict(os.environ),
        )


def _scripted_gateway(replies: list[CanonicalResponse], switch_after: int | None,
                      events: EventBus) -> tuple[EmbeddedGateway, list[int]]:
    gateway = EmbeddedGateway(
        GatewayConfig(
            model="zai-org/GLM-5.3",
            effort="low",
            api_key="upstream-secret",
            local_token="local-secret",
            base_url="https://example.test/v1",
            stage="first",
            harness="pi",
            switch_after=switch_after,
        ),
        events,
    )
    forwarded: list[int] = []

    def forward(request):
        forwarded.append(len(forwarded) + 1)
        return replies[min(len(forwarded), len(replies)) - 1]

    gateway._forward = forward  # type: ignore[method-assign]
    return gateway, forwarded


@pytest.mark.parametrize("switch_after", [1, 3])
def test_the_request_past_the_switch_point_never_reaches_the_provider(
    tmp_path: Path, switch_after: int
) -> None:
    """The stage ends after N tool replies whatever the harness.

    The harness used to be killed only once its own stdout said N, by which
    point Pi had already sent its next request. The provider billed that call
    and its usage landed after the stage's total had been taken.
    """
    ctx = replace(
        context(tmp_path, harness="pi"),
        plan=StagePlan("pi", "low", switch_after=switch_after),
    )
    events: list = []
    ctx.events.subscribe(events.append)
    gateway, forwarded = _scripted_gateway([_tool_reply()], switch_after, ctx.events)
    with gateway:
        started = time.monotonic()
        result = FakeAgentAdapter().run(replace(ctx, gateway=gateway))
        elapsed = time.monotonic() - started
    assert forwarded == list(range(1, switch_after + 1))
    assert result.status == "switch_limit"
    assert result.turns == switch_after
    assert result.model_turns == switch_after
    assert result.error == ""
    switches = [e for e in events if e.kind == EventKind.SWITCH_TRIGGERED]
    assert [e.data["turns"] for e in switches] == [switch_after]
    assert elapsed < 15


def test_an_agent_that_finishes_before_the_switch_point_completes(tmp_path: Path) -> None:
    ctx = context(tmp_path, harness="pi")
    gateway, forwarded = _scripted_gateway(
        [_tool_reply(), _tool_reply(), _text_reply()], 3, ctx.events
    )
    with gateway:
        result = FakeAgentAdapter().run(replace(ctx, gateway=gateway))
    assert forwarded == [1, 2, 3]
    assert result.status == "completed"
    assert (result.turns, result.model_turns) == (2, 3)
    assert result.final_text == "all done"


def test_a_summary_stage_is_never_held(tmp_path: Path) -> None:
    ctx = replace(context(tmp_path, harness="pi"), summary=True)
    gateway, forwarded = _scripted_gateway([_tool_reply(), _text_reply()], None, ctx.events)
    with gateway:
        result = FakeAgentAdapter().run(replace(ctx, gateway=gateway))
    assert forwarded == [1, 2]
    assert result.status == "completed"


def test_a_tool_process_holding_the_pipe_does_not_hang_the_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A process the harness left behind, outside its process group, can keep
    stdout open after the harness itself has gone."""
    monkeypatch.setattr(harnesses_module, "DRAIN_SECONDS", 0.5)

    class LeavesAProcessBehind(PiAdapter):
        def build_command(self, context):
            script = (
                "import json, subprocess, sys\n"
                "subprocess.Popen(['sleep', '30'], start_new_session=True)\n"
                "print(json.dumps({'type': 'message_end', 'message': "
                "{'role': 'assistant', 'content': 'done'}}), flush=True)\n"
            )
            return [sys.executable, "-c", script], dict(os.environ)

    started = time.monotonic()
    result = LeavesAProcessBehind().run(context(tmp_path, harness="pi"))
    assert time.monotonic() - started < 10
    assert result.status == "completed"
    assert result.final_text == "done"


@pytest.mark.parametrize(
    ("adapter", "events", "expected"),
    [
        (
            PiAdapter(),
            [{"type": "message_end", "message": {
                "role": "assistant", "stopReason": "error",
                "errorMessage": "400 This content was flagged for possible cybersecurity risk"}}],
            "400 This content was flagged for possible cybersecurity risk",
        ),
        (
            ClaudeAdapter(),
            [{"type": "result", "subtype": "error_during_execution", "is_error": True,
              "result": "API Error: 400 context too long"}],
            "error_during_execution: API Error: 400 context too long",
        ),
        (
            ClaudeAdapter(),
            [{"type": "result", "subtype": "success", "is_error": True,
              "result": "API Error: 400 LLM endpoint returned 400: flagged"}],
            "API Error: 400 LLM endpoint returned 400: flagged",
        ),
        (
            CodexAdapter(),
            [{"type": "error", "message": "Reconnecting... 1/5"},
             {"type": "turn.failed", "error": {"message": "unexpected status 400: bad request"}}],
            "unexpected status 400: bad request",
        ),
    ],
)
def test_each_harness_reports_why_it_failed(
    tmp_path: Path, adapter, events: list[dict], expected: str
) -> None:
    ctx = context(tmp_path, harness=adapter.name)
    result = HarnessResult(harness=adapter.name, status="running", returncode=0,
                           session_id=ctx.session_id)
    for event in events:
        adapter.consume_event(event, result, ctx)
    assert result.error == expected


def test_a_failure_the_harness_does_not_explain_takes_the_gateway_error(
    tmp_path: Path,
) -> None:
    class SilentFailure(PiAdapter):
        def build_command(self, context):
            return [sys.executable, "-c", "raise SystemExit(3)"], dict(os.environ)

    with _gateway() as gateway:
        gateway.last_error = "gateway returned 400 to the agent: flagged"
        result = SilentFailure().run(context(tmp_path, harness="pi", gateway=gateway))
    assert result.status == "failed"
    assert result.error == "gateway returned 400 to the agent: flagged"


@pytest.mark.parametrize(
    ("exit_code", "expected"),
    [(-9, "killed by SIGKILL, which cascade did not send"),
     (143, "killed by SIGTERM, which cascade did not send"),
     (1, "plain crash line")],
)
def test_a_kill_cascade_did_not_send_is_named(
    tmp_path: Path, exit_code: int, expected: str
) -> None:
    """The agent's own `pkill -f` matching its argv, or the OOM killer."""

    class Dies(PiAdapter):
        def build_command(self, context):
            script = (
                "import os, signal, sys\n"
                "print('plain crash line', flush=True)\n"
                f"code = {exit_code}\n"
                "os.kill(os.getpid(), -code) if code < 0 else sys.exit(code)\n"
            )
            return [sys.executable, "-c", script], dict(os.environ)

    result = Dies().run(context(tmp_path, harness="pi"))
    assert result.status == "failed"
    assert result.error == expected


def test_a_reply_cut_off_before_any_text_explains_the_missing_answer(
    tmp_path: Path,
) -> None:
    class CutOff(PiAdapter):
        def build_command(self, context):
            event = {"type": "message_end", "message": {
                "role": "assistant", "stopReason": "length",
                "content": [{"type": "thinking", "thinking": "..."}]}}
            script = f"import json; print(json.dumps({event!r}))"
            return [sys.executable, "-c", script], dict(os.environ)

    result = CutOff().run(context(tmp_path, harness="pi"))
    assert result.status == "completed"
    assert result.final_text == ""
    assert result.error == "the reply reached the output token limit before any text"


def test_a_stage_that_succeeds_after_a_retry_carries_no_error(tmp_path: Path) -> None:
    class RetriedThenFine(CodexAdapter):
        def build_command(self, context):
            script = (
                "import json\n"
                "print(json.dumps({'type': 'error', 'message': 'Reconnecting... 1/5'}))\n"
                "print(json.dumps({'type': 'item.completed', 'item': "
                "{'type': 'agent_message', 'text': 'done'}}))\n"
            )
            return [sys.executable, "-c", script], dict(os.environ)

    result = RetriedThenFine().run(context(tmp_path, harness="codex"))
    assert result.status == "completed"
    assert result.error == ""


def test_pi_recovering_from_a_context_overflow_is_not_a_failure(tmp_path: Path) -> None:
    """X-05: Pi compacts after the provider's overflow error and resends."""
    ctx = context(tmp_path, harness="pi")
    result = HarnessResult(harness="pi", status="running", returncode=0, session_id="s")
    adapter = PiAdapter()
    adapter.consume_event(
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [],
                "stopReason": "error",
                "errorMessage": "400 maximum context length is 48000 tokens",
            },
        },
        result,
        ctx,
    )
    assert result.status == "failed"
    adapter.consume_event(
        {"type": "compaction_end", "reason": "overflow", "result": {"summary": "s"}},
        result,
        ctx,
    )
    assert (result.status, result.error) == ("running", "")


def test_stopping_a_harness_stops_commands_it_started_in_their_own_session(
    tmp_path: Path,
) -> None:
    """X-17: Codex runs each tool command outside the harness's process group."""
    import subprocess

    from harness_cascade.harnesses import _kill_process_group

    pid_file = tmp_path / "pid"
    proc = subprocess.Popen(
        ["bash", "-c", f"setsid sleep 300 & echo $! > {pid_file}; wait"],
        start_new_session=True,
        text=True,
    )
    for _ in range(100):
        if pid_file.exists() and pid_file.read_text().strip():
            break
        time.sleep(0.05)
    escaped = int(pid_file.read_text())
    assert os.getsid(escaped) != os.getsid(proc.pid)
    _kill_process_group(proc)
    time.sleep(0.2)
    try:
        with open(f"/proc/{escaped}/stat", encoding="utf-8") as handle:
            state = handle.read().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        state = "gone"
    assert state in {"gone", "Z", "X"}


LONG_PROMPT = "Fix the parser — every case.\n" * 8_000


@pytest.mark.parametrize(
    ("adapter", "harness", "tail"),
    [
        (ClaudeAdapter(), "claude-code", ["-p"]),
        (CodexAdapter(), "codex", ["-"]),
        (PiAdapter(), "pi", ["--session-id", "00000000-0000-4000-8000-000000000001"]),
    ],
)
def test_a_prompt_too_long_for_argv_goes_to_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, adapter, harness: str, tail: list[str]
) -> None:
    monkeypatch.setenv("CASCADE_CODEX_CATALOG", "0")
    assert len(LONG_PROMPT.encode()) > harnesses_module.MAX_ARG_STRLEN
    with _gateway() as gateway:
        short, _ = adapter.build_command(
            context(tmp_path, harness=harness, gateway=gateway, prompt="Fix auth.py")
        )
        long, _ = adapter.build_command(
            context(tmp_path, harness=harness, gateway=gateway, prompt=LONG_PROMPT)
        )
    assert short[-1] == "Fix auth.py"
    assert long[-len(tail):] == tail
    assert max(len(part.encode()) for part in long) < harnesses_module.PROMPT_ARG_BYTES


def test_the_harness_reads_the_whole_long_prompt_from_stdin(tmp_path: Path) -> None:
    class EchoesStdin(PiAdapter):
        def build_command(self, context):
            script = (
                "import hashlib, json, sys\n"
                "data = sys.stdin.buffer.read()\n"
                "print(json.dumps({'type': 'message_end', 'message': {'role': 'assistant',"
                " 'content': hashlib.sha256(data).hexdigest()}}), flush=True)\n"
            )
            return [sys.executable, "-c", script], dict(os.environ)

    import hashlib

    result = EchoesStdin().run(context(tmp_path, harness="pi", prompt=LONG_PROMPT))
    assert result.status == "completed"
    assert result.final_text == hashlib.sha256(LONG_PROMPT.encode()).hexdigest()


def test_a_short_prompt_leaves_the_harness_stdin_empty(tmp_path: Path) -> None:
    class EchoesStdin(PiAdapter):
        def build_command(self, context):
            script = (
                "import json, sys\n"
                "print(json.dumps({'type': 'message_end', 'message': {'role': 'assistant',"
                " 'content': repr(sys.stdin.read())}}), flush=True)\n"
            )
            return [sys.executable, "-c", script], dict(os.environ)

    result = EchoesStdin().run(context(tmp_path, harness="pi", prompt="Fix auth.py"))
    assert result.final_text == "''"


def test_an_argument_over_the_kernel_limit_is_a_clear_error(tmp_path: Path) -> None:
    class OversizeArgument(PiAdapter):
        def build_command(self, context):
            return [sys.executable, "-c", "pass", "x" * 200_000], dict(os.environ)

    with pytest.raises(HarnessError, match="200000 bytes and Linux allows 131072"):
        OversizeArgument().run(context(tmp_path, harness="pi"))
