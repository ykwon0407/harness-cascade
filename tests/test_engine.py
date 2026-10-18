import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import pytest

import harness_cascade.engine as engine_module
from harness_cascade.config import HandoffPlan, RunPlan, StagePlan
from harness_cascade.credentials import CredentialSet
from harness_cascade.engine import (
    HandoffEngine,
    build_second_prompt,
    packaged_prompt,
    workspace_snapshot,
)
from harness_cascade.events import EventBus, EventKind
from harness_cascade.harnesses import HarnessResult


class FakeAdapter:
    def __init__(self, name: str, calls: list[tuple[str, str, bool]]) -> None:
        self.name = name
        self.calls = calls

    def run(self, context):
        self.calls.append((self.name, context.stage, context.summary))
        text = "handoff summary" if context.summary else f"{self.name} final"
        return HarnessResult(
            harness=self.name,
            status="completed",
            returncode=0,
            session_id=context.session_id,
            turns=2,
            final_text=text,
            commands=["pytest"],
            usage={
                "input_tokens": 10,
                "output_tokens": 5,
                "reasoning_tokens": 2,
            },
        )


@pytest.mark.parametrize("first", ["pi", "claude-code", "codex"])
@pytest.mark.parametrize("second", ["pi", "claude-code", "codex"])
def test_all_nine_handoff_combinations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first: str,
    second: str,
) -> None:
    calls: list[tuple[str, str, bool]] = []
    monkeypatch.setattr(
        engine_module,
        "adapter_for",
        lambda name: FakeAdapter(name, calls),
    )
    plan = RunPlan(
        first=StagePlan(first, "low", switch_after=2),
        second=StagePlan(second, "high"),
        workspace=tmp_path,
        model="zai-org/GLM-5.3",
    )
    bus = EventBus("matrix")
    events = []
    bus.subscribe(events.append)
    engine = HandoffEngine(
        plan,
        CredentialSet("together", "secret", "test"),
        events=bus,
        runs_root=tmp_path / "runs",
    )
    monkeypatch.setattr(engine, "_gateway", lambda *args, **kwargs: nullcontext(None))
    result = engine.run("Fix auth.py")
    assert result["status"] == "completed"
    assert result["final_text"] == f"{second} final"
    assert result["handoff"]["final_text"] == "handoff summary"
    assert result["usage"]["total_tokens"] == 45
    assert result["usage"]["reasoning_tokens"] == 6
    assert result["configuration"]["model"] == "zai-org/GLM-5.3"
    assert (engine.run_dir / "configuration.json").is_file()
    assert calls == [
        (first, "first", False),
        (first, "handoff", True),
        (second, "second", False),
    ]
    assert any(event.kind == EventKind.HANDOFF_SUMMARY for event in events)


def test_failed_handoff_does_not_start_second_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FailingHandoffAdapter:
        name = "pi"

        def run(self, context):
            calls.append(context.stage)
            if context.summary:
                return HarnessResult(
                    harness=self.name,
                    status="failed",
                    returncode=1,
                    session_id=context.session_id,
                    final_text="API Error: invalid tool history",
                )
            return HarnessResult(
                harness=self.name,
                status="switch_limit",
                returncode=1,
                session_id=context.session_id,
                final_text="partial work",
            )

    monkeypatch.setattr(
        engine_module,
        "adapter_for",
        lambda _: FailingHandoffAdapter(),
    )
    plan = RunPlan(
        first=StagePlan("pi", "low", switch_after=1),
        second=StagePlan("codex", "low"),
        workspace=tmp_path,
        model="mock/model",
    )
    engine = HandoffEngine(
        plan,
        CredentialSet("together", "secret", "test"),
        runs_root=tmp_path / "runs",
    )
    monkeypatch.setattr(engine, "_gateway", lambda *args, **kwargs: nullcontext(None))

    with pytest.raises(engine_module.EngineError, match="handoff summary.*failed"):
        engine.run("Fix the task")
    assert calls == ["first", "handoff"]


def test_a_summary_written_only_as_reasoning_still_reaches_the_second_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GLM wrote 65 of its before-fix summaries entirely as reasoning."""
    calls: list[tuple[str, str, bool]] = []

    class ReasoningSummaryAdapter(FakeAdapter):
        def run(self, context):
            result = super().run(context)
            if context.summary:
                result.final_text = ""
                result.final_reasoning = "# Transfer Record\nparser done, tests pending"
            return result

    monkeypatch.setattr(
        engine_module,
        "adapter_for",
        lambda name: ReasoningSummaryAdapter(name, calls),
    )
    plan = RunPlan(
        first=StagePlan("pi", "high", switch_after=2),
        second=StagePlan("codex", "high"),
        workspace=tmp_path,
        model="zai-org/GLM-5.3",
    )
    bus = EventBus("reasoning-summary")
    events = []
    bus.subscribe(events.append)
    engine = HandoffEngine(
        plan,
        CredentialSet("together", "secret", "test"),
        events=bus,
        runs_root=tmp_path / "runs",
    )
    monkeypatch.setattr(engine, "_gateway", lambda *args, **kwargs: nullcontext(None))
    result = engine.run("Fix auth.py")
    assert result["status"] == "completed"
    assert "parser done, tests pending" in (engine.run_dir / "second-prompt.md").read_text()
    assert any(event.kind == EventKind.WARNING for event in events)


def test_reasoning_that_ran_to_the_cap_is_cut_to_fit_one_argument(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DeepSeek passed 222k characters of reasoning to claude, which failed with E2BIG."""
    calls: list[tuple[str, str, bool]] = []

    class CappedReasoningAdapter(FakeAdapter):
        def run(self, context):
            result = super().run(context)
            if context.summary:
                result.final_text = ""
                result.final_reasoning = "# Transfer Record\nencoder half done\n" + "x" * 200_000
            return result

    monkeypatch.setattr(
        engine_module,
        "adapter_for",
        lambda name: CappedReasoningAdapter(name, calls),
    )
    plan = RunPlan(
        first=StagePlan("pi", "high", switch_after=2),
        second=StagePlan("claude-code", "high"),
        workspace=tmp_path,
        model="deepseek-ai/DeepSeek-V4.1",
    )
    engine = HandoffEngine(
        plan,
        CredentialSet("together", "secret", "test"),
        events=EventBus("capped-reasoning"),
        runs_root=tmp_path / "runs",
    )
    monkeypatch.setattr(engine, "_gateway", lambda *args, **kwargs: nullcontext(None))
    engine.run("Fix auth.py")
    second_prompt = (engine.run_dir / "second-prompt.md").read_text()
    assert "encoder half done" in second_prompt
    assert len(second_prompt.encode()) < 128 * 1024


def test_second_prompt_contains_mechanical_state() -> None:
    prompt = build_second_prompt(
        "Fix auth.py",
        "Changed validation.",
        {
            "cwd": "/repo",
            "git_status": " M auth.py",
            "git_diff_stat": "auth.py | 2 +-",
            "recent_commands": ["pytest"],
        },
    )
    assert "Fix auth.py" in prompt
    assert "Changed validation." in prompt
    assert "M auth.py" in prompt
    assert "pytest" in prompt
    assert "hints, not ground truth" in prompt


def test_custom_continue_prompt_template_is_substituted() -> None:
    prompt = build_second_prompt(
        "Original objective",
        "Transfer facts",
        {
            "cwd": "/repo",
            "git_status": " M file.txt",
            "git_diff_stat": "file.txt | 1 +",
            "recent_commands": ["check"],
        },
        template=(
            "$original_prompt\n$handoff_summary\n$workspace\n"
            "$git_status\n$git_diff_stat\n$recent_commands"
        ),
    )
    assert prompt.splitlines() == [
        "Original objective",
        "Transfer facts",
        "/repo",
        " M file.txt",
        "file.txt | 1 +",
        "- check",
    ]


def test_non_git_workspace_is_not_counted_as_a_changed_file(tmp_path: Path) -> None:
    snapshot = workspace_snapshot(tmp_path, [])
    assert snapshot["changed_files"] == 0
    assert snapshot["git_status"] == "(not a git repository)"
    assert snapshot["git_diff_stat"] == "(not available)"


def test_packaged_default_prompts_are_available() -> None:
    summary = packaged_prompt("handoff-summary.md")
    assert "Report observable" in summary
    assert "facts only" in summary
    assert "$original_prompt" in packaged_prompt("handoff-continue.md")


class ScriptedAdapter:
    """First stage ends with `first_status`; every other stage completes."""

    def __init__(self, calls: list[str], first_status: str, error: str = "") -> None:
        self.calls = calls
        self.first_status = first_status
        self.error = error
        self.name = "pi"

    def run(self, context):
        self.calls.append(context.stage)
        first = context.stage == "first"
        return HarnessResult(
            harness=self.name,
            status=self.first_status if first else "completed",
            returncode=0,
            session_id=context.session_id,
            final_text="summary" if context.summary else f"{context.stage} answer",
            error=self.error if first else "",
        )


def _engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, adapter, *,
            skip_second: bool = False, events: list | None = None) -> HandoffEngine:
    monkeypatch.setattr(engine_module, "adapter_for", lambda _: adapter)
    plan = RunPlan(
        first=StagePlan("pi", "low", switch_after=5),
        second=StagePlan("claude-code", "low"),
        handoff=HandoffPlan(skip_second_if_finished=skip_second),
        workspace=tmp_path,
        model="zai-org/GLM-5.3",
    )
    bus = EventBus("scripted")
    if events is not None:
        bus.subscribe(events.append)
    engine = HandoffEngine(
        plan, CredentialSet("together", "secret", "test"), events=bus,
        runs_root=tmp_path / "runs",
    )
    monkeypatch.setattr(engine, "_gateway", lambda *args, **kwargs: nullcontext(None))
    return engine


@pytest.mark.parametrize(
    ("skip_second", "first_status", "stages", "skipped"),
    [
        (False, "completed", ["first", "handoff", "second"], False),
        (True, "completed", ["first"], True),
        (True, "switch_limit", ["first", "handoff", "second"], False),
    ],
)
def test_the_second_harness_is_skipped_only_when_asked_and_first_finished(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    skip_second: bool,
    first_status: str,
    stages: list[str],
    skipped: bool,
) -> None:
    calls: list[str] = []
    events: list = []
    engine = _engine(tmp_path, monkeypatch, ScriptedAdapter(calls, first_status),
                     skip_second=skip_second, events=events)
    result = engine.run("Fix auth.py")
    assert calls == stages
    assert result["status"] == "completed"
    assert result["second_skipped"] is skipped
    assert result["final_text"] == ("first answer" if skipped else "second answer")
    assert (result["second"] is None) is skipped
    assert any(e.kind == EventKind.HANDOFF_SKIPPED for e in events) is skipped
    assert result["configuration"]["skip_second_if_finished"] is skip_second


def test_a_failed_stage_says_why(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    cause = "400 This content was flagged for possible cybersecurity risk"
    engine = _engine(tmp_path, monkeypatch, ScriptedAdapter(calls, "failed", cause))
    with pytest.raises(engine_module.EngineError) as raised:
        engine.run("Fix auth.py")
    assert str(raised.value) == f"first harness pi failed (exit 0): {cause}"
    written = json.loads((engine.run_dir / "result.json").read_text())
    assert written["error"].endswith(cause)
    assert written["first"]["error"] == cause


def test_a_model_call_that_finishes_after_its_stage_is_still_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class LateCall(ScriptedAdapter):
        def run(self, context):
            usage = {"input_tokens": 100 if context.stage == "first" else 40,
                     "output_tokens": 1}
            bus = context.events
            bus.emit(EventKind.MODEL_CALL_FINISHED, stage=context.stage, usage=usage)
            if context.stage == "second":
                bus.emit(EventKind.MODEL_CALL_FINISHED, stage="first",
                         usage={"input_tokens": 7, "output_tokens": 1})
            return super().run(context)

    events: list = []
    engine = _engine(tmp_path, monkeypatch, LateCall([], "switch_limit"), events=events)
    result = engine.run("Fix auth.py")
    assert result["first"]["usage"]["input_tokens"] == 107
    assert result["usage"]["input_tokens"] == 107 + 40 + 40
    assert any(
        e.kind == EventKind.WARNING and "after its stage ended" in e.data["message"]
        for e in events
    )


def test_solo_empty_generation_is_an_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptyAdapter:
        name = "pi"

        def run(self, context):
            return HarnessResult(
                harness="pi",
                status="completed",
                returncode=0,
                session_id=context.session_id,
                final_text="",
            )

    monkeypatch.setattr(engine_module, "adapter_for", lambda _: EmptyAdapter())
    plan = RunPlan(
        first=StagePlan("pi", "low"),
        workspace=tmp_path,
        model="zai-org/GLM-5.3",
    )
    engine = HandoffEngine(
        plan,
        CredentialSet("together", "secret", "test"),
        runs_root=tmp_path / "runs",
    )
    monkeypatch.setattr(engine, "_gateway", lambda *args, **kwargs: nullcontext(None))
    with pytest.raises(engine_module.EngineError, match="without an assistant result"):
        engine.run("Fix auth.py")


def test_a_solo_stage_stopped_at_a_switch_point_is_not_a_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """X-10: a truncated agent's last narration ("Step 3:") is not an answer."""
    calls: list[str] = []
    engine = _engine(tmp_path, monkeypatch, ScriptedAdapter(calls, "switch_limit"))
    engine.plan = replace(engine.plan, second=None)
    with pytest.raises(engine_module.EngineError, match="stopped at the switch point"):
        engine.run("Fix auth.py")
    assert json.loads((engine.run_dir / "result.json").read_text())["status"] == "failed"


def test_an_interrupted_run_still_records_what_it_spent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """X-18: Ctrl-C or a benchmark's SIGTERM in the middle of a stage."""

    class Interrupted(ScriptedAdapter):
        def run(self, context):
            context.events.emit(
                EventKind.MODEL_CALL_FINISHED,
                stage=context.stage,
                harness=self.name,
                usage={"input_tokens": 100, "output_tokens": 7},
            )
            raise KeyboardInterrupt

    engine = _engine(tmp_path, monkeypatch, Interrupted([], "completed"))
    with pytest.raises(KeyboardInterrupt):
        engine.run("Fix auth.py")
    written = json.loads((engine.run_dir / "result.json").read_text())
    assert written["status"] == "interrupted"
    assert written["usage"]["input_tokens"] == 100


def test_a_workspace_that_does_not_exist_is_not_created(tmp_path: Path) -> None:
    """X-25: a mistyped path or an unmounted volume must not "complete"."""
    missing = tmp_path / "does" / "not" / "exist"
    plan = RunPlan(first=StagePlan("pi", "low"), workspace=missing, model="m")
    with pytest.raises(engine_module.EngineError, match="not a directory"):
        HandoffEngine(plan, CredentialSet("together", "secret", "test"))
    assert not missing.exists()


def test_events_never_carry_the_key() -> None:
    bus = EventBus("redact")
    seen: list = []
    bus.subscribe(seen.append)
    bus.redact("sk-test-0123456789")
    bus.emit(EventKind.WARNING, message="bad key sk-test-0123456789", nested={"a": ["sk-test-0123456789"]})
    assert "sk-test-0123456789" not in seen[0].to_json()


def test_a_huge_workspace_change_still_makes_a_bounded_second_prompt(tmp_path: Path) -> None:
    import subprocess

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.test")
    git("config", "user.name", "t")
    for index in range(600):
        (tmp_path / f"tracked{index}.txt").write_text("a\n")
    git("add", ".")
    git("commit", "-qm", "base")
    for index in range(600):
        (tmp_path / f"tracked{index}.txt").write_text("b\n")
    for index in range(20_000):
        (tmp_path / f"untracked{index}.txt").write_text("")
    heredoc = "cat > big.py <<'EOF'\n" + "x = 1\n" * 2_000 + "EOF"

    snapshot = workspace_snapshot(tmp_path, [heredoc] * 25)
    assert snapshot["changed_files"] == 20_600
    status = snapshot["git_status"].splitlines()
    assert len(status) == engine_module.SNAPSHOT_LINES + 1
    assert status[-1] == "... and 20400 more lines"
    stat = snapshot["git_diff_stat"].splitlines()
    assert stat[-2] == "... and 400 more lines"
    assert "600 files changed" in stat[-1]
    assert len(snapshot["recent_commands"]) == 20
    assert all(
        len(item) < engine_module.COMMAND_CHARS + 50 for item in snapshot["recent_commands"]
    )
    assert snapshot["recent_commands"][0].startswith("cat > big.py")
    assert snapshot["recent_commands"][0].endswith("EOF")

    summary = "progress notes\n" * 20_000
    prompt = build_second_prompt("Original objective", summary, snapshot)
    assert len(prompt.encode()) < 100_000
    assert "characters cut" in prompt


def test_clip_text_keeps_both_ends() -> None:
    from harness_cascade.engine import clip_text

    assert clip_text("short", 10) == "short"
    clipped = clip_text("A" * 50 + "Z" * 50, 40)
    assert clipped.startswith("A" * 30) and clipped.endswith("Z" * 10)
    assert "[... 60 characters cut ...]" in clipped
