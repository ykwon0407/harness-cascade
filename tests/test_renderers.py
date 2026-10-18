from io import StringIO

from harness_cascade.events import EventBus, EventKind
from harness_cascade.renderers import PrettyRenderer, print_final_result


class TtyBuffer(StringIO):
    def isatty(self) -> bool:
        return True


def test_pretty_renderer_shows_settings_and_distinct_stage_colors() -> None:
    stream = TtyBuffer()
    bus = EventBus("run123")
    bus.subscribe(PrettyRenderer(stream, color="always"))
    bus.emit(
        EventKind.RUN_STARTED,
        workspace="/tmp/demo",
        provider="together",
        model="zai-org/GLM-5.3",
        first={"harness": "pi", "effort": "low", "switch_after": 1},
        second={"harness": "claude-code", "effort": "low"},
        output_format="pretty",
        artifacts="/tmp/demo/.cascade/runs/run123",
        summary_prompt="packaged default",
        continue_prompt="packaged default",
    )
    for stage, harness in (
        ("first", "pi"),
        ("handoff", "pi"),
        ("second", "claude-code"),
    ):
        bus.emit(
            EventKind.STAGE_STARTED,
            stage=stage,
            harness=harness,
            effort="low",
        )
        bus.emit(
            EventKind.ASSISTANT_TEXT,
            stage=stage,
            harness=harness,
            text=stage,
        )
        bus.emit(
            EventKind.STAGE_FINISHED,
            stage=stage,
            harness=harness,
            status="completed",
            turns=1,
            duration_seconds=1.25,
            usage={
                "input_tokens": 10,
                "output_tokens": 5,
                "reasoning_tokens": 2,
                "total_tokens": 15,
            },
        )
    bus.emit(
        EventKind.WORKSPACE_SNAPSHOT,
        stage="handoff",
        changed_files=1,
        git_status=" M auth.py",
    )
    bus.emit(
        EventKind.RUN_FINISHED,
        duration_seconds=3.75,
        usage={"input_tokens": 30, "output_tokens": 15, "total_tokens": 45},
        stage_usage={
            stage: {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            for stage in ("first", "handoff", "second")
        },
        artifacts="/tmp/demo/.cascade/runs/run123",
    )

    output = stream.getvalue()
    assert "CASCADE START" in output
    assert "Session ID  run123" in output
    assert "pi:low  →  claude-code:low" in output
    assert "Workspace" not in output
    assert "Prompts" not in output
    assert "FIRST HARNESS" in output
    assert "HANDOFF SUMMARY" in output
    assert "SECOND HARNESS" in output
    assert "WORKSPACE CHECKPOINT" not in output
    assert "CASCADE COMPLETED" in output
    assert "45 total" in output
    assert "work units" not in output
    assert "╰─ completed · 1.2s" in output
    assert "\033[96m" in output
    assert "\033[93m" in output
    assert "\033[92m" in output
    assert "\033[95m" in output


def test_pretty_mode_does_not_dump_result_json(
    monkeypatch,
) -> None:
    stream = StringIO()
    monkeypatch.setattr("sys.stdout", stream)
    print_final_result(
        {"run_id": "run123", "status": "completed", "final_text": "done"},
        output_format="pretty",
    )
    assert stream.getvalue() == ""


def test_pretty_renderer_numbers_every_turn_up_to_the_switch_point() -> None:
    # Turns 2 and 3 poll a running command and turn 5 asks for one Codex
    # refuses: Codex reports no tool for any of them.
    stream = StringIO()
    renderer = PrettyRenderer(stream, color="never")
    bus = EventBus("run123")
    bus.subscribe(renderer)
    bus.emit(EventKind.STAGE_STARTED, stage="first", harness="codex", effort="high")
    calls = {1: ["exec_command"], 2: ["write_stdin"], 3: ["write_stdin"], 4: ["exec_command"], 5: ["exec_command"]}
    for reply, tools in calls.items():
        bus.emit(
            EventKind.TURN_STARTED,
            stage="first",
            harness="codex",
            turn=reply,
            limit=5,
            tools=tools,
        )
        if reply in (1, 4):
            bus.emit(
                EventKind.TOOL_CALL_STARTED,
                stage="first",
                harness="codex",
                name="command_execution",
                detail="ls",
            )
        if reply in (3, 4):
            bus.emit(
                EventKind.TOOL_CALL_FINISHED,
                stage="first",
                harness="codex",
                name="command_execution",
                detail="ls",
                failed=False,
            )
    bus.emit(
        EventKind.TOOL_CALL_FINISHED,
        stage="first",
        harness="codex",
        name="exec_command",
        detail="Codex refused to run it: rm -f style commands are not permitted",
        failed=True,
    )
    bus.emit(EventKind.SWITCH_TRIGGERED, stage="first", harness="codex", turns=5)

    output = stream.getvalue()
    headers = [line for line in output.splitlines() if "▸ turn" in line]
    assert headers == [
        "│ ▸ turn 1/5 · exec_command",
        "│ ▸ turn 2/5 · write_stdin",
        "│ ▸ turn 3/5 · write_stdin",
        "│ ▸ turn 4/5 · exec_command",
        "│ ▸ turn 5/5 · exec_command",
    ]
    assert "tool" not in output
    assert "✓ command_execution finished" in output
    assert "✗ exec_command failed" in output
    assert "Codex refused to run it" in output
    assert "boundary reached after 5 turns" in output


def test_pretty_renderer_says_why_a_model_call_attempt_failed() -> None:
    stream = StringIO()
    bus = EventBus("run123")
    bus.subscribe(PrettyRenderer(stream, color="never"))
    bus.emit(
        EventKind.WARNING,
        stage="second",
        harness="codex",
        message="model call attempt failed",
        attempt=1,
        of=4,
        reason="ReadError: [Errno 104] Connection reset by peer",
        seconds=0.538,
        retrying=True,
    )
    assert stream.getvalue().strip() == (
        "cascade ! model call attempt 1 of 4 failed after 0.5s: "
        "ReadError: [Errno 104] Connection reset by peer; retrying"
    )


def test_claude_code_stopping_itself_at_the_switch_point_reads_the_same() -> None:
    stream = StringIO()
    bus = EventBus("run123")
    bus.subscribe(PrettyRenderer(stream, color="never"))
    bus.emit(EventKind.STAGE_STARTED, stage="first", harness="claude-code", effort="low")
    bus.emit(
        EventKind.STAGE_FINISHED,
        stage="first",
        harness="claude-code",
        status="switch_limit",
        turns=3,
        duration_seconds=0.9,
    )
    assert stream.getvalue().splitlines()[-1] == (
        "╰─ configured handoff boundary reached after 3 turns"
    )


def test_a_stage_that_finishes_on_its_own_says_after_how_many_turns() -> None:
    stream = StringIO()
    bus = EventBus("run123")
    bus.subscribe(PrettyRenderer(stream, color="never"))
    for stage, turns in (("first", 4), ("handoff", 0)):
        bus.emit(EventKind.STAGE_STARTED, stage=stage, harness="pi", effort="low")
        bus.emit(
            EventKind.STAGE_FINISHED,
            stage=stage,
            harness="pi",
            status="completed",
            turns=turns,
            duration_seconds=41.4,
        )
    lines = stream.getvalue().splitlines()
    assert lines[2:4] == ["│ finished at 4 turns", "╰─ completed · 41.4s"]
    assert "finished at 0" not in stream.getvalue()

    colored = StringIO()
    bus = EventBus("run123")
    bus.subscribe(PrettyRenderer(colored, color="always"))
    bus.emit(EventKind.STAGE_STARTED, stage="second", harness="pi", effort="low")
    bus.emit(EventKind.STAGE_FINISHED, stage="second", harness="pi", status="completed", turns=1)
    assert "\033[1m\033[97m│ finished at 1 turn\033[0m" in colored.getvalue()


def test_pretty_renderer_numbers_turns_without_a_limit() -> None:
    stream = StringIO()
    renderer = PrettyRenderer(stream, color="never")
    bus = EventBus("run123")
    bus.subscribe(renderer)
    bus.emit(EventKind.STAGE_STARTED, stage="second", harness="pi", effort="high")
    bus.emit(
        EventKind.TURN_STARTED,
        stage="second",
        harness="pi",
        turn=1,
        limit=None,
        tools=["bash", "read"],
    )
    assert "│ ▸ turn 1 · bash, read" in stream.getvalue()


def test_pretty_renderer_highlights_the_tool_name() -> None:
    stream = StringIO()
    renderer = PrettyRenderer(stream, color="always")
    bus = EventBus("run123")
    bus.subscribe(renderer)
    bus.emit(
        EventKind.STAGE_STARTED,
        stage="first",
        harness="claude-code",
        effort="low",
    )
    bus.emit(
        EventKind.TOOL_CALL_STARTED,
        stage="first",
        harness="claude-code",
        turn=1,
        name="Edit",
        detail="{}",
    )
    bus.emit(
        EventKind.TOOL_CALL_FINISHED,
        stage="first",
        harness="claude-code",
        name="tool",
        detail="updated",
    )

    output = stream.getvalue()
    highlighted = "\033[1m\033[97mEdit"
    assert output.count(highlighted) == 2


def test_pretty_renderer_wraps_more_tool_output() -> None:
    stream = StringIO()
    renderer = PrettyRenderer(stream, color="never")
    bus = EventBus("run123")
    bus.subscribe(renderer)
    bus.emit(
        EventKind.STAGE_STARTED,
        stage="first",
        harness="claude-code",
        effort="low",
    )
    bus.emit(
        EventKind.TOOL_CALL_STARTED,
        stage="first",
        harness="claude-code",
        name="WebSearch",
        detail='{"query":"boosting"}',
    )
    detail = "result-start " + ("useful evidence " * 30) + "result-end"
    bus.emit(
        EventKind.TOOL_CALL_FINISHED,
        stage="first",
        harness="claude-code",
        name="tool",
        detail=detail,
    )

    output = stream.getvalue()
    assert "result-start" in output
    assert "result-end" in output
    assert output.count("│   ↳") > 1
