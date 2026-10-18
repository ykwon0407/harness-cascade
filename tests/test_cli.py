import json
from pathlib import Path

import pytest

from harness_cascade.cli import _prompt, run
from harness_cascade.config import PlanError


def test_prompt_file_preserves_shell_sensitive_math(tmp_path: Path) -> None:
    prompt_file = tmp_path / "task.md"
    prompt_file.write_text(
        r"Rates are $2$ and $7$; find $\frac{m}{n}$ and $m+n$.",
        encoding="utf-8",
    )
    assert _prompt(None, prompt_file) == (
        r"Rates are $2$ and $7$; find $\frac{m}{n}$ and $m+n$."
    )


def test_prompt_and_prompt_file_are_mutually_exclusive(tmp_path: Path) -> None:
    prompt_file = tmp_path / "task.md"
    prompt_file.write_text("task", encoding="utf-8")
    with pytest.raises(PlanError, match="either"):
        _prompt("inline", prompt_file)


def test_invalid_model_effort_stops_before_credentials(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        run(
            [
                "--first",
                "pi:xhigh",
                "--provider",
                "together",
                "--model",
                "openai/gpt-oss-120b",
                "--dry-run",
                "-p",
                "test",
            ]
        )
    error = capsys.readouterr().err
    assert "does not support effort 'xhigh'" in error
    assert "available: low, medium, high" in error


def test_valid_profile_is_in_dry_run_without_credentials(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        run(
            [
                "--first",
                "codex:max",
                "--provider",
                "openai",
                "--model",
                "gpt-5.6",
                "--dry-run",
                "-p",
                "test",
            ]
        )
        == 0
    )
    assert '"model_profile": "packaged:model_profiles.yaml"' in capsys.readouterr().out


def test_provider_is_inferred_and_together_is_preferred(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        run(
            [
                "--first",
                "pi:low",
                "--model",
                "zai-org/GLM-5.3",
                "--dry-run",
                "-p",
                "test",
            ]
        )
        == 0
    )
    assert '"provider": "together"' in capsys.readouterr().out


def _dry_run(capsys: pytest.CaptureFixture[str], *extra: str) -> dict:
    assert run(["--first", "pi:low", "--model", "zai-org/GLM-5.3", "--dry-run",
                "-p", "test", *extra]) == 0
    return json.loads(capsys.readouterr().out)


def test_the_second_harness_runs_unless_told_to_skip(
    capsys: pytest.CaptureFixture[str],
) -> None:
    handoff = ("--second", "cc:low", "--switch-after", "5")
    assert _dry_run(capsys, *handoff)["skip_second_if_finished"] is False
    assert _dry_run(capsys, *handoff, "--skip-second-if-finished")[
        "skip_second_if_finished"
    ] is True
    assert _dry_run(capsys, *handoff, "--no-skip-second-if-finished")[
        "skip_second_if_finished"
    ] is False


def test_skipping_the_second_harness_needs_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        run(["--first", "pi:low", "--model", "zai-org/GLM-5.3", "--dry-run",
             "--skip-second-if-finished", "-p", "test"])
    assert "needs --second" in capsys.readouterr().err


def test_every_harness_compacts_at_85_percent_unless_told_otherwise(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _dry_run(capsys)["compact_at_percent"] == 85
    assert _dry_run(capsys, "--compact-at", "70")["compact_at_percent"] == 70
    assert _dry_run(capsys, "--compact-at", "0")["compact_at_percent"] is None
    with pytest.raises(SystemExit):
        run(["--first", "pi:low", "--model", "zai-org/GLM-5.3", "--dry-run",
             "--compact-at", "100", "-p", "test"])
    assert "whole percent" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra",
    [
        ["--switch-after", "3"],
        ["--workspace", "/does/not/exist"],
        ["--plan", "examples/SWITCH.yaml"],
    ],
)
def test_runs_that_would_silently_do_something_else_are_refused(extra: list[str]) -> None:
    """X-10 (a solo cap), X-25 (a missing workspace) and the unfinished --plan."""
    with pytest.raises(SystemExit):
        run(["--first", "pi:low", "--model", "zai-org/GLM-5.3", "--dry-run",
             "-p", "test", *extra])


@pytest.mark.parametrize(
    ("provider", "endpoint", "warned"),
    [
        ("together", "https://api.together.ai/v1", []),
        ("together", "http://203.0.113.7/v1", ["203.0.113.7, not api.together.ai", "plain HTTP"]),
        ("openai-compatible", "http://127.0.0.1:8000/v1", []),
        ("openai-compatible", "https://llm.example/v1", []),
    ],
)
def test_a_key_sent_somewhere_unexpected_is_announced(
    capsys: pytest.CaptureFixture[str], provider: str, endpoint: str, warned: list[str]
) -> None:
    from harness_cascade.cli import _warn_about_endpoint

    _warn_about_endpoint(provider, endpoint)
    err = capsys.readouterr().err
    assert len(err.splitlines()) == len(warned)
    assert all(text in err for text in warned)
