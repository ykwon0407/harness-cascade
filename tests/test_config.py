from pathlib import Path

import pytest

from harness_cascade.config import PlanError, load_plan, parse_harness_spec


@pytest.mark.parametrize(
    ("value", "harness", "effort"),
    [
        ("pi-low", "pi", "low"),
        ("CC-high", "claude-code", "high"),
        ("claude-code:xhigh", "claude-code", "xhigh"),
        ("codex:max", "codex", "max"),
    ],
)
def test_parse_harness_spec(value: str, harness: str, effort: str) -> None:
    result = parse_harness_spec(value, switch_after=3)
    assert result.harness == harness
    assert result.effort == effort
    assert result.switch_after == 3


def test_rejects_unknown_harness() -> None:
    with pytest.raises(PlanError, match="unknown harness"):
        parse_harness_spec("other-low")


def test_effort_is_required_in_harness_specs() -> None:
    with pytest.raises(PlanError, match="effort is required"):
        parse_harness_spec("pi")


def test_yaml_requires_effort_and_allows_provider_inference(tmp_path: Path) -> None:
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(
        "version: 1\nmodel: model\nfirst:\n  harness: pi\n",
        encoding="utf-8",
    )
    with pytest.raises(PlanError, match="stage.effort is required"):
        load_plan(plan_file)
    plan_file.write_text(
        "version: 1\nmodel: model\nfirst:\n  harness: pi\n  effort: low\n",
        encoding="utf-8",
    )
    assert load_plan(plan_file).provider == ""


def test_loads_relative_yaml_paths(tmp_path: Path) -> None:
    summary = tmp_path / "summary.md"
    summary.write_text("Summarize.", encoding="utf-8")
    continuation = tmp_path / "continue.md"
    continuation.write_text("Continue $original_prompt", encoding="utf-8")
    plan_file = tmp_path / "SWITCH.yaml"
    plan_file.write_text(
        """
version: 1
provider: openai
model: zai-org/GLM-5.3
llm_endpoint: https://llm.example/v1
first:
  harness: pi
  effort: low
  switch_after: 4
handoff:
  summary_prompt: summary.md
  continue_prompt: continue.md
second:
  harness: codex
  effort: high
output:
  format: jsonl
  color: always
""",
        encoding="utf-8",
    )
    plan = load_plan(plan_file)
    assert plan.first.harness == "pi"
    assert plan.first.switch_after == 4
    assert plan.second is not None
    assert plan.second.harness == "codex"
    assert plan.handoff.summary_prompt == summary
    assert plan.handoff.continue_prompt == continuation
    assert plan.workspace == tmp_path
    assert plan.provider == "openai"
    assert plan.base_url == "https://llm.example/v1"
    assert plan.output.format == "jsonl"
    assert plan.output.color == "always"
    assert plan.handoff.skip_second_if_finished is False


def test_yaml_can_skip_the_second_harness_after_a_natural_finish(tmp_path: Path) -> None:
    plan_file = tmp_path / "plan.yaml"
    base = (
        "version: 1\nmodel: model\n"
        "first:\n  harness: pi\n  effort: low\n  switch_after: 5\n"
    )
    plan_file.write_text(
        base + "second: cc:low\nhandoff:\n  skip_second_if_finished: true\n",
        encoding="utf-8",
    )
    assert load_plan(plan_file).handoff.skip_second_if_finished is True
    plan_file.write_text(
        base + "second: cc:low\nhandoff:\n  skip_second_if_finished: 'no'\n",
        encoding="utf-8",
    )
    with pytest.raises(PlanError, match="true or false"):
        load_plan(plan_file)
    plan_file.write_text(
        base + "handoff:\n  skip_second_if_finished: true\n", encoding="utf-8"
    )
    with pytest.raises(PlanError, match="needs a second stage"):
        load_plan(plan_file)


@pytest.mark.parametrize(
    ("line", "expected"),
    [("", 85), ("compact_at_percent: 70\n", 70), ("compact_at_percent: 0\n", None),
     ("compact_at_percent: null\n", None)],
)
def test_yaml_sets_the_compaction_point(tmp_path: Path, line: str, expected: int | None) -> None:
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text("version: 1\nmodel: model\nfirst: pi:low\n" + line, encoding="utf-8")
    assert load_plan(plan_file).compact_at_percent == expected


@pytest.mark.parametrize("value", ["100", "-5", "'85'", "true", "0.85"])
def test_yaml_rejects_a_compaction_point_that_is_not_a_percent(tmp_path: Path, value: str) -> None:
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(
        f"version: 1\nmodel: model\nfirst: pi:low\ncompact_at_percent: {value}\n",
        encoding="utf-8",
    )
    with pytest.raises(PlanError, match="whole percent"):
        load_plan(plan_file)
