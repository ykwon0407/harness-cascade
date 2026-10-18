"""Run-plan parsing and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .models import ModelProfile

HARNESS_ALIASES = {
    "cc": "claude-code",
    "claude": "claude-code",
    "claude-code": "claude-code",
    "pi": "pi",
    "codex": "codex",
}
EFFORTS = {"none", "off", "minimal", "low", "medium", "high", "xhigh", "max"}
DEFAULT_COMPACT_AT_PERCENT = 85
PROVIDER_ALIASES = {
    "generic": "openai-compatible",
    "openai": "openai",
    "anthropic": "anthropic",
    "openai-compatible": "openai-compatible",
    "together": "together",
}


class PlanError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class StagePlan:
    harness: str
    effort: str
    switch_after: int | None = None


@dataclass(frozen=True, slots=True)
class HandoffPlan:
    summary_prompt: Path | None = None
    continue_prompt: Path | None = None
    include: tuple[str, ...] = (
        "original_prompt",
        "git_status",
        "git_diff",
        "recent_commands",
    )
    # A first agent that ends on its own before the switch point has given its
    # answer. By default the second agent still runs, so every handoff
    # condition puts two agents on every task; set this to end the run there.
    skip_second_if_finished: bool = False


@dataclass(frozen=True, slots=True)
class OutputPlan:
    format: str = "pretty"
    save_trajectory: bool = True
    color: str = "always"


@dataclass(frozen=True, slots=True)
class RunPlan:
    first: StagePlan
    second: StagePlan | None = None
    handoff: HandoffPlan = field(default_factory=HandoffPlan)
    output: OutputPlan = field(default_factory=OutputPlan)
    workspace: Path = Path(".")
    provider: str = "together"
    model: str = ""
    base_url: str = ""
    timeout_seconds: int = 1800
    model_profiles: Path | None = None
    model_profile: ModelProfile | None = None
    # Every harness is told the model's real window and compacts once its
    # context passes this share of it. Left to themselves they disagree:
    # Claude Code assumes 200,000 for a model it does not know, Codex 272,000,
    # and Pi never compacts mid-task at all. None keeps each harness's own.
    compact_at_percent: int | None = DEFAULT_COMPACT_AT_PERCENT


def compact_at_percent(value: Any, source: str) -> int | None:
    if value is None or value == 0:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 99:
        raise PlanError(f"{source} must be a whole percent from 1 to 99, or 0 for each harness's own")
    return value


def parse_harness_spec(value: str, *, switch_after: int | None = None) -> StagePlan:
    raw = value.strip()
    if not raw:
        raise PlanError("harness must not be empty")

    effort = ""
    harness_token = raw
    if ":" in raw:
        harness_token, effort = raw.rsplit(":", 1)
    else:
        lowered = raw.lower()
        for candidate in sorted(EFFORTS, key=len, reverse=True):
            suffix = f"-{candidate}"
            if lowered.endswith(suffix):
                harness_token = raw[: -len(suffix)]
                effort = candidate
                break
    if not effort:
        raise PlanError(
            f"effort is required in harness {value!r}; use HARNESS:EFFORT"
        )

    harness = HARNESS_ALIASES.get(harness_token.strip().lower())
    if harness is None:
        accepted = ", ".join(sorted(HARNESS_ALIASES))
        raise PlanError(f"unknown harness {harness_token!r}; expected one of: {accepted}")
    effort = effort.strip().lower()
    if effort not in EFFORTS:
        raise PlanError(f"unknown effort {effort!r}; expected one of: {', '.join(sorted(EFFORTS))}")
    if switch_after is not None and switch_after < 1:
        raise PlanError("switch_after must be at least 1")
    return StagePlan(harness=harness, effort=effort, switch_after=switch_after)


def _stage_from_mapping(value: Any, *, default_switch: int | None = None) -> StagePlan:
    if isinstance(value, str):
        return parse_harness_spec(value, switch_after=default_switch)
    if not isinstance(value, dict):
        raise PlanError("stage must be a harness string or mapping")
    harness = str(value.get("harness") or "").strip()
    if "effort" not in value or not str(value.get("effort") or "").strip():
        raise PlanError("stage.effort is required")
    effort = str(value["effort"]).strip()
    stop = value.get("stop") or {}
    switch_after = value.get("switch_after", default_switch)
    if isinstance(stop, dict):
        switch_after = stop.get("tool_turns", switch_after)
    spec = f"{harness}:{effort}"
    return parse_harness_spec(spec, switch_after=int(switch_after) if switch_after else None)


def load_plan(path: str | Path) -> RunPlan:
    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise PlanError(f"could not read plan {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise PlanError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise PlanError("plan root must be a mapping")
    if int(raw.get("version") or 1) != 1:
        raise PlanError("only plan version 1 is supported")

    first_raw = raw.get("first")
    second_raw = raw.get("second")
    if first_raw is None:
        stages = raw.get("stages")
        if isinstance(stages, list) and stages:
            executable = [item for item in stages if isinstance(item, (str, dict)) and not (isinstance(item, dict) and "handoff" in item)]
            first_raw = executable[0] if executable else None
            second_raw = executable[1] if len(executable) > 1 else None
    if first_raw is None:
        raise PlanError("plan requires first or stages[0]")

    first = _stage_from_mapping(first_raw)
    second = _stage_from_mapping(second_raw) if second_raw is not None else None
    if second is not None and first.switch_after is None:
        raise PlanError("a two-stage plan requires first.switch_after")

    handoff_raw = raw.get("handoff") or {}
    if not isinstance(handoff_raw, dict):
        raise PlanError("handoff must be a mapping")
    prompt_value = handoff_raw.get("summary_prompt") or handoff_raw.get("prompt_file")
    continue_value = handoff_raw.get("continue_prompt")
    summary_prompt = None
    continue_prompt = None
    if prompt_value:
        summary_prompt = (config_path.parent / str(prompt_value)).resolve()
    if continue_value:
        continue_prompt = (config_path.parent / str(continue_value)).resolve()
    include = tuple(str(item) for item in handoff_raw.get("include") or HandoffPlan().include)
    skip_second = handoff_raw.get("skip_second_if_finished", False)
    if not isinstance(skip_second, bool):
        raise PlanError("handoff.skip_second_if_finished must be true or false")
    if skip_second and second is None:
        raise PlanError("handoff.skip_second_if_finished needs a second stage")

    output_raw = raw.get("output") or {}
    if not isinstance(output_raw, dict):
        raise PlanError("output must be a mapping")
    output_format = str(output_raw.get("format") or "pretty")
    if output_format not in {"pretty", "text", "jsonl"}:
        raise PlanError("output.format must be pretty, text, or jsonl")
    output_color = str(output_raw.get("color") or "always")
    if output_color not in {"auto", "always", "never"}:
        raise PlanError("output.color must be auto, always, or never")

    workspace_raw = Path(str(raw.get("workspace") or ".")).expanduser()
    workspace = workspace_raw if workspace_raw.is_absolute() else (config_path.parent / workspace_raw)
    provider_value = str(raw.get("provider") or "").strip().lower()
    provider = PROVIDER_ALIASES.get(provider_value) if provider_value else ""
    if provider_value and provider is None:
        raise PlanError(
            "provider must be openai, anthropic, together, or openai-compatible"
        )
    model = str(raw.get("model") or "").strip()
    if not model:
        raise PlanError("plan requires model")
    profiles_value = raw.get("model_profiles")
    model_profiles = None
    if profiles_value:
        profiles_path = Path(str(profiles_value)).expanduser()
        model_profiles = (
            profiles_path
            if profiles_path.is_absolute()
            else (config_path.parent / profiles_path)
        ).resolve()

    return RunPlan(
        first=first,
        second=second,
        handoff=HandoffPlan(
            summary_prompt=summary_prompt,
            continue_prompt=continue_prompt,
            include=include,
            skip_second_if_finished=skip_second,
        ),
        output=OutputPlan(
            format=output_format,
            save_trajectory=bool(output_raw.get("save_trajectory", True)),
            color=output_color,
        ),
        workspace=workspace.resolve(),
        provider=provider,
        model=model,
        base_url=str(
            raw.get("llm_endpoint") or raw.get("base_url") or ""
        ).rstrip("/"),
        timeout_seconds=int(raw.get("timeout_seconds") or 1800),
        model_profiles=model_profiles,
        compact_at_percent=compact_at_percent(
            raw.get("compact_at_percent", DEFAULT_COMPACT_AT_PERCENT),
            "compact_at_percent",
        ),
    )
