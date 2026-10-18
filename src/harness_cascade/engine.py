"""Two-stage handoff state machine, independent of benchmark infrastructure."""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from dataclasses import asdict
from importlib import resources
from pathlib import Path
from string import Template
from typing import Any

from .config import RunPlan, StagePlan
from .credentials import CredentialSet
from .events import CascadeEvent, EventBus, EventKind
from .gateway import EmbeddedGateway, GatewayConfig
from .harnesses import HarnessContext, HarnessResult, adapter_for, new_session_id

PROMPT_PACKAGE = "harness_cascade.prompts"
# Written summaries in the TB 2.1 runs peaked at about 10k characters.
REASONING_SUMMARY_CHARS = 12_000
# Bounds on what the second prompt carries over. Tens of thousands of
# untracked files, or an agent that writes files through heredocs, would
# otherwise fill the second harness's context window before it starts.
SUMMARY_CHARS = 48_000
SNAPSHOT_LINES = 200
COMMAND_CHARS = 1_000


class EngineError(RuntimeError):
    pass


def _stage_failure(label: str, result: HarnessResult) -> str:
    message = f"{label} {result.harness} {result.status} (exit {result.returncode})"
    return f"{message}: {result.error}" if result.error else message


def _no_answer(label: str, result: HarnessResult) -> str:
    message = f"{label} {result.harness} exited without an assistant result"
    return f"{message}: {result.error}" if result.error else message


def combined_usage(*usages: dict[str, int]) -> dict[str, int]:
    combined = {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }
    for usage in usages:
        for key in combined:
            combined[key] += int(usage.get(key) or 0)
    combined["total_tokens"] = (
        combined["input_tokens"] + combined["output_tokens"]
    )
    return combined


def clip_text(text: str, limit: int) -> str:
    """`text` cut to about `limit` characters, keeping its start and its end."""
    if len(text) <= limit:
        return text
    head = limit * 3 // 4
    tail = limit - head
    return (
        f"{text[:head]}\n[... {len(text) - head - tail} characters cut ...]\n"
        f"{text[len(text) - tail:]}"
    )


def clip_lines(text: str, limit: int, *, keep_last: bool = False) -> str:
    """The first `limit` lines of `text`, and its last when `keep_last`."""
    lines = text.splitlines()
    if len(lines) <= limit:
        return text
    last = [lines[-1]] if keep_last else []
    cut = len(lines) - limit - len(last)
    return "\n".join([*lines[:limit], f"... and {cut} more lines", *last])


def workspace_snapshot(workspace: Path, commands: list[str]) -> dict[str, Any]:
    def run_git(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        return (result.stdout or result.stderr).strip()

    is_git = run_git("rev-parse", "--is-inside-work-tree") == "true"
    status = run_git("status", "--short") if is_git else "(not a git repository)"
    diff = run_git("diff", "--stat") if is_git else "(not available)"
    changed_files = (
        len([line for line in status.splitlines() if line.strip()]) if is_git else 0
    )
    return {
        "cwd": str(workspace),
        "git_status": clip_lines(status, SNAPSHOT_LINES)
        or "(clean or not a git repository)",
        # The last line of --stat is the total.
        "git_diff_stat": clip_lines(diff, SNAPSHOT_LINES, keep_last=True) or "(empty)",
        "recent_commands": [clip_text(item, COMMAND_CHARS) for item in commands[-20:]],
        "changed_files": changed_files,
    }


def build_second_prompt(
    original_prompt: str,
    summary: str,
    snapshot: dict[str, Any],
    *,
    template: str | None = None,
) -> str:
    commands = "\n".join(f"- {item}" for item in snapshot["recent_commands"]) or "(none)"
    prompt_template = template or packaged_prompt("handoff-continue.md")
    return Template(prompt_template).safe_substitute(
        original_prompt=original_prompt.strip(),
        handoff_summary=clip_text(summary.strip(), SUMMARY_CHARS),
        workspace=snapshot["cwd"],
        git_status=snapshot["git_status"],
        git_diff_stat=snapshot["git_diff_stat"],
        recent_commands=commands,
    )


def packaged_prompt(name: str) -> str:
    return resources.files(PROMPT_PACKAGE).joinpath(name).read_text(encoding="utf-8")


def prompt_from_path(path: Path | None, default_name: str) -> str:
    if path is None:
        return packaged_prompt(default_name)
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EngineError(f"could not read prompt {path}: {exc}") from exc


class HandoffEngine:
    def __init__(
        self,
        plan: RunPlan,
        credentials: CredentialSet,
        *,
        events: EventBus | None = None,
        runs_root: Path | None = None,
    ) -> None:
        self.plan = plan
        self.credentials = credentials
        self.run_id = events.run_id if events else uuid.uuid4().hex[:12]
        self.events = events or EventBus(self.run_id)
        self.events.redact(credentials.api_key)
        self._gateway_usage: dict[str, dict[str, int]] = {}
        self._gateways: dict[str, EmbeddedGateway] = {}
        self.events.subscribe(self._capture_gateway_usage)
        if not plan.workspace.is_dir():
            raise EngineError(f"workspace is not a directory: {plan.workspace}")
        self.runs_root = runs_root or (plan.workspace / ".cascade" / "runs")
        self.run_dir = self.runs_root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _capture_gateway_usage(self, event: CascadeEvent) -> None:
        if event.kind != EventKind.MODEL_CALL_FINISHED or not event.stage:
            return
        usage = event.data.get("usage")
        if not isinstance(usage, dict):
            return
        current = self._gateway_usage.get(event.stage, {})
        self._gateway_usage[event.stage] = combined_usage(current, usage)

    def _gateway(self, stage_name: str, stage: StagePlan, *, summary: bool = False):
        token = self.credentials.local_session_token()
        profile = self.plan.model_profile
        protocol = (
            profile.protocol
            if profile
            else {
                "openai": "openai-responses",
                "anthropic": "anthropic-messages",
            }.get(self.plan.provider, "openai-chat")
        )
        return EmbeddedGateway(
            GatewayConfig(
                model=self.plan.model,
                effort=stage.effort,
                api_key=self.credentials.api_key,
                local_token=token,
                base_url=self.plan.base_url,
                provider=self.plan.provider,
                protocol=protocol,
                supported_efforts=profile.efforts if profile else (),
                profile_context_window=profile.context_window if profile else None,
                profile_max_output_tokens=(
                    profile.max_output_tokens if profile else None
                ),
                request_timeout_seconds=self.plan.timeout_seconds,
                disable_tools=summary,
                stage=stage_name,
                harness=stage.harness,
                state_path=str(self.run_dir / "state" / "opaque-provider-state.json"),
                switch_after=None if summary else stage.switch_after,
                compact_at_percent=self.plan.compact_at_percent,
            ),
            self.events,
        )

    def _settle_usage(self, stages: dict[str, HarnessResult | None]) -> dict[str, int]:
        """Fold in model calls that finished after their stage did.

        A call cut off by a kill can still be answered, and billed, once the
        engine has moved on; its usage arrives late and would otherwise be
        missing from the stage that made it. Returns the calls still open.
        """
        for name, result in stages.items():
            latest = self._gateway_usage.get(name)
            if result is None or not latest or latest == result.usage:
                continue
            before = int(result.usage.get("input_tokens") or 0)
            result.usage = dict(latest)
            self.events.emit(
                EventKind.WARNING,
                stage=name,
                harness=result.harness,
                message="a model call finished after its stage ended; its usage is now counted",
                added_input_tokens=int(latest.get("input_tokens") or 0) - before,
            )
        open_calls = {
            name: gateway.calls_in_flight
            for name, gateway in self._gateways.items()
            if gateway.calls_in_flight
        }
        if open_calls:
            self.events.emit(
                EventKind.WARNING,
                message=(
                    "model calls were still unanswered when the run ended; "
                    "the provider may bill them, and their usage is not counted"
                ),
                calls=open_calls,
            )
        return open_calls

    def _run_stage(
        self,
        *,
        name: str,
        stage: StagePlan,
        prompt: str,
        session_id: str,
        resume: bool = False,
        summary: bool = False,
    ) -> HarnessResult:
        adapter = adapter_for(stage.harness)
        stage_started = time.perf_counter()
        self.events.emit(
            EventKind.STAGE_STARTED,
            stage=name,
            harness=stage.harness,
            effort=stage.effort,
            switch_after=stage.switch_after,
            resume=resume,
            summary=summary,
        )
        with self._gateway(name, stage, summary=summary) as gateway:
            if gateway is not None:
                self._gateways[name] = gateway
            result = adapter.run(
                HarnessContext(
                    stage=name,
                    plan=stage,
                    model=self.plan.model,
                    prompt=prompt,
                    workspace=self.plan.workspace,
                    state_dir=self.run_dir / "state",
                    log_path=self.run_dir / f"{name}.log",
                    credentials=self.credentials,
                    events=self.events,
                    gateway=gateway,
                    session_id=session_id,
                    timeout_seconds=self.plan.timeout_seconds,
                    resume=resume,
                    summary=summary,
                )
            )
        if name in self._gateway_usage:
            result.usage = dict(self._gateway_usage[name])
        duration_seconds = round(time.perf_counter() - stage_started, 3)
        self.events.emit(
            EventKind.STAGE_FINISHED,
            stage=name,
            harness=stage.harness,
            status=result.status,
            returncode=result.returncode,
            turns=result.turns,
            model_turns=result.model_turns,
            session_id=result.session_id,
            usage=combined_usage(result.usage),
            duration_seconds=duration_seconds,
            log_path=result.log_path,
            **({"error": result.error} if result.error else {}),
        )
        return result

    def _summary_prompt(self) -> str:
        return prompt_from_path(
            self.plan.handoff.summary_prompt,
            "handoff-summary.md",
        )

    def _continue_prompt(self) -> str:
        return prompt_from_path(
            self.plan.handoff.continue_prompt,
            "handoff-continue.md",
        )

    def _configuration(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "provider": self.plan.provider,
            "model": self.plan.model,
            "model_profile": (
                self.plan.model_profile.source if self.plan.model_profile else None
            ),
            "upstream_protocol": (
                self.plan.model_profile.protocol
                if self.plan.model_profile
                else {
                    "openai": "openai-responses",
                    "anthropic": "anthropic-messages",
                }.get(self.plan.provider, "openai-chat")
            ),
            "llm_endpoint": self.plan.base_url,
            "workspace": str(self.plan.workspace),
            "first": {
                "harness": self.plan.first.harness,
                "effort": self.plan.first.effort,
                "switch_after": self.plan.first.switch_after,
            },
            "second": (
                {
                    "harness": self.plan.second.harness,
                    "effort": self.plan.second.effort,
                }
                if self.plan.second
                else None
            ),
            "skip_second_if_finished": self.plan.handoff.skip_second_if_finished,
            "compact_at_percent": self.plan.compact_at_percent,
            "output": {
                "format": self.plan.output.format,
                "color": self.plan.output.color,
                "save_trajectory": self.plan.output.save_trajectory,
            },
            "prompts": {
                "summary": (
                    str(self.plan.handoff.summary_prompt)
                    if self.plan.handoff.summary_prompt
                    else "packaged default"
                ),
                "continue": (
                    str(self.plan.handoff.continue_prompt)
                    if self.plan.handoff.continue_prompt
                    else "packaged default"
                ),
            },
            "timeout_seconds": self.plan.timeout_seconds,
            "artifacts": str(self.run_dir),
        }

    def run(self, prompt: str) -> dict[str, Any]:
        started = time.perf_counter()
        if not self.plan.workspace.is_dir():
            raise EngineError(f"workspace is not a directory: {self.plan.workspace}")
        configuration = self._configuration()
        (self.run_dir / "configuration.json").write_text(
            json.dumps(configuration, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self.events.emit(
            EventKind.RUN_STARTED,
            prompt=prompt,
            workspace=configuration["workspace"],
            model=configuration["model"],
            provider=configuration["provider"],
            first=configuration["first"],
            second=configuration["second"],
            skip_second_if_finished=configuration["skip_second_if_finished"],
            compact_at_percent=configuration["compact_at_percent"],
            output_format=self.plan.output.format,
            color=self.plan.output.color,
            artifacts=configuration["artifacts"],
            summary_prompt=configuration["prompts"]["summary"],
            continue_prompt=configuration["prompts"]["continue"],
        )
        first_session = new_session_id()
        first: HarnessResult | None = None
        summary_result: HarnessResult | None = None
        second: HarnessResult | None = None
        summary = ""
        snapshot: dict[str, Any] = {}
        try:
            first = self._run_stage(
                name="first",
                stage=self.plan.first,
                prompt=prompt,
                session_id=first_session,
            )
            if first.status in {"failed", "timeout"}:
                raise EngineError(_stage_failure("first harness", first))
            if self.plan.second is None and first.status == "switch_limit":
                raise EngineError(
                    "solo harness was stopped at the switch point, before it finished"
                )
            skip_second = (
                self.plan.second is not None
                and self.plan.handoff.skip_second_if_finished
                and first.status == "completed"
            )
            if skip_second:
                self.events.emit(
                    EventKind.HANDOFF_SKIPPED,
                    stage="handoff",
                    harness=self.plan.first.harness,
                    reason="first harness finished before the switch point",
                    turns=first.turns,
                    switch_after=self.plan.first.switch_after,
                )
            if (self.plan.second is None or skip_second) and not first.final_text:
                raise EngineError(
                    _no_answer(
                        f"{'solo' if self.plan.second is None else 'first'} harness",
                        first,
                    )
                )

            if self.plan.second is not None and not skip_second:
                self.events.emit(
                    EventKind.HANDOFF_STARTED,
                    stage="handoff",
                    harness=self.plan.first.harness,
                )
                summary_stage = StagePlan(
                    harness=self.plan.first.harness,
                    effort=self.plan.first.effort,
                    switch_after=None,
                )
                summary_result = self._run_stage(
                    name="handoff",
                    stage=summary_stage,
                    prompt=self._summary_prompt(),
                    session_id=first.session_id,
                    resume=True,
                    summary=True,
                )
                if summary_result.status in {"failed", "timeout"}:
                    raise EngineError(
                        _stage_failure("handoff summary with", summary_result)
                    )
                summary = summary_result.final_text
                if not summary.strip() and summary_result.final_reasoning.strip():
                    # Some models write the whole summary as reasoning and leave
                    # the reply empty. The summary is this harness's request, not
                    # part of the agent being measured, so what was written is
                    # passed on wherever it landed. Reasoning that ran to the
                    # output cap goes on working after the summary.
                    summary = summary_result.final_reasoning[:REASONING_SUMMARY_CHARS]
                    self.events.emit(
                        EventKind.WARNING,
                        stage="handoff",
                        harness=self.plan.first.harness,
                        message="handoff summary taken from reasoning; the reply had no text",
                    )
                if len(summary.strip()) > SUMMARY_CHARS:
                    self.events.emit(
                        EventKind.WARNING,
                        stage="handoff",
                        harness=self.plan.first.harness,
                        message=(
                            f"handoff summary of {len(summary.strip())} characters "
                            f"cut to {SUMMARY_CHARS} for the second prompt"
                        ),
                    )
                if not summary.strip():
                    raise EngineError(
                        "first harness produced no handoff summary"
                        + (f": {summary_result.error}" if summary_result.error else "")
                    )
                self.events.emit(
                    EventKind.HANDOFF_SUMMARY,
                    stage="handoff",
                    harness=self.plan.first.harness,
                    text=summary,
                )
                snapshot = workspace_snapshot(self.plan.workspace, first.commands)
                self.events.emit(
                    EventKind.WORKSPACE_SNAPSHOT,
                    stage="handoff",
                    harness=self.plan.first.harness,
                    **snapshot,
                )
                second_prompt = build_second_prompt(
                    prompt,
                    summary,
                    snapshot,
                    template=self._continue_prompt(),
                )
                (self.run_dir / "handoff-summary.md").write_text(
                    summary + "\n",
                    encoding="utf-8",
                )
                (self.run_dir / "second-prompt.md").write_text(
                    second_prompt,
                    encoding="utf-8",
                )
                second = self._run_stage(
                    name="second",
                    stage=self.plan.second,
                    prompt=second_prompt,
                    session_id=new_session_id(),
                )
                if second.status in {"failed", "timeout"}:
                    raise EngineError(_stage_failure("second harness", second))
                if second.status == "switch_limit":
                    raise EngineError(
                        "second harness was stopped at a switch point, before it finished"
                    )
                if not second.final_text:
                    raise EngineError(_no_answer("second harness", second))

            final = second or first
            open_calls = self._settle_usage(
                {"first": first, "handoff": summary_result, "second": second}
            )
            first_usage = combined_usage(first.usage)
            handoff_usage = combined_usage(
                summary_result.usage if summary_result else {}
            )
            second_usage = combined_usage(second.usage if second else {})
            total_usage = combined_usage(
                first_usage,
                handoff_usage,
                second_usage,
            )
            result = {
                "run_id": self.run_id,
                "status": "completed",
                "configuration": configuration,
                "first": _result_dict(first),
                "handoff": _result_dict(summary_result),
                "second": _result_dict(second),
                "second_skipped": skip_second,
                "handoff_summary": summary,
                "workspace_snapshot": snapshot,
                "final_text": final.final_text,
                "trajectory": str(self.run_dir / "trajectory.jsonl"),
                "duration_seconds": round(time.perf_counter() - started, 3),
                "usage": total_usage,
            }
            if open_calls:
                result["unanswered_model_calls"] = open_calls
            (self.run_dir / "result.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            self.events.emit(
                EventKind.RUN_FINISHED,
                duration_seconds=result["duration_seconds"],
                final_text=result["final_text"],
                usage=total_usage,
                stage_usage={
                    "first": first_usage,
                    "handoff": handoff_usage,
                    "second": second_usage,
                },
                artifacts=str(self.run_dir),
            )
            return result
        except BaseException as exc:
            stages = {"first": first, "handoff": summary_result, "second": second}
            open_calls = self._settle_usage(stages)
            # Ctrl-C, or SIGTERM/SIGHUP from the CLI: a benchmark's time limit.
            interrupted = isinstance(exc, KeyboardInterrupt)
            message = "interrupted" if interrupted else str(exc)
            self.events.emit(EventKind.RUN_FAILED, message=message)
            failure = {
                "run_id": self.run_id,
                "status": "interrupted" if interrupted else "failed",
                "configuration": configuration,
                "error": message if interrupted else f"{type(exc).__name__}: {exc}",
                "first": _result_dict(first),
                "handoff": _result_dict(summary_result),
                "second": _result_dict(second),
                # A stage cut off in the middle has no result; the gateway
                # counted every call it answered.
                "usage": combined_usage(
                    *(
                        result.usage if result else self._gateway_usage.get(name, {})
                        for name, result in stages.items()
                    )
                ),
                "duration_seconds": round(time.perf_counter() - started, 3),
            }
            if open_calls:
                failure["unanswered_model_calls"] = open_calls
            (self.run_dir / "result.json").write_text(
                json.dumps(failure, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            raise


def _result_dict(result: HarnessResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    payload = asdict(result)
    payload.pop("raw_lines", None)
    payload.pop("pending_tools", None)
    return payload
