"""Command-line entry point."""

from __future__ import annotations

import argparse
import contextlib
import ipaddress
import json
import os
import signal
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

from . import __version__
from .config import (
    DEFAULT_COMPACT_AT_PERCENT,
    HandoffPlan,
    OutputPlan,
    PlanError,
    PROVIDER_ALIASES,
    RunPlan,
    compact_at_percent,
    parse_harness_spec,
)
from .credentials import (
    VENDOR_ENDPOINTS,
    CredentialError,
    lock_down_process_credentials,
    resolve_credentials,
    validate_api_key,
)
from .engine import HandoffEngine
from .events import EventBus
from .models import (
    ModelProfileError,
    load_model_registry,
    resolve_model_profile,
)
from .renderers import (
    JsonlRenderer,
    PrettyRenderer,
    TrajectoryWriter,
    print_final_result,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cascade",
        description="Run one agent harness or hand work from one harness to another.",
    )
    parser.add_argument("-p", "--prompt", help="headless task prompt; stdin is used when omitted")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="read the task prompt from a file (safe for $, quotes, and multiline text)",
    )
    # --plan (YAML run plans, config.load_plan) is switched off until plans
    # are finished.
    parser.add_argument(
        "--first",
        "--first-harness",
        "-first-harness",
        dest="first",
        help="first harness, e.g. pi:low, CC-low, codex:high",
    )
    parser.add_argument(
        "--second",
        "--second-harness",
        "-second-harness",
        dest="second",
        help="second harness; omit for a solo run",
    )
    parser.add_argument(
        "--switch-after",
        "--n-to-switch",
        "-n-to-switch",
        dest="switch_after",
        type=int,
        help=(
            "hand off after this many first-stage model replies that called a "
            "tool; the same unit for every harness"
        ),
    )
    parser.add_argument(
        "--skip-second-if-finished",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "end the run when the first harness finishes on its own before the "
            "switch point; by default the second harness always runs"
        ),
    )
    parser.add_argument(
        "--summary-prompt",
        "--handoff-prompt",
        dest="summary_prompt",
        type=Path,
        help="custom Markdown prompt used to create the transfer record",
    )
    parser.add_argument(
        "--continue-prompt",
        type=Path,
        help="custom Markdown template passed to the next agent",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("CASCADE_MODEL", ""),
        help="model id; defaults to $CASCADE_MODEL",
    )
    parser.add_argument(
        "--provider",
        choices=["generic", "openai", "anthropic", "openai-compatible", "together"],
    )
    parser.add_argument(
        "--llm-endpoint",
        "--base-url",
        dest="base_url",
        help="provider API base URL; the provider's own when omitted",
    )
    parser.add_argument("--workspace", type=Path)
    parser.add_argument(
        "--env-file",
        type=Path,
        help="read API keys from this file; the process environment wins",
    )
    parser.add_argument(
        "--model-profiles",
        type=Path,
        help="YAML file adding or overriding exact provider/model effort profiles",
    )
    parser.add_argument(
        "--output-format",
        choices=["pretty", "text", "jsonl"],
        help="pretty progress, final text only, or machine-readable events",
    )
    parser.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        help="ANSI color policy for pretty output",
    )
    parser.add_argument(
        "--compact-at",
        dest="compact_at",
        type=int,
        metavar="PERCENT",
        help=(
            "every harness compacts once its context passes this percent of the "
            f"model's window (default {DEFAULT_COMPACT_AT_PERCENT}); 0 leaves "
            "each harness its own window and threshold"
        ),
    )
    parser.add_argument("--timeout", type=int, default=0, help="stage timeout in seconds")
    parser.add_argument("--no-preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the resolved plan")
    parser.add_argument("--version", action="version", version=f"cascade {__version__}")
    return parser


def _prompt(value: str | None, prompt_file: Path | None = None) -> str:
    if value is not None and prompt_file is not None:
        raise PlanError("use either --prompt or --prompt-file, not both")
    if prompt_file is not None:
        try:
            value = prompt_file.expanduser().read_text(encoding="utf-8")
        except OSError as exc:
            raise PlanError(f"could not read prompt file {prompt_file}: {exc}") from exc
    if value is not None:
        if not value.strip():
            raise PlanError("prompt must not be empty")
        return value
    if sys.stdin.isatty():
        raise PlanError("-p/--prompt is required when stdin is a terminal")
    value = sys.stdin.read()
    if not value.strip():
        raise PlanError("stdin contained no prompt")
    return value


def _direct_plan(args: argparse.Namespace) -> RunPlan:
    if not args.first:
        raise PlanError("--first is required")
    if not args.model:
        raise PlanError("--model is required")
    first = parse_harness_spec(args.first, switch_after=args.switch_after)
    second = parse_harness_spec(args.second) if args.second else None
    if second is not None and first.switch_after is None:
        raise PlanError("--switch-after is required with --second")
    if second is None and first.switch_after is not None:
        raise PlanError(
            "--switch-after needs --second; a solo run is not cut short"
        )
    workspace = (args.workspace or Path.cwd()).expanduser().resolve()
    if not workspace.is_dir():
        raise PlanError(f"workspace is not a directory: {workspace}")
    return RunPlan(
        first=first,
        second=second,
        handoff=HandoffPlan(
            summary_prompt=args.summary_prompt.expanduser().resolve()
            if args.summary_prompt
            else None,
            continue_prompt=args.continue_prompt.expanduser().resolve()
            if args.continue_prompt
            else None,
        ),
        output=OutputPlan(
            format=args.output_format or "pretty",
            color=args.color or "always",
        ),
        workspace=workspace,
        provider=PROVIDER_ALIASES[args.provider] if args.provider else "",
        model=args.model,
        base_url=(args.base_url or "").rstrip("/"),
        timeout_seconds=args.timeout or 1800,
        model_profiles=args.model_profiles.expanduser().resolve()
        if args.model_profiles
        else None,
    )


def _resolved_plan(args: argparse.Namespace) -> RunPlan:
    plan = _direct_plan(args)
    if args.compact_at is not None:
        plan = replace(
            plan,
            compact_at_percent=compact_at_percent(args.compact_at, "--compact-at"),
        )
    if args.skip_second_if_finished is not None:
        if args.skip_second_if_finished and plan.second is None:
            raise PlanError("--skip-second-if-finished needs --second")
        plan = replace(
            plan,
            handoff=replace(
                plan.handoff,
                skip_second_if_finished=args.skip_second_if_finished,
            ),
        )
    try:
        if not plan.provider:
            inferred_provider = load_model_registry(
                plan.model_profiles
            ).infer_provider(plan.model)
            plan = replace(plan, provider=inferred_provider)
        profile = resolve_model_profile(
            plan.provider,
            plan.model,
            plan.first.effort,
            override=plan.model_profiles,
        )
        if plan.second is not None:
            resolve_model_profile(
                plan.provider,
                plan.model,
                plan.second.effort,
                override=plan.model_profiles,
            )
    except ModelProfileError as exc:
        raise PlanError(str(exc)) from exc
    return replace(plan, model_profile=profile)


def _print_dry_run(plan: RunPlan) -> None:
    payload = {
        "first": {
            "harness": plan.first.harness,
            "effort": plan.first.effort,
            "switch_after": plan.first.switch_after,
        },
        "second": (
            {"harness": plan.second.harness, "effort": plan.second.effort}
            if plan.second
            else None
        ),
        "skip_second_if_finished": plan.handoff.skip_second_if_finished,
        "compact_at_percent": plan.compact_at_percent,
        "summary_prompt": str(plan.handoff.summary_prompt)
        if plan.handoff.summary_prompt
        else "packaged default",
        "continue_prompt": str(plan.handoff.continue_prompt)
        if plan.handoff.continue_prompt
        else "packaged default",
        "workspace": str(plan.workspace),
        "provider": plan.provider,
        "model": plan.model,
        "model_profile": plan.model_profile.source if plan.model_profile else None,
        "llm_endpoint": plan.base_url,
        "output_format": plan.output.format,
        "color": plan.output.color,
        "uses_external_protocol_bridge": False,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        plan = _resolved_plan(args)
        prompt = _prompt(args.prompt, args.prompt_file)
        if args.dry_run:
            _print_dry_run(plan)
            return 0

        env_file = args.env_file.expanduser().resolve() if args.env_file else None
        if env_file is not None and env_file.is_relative_to(plan.workspace):
            _warn(f"{env_file} is inside the workspace, where the agent can read it")
        credentials = resolve_credentials(
            plan.provider,
            env_file=env_file,
        )
        if credentials.ignored_endpoint and not plan.base_url:
            _warn(
                f"{credentials.ignored_endpoint} is ignored for {credentials.provider}; "
                "pass --llm-endpoint to send its key elsewhere"
            )
        endpoint = (
            plan.base_url
            or credentials.endpoint
            or VENDOR_ENDPOINTS.get(credentials.provider, "")
        )
        if not endpoint:
            raise CredentialError(
                "LLM_END_POINT (or LLM_ENDPOINT/--llm-endpoint) is required "
                "when using generic API_KEY credentials"
            )
        _warn_about_endpoint(credentials.provider, endpoint)
        plan = replace(
            plan,
            provider=plan.provider,
            base_url=endpoint.rstrip("/"),
        )
        if not args.no_preflight:
            validate_api_key(credentials, endpoint=plan.base_url)
        lock_down_process_credentials(credentials)

        run_id = uuid.uuid4().hex[:12]
        events = EventBus(run_id)
        if plan.output.format == "pretty":
            events.subscribe(PrettyRenderer(color=plan.output.color))
        elif plan.output.format == "jsonl":
            events.subscribe(JsonlRenderer())
        engine = HandoffEngine(plan, credentials, events=events)
        trajectory = TrajectoryWriter(engine.run_dir / "trajectory.jsonl")
        if plan.output.save_trajectory:
            events.subscribe(trajectory)
        try:
            with _stopped_by_signals():
                result = engine.run(prompt)
        finally:
            trajectory.close()
        print_final_result(result, output_format=plan.output.format)
        return 0
    except (PlanError, CredentialError) as exc:
        parser.error(str(exc))
    except Terminated as exc:
        sys.stderr.write(f"cascade: stopped by {signal.Signals(exc.signum).name}\n")
        return 128 + exc.signum
    except KeyboardInterrupt:
        sys.stderr.write("cascade: interrupted\n")
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports unexpected failures.
        sys.stderr.write(f"cascade: {type(exc).__name__}: {exc}\n")
        return 1
    return 1


class Terminated(KeyboardInterrupt):
    """SIGTERM or SIGHUP, taken like Ctrl-C so the harness is stopped too.

    A benchmark's wall-clock limit arrives as SIGTERM. Left to the default
    action, cascade dies at once and the harness, in its own session, goes on
    working in the workspace that is about to be graded.
    """

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


@contextlib.contextmanager
def _stopped_by_signals():
    def stop(signum: int, _frame: object) -> None:
        raise Terminated(signum)

    previous = {
        signum: signal.signal(signum, stop) for signum in (signal.SIGTERM, signal.SIGHUP)
    }
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _warn(message: str) -> None:
    sys.stderr.write(f"cascade: warning: {message}\n")


def _warn_about_endpoint(provider: str, endpoint: str) -> None:
    url = urlparse(endpoint)
    vendor = VENDOR_ENDPOINTS.get(provider)
    if vendor and url.hostname != urlparse(vendor).hostname:
        _warn(f"the {provider} key goes to {url.hostname}, not {urlparse(vendor).hostname}")
    if url.scheme == "http" and not _is_loopback(url.hostname or ""):
        _warn(f"{endpoint} is plain HTTP; the key crosses the network unencrypted")


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main() -> None:
    raise SystemExit(run())
