"""How much room the model actually has, asked of the provider where it will say.

An agent that manages its own context plans around these two numbers: Pi
compacts when it believes it is running out, Claude Code decides what to keep,
Codex sets an auto-compact threshold. So a window written here by hand is not a
cosmetic error. It used to say 200,000 for the Anthropic line, which the API
reports at 1,000,000 -- five times too small, which means the agent would throw
away work that still fit.

Hence: ask the provider, fall back to the model profile, and fall back again to
a constant. The constants are exactly the values that were hardcoded before this
module existed, so a provider that cannot be reached costs only the accuracy
this module was added to gain, never a working run.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any

import httpx

from .credentials import models_url

# What the code claimed before it learned to ask. Low enough that every model
# we have run accepts it, which is what makes it safe as a last resort.
FALLBACK_CONTEXT_WINDOW = 200_000
FALLBACK_MAX_OUTPUT_TOKENS = 65_536

# A campaign-level ceiling on output, below whatever the model allows. The
# gateway hands the agent a reply only once it is complete, and the agents are
# set to wait as long as the gateway does, so what bounds a long generation is
# the gateway's per-attempt ceiling; a lower cap keeps each reply inside that.
MAX_OUTPUT_ENV = "CASCADE_MAX_OUTPUT_TOKENS"


def output_ceiling() -> int | None:
    try:
        value = int(os.environ.get(MAX_OUTPUT_ENV, ""))
    except ValueError:
        return None
    return value if value > 0 else None


@dataclass(frozen=True, slots=True)
class ModelLimits:
    context_window: int
    max_output_tokens: int
    # Which of the three sources each number came from, so a surprising agent
    # decision can be traced back to the number that caused it.
    source: str
    # Places the provider and the profile disagree. Non-empty means one of them
    # is stale and someone should look.
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _Reported:
    """What a provider said, straight out of its JSON.

    Either field may be absent, and neither is trusted until `_positive` has
    looked at it: a window of 0, or of "1000000", is worse than no answer,
    because the fallback is at least known to work.
    """

    context_window: Any = None
    max_output_tokens: Any = None


def _positive(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _anthropic(model: str, api_key: str, base_url: str, timeout: float) -> _Reported:
    """`GET /v1/models/{id}`, which reports both limits for every model."""
    response = httpx.get(
        f"{models_url(base_url, 'anthropic')}/{model}",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    return _Reported(
        context_window=body.get("max_input_tokens"),
        max_output_tokens=body.get("max_tokens"),
    )


def _together(model: str, api_key: str, base_url: str, timeout: float) -> _Reported:
    """`GET /v1/models`, which reports a window but no output cap.

    There is no per-model route, so the whole list comes back and the entry has
    to be picked out of it.
    """
    response = httpx.get(
        models_url(base_url),
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    entries = body.get("data") if isinstance(body, dict) else body
    for entry in entries or ():
        if isinstance(entry, dict) and entry.get("id") == model:
            return _Reported(context_window=entry.get("context_length"))
    return _Reported()


# OpenAI is absent on purpose. `GET /v1/models/{id}` answers with an id, an
# owner and a shutdown date -- no limits of any kind -- and the only other place
# OpenAI publishes a window is the catalog compiled into the Codex binary, which
# carries the gpt-5.x slugs and not the models we run. So for OpenAI the profile
# is the best source there is, and it records where its number came from.
_PROVIDERS = {"anthropic": _anthropic, "together": _together}

_cache: dict[tuple[str, str, str], ModelLimits] = {}
_cache_lock = threading.Lock()


def reset_cache() -> None:
    with _cache_lock:
        _cache.clear()


def discover_model_limits(
    *,
    provider: str,
    model: str,
    api_key: str,
    base_url: str,
    profile_context_window: int | None = None,
    profile_max_output_tokens: int | None = None,
    timeout: float = 10.0,
) -> ModelLimits:
    """Resolve both limits once per process, preferring the provider's answer.

    The provider wins over the profile because the profile is a snapshot and
    the provider is current; when they disagree the profile is the stale one.
    The disagreement is still recorded rather than swallowed, because a profile
    that has drifted is worth fixing even though the run no longer depends on
    it.
    """
    key = (provider, model, base_url)
    with _cache_lock:
        cached = _cache.get(key)
    if cached is not None:
        return cached

    reported = _Reported()
    # Distinguishes "this provider publishes nothing" from "we could not reach
    # it", because only the second means the number in hand might be stale.
    fell_back_blind = False
    lookup = _PROVIDERS.get(provider)
    if lookup is not None and api_key:
        try:
            reported = lookup(model, api_key, base_url, timeout)
        except Exception:
            fell_back_blind = True
            # Every failure here is the same failure: we did not learn
            # anything. A missing key, a 404 on an id the provider does not
            # publish, a network that is not there during a container start --
            # none of them are worth failing a trial over when a usable number
            # is already in hand.
            reported = _Reported()

    notes: list[str] = []
    sources: list[str] = []

    def resolve(name: str, raw: Any, profile: int | None, default: int) -> int:
        live = _positive(raw)
        if live is not None and profile is not None and live != profile:
            notes.append(
                f"{provider}/{model} reports {name} {live:,} but the model "
                f"profile says {profile:,}; using {live:,} and the profile "
                "should be corrected"
            )
        qualifier = "(provider unreachable)" if fell_back_blind else ""
        if live is not None:
            sources.append(f"{name}=provider")
            return live
        if profile is not None:
            sources.append(f"{name}=profile{qualifier}")
            return profile
        sources.append(f"{name}=default{qualifier}")
        return default

    limits = ModelLimits(
        context_window=resolve(
            "context_window",
            reported.context_window,
            profile_context_window,
            FALLBACK_CONTEXT_WINDOW,
        ),
        max_output_tokens=resolve(
            "max_output_tokens",
            reported.max_output_tokens,
            profile_max_output_tokens,
            FALLBACK_MAX_OUTPUT_TOKENS,
        ),
        source=" ".join(sources),
        notes=tuple(notes),
    )
    with _cache_lock:
        _cache.setdefault(key, limits)
        return _cache[key]
