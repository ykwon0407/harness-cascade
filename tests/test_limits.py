from __future__ import annotations

import pytest

from harness_cascade import limits
from harness_cascade.limits import (
    FALLBACK_CONTEXT_WINDOW,
    FALLBACK_MAX_OUTPUT_TOKENS,
    ModelLimits,
    _Reported,
    discover_model_limits,
)


def serve(monkeypatch: pytest.MonkeyPatch, reported: _Reported | Exception) -> list[str]:
    """Stand in for a provider, recording that it was asked."""
    asked: list[str] = []

    def lookup(model: str, api_key: str, base_url: str, timeout: float) -> _Reported:
        asked.append(model)
        if isinstance(reported, Exception):
            raise reported
        return reported

    monkeypatch.setattr(limits, "_PROVIDERS", {"anthropic": lookup})
    return asked


def resolve(**overrides) -> ModelLimits:
    values = {
        "provider": "anthropic",
        "model": "claude-fable-5",
        "api_key": "secret",
        "base_url": "https://example.test",
    }
    values.update(overrides)
    return discover_model_limits(**values)


def test_the_provider_outranks_a_stale_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """The profile is a snapshot; the provider is what will be enforced.

    This is the case that motivated the module. The profile said 200,000 for
    the Anthropic line -- a number Claude Code had reported, which turned out
    to be Claude Code echoing back the window our own gateway advertised --
    against a real limit five times larger.
    """
    serve(monkeypatch, _Reported(context_window=1_000_000, max_output_tokens=128_000))
    resolved = resolve(profile_context_window=200_000, profile_max_output_tokens=64_000)
    assert (resolved.context_window, resolved.max_output_tokens) == (1_000_000, 128_000)


def test_a_disagreement_is_reported_rather_than_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Silently winning is how the profile stays wrong."""
    serve(monkeypatch, _Reported(context_window=1_000_000))
    notes = resolve(profile_context_window=200_000).notes
    assert len(notes) == 1
    assert "200,000" in notes[0] and "1,000,000" in notes[0]


def test_agreement_is_not_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    serve(monkeypatch, _Reported(context_window=1_000_000))
    assert resolve(profile_context_window=1_000_000).notes == ()


def test_an_unreachable_provider_falls_back_to_the_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A container starting without a route out must not lose the trial."""
    serve(monkeypatch, RuntimeError("no route to host"))
    resolved = resolve(profile_context_window=272_000)
    assert resolved.context_window == 272_000
    assert "unreachable" in resolved.source


def test_a_provider_that_reports_nothing_still_yields_usable_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every path has to end in a number an agent can plan against."""
    serve(monkeypatch, _Reported())
    resolved = resolve()
    assert resolved.context_window == FALLBACK_CONTEXT_WINDOW
    assert resolved.max_output_tokens == FALLBACK_MAX_OUTPUT_TOKENS


def test_a_provider_with_no_published_limits_is_never_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI publishes neither limit, so the profile is the best there is."""
    asked = serve(monkeypatch, _Reported(context_window=1_000_000))
    resolved = resolve(provider="openai", profile_context_window=272_000)
    assert asked == []
    assert resolved.context_window == 272_000
    # A provider that was never asked is not a provider we failed to reach,
    # and reading it that way would send someone looking for a network fault.
    assert "unreachable" not in resolved.source


def test_the_lookup_happens_once_per_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every stage of every trial starts a gateway; only one may pay for it."""
    asked = serve(monkeypatch, _Reported(context_window=1_000_000))
    resolve()
    resolve()
    assert len(asked) == 1


def test_partial_answers_take_the_profile_for_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Together reports a window and no output cap; that must not lose the cap."""
    serve(monkeypatch, _Reported(context_window=131_072))
    resolved = resolve(profile_max_output_tokens=32_000)
    assert (resolved.context_window, resolved.max_output_tokens) == (131_072, 32_000)


@pytest.mark.parametrize("bogus", [0, -1, "1000000", None, True])
def test_nonsense_from_a_provider_is_declined(
    monkeypatch: pytest.MonkeyPatch, bogus: object
) -> None:
    """A window of zero or "1000000" would be worse than the fallback."""
    serve(monkeypatch, _Reported(context_window=bogus))  # type: ignore[arg-type]
    assert resolve().context_window == FALLBACK_CONTEXT_WINDOW
