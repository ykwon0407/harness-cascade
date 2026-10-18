from __future__ import annotations

import pytest

from harness_cascade import limits


@pytest.fixture(autouse=True)
def offline_model_limits(monkeypatch: pytest.MonkeyPatch):
    """Keep limit discovery off the network, and out of the next test's way.

    discover_model_limits runs on every gateway start, so without this every
    test that starts one would send its placeholder credentials to whatever
    host it named. The cache reset matters just as much: it is keyed by
    provider, model and endpoint, which tests reuse while varying the profile
    values around them. Tests that exercise discovery itself put _PROVIDERS
    back.
    """
    limits.reset_cache()
    monkeypatch.setattr(limits, "_PROVIDERS", {})
    yield
    limits.reset_cache()


@pytest.fixture(autouse=True)
def unstreamed_upstream(monkeypatch: pytest.MonkeyPatch):
    """Most tests fake the endpoint with `httpx.post`, the unstreamed path.

    Streaming is the default, and the tests that exercise it turn it back on
    and fake `httpx.stream` instead.
    """
    monkeypatch.setenv("CASCADE_UPSTREAM_STREAM", "0")
