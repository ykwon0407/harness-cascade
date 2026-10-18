from pathlib import Path

import pytest

from harness_cascade.models import (
    ModelProfileError,
    load_model_registry,
    resolve_model_profile,
)


def test_packaged_profiles_cover_cascade_agent_models() -> None:
    for model in (
        "zai-org/GLM-5.3",
        "zai-org/GLM-5.3-Flash",
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "zainhas/dsv41",
    ):
        profile = resolve_model_profile("together", model, "low")
        assert profile.protocol == "openai-chat"


def test_profile_rejects_unknown_model_and_effort() -> None:
    with pytest.raises(ModelProfileError, match="no model profile"):
        resolve_model_profile("together", "unknown/model", "low")
    with pytest.raises(ModelProfileError, match="does not support effort"):
        resolve_model_profile("together", "zai-org/GLM-5.3", "medium")


# Measured against the self-hosted endpoint at n=20 per level; see
# docs/reasoning-effort.md. Guards the ladders against silent edits.
MEASURED_TOGETHER_EFFORTS = {
    "zai-org/GLM-5.3-Flash": ("none", "low", "high", "max"),
    "deepseek-ai/DeepSeek-V4.1-Flash": ("none", "low", "high", "xhigh", "max"),
    "Qwen/Qwen3.8-27B": ("none", "low", "medium", "xhigh"),
    "deepseek-ai/DeepSeek-V4-Flash-0731": (
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ),
}


@pytest.mark.parametrize(("model", "efforts"), MEASURED_TOGETHER_EFFORTS.items())
def test_together_ladders_match_measured_endpoint_support(
    model: str, efforts: tuple[str, ...]
) -> None:
    profile = load_model_registry().profile("together", model)
    assert profile.efforts == efforts
    for effort in efforts:
        resolve_model_profile("together", model, effort)


@pytest.mark.parametrize(
    ("model", "effort"),
    [
        # Rejected by the endpoint with a 400.
        ("Qwen/Qwen3.8-27B", "minimal"),
        ("Qwen/Qwen3.8-27B", "high"),
        ("Qwen/Qwen3.8-27B", "max"),
        ("deepseek-ai/DeepSeek-V4.1-Flash", "minimal"),
        ("deepseek-ai/DeepSeek-V4.1-Flash", "medium"),
        # Accepted by the endpoint but served at the same budget as `max`, so the
        # profile must not advertise them as separate levels.
        ("zai-org/GLM-5.3-Flash", "minimal"),
        ("zai-org/GLM-5.3-Flash", "medium"),
        ("zai-org/GLM-5.3-Flash", "xhigh"),
    ],
)
def test_together_ladders_exclude_unusable_efforts(model: str, effort: str) -> None:
    with pytest.raises(ModelProfileError, match="does not support effort"):
        resolve_model_profile("together", model, effort)


def test_every_public_model_records_its_window_and_output_limit() -> None:
    profiles = load_model_registry().profiles.values()
    missing = {
        p.model
        for p in profiles
        if p.context_window is None or p.max_output_tokens is None
    }
    # A private deployment with no published figures.
    assert missing == {"zainhas/dsv41"}
    for profile in profiles:
        if profile.context_window and profile.max_output_tokens:
            assert profile.max_output_tokens <= profile.context_window


def test_override_adds_openai_compatible_model(tmp_path: Path) -> None:
    override = tmp_path / "profiles.yaml"
    override.write_text(
        """
version: 1
providers:
  openai-compatible:
    protocol: openai-chat
    models:
      custom/model:
        efforts: [low, high]
""",
        encoding="utf-8",
    )
    profile = resolve_model_profile(
        "openai-compatible",
        "custom/model",
        "high",
        override=override,
    )
    assert profile.model == "custom/model"
    assert profile.source == str(override)


def test_inference_prefers_together_when_model_overlaps(tmp_path: Path) -> None:
    override = tmp_path / "profiles.yaml"
    override.write_text(
        """
version: 1
providers:
  together:
    protocol: openai-chat
    models:
      gpt-5.6:
        efforts: [low, high]
""",
        encoding="utf-8",
    )
    assert load_model_registry(override).infer_provider("gpt-5.6") == "together"


def test_a_model_the_user_listed_is_routed_as_the_user_listed_it(tmp_path: Path) -> None:
    """X-09: the packaged Together entry for the same id would send the
    Together key to the user's endpoint."""
    override = tmp_path / "profiles.yaml"
    override.write_text(
        """
version: 1
providers:
  openai-compatible:
    protocol: openai-chat
    models:
      zai-org/GLM-5.3-Flash:
        efforts: [low]
""",
        encoding="utf-8",
    )
    registry = load_model_registry(override)
    assert registry.infer_provider("zai-org/GLM-5.3-Flash") == "openai-compatible"
    assert load_model_registry().infer_provider("zai-org/GLM-5.3-Flash") == "together"
