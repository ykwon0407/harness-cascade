"""Exact provider/model capability profiles."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


PACKAGED_SOURCE = "packaged:model_profiles.yaml"


class ModelProfileError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ModelProfile:
    provider: str
    model: str
    protocol: str
    efforts: tuple[str, ...]
    source: str
    # Agents that manage their own context need to be told how much room they
    # have. Both are optional because a wrong number is worse than no number:
    # too small and the agent compacts away work that still fits, too large and
    # it walks into an upstream rejection. Only set them where the value has a
    # source, and prefer limits.discover_model_limits, which asks the provider
    # and treats these as the offline fallback.
    context_window: int | None = None
    max_output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ModelRegistry:
    profiles: dict[tuple[str, str], ModelProfile]

    def infer_provider(self, model: str) -> str:
        candidates = [
            profile for profile in self.profiles.values() if profile.model == model
        ]
        # A model the user listed in their own profiles is theirs to route; the
        # packaged Together entry for the same id would send the Together key
        # to the user's endpoint.
        own = [profile for profile in candidates if profile.source != PACKAGED_SOURCE]
        providers = sorted({profile.provider for profile in own or candidates})
        if "together" in providers:
            return "together"
        if len(providers) == 1:
            return providers[0]
        if not providers:
            raise ModelProfileError(
                f"no model profile for {model!r}; specify --provider and "
                "provide --model-profiles"
            )
        raise ModelProfileError(
            f"model {model!r} is available through multiple providers: "
            f"{', '.join(providers)}; specify --provider"
        )

    def profile(self, provider: str, model: str) -> ModelProfile:
        key = (provider, model)
        profile = self.profiles.get(key)
        if profile is not None:
            return profile
        available = sorted(
            candidate.model
            for candidate in self.profiles.values()
            if candidate.provider == provider
        )
        suffix = (
            f"; available {provider} models: {', '.join(available)}"
            if available
            else f"; no built-in {provider} models (provide --model-profiles)"
        )
        raise ModelProfileError(
            f"no model profile for provider {provider!r}, model {model!r}{suffix}"
        )


def _read_yaml(path: Path | None) -> tuple[dict[str, Any], str]:
    if path is None:
        resource = resources.files("harness_cascade").joinpath("model_profiles.yaml")
        source = PACKAGED_SOURCE
        text = resource.read_text(encoding="utf-8")
    else:
        resolved = path.expanduser().resolve()
        source = str(resolved)
        try:
            text = resolved.read_text(encoding="utf-8")
        except OSError as exc:
            raise ModelProfileError(
                f"could not read model profiles {resolved}: {exc}"
            ) from exc
    try:
        raw = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ModelProfileError(f"invalid model profiles YAML in {source}: {exc}") from exc
    if not isinstance(raw, dict) or int(raw.get("version") or 0) != 1:
        raise ModelProfileError(f"{source} must be a version 1 model profile mapping")
    return raw, source


def _token_limit(model_raw: dict[str, Any], name: str, where: str) -> int | None:
    raw = model_raw.get(name)
    if raw is not None and (not isinstance(raw, int) or isinstance(raw, bool) or raw <= 0):
        raise ModelProfileError(f"{where} has invalid {name} {raw!r}")
    return raw


def _profiles(raw: dict[str, Any], source: str) -> dict[tuple[str, str], ModelProfile]:
    providers = raw.get("providers")
    if not isinstance(providers, dict):
        raise ModelProfileError(f"{source}: providers must be a mapping")
    result: dict[tuple[str, str], ModelProfile] = {}
    for provider, provider_raw in providers.items():
        if not isinstance(provider_raw, dict):
            raise ModelProfileError(f"{source}: provider {provider!r} must be a mapping")
        protocol = str(provider_raw.get("protocol") or "").strip()
        if protocol not in {"openai-responses", "anthropic-messages", "openai-chat"}:
            raise ModelProfileError(
                f"{source}: provider {provider!r} has invalid protocol {protocol!r}"
            )
        models = provider_raw.get("models") or {}
        if not isinstance(models, dict):
            raise ModelProfileError(f"{source}: {provider}.models must be a mapping")
        for model, model_raw in models.items():
            if not isinstance(model_raw, dict):
                raise ModelProfileError(
                    f"{source}: profile {provider}/{model} must be a mapping"
                )
            efforts_raw = model_raw.get("efforts")
            if not isinstance(efforts_raw, list) or not efforts_raw:
                raise ModelProfileError(
                    f"{source}: profile {provider}/{model} requires non-empty efforts"
                )
            efforts = tuple(str(value).strip() for value in efforts_raw)
            if any(not value for value in efforts) or len(set(efforts)) != len(efforts):
                raise ModelProfileError(
                    f"{source}: profile {provider}/{model} has invalid efforts"
                )
            where = f"{source}: profile {provider}/{model}"
            key = (str(provider), str(model))
            result[key] = ModelProfile(
                provider=key[0],
                model=key[1],
                protocol=protocol,
                efforts=efforts,
                source=source,
                context_window=_token_limit(model_raw, "context_window", where),
                max_output_tokens=_token_limit(model_raw, "max_output_tokens", where),
            )
    return result


def load_model_registry(override: Path | None = None) -> ModelRegistry:
    packaged_raw, packaged_source = _read_yaml(None)
    profiles = _profiles(packaged_raw, packaged_source)
    if override is not None:
        override_raw, override_source = _read_yaml(override)
        profiles.update(_profiles(override_raw, override_source))
    return ModelRegistry(profiles)


def resolve_model_profile(
    provider: str,
    model: str,
    effort: str,
    *,
    override: Path | None = None,
) -> ModelProfile:
    profile = load_model_registry(override).profile(provider, model)
    if effort not in profile.efforts:
        raise ModelProfileError(
            f"{model} does not support effort {effort!r} through {provider}; "
            f"available: {', '.join(profile.efforts)}"
        )
    return profile
