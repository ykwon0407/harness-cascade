"""Credential resolution, validation, and redaction."""

from __future__ import annotations

import ctypes
import os
import secrets
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import httpx

SECRET_FILE_ENV = "CASCADE_API_KEY_FILE"
PR_SET_DUMPABLE = 4
APPROVED_ENV_KEYS = {
    "API_KEY",
    "LLM_END_POINT",
    "LLM_ENDPOINT",
    "TOGETHER_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
}
# A vendor's key never falls through to an endpoint of the user's choosing.
PROVIDER_ENV_KEYS = {
    "generic": ("API_KEY",),
    "openai": ("OPENAI_API_KEY", "API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY", "API_KEY"),
    "openai-compatible": ("API_KEY",),
    "together": ("TOGETHER_API_KEY", "API_KEY"),
}
# Providers whose endpoint comes only from --llm-endpoint. A stray
# LLM_END_POINT in a shell or a task's .env would otherwise send their key
# wherever it points.
VENDOR_ENDPOINTS = {
    "together": "https://api.together.ai/v1",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
}
_CREDENTIAL_ENV_NAMES = frozenset(
    {
        *APPROVED_ENV_KEYS,
        SECRET_FILE_ENV,
        "LLM_API_KEY",
        "DSV41_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    }
    - {"LLM_END_POINT", "LLM_ENDPOINT"}
)
_CREDENTIAL_ENV_SUFFIXES = (
    "_API_KEY",
    "_AUTH_TOKEN",
    "_ACCESS_TOKEN",
    "_SECRET_ACCESS_KEY",
)


class CredentialError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CredentialSet:
    provider: str
    api_key: str
    source: str
    endpoint: str = ""
    endpoint_source: str = ""
    # The LLM_END_POINT variable that was set but does not apply here.
    ignored_endpoint: str = ""
    # The CASCADE_API_KEY_FILE the key was read from, wiped once agents start.
    secret_file: str = ""

    def local_session_token(self) -> str:
        return f"cascade-{secrets.token_urlsafe(32)}"


def load_approved_dotenv(path: Path, environ: dict[str, str] | None = None) -> None:
    """Load only credential keys, without overriding the process environment."""
    target = os.environ if environ is None else environ
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key not in APPROVED_ENV_KEYS or key in target:
            continue
        target[key] = value.strip().strip("'").strip('"')


def resolve_credentials(
    provider: str,
    *,
    environ: Mapping[str, str] | None = None,
    env_file: Path | None = None,
) -> CredentialSet:
    provider = provider.lower()
    if provider == "generic":
        provider = "openai-compatible"
    key_names = PROVIDER_ENV_KEYS.get(provider)
    if key_names is None:
        raise CredentialError(f"unsupported provider {provider!r}")

    values = dict(os.environ if environ is None else environ)
    if env_file is not None:
        load_approved_dotenv(env_file, values)
    key_name, key, secret_file = _resolve_api_key(key_names, values)
    if not key:
        location = f" or add it to {env_file}" if env_file else ""
        expected = " or ".join((*key_names, SECRET_FILE_ENV))
        raise CredentialError(f"{expected} is not set{location}")
    endpoint_name = next(
        (
            name
            for name in ("LLM_END_POINT", "LLM_ENDPOINT")
            if values.get(name, "").strip()
        ),
        "",
    )
    if endpoint_name and provider in VENDOR_ENDPOINTS:
        return CredentialSet(
            provider=provider,
            api_key=key,
            source=f"environment:{key_name}",
            ignored_endpoint=endpoint_name,
            secret_file=secret_file,
        )
    endpoint = values.get(endpoint_name, "").strip().rstrip("/")
    return CredentialSet(
        provider=provider,
        api_key=key,
        source=f"environment:{key_name}",
        endpoint=endpoint,
        endpoint_source=f"environment:{endpoint_name}" if endpoint_name else "",
        secret_file=secret_file,
    )


def models_url(endpoint: str, provider: str = "openai-compatible") -> str:
    base = endpoint.rstrip("/")
    if provider == "anthropic":
        if base.endswith("/v1"):
            return f"{base}/models"
        return f"{base}/v1/models"
    if base.endswith("/chat/completions"):
        base = base.removesuffix("/chat/completions")
    if base.endswith("/v1"):
        return f"{base}/models"
    return f"{base}/v1/models"


def validate_api_key(
    credentials: CredentialSet,
    *,
    endpoint: str,
    timeout: float = 10.0,
) -> None:
    """Fail before launching an agent when an endpoint rejects the credential."""
    try:
        headers = {"Authorization": f"Bearer {credentials.api_key}"}
        if credentials.provider == "anthropic":
            headers = {
                "x-api-key": credentials.api_key,
                "anthropic-version": "2023-06-01",
            }
        response = httpx.get(
            models_url(endpoint, credentials.provider),
            headers=headers,
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        raise CredentialError(f"LLM endpoint preflight failed: {exc}") from exc
    if response.status_code in {401, 403}:
        raise CredentialError(
            f"LLM endpoint rejected {credentials.source} with HTTP {response.status_code}"
        )
    if response.status_code in {404, 405}:
        return
    if response.status_code >= 400:
        raise CredentialError(
            f"LLM endpoint preflight returned HTTP {response.status_code}: "
            f"{redact(response.text[:200], credentials)}"
        )


def redact(text: str, credentials: CredentialSet | None) -> str:
    if credentials is None or not credentials.api_key:
        return text
    return redact_secret(text, credentials.api_key)


def redact_secret(text: str, secret: str) -> str:
    if not secret:
        return text
    return text.replace(secret, "[REDACTED]")


def is_credential_env(name: str) -> bool:
    if name in _CREDENTIAL_ENV_NAMES:
        return True
    return name.endswith(_CREDENTIAL_ENV_SUFFIXES)


def scrub_credential_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Copy an environment without upstream API keys or secret-file paths."""
    values = os.environ if environ is None else environ
    return {key: value for key, value in values.items() if not is_credential_env(key)}


def lock_down_process_credentials(
    credentials: CredentialSet | None = None,
    environ: dict[str, str] | None = None,
) -> None:
    """Put the key out of the agents' reach once it is in memory.

    The agents run as this process's user. Deleting variables only edits
    libc's copy; /proc/<pid>/environ still shows the block the process started
    with, so the process is also made non-dumpable, which closes its /proc
    files to other processes of the same user. Children regain dumpability at
    exec. The key file, when the key came from one, is wiped.
    """
    target = os.environ if environ is None else environ
    for key in [name for name in target if is_credential_env(name)]:
        del target[key]
    if environ is None:
        _make_undumpable()
    if credentials is not None and credentials.secret_file:
        _shred_secret_file(Path(credentials.secret_file))


def _make_undumpable() -> None:
    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0)
    except (OSError, AttributeError):
        return


def _resolve_api_key(
    key_names: tuple[str, ...],
    values: Mapping[str, str],
) -> tuple[str, str, str]:
    secret_file = values.get(SECRET_FILE_ENV, "").strip()
    if secret_file:
        path = Path(secret_file).expanduser()
        if path.is_symlink():
            raise CredentialError(
                f"{SECRET_FILE_ENV} must not be a symlink, since cascade wipes "
                f"the file after reading it: {path}"
            )
        try:
            key = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise CredentialError(
                f"{SECRET_FILE_ENV} is not readable: {path}"
            ) from exc
        if key:
            return SECRET_FILE_ENV, key, str(path)
    key_name = next((name for name in key_names if values.get(name, "").strip()), "")
    return key_name, values.get(key_name, "").strip(), ""


def _shred_secret_file(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    except OSError:
        return
    try:
        size = max(os.fstat(fd).st_size, 1)
        os.write(fd, b"\0" * size)
        os.fsync(fd)
        os.ftruncate(fd, 0)
        os.fchmod(fd, 0o000)
    except OSError:
        return
    finally:
        os.close(fd)
