from pathlib import Path

import pytest

from harness_cascade.credentials import (
    SECRET_FILE_ENV,
    CredentialError,
    lock_down_process_credentials,
    models_url,
    resolve_credentials,
    scrub_credential_env,
)


def test_generic_api_key_and_endpoint_are_resolved() -> None:
    credentials = resolve_credentials(
        "openai-compatible",
        environ={
            "API_KEY": "generic-secret",
            "LLM_END_POINT": "https://llm.example/v1/",
        },
    )
    assert credentials.provider == "openai-compatible"
    assert credentials.api_key == "generic-secret"
    assert credentials.endpoint == "https://llm.example/v1"
    assert credentials.source == "environment:API_KEY"


def test_llm_endpoint_spelling_alias_is_accepted() -> None:
    credentials = resolve_credentials(
        "openai-compatible",
        environ={
            "API_KEY": "generic-secret",
            "LLM_ENDPOINT": "https://api.openai.com/v1",
        },
    )
    assert credentials.endpoint == "https://api.openai.com/v1"
    assert credentials.endpoint_source == "environment:LLM_ENDPOINT"


def test_legacy_together_key_selects_together_without_endpoint() -> None:
    credentials = resolve_credentials(
        "together",
        environ={"TOGETHER_API_KEY": "together-secret"},
    )
    assert credentials.provider == "together"
    assert credentials.endpoint == ""


def test_process_environment_takes_precedence_over_dotenv(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "API_KEY=file-secret\nLLM_END_POINT=https://file.example/v1\n",
        encoding="utf-8",
    )
    credentials = resolve_credentials(
        "openai-compatible",
        environ={
            "API_KEY": "process-secret",
            "LLM_END_POINT": "https://process.example/v1",
        },
        env_file=env_file,
    )
    assert credentials.api_key == "process-secret"
    assert credentials.endpoint == "https://process.example/v1"


def test_openai_and_anthropic_use_provider_specific_keys() -> None:
    openai = resolve_credentials(
        "openai",
        environ={"OPENAI_API_KEY": "openai-secret", "API_KEY": "fallback"},
    )
    anthropic = resolve_credentials(
        "anthropic",
        environ={"ANTHROPIC_API_KEY": "anthropic-secret", "API_KEY": "fallback"},
    )
    assert openai.provider == "openai"
    assert openai.api_key == "openai-secret"
    assert anthropic.provider == "anthropic"
    assert anthropic.api_key == "anthropic-secret"


def test_models_url_accepts_base_or_chat_completions_url() -> None:
    assert models_url("https://llm.example/v1") == "https://llm.example/v1/models"
    assert (
        models_url("https://llm.example/v1/chat/completions")
        == "https://llm.example/v1/models"
    )
    assert models_url("https://llm.example") == "https://llm.example/v1/models"


def test_secret_file_is_used_then_shredded(tmp_path: Path) -> None:
    secret = tmp_path / "cascade-api-key"
    secret.write_text("file-secret\n", encoding="utf-8")
    secret.chmod(0o600)
    credentials = resolve_credentials(
        "together",
        environ={SECRET_FILE_ENV: str(secret)},
    )
    assert credentials.api_key == "file-secret"
    assert credentials.source == f"environment:{SECRET_FILE_ENV}"
    lock_down_process_credentials(credentials, {SECRET_FILE_ENV: str(secret)})
    assert secret.stat().st_size == 0
    assert secret.stat().st_mode & 0o777 == 0


def test_a_key_file_that_was_not_the_source_is_left_alone(tmp_path: Path) -> None:
    secret = tmp_path / "cascade-api-key"
    secret.write_text("", encoding="utf-8")
    credentials = resolve_credentials(
        "together",
        environ={SECRET_FILE_ENV: str(secret), "TOGETHER_API_KEY": "together-secret"},
    )
    assert credentials.secret_file == ""
    lock_down_process_credentials(credentials, {SECRET_FILE_ENV: str(secret)})
    assert secret.stat().st_mode & 0o777 != 0


def test_a_symlinked_key_file_is_refused_rather_than_wiped_through(tmp_path: Path) -> None:
    target = tmp_path / "real-key"
    target.write_text("file-secret\n", encoding="utf-8")
    link = tmp_path / "cascade-api-key"
    link.symlink_to(target)
    with pytest.raises(CredentialError, match="symlink"):
        resolve_credentials("together", environ={SECRET_FILE_ENV: str(link)})
    assert target.read_text(encoding="utf-8") == "file-secret\n"


def test_a_stray_endpoint_does_not_redirect_a_vendor_key() -> None:
    """X-08: LLM_END_POINT in a shell or a task's .env is not an instruction."""
    credentials = resolve_credentials(
        "together",
        environ={"TOGETHER_API_KEY": "together-secret", "LLM_END_POINT": "http://10.0.0.9/v1"},
    )
    assert credentials.endpoint == ""
    assert credentials.ignored_endpoint == "LLM_END_POINT"


def test_a_vendor_key_never_falls_through_to_a_generic_endpoint() -> None:
    with pytest.raises(CredentialError):
        resolve_credentials(
            "openai-compatible",
            environ={"OPENAI_API_KEY": "openai-secret", "TOGETHER_API_KEY": "together-secret"},
        )


def test_scrub_drops_upstream_keys_but_keeps_local_token() -> None:
    cleaned = scrub_credential_env(
        {
            "PATH": "/usr/bin",
            "TOGETHER_API_KEY": "together-secret",
            "LLM_API_KEY": "llm-secret",
            "CASCADE_API_KEY_FILE": "/run/secrets/cascade-api-key",
            "CASCADE_GATEWAY_TOKEN": "local-secret",
            "HOME": "/root",
        }
    )
    assert cleaned == {
        "PATH": "/usr/bin",
        "CASCADE_GATEWAY_TOKEN": "local-secret",
        "HOME": "/root",
    }
