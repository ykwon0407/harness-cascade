"""Optional tests using the real installed Claude and Codex executables.

Run explicitly:
    CASCADE_RUN_CLI_E2E=1 uv run pytest tests/test_cli_e2e.py -v -s
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from harness_cascade.config import RunPlan, StagePlan
from harness_cascade.credentials import CredentialSet
from harness_cascade.engine import HandoffEngine
from harness_cascade.events import EventBus

pytestmark = pytest.mark.skipif(
    os.environ.get("CASCADE_RUN_CLI_E2E") != "1",
    reason="set CASCADE_RUN_CLI_E2E=1 to run installed-CLI tests",
)


class MockTogetherHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    requests: ClassVar[list[dict]] = []

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.requests.append(body)
        call_number = len(self.requests)
        text = "TRANSFER READY" if call_number == 2 else "DONE"
        payload = {
            "id": f"chatcmpl_mock_{call_number}",
            "choices": [
                {
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def mock_together():
    MockTogetherHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockTogetherHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def run_plan(
    tmp_path: Path,
    base_url: str,
    first: str,
    second: str | None = None,
) -> dict:
    plan = RunPlan(
        first=StagePlan(first, "low", switch_after=1 if second else None),
        second=StagePlan(second, "low") if second else None,
        workspace=tmp_path,
        model="mock/model",
        base_url=base_url,
        timeout_seconds=60,
    )
    return HandoffEngine(
        plan,
        CredentialSet("together", "mock-key", "test"),
        events=EventBus(f"{first}-{second or 'solo'}"),
        runs_root=tmp_path / ".runs",
    ).run("Reply exactly DONE. Do not use tools.")


@pytest.mark.parametrize(
    ("harness", "executable"),
    [("pi", "pi"), ("claude-code", "claude"), ("codex", "codex")],
)
def test_real_cli_solo(
    tmp_path: Path,
    mock_together: str,
    harness: str,
    executable: str,
) -> None:
    if shutil.which(executable) is None:
        pytest.skip(f"{executable} is not installed")
    result = run_plan(tmp_path, mock_together, harness)
    assert result["status"] == "completed"
    assert result["final_text"] == "DONE"
    assert len(MockTogetherHandler.requests) == 1
    assert MockTogetherHandler.requests[0]["stream"] is False
    assert result["usage"]["total_tokens"] == 12


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (first, second)
        for first in ("claude-code", "pi", "codex")
        for second in ("claude-code", "pi", "codex")
    ],
)
def test_real_cli_cross_handoff(
    tmp_path: Path,
    mock_together: str,
    first: str,
    second: str,
) -> None:
    required = {"claude-code": "claude", "pi": "pi", "codex": "codex"}
    for harness in (first, second):
        if shutil.which(required[harness]) is None:
            pytest.skip(f"{required[harness]} is not installed")
    result = run_plan(tmp_path, mock_together, first, second)
    assert result["status"] == "completed"
    assert result["handoff_summary"] == "TRANSFER READY"
    assert result["final_text"] == "DONE"
    assert len(MockTogetherHandler.requests) == 3
    assert result["handoff"]["usage"]["total_tokens"] == 12
    assert result["usage"]["total_tokens"] == 36
