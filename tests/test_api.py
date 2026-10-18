import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from harness_cascade import run

_DRIVER = """\
import os
import sys

from harness_cascade import run

os.environ["PYTHONPATH"] = (
    sys.argv[1] + os.pathsep + os.environ.get("PYTHONPATH", "")
)
raise SystemExit(run(sys.argv[2:]))
"""

_STUB_MAIN = """\
import os
import signal
import sys

stop = int(sys.argv[1])
ready_path = sys.argv[2]
record_path = sys.argv[3]
seen: list[int] = []


def on_stop(signum: int, _frame: object) -> None:
    seen.append(signum)


signal.signal(stop, on_stop)
with open(ready_path, "w", encoding="utf-8") as handle:
    handle.write(f"{os.getpid()}\\n")
    handle.flush()
while not seen:
    signal.pause()
with open(record_path, "w", encoding="utf-8") as handle:
    handle.write("".join(f"{item}\\n" for item in seen))
raise SystemExit(128 + seen[0])
"""

_STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


def _caller_stop_handlers() -> dict[int, object]:
    return {signum: signal.getsignal(signum) for signum in _STOP_SIGNALS}


def test_dry_run_returns_the_cli_plan(capfd: pytest.CaptureFixture[str]) -> None:
    handlers = _caller_stop_handlers()
    code = run(
        [
            "--first",
            "pi:low",
            "--model",
            "zai-org/GLM-5.3",
            "--dry-run",
            "-p",
            "test",
        ]
    )
    assert _caller_stop_handlers() == handlers
    assert code == 0
    plan = json.loads(capfd.readouterr().out)
    assert plan["first"] == {"harness": "pi", "effort": "low", "switch_after": None}
    assert plan["second"] is None
    assert plan["provider"] == "together"
    assert plan["model"] == "zai-org/GLM-5.3"
    assert plan["skip_second_if_finished"] is False


def test_invalid_effort_exits_2(capfd: pytest.CaptureFixture[str]) -> None:
    code = run(
        [
            "--first",
            "pi:xhigh",
            "--provider",
            "together",
            "--model",
            "openai/gpt-oss-120b",
            "--dry-run",
            "-p",
            "test",
        ]
    )
    assert code == 2
    err = capfd.readouterr().err
    assert err.startswith("usage: cascade ")
    assert err.endswith(
        "cascade: error: openai/gpt-oss-120b does not support effort 'xhigh' "
        "through together; available: low, medium, high\n"
    )


def test_second_harness_is_forwarded(capfd: pytest.CaptureFixture[str]) -> None:
    code = run(
        [
            "--first",
            "codex:high",
            "--second",
            "pi:high",
            "--switch-after",
            "5",
            "--model",
            "zai-org/GLM-5.3",
            "--dry-run",
            "-p",
            "Find and fix the bug in auth.py",
        ]
    )
    assert code == 0
    plan = json.loads(capfd.readouterr().out)
    assert plan["first"] == {"harness": "codex", "effort": "high", "switch_after": 5}
    assert plan["second"] == {"harness": "pi", "effort": "high"}


def test_a_bare_string_is_rejected() -> None:
    with pytest.raises(TypeError, match="not str or bytes"):
        run("--dry-run")
    with pytest.raises(TypeError, match="not str or bytes"):
        run(b"--dry-run")


def _live_children() -> set[int]:
    listed = subprocess.run(
        ["ps", "-o", "pid=,ppid="],
        capture_output=True,
        text=True,
        check=True,
    )
    parent = os.getpid()
    found: set[int] = set()
    for line in listed.stdout.splitlines():
        pid_text, ppid_text = line.split()
        if int(ppid_text) == parent:
            found.add(int(pid_text))
    return found


def test_a_worker_thread_is_refused_before_a_child(
    capfd: pytest.CaptureFixture[str],
) -> None:
    args = [
        "--first",
        "pi:low",
        "--model",
        "zai-org/GLM-5.3",
        "--dry-run",
        "-p",
        "test",
    ]
    caught: list[BaseException] = []

    def call() -> None:
        try:
            run(args)
        except BaseException as exc:
            caught.append(exc)

    before = _live_children()
    worker = threading.Thread(target=call)
    worker.start()
    worker.join(timeout=10)
    captured = capfd.readouterr()
    assert not worker.is_alive()
    assert len(caught) == 1
    assert type(caught[0]) is RuntimeError
    assert str(caught[0]) == (
        "harness_cascade.run must run on Python's main thread "
        "so it can relay stop signals"
    )
    assert captured.out == ""
    assert captured.err == ""
    assert _live_children() <= before

    assert run(args) == 0
    assert json.loads(capfd.readouterr().out)["model"] == "zai-org/GLM-5.3"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for_ready(path: Path, proc: subprocess.Popen[str], deadline: float) -> int:
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError("stub did not become ready")
        if proc.poll() is not None:
            raise AssertionError(f"driver exited early with {proc.returncode}")
        if path.exists() and path.stat().st_size > 0:
            text = path.read_text(encoding="utf-8")
            if "\n" in text and text.strip():
                return int(text.strip())
        select.select([], [], [], min(0.05, remaining))


@pytest.mark.parametrize(
    "signum",
    _STOP_SIGNALS,
    ids=[signal.Signals(number).name for number in _STOP_SIGNALS],
)
def test_stop_signal_is_forwarded_once(signum: int, tmp_path: Path) -> None:
    stub = tmp_path / "stub"
    package = stub / "harness_cascade"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "__main__.py").write_text(_STUB_MAIN, encoding="utf-8")
    driver = tmp_path / "driver.py"
    driver.write_text(_DRIVER, encoding="utf-8")
    ready = tmp_path / "ready"
    record = tmp_path / "signals"

    proc = subprocess.Popen(
        [
            sys.executable,
            str(driver),
            str(stub),
            str(signum),
            str(ready),
            str(record),
        ],
        text=True,
    )
    child_pid: int | None = None
    try:
        child_pid = _wait_for_ready(ready, proc, time.monotonic() + 10)
        os.kill(proc.pid, signum)
        code = proc.wait(timeout=10)
        assert code == 128 + signum
        assert record.read_text(encoding="utf-8") == f"{signum}\n"
        assert not _alive(proc.pid)
        assert not _alive(child_pid)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        if child_pid is not None and _alive(child_pid):
            os.kill(child_pid, signal.SIGKILL)
