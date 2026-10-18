"""Codex inspects a bug for two tool turns, then hands off to Pi, which fixes it.

    python examples/01-python-api/run.py            # real run
    python examples/01-python-api/run.py --dry-run  # print the plan only

Needs harness-cascade installed in this Python, `codex` and `pi` on PATH, and
TOGETHER_API_KEY set.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import harness_cascade

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cost import report  # noqa: E402

AUTH_PY = '''\
def check_password(stored: str, given: str) -> bool:
    """Return True only when the given password matches the stored one."""
    return stored.startswith(given)
'''


def make_workspace() -> Path:
    workspace = Path(tempfile.mkdtemp(prefix="cascade-01-"))
    (workspace / "auth.py").write_text(AUTH_PY, encoding="utf-8")
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "init"]):
        subprocess.run(["git", *args], cwd=workspace, check=True)
    return workspace


def main() -> int:
    workspace = make_workspace()
    code = harness_cascade.run([
        "--first", "codex:high",
        "--second", "pi:high",
        "--switch-after", "2",
        "--model", "zai-org/GLM-5.3",
        "--workspace", str(workspace),
        "-p", "Find and fix the bug in auth.py",
        *sys.argv[1:],
    ])
    print(f"\nexit code: {code}")
    if "--dry-run" in sys.argv[1:]:
        shutil.rmtree(workspace)
        return code
    print(f"workspace: {workspace}")

    runs = sorted((workspace / ".cascade" / "runs").glob("*"))
    if runs:
        print(f"artifacts: {runs[-1]}\n")
        print(report(runs[-1]))
        print("\nWhat the agents changed:", flush=True)
        subprocess.run(["git", "--no-pager", "diff"], cwd=workspace)
    return code


if __name__ == "__main__":
    sys.exit(main())
