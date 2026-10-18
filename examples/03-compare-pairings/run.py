"""Run the same task with Codex -> Pi and Pi -> Codex, then compare the results.

    python examples/03-compare-pairings/run.py            # real runs
    python examples/03-compare-pairings/run.py --dry-run  # print both plans only

Each pairing gets its own fresh copy of a small module with three bugs. After
each run the script executes the tests itself, so the score doesn't depend on
what the agents claim.

Needs harness-cascade installed in this Python, `codex` and `pi` on PATH, and
TOGETHER_API_KEY set.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import harness_cascade

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cost import cost  # noqa: E402

PAIRINGS = [("codex:high", "pi:high"), ("pi:high", "codex:high")]
SWITCH_AFTER = "3"
PROMPT = (
    "The tests in test_stats.py fail. Fix stats.py so that "
    "`python3 test_stats.py` passes. Do not edit the tests."
)

STATS_PY = '''\
def mean(values):
    return sum(values) / len(values) - 1


def median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def mode(values):
    """Most common value; ties go to the smallest."""
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=counts.get)
'''

TEST_STATS_PY = '''\
from stats import mean, median, mode

assert mean([1, 2, 3, 4]) == 2.5
assert median([3, 1, 2]) == 2
assert median([4, 1, 3, 2]) == 2.5
assert mode([3, 3, 1, 1, 2]) == 1
assert mode([5]) == 5
print("all tests passed")
'''


def make_workspace(label: str) -> Path:
    workspace = Path(tempfile.mkdtemp(prefix=f"cascade-03-{label}-"))
    (workspace / "stats.py").write_text(STATS_PY, encoding="utf-8")
    (workspace / "test_stats.py").write_text(TEST_STATS_PY, encoding="utf-8")
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "init"]):
        subprocess.run(["git", *args], cwd=workspace, check=True)
    return workspace


def score(workspace: Path) -> dict:
    untouched = (workspace / "test_stats.py").read_text(encoding="utf-8") == TEST_STATS_PY
    passed = (
        subprocess.run(
            [sys.executable, "test_stats.py"], cwd=workspace, capture_output=True
        ).returncode
        == 0
    )
    row = {"tests pass": passed, "tests untouched": untouched}
    runs = sorted((workspace / ".cascade" / "runs").glob("*"))
    if runs:
        result = json.loads((runs[-1] / "result.json").read_text(encoding="utf-8"))
        row["turns"] = "{} + {}".format(
            (result.get("first") or {}).get("turns", "-"),
            (result.get("second") or {}).get("turns", "-"),
        )
        row["tokens"] = f"{result['usage']['total_tokens']:,}"
        row["cost"] = f"${cost(result['usage']):.4f}"
        row["artifacts"] = str(runs[-1])
    return row


def main() -> int:
    rows = []
    for first, second in PAIRINGS:
        label = f"{first.split(':')[0]}-{second.split(':')[0]}"
        workspace = make_workspace(label)
        print(f"\n=== {first} -> {second}  ({workspace})", flush=True)
        code = harness_cascade.run([
            "--first", first,
            "--second", second,
            "--switch-after", SWITCH_AFTER,
            "--model", "zai-org/GLM-5.3",
            "--workspace", str(workspace),
            "--output-format", "text",
            "-p", PROMPT,
            *sys.argv[1:],
        ])
        if "--dry-run" in sys.argv[1:]:
            shutil.rmtree(workspace)
            continue
        rows.append({"pairing": f"{first} -> {second}", "exit": code, **score(workspace)})

    if not rows:
        return 0
    print("\n=== Comparison")
    for row in rows:
        print()
        for key, value in row.items():
            print(f"  {key:<16} {value}")
    return 0 if all(row["exit"] == 0 for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
