# Examples

Each example creates a small Git repository with a planted bug in a temporary
directory, points cascade at it, and prints the exit code, the run's artifact
directory, the cost, and the diff the agents made. Nothing in this repository is
modified.

| Example | What it shows | Harnesses | Time | Cost |
|---|---|---|---|---|
| [01-python-api](01-python-api/) | `harness_cascade.run` from Python, one handoff | Codex → Pi | 10–60 s | ~$0.02–0.03 |
| [02-cli-solo](02-cli-solo/) | The `cascade` CLI with one agent, plain-text output | Pi | under 30 s | ~$0.01 |
| [03-compare-pairings](03-compare-pairings/) | The same task in both orders, scored by its tests | Codex ↔ Pi | under 1 min | ~$0.04 |
| [04-custom-prompts](04-custom-prompts/) | Your own `--summary-prompt` and `--continue-prompt` | Claude Code → Pi | ~15 s | ~$0.01–0.02 |

Times and costs were measured in October 2026 with `zai-org/GLM-5.3` on
Together AI ($1.40 per million input tokens, $0.26 cached, $4.40 output). Agents
don't take the same path twice, so expect some spread. Every example accepts
`--dry-run`, which prints the plan and costs nothing.

## Setup

From the repository root, with Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```bash
uv tool install .     # the `cascade` CLI, for 02 and 04
uv sync               # .venv with the package installed, for 01 and 03
export TOGETHER_API_KEY=...
```

You also need the harness CLIs each example uses (`codex`, `pi`, `claude`) on
your `PATH`.

## Reading the results

[`cost.py`](cost.py) prints the tokens and cost of any run:

```bash
python3 examples/cost.py /path/to/workspace/.cascade/runs/<run-id>
```

Each run directory also holds `result.json` (status, final answer, usage),
one log per stage, `handoff-summary.md` (the note the first agent wrote), and
`second-prompt.md` (exactly what the second agent received). The temporary
workspaces are kept so you can inspect them; delete them when done.
