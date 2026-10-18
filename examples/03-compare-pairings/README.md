# 03 · Compare pairings: Codex → Pi vs. Pi → Codex

`stats.py` has three bugs (`mean`, `median` for even lengths, `mode` ties) and
`test_stats.py` checks them. The script runs the same task twice, each time on a
fresh copy: Codex then Pi, and Pi then Codex, handing off after three tool
turns. Afterwards it runs the tests itself rather than trusting what the agents
report.

```bash
.venv/bin/python examples/03-compare-pairings/run.py --dry-run   # both plans, free
.venv/bin/python examples/03-compare-pairings/run.py
```

Needs: the package installed in that Python (see [Setup](../README.md#setup)),
`codex` and `pi` on `PATH`, `TOGETHER_API_KEY`.

**Cost and time:** about 45k tokens for both runs, about $0.04, under a
minute.

## What you'll see

Each run prints only its final answer (`--output-format text`). Then a
comparison:

```
=== Comparison

  pairing          codex:high -> pi:high
  exit             0
  tests pass       True
  tests untouched  True
  turns            0 + 4
  tokens           24,486
  cost             $0.0248

  pairing          pi:high -> codex:high
  exit             0
  tests pass       True
  tests untouched  True
  turns            3 + 1
  tokens           22,001
  cost             $0.0141
```

- **tests pass**: `python3 test_stats.py` succeeded after the run.
- **tests untouched**: the agents did not edit the test file to make it pass.
- **turns**: tool turns in the first and second stage. The first stage stops
  at three.

`0` turns in a stage means that agent did no work, and the other one did it
all. In the run above, Codex as the first stage ended after one reply: GLM-5.3
wrote its planning tool call as plain text, which Codex took as a final answer.
The run still reports `completed`, so check the turns rather than the exit code
when comparing pairings.

## Things to try

- Add `("claude:high", "pi:high")` to `PAIRINGS`.
- Change `SWITCH_AFTER` and see where each pairing spends its tokens.
- Run it a few times. One run per pairing says little, because agents rarely
  take the same path twice.
