# 01 · Python API: Codex hands off to Pi

`auth.py` accepts any prefix of a password (`stored.startswith(given)`). Codex
inspects the code for two tool turns, writes a handoff note, and Pi finishes the
fix, all started from Python with `harness_cascade.run`.

```bash
.venv/bin/python examples/01-python-api/run.py --dry-run   # plan only, free
.venv/bin/python examples/01-python-api/run.py
```

Needs: the package installed in that Python (see [Setup](../README.md#setup)),
`codex` and `pi` on `PATH`, `TOGETHER_API_KEY`.

**Cost and time:** about 28k tokens, $0.02–0.03, 10–60 seconds.

## What you'll see

1. A **CASCADE START** box with the route `codex:high → pi:high`.
2. **FIRST HARNESS · codex**: two turns of shell commands that list and read
   `auth.py`, ending with "configured handoff boundary reached after 2 turns".
3. **HANDOFF SUMMARY**: Codex's note, which usually names the bug and the fix
   (`stored == given`) but says nothing has been changed yet.
4. **SECOND HARNESS · pi**: Pi reads the file, edits it, and checks a few
   inputs.
5. A **CASCADE COMPLETED** box with tokens per stage, then the script's own
   summary:

```
exit code: 0
workspace: /tmp/cascade-01-...
artifacts: /tmp/cascade-01-.../.cascade/runs/<run-id>

run <run-id>  status: completed
  first        6,292 tokens  $0.0029
  handoff      5,784 tokens  $0.0040
  second      15,943 tokens  $0.0104
  total       28,019 tokens  $0.0173  (22,464 input tokens cached)

What the agents changed:
-    return stored.startswith(given)
+    return stored == given
```

## Things to try

- Change `--switch-after` to `1` or `5` and see how much the second agent
  still has to do.
- Swap `codex:high` and `pi:high`.
- Read `second-prompt.md` in the artifacts to see exactly what Pi was given.
