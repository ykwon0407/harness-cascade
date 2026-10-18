# 04 · Custom handoff prompts: Claude Code → Pi

`restock()` mutates its input and raises `KeyError` for new items. Claude Code
starts, then Pi continues, but the handoff uses the two prompt files in this
folder instead of the packaged defaults:

- [`handoff-summary.md`](handoff-summary.md), passed as `--summary-prompt`.
  This prompt is sent to the first agent, with tools off, to write its note.
  This version asks for exactly five lines: Done, Evidence, Not done, Risk,
  Next step.
- [`handoff-continue.md`](handoff-continue.md), passed as `--continue-prompt`.
  This is the second agent's first message. It can use `$original_prompt`,
  `$handoff_summary`, `$workspace`, `$git_status`, `$git_diff_stat` and
  `$recent_commands`.

The packaged defaults are in `src/harness_cascade/prompts/`. Copy them as a
starting point for your own.

```bash
bash examples/04-custom-prompts/run.sh --dry-run   # plan only, free
bash examples/04-custom-prompts/run.sh
```

Needs: `cascade`, `claude` and `pi` on `PATH`, `TOGETHER_API_KEY`.

**Cost and time:** 10k–20k tokens, $0.01–0.02, about 15 seconds.

## What you'll see

The usual progress view, where the **HANDOFF SUMMARY** box now has the
five-line shape. The script then prints the note again on its own:

```
Handoff note written with the custom prompt:
- **Done:** nothing — inventory.py was read but not yet modified
- **Evidence:** none — only `ls`/`find` and file read performed; no tests run
- **Not done:** fix restock() to (1) not mutate its input and (2) not raise KeyError ...
- **Risk:** a "fix" that copies the dict shallowly ...
- **Next step:** open inventory.py and rewrite restock() ...
```

Then the diff, typically a copy of `stock` plus `stock.get(item, 0)`. Pi may also
leave a new, untracked test file.

## Things to try

- Run the same task without the two flags and compare the default note.
- Remove `$handoff_summary` from `handoff-continue.md` to see how Pi does
  with only the mechanical snapshot.
