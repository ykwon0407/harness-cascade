# 02 · CLI: one agent, no handoff

`slugify("Hello,  World!")` should give `hello-world`, but the code only swaps
spaces for hyphens. Pi works alone, started from the `cascade` CLI with
`--output-format text`, which prints only the final answer.

```bash
bash examples/02-cli-solo/run.sh --dry-run   # plan only, free
bash examples/02-cli-solo/run.sh
```

Needs: `cascade` and `pi` on `PATH`, `TOGETHER_API_KEY`.

**Cost and time:** about 7k tokens, about $0.01, under 30 seconds.

## What you'll see

No progress boxes, because of `--output-format text`. After a short wait, Pi's
final answer: what was wrong and how it checked the fix. Then:

```
exit code: 0
run <run-id>  status: completed
  first        7,244 tokens  $0.0063
  total        7,244 tokens  $0.0063  (4,928 input tokens cached)

What the agent changed:
-    return title.lower().replace(" ", "-")
+    words = [word for word in title.lower().split() if word]
+    ...
```

The exact fix varies between runs.

## Things to try

- Drop `--output-format text` to see the colorful progress view, or use
  `--output-format jsonl` to get one event per line for your own tooling.
- Replace `pi:high` with `codex:high` or `claude:high`.
- Add `--second codex:high --switch-after 1` to turn it into a handoff.
