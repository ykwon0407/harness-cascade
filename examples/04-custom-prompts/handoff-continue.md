Continue the original task in the same workspace.

Treat the transfer record and mechanical snapshot as hints rather than ground
truth. Inspect the named artifacts and re-run the most relevant check before
changing anything. Trust files and reproducible output when they disagree.

## Original task

$original_prompt

## Previous agent transfer record

$handoff_summary

## Mechanical workspace snapshot

- Working directory: `$workspace`
- `git status --short`:

```text
$git_status
```

- `git diff --stat`:

```text
$git_diff_stat
```

- Recent commands:

$recent_commands

Complete and verify the original task. Report remaining uncertainty honestly.
