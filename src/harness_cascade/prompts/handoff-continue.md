Continue the original task in the same workspace.

The previous agent was stopped at a stage boundary. It did not necessarily
choose to hand off, so its work may be incomplete or mid-edit. The transfer
record and mechanical snapshot below are hints, not ground truth. They may be
missing, truncated, stale, or wrong. When they disagree, trust the files and
reproducible command output.

Before changing anything:

1. Inspect the workspace and the artifacts named in the transfer record.
2. Re-run the most relevant check or test when practical.
3. Keep prior work only when it is actually correct.
4. Continue toward the original objective. Do not optimize for the summary or
   start over unless the existing work is wrong or unusable.

## Original task

$original_prompt

## Previous agent transfer record

$handoff_summary

## Mechanical workspace snapshot

Working directory: `$workspace`

### `git status --short`

```text
$git_status
```

### `git diff --stat`

```text
$git_diff_stat
```

### Recent commands observed by Cascade

$recent_commands

The original task is the success criterion. Complete it, verify the result, and
report any remaining uncertainty honestly.
