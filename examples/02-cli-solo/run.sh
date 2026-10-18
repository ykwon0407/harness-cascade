#!/usr/bin/env bash
# One agent (Pi) works alone on a small bug, run from the shell.
#
#   bash examples/02-cli-solo/run.sh             # real run
#   bash examples/02-cli-solo/run.sh --dry-run   # print the plan only
#
# Needs `cascade` (uv tool install .) and `pi` on PATH, and TOGETHER_API_KEY.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
workspace="$(mktemp -d "${TMPDIR:-/tmp}/cascade-02-XXXXXX")"

cat > "$workspace/slugify.py" <<'EOF'
def slugify(title: str) -> str:
    """Lowercase, words joined by single hyphens: "Hello,  World!" -> "hello-world"."""
    return title.lower().replace(" ", "-")
EOF
git -C "$workspace" init -q
git -C "$workspace" add -A
git -C "$workspace" commit -qm init

set +e
cascade --first pi:high \
  --model zai-org/GLM-5.3 \
  --workspace "$workspace" \
  --output-format text \
  -p "slugify.py does not match its docstring. Fix it and check a few inputs." \
  "$@"
code=$?
set -e

echo
echo "exit code: $code"
case " $* " in *" --dry-run "*) rm -rf "$workspace"; exit "$code" ;; esac
echo "workspace: $workspace"
run="$(ls -dt "$workspace"/.cascade/runs/* 2>/dev/null | head -1 || true)"
if [ -n "$run" ]; then
  echo "artifacts: $run"
  echo
  python3 "$here/../cost.py" "$run"
  echo
  echo "What the agent changed:"
  git -C "$workspace" --no-pager diff
fi
exit "$code"
