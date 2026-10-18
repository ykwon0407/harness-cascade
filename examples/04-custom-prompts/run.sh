#!/usr/bin/env bash
# Claude Code starts, then Pi continues, using the handoff prompts in this
# folder instead of the packaged defaults.
#
#   bash examples/04-custom-prompts/run.sh             # real run
#   bash examples/04-custom-prompts/run.sh --dry-run   # print the plan only
#
# Needs `cascade` (uv tool install .), `claude` and `pi` on PATH, and
# TOGETHER_API_KEY.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
workspace="$(mktemp -d "${TMPDIR:-/tmp}/cascade-04-XXXXXX")"

cat > "$workspace/inventory.py" <<'EOF'
def restock(stock: dict[str, int], order: dict[str, int]) -> dict[str, int]:
    """Return a new dict with each ordered quantity added. Never mutates stock."""
    for item, qty in order.items():
        stock[item] = stock[item] + qty
    return stock
EOF
git -C "$workspace" init -q
git -C "$workspace" add -A
git -C "$workspace" commit -qm init

set +e
cascade --first claude:high --second pi:high --switch-after 2 \
  --model zai-org/GLM-5.3 \
  --summary-prompt "$here/handoff-summary.md" \
  --continue-prompt "$here/handoff-continue.md" \
  --workspace "$workspace" \
  -p "restock() in inventory.py mutates its input and raises KeyError for items not yet in stock. Fix both and verify." \
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
  echo "Handoff note written with the custom prompt:"
  cat "$run/handoff-summary.md"
  echo
  echo "What the agents changed:"
  git -C "$workspace" --no-pager diff
fi
exit "$code"
