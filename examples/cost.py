"""Print the token usage and estimated cost of a cascade run.

    python3 examples/cost.py <workspace>/.cascade/runs/<run-id>

Prices are Together AI's list prices for zai-org/GLM-5.3 (October 2026), in
USD per million tokens. Edit PRICES for another model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PRICES = {"input": 1.40, "cached_input": 0.26, "output": 4.40}


def cost(usage: dict) -> float:
    # input_tokens includes cached tokens; cache_read_tokens is the part of it
    # billed at the cached rate.
    cached = usage.get("cache_read_tokens", 0)
    fresh = usage.get("input_tokens", 0) - cached
    return (
        fresh * PRICES["input"]
        + cached * PRICES["cached_input"]
        + usage.get("output_tokens", 0) * PRICES["output"]
    ) / 1_000_000


def report(run_dir: Path) -> str:
    result = json.loads((Path(run_dir) / "result.json").read_text(encoding="utf-8"))
    lines = [f"run {result['run_id']}  status: {result['status']}"]
    for stage in ("first", "handoff", "second"):
        usage = (result.get(stage) or {}).get("usage")
        if usage:
            lines.append(
                f"  {stage:<8} {usage['total_tokens']:>8,} tokens  ${cost(usage):.4f}"
            )
    usage = result["usage"]
    lines.append(
        f"  {'total':<8} {usage['total_tokens']:>8,} tokens  ${cost(usage):.4f}"
        f"  ({usage.get('cache_read_tokens', 0):,} input tokens cached)"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(report(Path(sys.argv[1])))
