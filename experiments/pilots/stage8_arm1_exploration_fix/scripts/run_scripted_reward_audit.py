#!/usr/bin/env python3
"""Pre-training scripted Base Reward V2 audit for Stage 8 arm1.

Reused unchanged from Stage 7C-Q1 / arm0: arm1 uses the identical
active-time-cost coefficient (0.0005) -- only the exploration schedule
changes in arm1, not the reward -- so the same audit function applies
without modification.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

from thesis.pilots.stage7c_q1_scripted_audit import run_scripted_reward_audit  # noqa: E402


def main() -> int:
    result = run_scripted_reward_audit()
    out = Path(__file__).resolve().parents[1] / "logs" / "scripted_reward_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result.get("passed"):
        print("SCRIPTED AUDIT FAILED — do not start Stage 8 arm1 training", file=sys.stderr)
        return 1
    print("SCRIPTED AUDIT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
