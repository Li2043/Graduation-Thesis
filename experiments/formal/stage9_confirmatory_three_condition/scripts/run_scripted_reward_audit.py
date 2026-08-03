#!/usr/bin/env python3
"""Pre-training scripted Base Reward V2 audit for Stage 9.

Reused unchanged from Stage 7C-Q1 / every Stage 8 arm / the Stage 8 gate:
Stage 9 uses the identical active-time-cost coefficient (0.0005) and Base
Reward V2 composition for all three conditions -- only the PBRS shaping term
varies across conditions, which this base-reward audit does not (and should
not) cover. PBRS correctness is exercised separately by the smoke test
(`run_smoke_stage9.py`), which asserts non-zero shaping for mean_pbrs/
min_pbrs and exact-zero shaping for baseline.
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
        print("SCRIPTED AUDIT FAILED — do not start Stage 9 training", file=sys.stderr)
        return 1
    print("SCRIPTED AUDIT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
