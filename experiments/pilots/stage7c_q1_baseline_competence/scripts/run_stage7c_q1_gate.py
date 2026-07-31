#!/usr/bin/env python3
"""Stage 7C-Q1 competence gate CLI (PASS/FAIL/INVALID only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

from thesis.pilots.stage7c_q1_gate import evaluate_gate_from_csv  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed-checkpoint-csv", type=Path, required=True)
    p.add_argument("--integrity-ok", action="store_true", default=True)
    p.add_argument("--integrity-fail", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    integrity_ok = False if args.integrity_fail else bool(args.integrity_ok)
    result = evaluate_gate_from_csv(str(args.seed_checkpoint_csv), integrity_ok=integrity_ok)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    status = result.get("status")
    if status == "PASS":
        return 0
    if status == "FAIL":
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
