#!/usr/bin/env python3
"""Aggregate Stage 7C-Q1 episode CSVs into seed×checkpoint summary for the gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

from thesis.pilots.stage7c_q1_eval import compute_swap_eligibility  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes-glob", required=True, help="Glob of per-seed episode CSVs")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    paths = sorted(Path().glob(args.episodes_glob)) if "*" in args.episodes_glob else []
    if not paths:
        # allow absolute / recursive via Path.parent.glob
        root = Path(args.episodes_glob)
        if root.is_file():
            paths = [root]
        else:
            parent = root.parent if root.parent.exists() else Path(".")
            paths = sorted(parent.glob(root.name))
    if not paths:
        print(f"No episode CSVs matched {args.episodes_glob}", file=sys.stderr)
        return 2

    frames = [pd.read_csv(p) for p in paths]
    ep = pd.concat(frames, ignore_index=True)
    rows = []
    for (seed, step), g in ep.groupby(["master_seed", "checkpoint_step"]):
        episodes = g.to_dict(orient="records")
        rows.append(
            {
                "master_seed": int(seed),
                "checkpoint_step": int(step),
                "success_rate": float(g["success"].mean()),
                "collision_rate": float(g["collision"].mean()),
                "truncation_rate": float(g["truncation"].mean())
                if "truncation" in g
                else float(g.get("truncated", pd.Series([0.0])).mean()),
                "swap_eligibility": compute_swap_eligibility(episodes),
                "n_episodes": int(len(g)),
            }
        )
    out = pd.DataFrame(rows).sort_values(["master_seed", "checkpoint_step"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} rows={len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
