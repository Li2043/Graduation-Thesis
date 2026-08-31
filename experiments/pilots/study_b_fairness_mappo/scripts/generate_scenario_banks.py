#!/usr/bin/env python3
"""Generates and SHA-256-hashes Study B's frozen evaluation scenario banks
-- new_research_plan.md's "固定 evaluation banks" table (Q/M/H1/H0/OOD-S).
Run ONCE, before Phase 1 qualification begins, then never regenerate
(these banks must stay byte-identical for the rest of the study; hashes
let anyone verify that later)."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(REPO_SRC))

from thesis.study_b.scenario_generator import generate_scenario  # noqa: E402
from thesis.study_b.training_common import save_scenario_bank  # noqa: E402

ROLE_MEMBERS = {"ramp": ["V0", "V1"], "mainline": ["V2", "V3"]}

# (bank_name, size, traffic_type, seed_offset, kwargs). seed_offset keeps
# most banks' RNG streams disjoint (new_research_plan.md: "All banks use
# mutually disjoint RNG seeds"), EXCEPT H0 and H1 deliberately SHARE their
# offset (920_000): generate_scenario's role/ttc-slot shuffle and TTC
# jitter are drawn before traffic_type's fast/slow-vs-v_ref branching, so
# the same episode_seed makes H0[k]/H1[k] share identical geometry/
# identity-shuffle/wave structure and differ ONLY in target speed -- the
# explicit "H0 and H1 banks should be pairable... scenario k in H0 and
# scenario k in H1 use the same geometry, identity shuffle and nominal
# interaction-wave structure" requirement.
BANK_SPECS = [
    ("Q", 64, "heterogeneous", 900_000, {}),
    ("M", 64, "heterogeneous", 910_000, {}),
    ("H1", 256, "heterogeneous", 920_000, {}),
    ("H0", 256, "homogeneous", 920_000, {}),
    ("OOD_speed", 128, "heterogeneous", 940_000, {"v_slow": 17.0, "v_fast": 23.0}),
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "scenario_banks")
    p.add_argument("--merge-start", type=float, default=200.0)
    args = p.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hash_lines = []

    for name, size, traffic_type, seed_offset, extra_kwargs in BANK_SPECS:
        scenarios = [
            generate_scenario(
                scenario_id=f"{name}_{i:05d}",
                episode_seed=seed_offset + i,
                role_members=ROLE_MEMBERS,
                traffic_type=traffic_type,
                merge_start=args.merge_start,
                **extra_kwargs,
            )
            for i in range(size)
        ]
        path = args.output_dir / f"{name}.json"
        save_scenario_bank(scenarios, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hash_lines.append(f"{digest}  {path.name}")
        print(f"wrote {path} ({size} scenarios) sha256={digest}")

    (args.output_dir / "scenario_hashes.txt").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    print(f"hashes written to {args.output_dir / 'scenario_hashes.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
