"""M4-E -- role/speed counterbalancing gate (runbook sec 22), HighwayEnv
backend.

The abstract role x speed_class x ttc_slot counterbalancing is entirely
the (unmodified) generator's responsibility, already exercised by M4-D at
scale. What is NEW in this migration and needs its own check is
``StudyBHeterogeneousHighwayEnv._role_members()`` -- the physical-vehicle-
id ("V0".."V3") <-> role permutation, which did not exist in the legacy
backend (there, ``Stage10SymmetricMergeEnv.reset()`` did its own role
permutation internally; this wrapper reimplements that decision for the
new backend). This script runs many resets and tabulates, per physical
id, how often it lands in each (role, speed_class, ttc_slot) combination
-- no physical id may deterministically encode any of them.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv  # noqa: E402


def run(*, n_resets: int, master_seed: int) -> dict:
    env = StudyBHeterogeneousHighwayEnv()
    role_counts: dict[str, Counter] = {vid: Counter() for vid in ("V0", "V1", "V2", "V3")}
    speed_counts: dict[str, Counter] = {vid: Counter() for vid in ("V0", "V1", "V2", "V3")}
    slot_counts: dict[str, Counter] = {vid: Counter() for vid in ("V0", "V1", "V2", "V3")}

    for i in range(n_resets):
        seed = master_seed * 7919 + i
        env.reset(seed=seed, traffic_type="heterogeneous")
        for vid, spec in env._scenario.vehicles.items():  # noqa: SLF001
            role_counts[vid][spec.role] += 1
            speed_counts[vid][spec.speed_class] += 1
            slot_counts[vid][spec.ttc_slot] += 1

    def _balance_report(counts: dict[str, Counter]) -> dict:
        report = {}
        for vid, c in counts.items():
            total = sum(c.values())
            report[vid] = {k: v / total for k, v in c.items()}
        return report

    role_report = _balance_report(role_counts)
    speed_report = _balance_report(speed_counts)
    slot_report = _balance_report(slot_counts)

    # No physical id may deterministically encode a class: every fraction
    # must be bounded away from 0 and 1 (a generous [0.30, 0.70] band for
    # a 2-way split -- true balance is ~0.50, this only flags a real
    # skew/bug, not sampling noise at n_resets ~ a few thousand).
    def _all_balanced(report: dict) -> bool:
        return all(0.30 <= frac <= 0.70 for per_vid in report.values() for frac in per_vid.values())

    gate = (
        "PASS" if (_all_balanced(role_report) and _all_balanced(speed_report) and _all_balanced(slot_report))
        else "FAIL"
    )
    return {
        "n_resets": n_resets,
        "role_balance_by_vehicle_id": role_report,
        "speed_class_balance_by_vehicle_id": speed_report,
        "ttc_slot_balance_by_vehicle_id": slot_report,
        "gate": gate,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-resets", type=int, default=4000)
    parser.add_argument("--master-seed", type=int, default=900101)
    parser.add_argument(
        "--out-dir", type=Path,
        default=REPO_ROOT / "output" / "highwayenv_migration" / "validation",
    )
    args = parser.parse_args(argv)
    summary = run(n_resets=args.n_resets, master_seed=args.master_seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "M4_E_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
