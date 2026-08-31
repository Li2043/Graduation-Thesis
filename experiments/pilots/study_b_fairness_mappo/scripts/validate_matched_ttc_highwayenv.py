"""M4-D — matched-TTC generator gate (runbook sec 21), HighwayEnv backend.

Generates >=10,000 scenario initializations via the UNMODIFIED
``thesis.study_b.scenario_generator.generate_scenario`` (abstract,
backend-free), then for each one:

1. computes the NOMINAL front/rear |delta TTC| exactly as
   ``matched_ttc_deltas`` already does (this reuses, not reimplements,
   that function -- the generator itself did not change in this
   migration);
2. runs the scenario through the new HighwayEnv
   ``scenario_adapter.place_scenario``-equivalent spawn-longitudinal
   computation to confirm the physical adapter never rejects a valid
   scenario (``s_i < 0``) at this project's frozen
   ``before_merge_length`` -- this is the one thing that COULD fail from
   the migration itself, since the abstract TTC arithmetic is unchanged
   but the physical road-length budget is new.

Required (sec 21): |delta TTC| <= 0.5s for >= 95% of standard
heterogeneous scenarios; same-lane center distance >= 15m for 100% of
ACCEPTED scenarios (accepted = generator's own resampling already
guarantees this at the abstract level -- verified here, not assumed).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig  # noqa: E402
from thesis.study_b.envs.scenario_adapter import spawn_longitudinal  # noqa: E402
from thesis.study_b.scenario_generator import (  # noqa: E402
    generate_scenario,
    matched_ttc_deltas,
)


def run(*, n_scenarios: int, master_seed: int, before_merge_length: float, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ttc_rows = []
    spawn_rows = []
    delta_ttc_values: list[float] = []
    same_lane_violations = 0
    negative_spawn_count = 0

    role_members = {"ramp": ["V0", "V1"], "mainline": ["V2", "V3"]}

    for i in range(n_scenarios):
        seed = master_seed * 1_000_003 + i  # dedicated, disjoint stream from any training seed
        scenario = generate_scenario(
            scenario_id=f"m4d_{i:06d}",
            episode_seed=seed,
            role_members=role_members,
            traffic_type="heterogeneous",
            merge_start=before_merge_length,
        )
        deltas = matched_ttc_deltas(scenario)
        for slot, delta in deltas.items():
            delta_ttc_values.append(delta)
            ttc_rows.append({"scenario_id": scenario.scenario_id, "slot": slot, "delta_ttc": delta})

        # Same-lane gap check at the ABSTRACT level (route_position, which
        # is what the generator's own resampling loop enforces) --
        # cross-checked here independently rather than trusted blindly.
        by_role: dict[str, list[float]] = {}
        for v in scenario.vehicles.values():
            by_role.setdefault(v.role, []).append(v.route_position)
        min_gap = min(
            abs(a - b)
            for positions in by_role.values()
            for ai, a in enumerate(positions)
            for b in positions[ai + 1:]
        )
        if min_gap < 15.0:
            same_lane_violations += 1

        # Physical-adapter placement check: does the chosen
        # before_merge_length ever force a negative spawn coordinate for
        # this seed's target_speed/nominal_ttc combination?
        min_s = min(
            spawn_longitudinal(
                before_merge_length=before_merge_length,
                target_speed=v.target_speed,
                nominal_ttc=v.nominal_ttc,
            )
            for v in scenario.vehicles.values()
        )
        if min_s < 0:
            negative_spawn_count += 1
        spawn_rows.append({"scenario_id": scenario.scenario_id, "min_spawn_longitudinal": min_s})

    within_tol = sum(1 for d in delta_ttc_values if d <= 0.5)
    pct_within_tol = within_tol / len(delta_ttc_values)

    with (out_dir / "matched_ttc_validation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scenario_id", "slot", "delta_ttc"])
        w.writeheader()
        w.writerows(ttc_rows)

    with (out_dir / "spawn_validity.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scenario_id", "min_spawn_longitudinal"])
        w.writeheader()
        w.writerows(spawn_rows)

    summary = {
        "n_scenarios": n_scenarios,
        "n_delta_ttc_samples": len(delta_ttc_values),
        "pct_within_0.5s_tolerance": pct_within_tol,
        "same_lane_gap_violations": same_lane_violations,
        "negative_spawn_count": negative_spawn_count,
        "before_merge_length": before_merge_length,
        "gate": "PASS" if (pct_within_tol >= 0.95 and same_lane_violations == 0 and negative_spawn_count == 0) else "FAIL",
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-scenarios", type=int, default=10_000)
    parser.add_argument("--master-seed", type=int, default=900101)
    parser.add_argument(
        "--before-merge-length", type=float,
        default=ThesisHighwayMergeEnvConfig().before_merge_length,
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=REPO_ROOT / "output" / "highwayenv_migration" / "validation",
    )
    args = parser.parse_args(argv)
    summary = run(
        n_scenarios=args.n_scenarios, master_seed=args.master_seed,
        before_merge_length=args.before_merge_length, out_dir=args.out_dir,
    )
    print(summary)
    (args.out_dir / "M4_D_SUMMARY.json").write_text(
        __import__("json").dumps(summary, indent=2), encoding="utf-8"
    )
    return 0 if summary["gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
