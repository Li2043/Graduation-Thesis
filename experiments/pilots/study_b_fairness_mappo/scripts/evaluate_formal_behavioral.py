#!/usr/bin/env python3
"""Behavioural-measures evaluation (merge order, hard-brake count) for the
18 formal welfare runs -- fills the two gaps identified in Section 4.5.4's
behavioural-measures list that evaluate_formal_welfare.py did not capture
(worst-off identity and burden by class were already covered there).

Reuses q_ensemble.py's load_ensemble_agents/select_ensemble_actions
(unmodified) for action selection and utility.py's EpisodeVehicleTrace
(unmodified, its hard_brake_count() method already exists but was unused)
for the hard-brake metric. Merge order is derived from step_info["exit_event"],
the same field stage_q_ensemble_gate.py already uses for this purpose.

Read-only: loads frozen checkpoints, writes CSVs, never trains."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

# DENSE-REWARD COPY (2026-08-26): see evaluate_formal_welfare.py's identical
# comment -- was a stale hardcoded Path(r"F:\正式训练"), replaced with
# portable parents[]-based resolution. No scientific logic touched.
BUNDLE_ROOT = Path(__file__).resolve().parents[5]
PROJECT_ROOT = BUNDLE_ROOT / "project"
SB_SCRIPTS = PROJECT_ROOT / "experiments" / "pilots" / "study_b_fairness_mappo" / "scripts"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SB_SCRIPTS))

from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig  # noqa: E402
from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv, StudyBHighwayWrapperConfig  # noqa: E402
from thesis.study_b.q_ensemble import ensemble_window_for_stage_end, load_ensemble_agents, select_ensemble_actions  # noqa: E402
from thesis.study_b.training_common import load_scenario_bank  # noqa: E402

SEEDS = [900101, 900102, 900103, 900104, 910101, 910102]
CONDITIONS = ["mean", "ggi", "maximin"]
STAGE_END = 2_000_000
WINDOW = ensemble_window_for_stage_end(STAGE_END)
CKPT_ROOT = BUNDLE_ROOT / "checkpoints" / "formal_runs"
BANK_ROOT = BUNDLE_ROOT / "scenario_banks"
OUT_ROOT = BUNDLE_ROOT / "outputs" / "welfare_analysis"

HARD_BRAKE_THRESHOLD = -3.0
# NOT the -3.5 default in EpisodeVehicleTrace.hard_brake_count(): under the
# R=50m migration's realized-acceleration clipping (MIN_REALIZED_ACCEL_MPS2 =
# -3.0, MAX = +2.0, confirmed this session via direct trace inspection --
# 181/5421 sampled accelerations land exactly at -3.0, none below it), -3.5
# is physically unreachable and would silently return 0 for every vehicle in
# every episode. -3.0 (the frozen physical floor itself, not a value chosen
# by looking at this run's results) is used instead: "hard brake" = braking
# at the maximum deceleration the environment allows.

CSV_FIELDS = [
    "run_id", "condition", "seed", "bank", "scenario_id", "term_reason",
    "completion", "collision", "timeout", "merge_order",
] + [f"{p}_{v}" for v in ("V0", "V1", "V2", "V3") for p in ("role", "speed_class", "merge_position", "hard_brake_count")]


def checkpoint_paths_for(run_id: str, seed: int) -> dict[int, Path]:
    d = CKPT_ROOT / run_id / f"seed_{seed}_Formal_{run_id.split('_', 1)[0]}"
    return {s: d / f"ckpt_step_{s}.pt" for s in WINDOW}


def run_one(run_id: str, condition: str, seed: int, bank_name: str) -> list[dict]:
    stage_name = f"Formal_{condition}"
    ckpt_paths = checkpoint_paths_for(run_id, seed)
    agents = load_ensemble_agents(
        seed=seed, checkpoint_paths=ckpt_paths, expected_steps=WINDOW,
        expected_stage_by_step=dict.fromkeys(WINDOW, stage_name),
    )
    scenarios = load_scenario_bank(BANK_ROOT / f"{bank_name}.json")
    env_config = ThesisHighwayMergeEnvConfig(episode_max_steps=200, action_representation="meta_speed")
    env = StudyBHeterogeneousHighwayEnv(StudyBHighwayWrapperConfig(env_config=env_config, local_sensing_range_m=50.0))

    rows = []
    for scenario in scenarios:
        obs, _info = env.reset(seed=0, scenario=scenario)
        term_reason = "truncation"
        completion_step: dict[str, int | None] = dict.fromkeys(env.active_vehicle_ids)
        for t in range(200):
            actions = select_ensemble_actions(agents, obs)
            obs, _reward, terminated, truncated, step_info = env.step(actions)
            for vid, done in step_info["exit_event"].items():
                if done and completion_step[vid] is None:
                    completion_step[vid] = t + 1
            if terminated:
                term_reason = "collision" if step_info["collision_event"] else "success"
                break
            if truncated:
                term_reason = "truncation"
                break

        merge_order = sorted(
            (vid for vid in completion_step if completion_step[vid] is not None),
            key=lambda vid: completion_step[vid],
        )
        merge_order_str = ">".join(merge_order) if merge_order else "DNF"
        merge_position = {vid: (merge_order.index(vid) + 1) for vid in merge_order}

        traces = env.episode_traces()
        row = {
            "run_id": run_id, "condition": condition, "seed": seed, "bank": bank_name,
            "scenario_id": scenario.scenario_id, "term_reason": term_reason,
            "completion": int(term_reason == "success"), "collision": int(term_reason == "collision"),
            "timeout": int(term_reason == "truncation"), "merge_order": merge_order_str,
        }
        for vid in ("V0", "V1", "V2", "V3"):
            row[f"role_{vid}"] = scenario.vehicles[vid].role
            row[f"speed_class_{vid}"] = scenario.vehicles[vid].speed_class
            row[f"merge_position_{vid}"] = merge_position.get(vid)  # None if DNF (collision/timeout, never exited)
            row[f"hard_brake_count_{vid}"] = traces[vid].hard_brake_count(threshold=HARD_BRAKE_THRESHOLD)
        rows.append(row)
    return rows


def all_jobs() -> list[tuple[str, str, int, str]]:
    jobs = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            run_id = f"{condition}_{seed}"
            banks = ["H1"] if condition != "mean" else ["H0", "H1"]
            for bank in banks:
                jobs.append((run_id, condition, seed, bank))
    return jobs


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = all_jobs()
    my_jobs = [j for i, j in enumerate(jobs) if i % args.num_shards == args.shard_index]

    all_rows = []
    for run_id, condition, seed, bank in my_jobs:
        print(f"[evaluate_formal_behavioral shard {args.shard_index}] {run_id} on {bank} ...", flush=True)
        rows = run_one(run_id, condition, seed, bank)
        all_rows.extend(rows)
        n = len(rows)
        avg_hb = sum(r[f"hard_brake_count_{v}"] for r in rows for v in ("V0", "V1", "V2", "V3")) / (4 * n)
        print(f"  n={n} mean_hard_brakes_per_vehicle={avg_hb:.3f}", flush=True)

    out_csv = OUT_ROOT / f"formal_behavioral_evaluation_shard{args.shard_index}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nwrote {len(all_rows)} rows -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
