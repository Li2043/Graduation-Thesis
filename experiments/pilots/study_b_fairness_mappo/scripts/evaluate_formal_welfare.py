#!/usr/bin/env python3
"""Ensemble-based (K(2,000,000), epsilon=0, R=50m) full utility/burden
evaluation for the 18 formal welfare runs -- the piece evaluate_formal.py
(task-outcome gate only) doesn't compute. Reuses q_ensemble.py's
load_ensemble_agents/select_ensemble_actions (unmodified) for action
selection, and utility.py's episode_utilities/episode_burdens/
gini_coefficient/generalized_gini_welfare (unmodified) for the welfare
metrics. Read-only: loads frozen checkpoints, writes CSVs, never trains."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

# DENSE-REWARD COPY (2026-08-26): was hardcoded Path(r"F:\正式训练") in the
# source formal-campaign bundle -- that literal path was already stale even
# there (real bundle root was F:\正式训练_seed_replication_v1), and would
# silently resolve to nothing on this machine/copy. Replaced with the same
# parents[]-based resolution the sibling training script
# (train_curriculum_stage_highwayenv.py) already uses, so this file locates
# itself correctly no matter which drive/folder the bundle lives in. No
# scientific logic touched.
BUNDLE_ROOT = Path(__file__).resolve().parents[5]
PROJECT_ROOT = BUNDLE_ROOT / "project"
SB_SCRIPTS = PROJECT_ROOT / "experiments" / "pilots" / "study_b_fairness_mappo" / "scripts"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SB_SCRIPTS))

from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig  # noqa: E402
from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv, StudyBHighwayWrapperConfig  # noqa: E402
from thesis.study_b.q_ensemble import ensemble_window_for_stage_end, load_ensemble_agents, select_ensemble_actions  # noqa: E402
from thesis.study_b.training_common import load_scenario_bank  # noqa: E402
from thesis.study_b.utility import episode_burdens, episode_utilities, generalized_gini_welfare, gini_coefficient  # noqa: E402

SEEDS = [900101, 900102, 900103, 900104, 910101, 910102]
CONDITIONS = ["mean", "ggi", "maximin"]
STAGE_END = 2_000_000
WINDOW = ensemble_window_for_stage_end(STAGE_END)  # (1850000, 1900000, 1950000, 2000000)
CKPT_ROOT = BUNDLE_ROOT / "checkpoints" / "formal_runs"
BANK_ROOT = BUNDLE_ROOT / "scenario_banks"
OUT_ROOT = BUNDLE_ROOT / "outputs" / "welfare_analysis"

CSV_FIELDS = [
    "run_id", "condition", "seed", "bank", "scenario_id", "traffic_type", "term_reason",
    "completion", "collision", "timeout", "mean_U", "min_U", "min_U_vehicle", "min_U_role",
    "min_U_speed_class", "ggi", "gini", "C_max", "C_mean", "episode_length",
] + [f"{p}_{v}" for v in ("V0", "V1", "V2", "V3") for p in ("role", "speed_class", "U", "C")]


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
        steps_taken = 0
        for _t in range(200):
            actions = select_ensemble_actions(agents, obs)
            obs, _reward, terminated, truncated, step_info = env.step(actions)
            steps_taken += 1
            if terminated:
                term_reason = "collision" if step_info["collision_event"] else "success"
                break
            if truncated:
                term_reason = "truncation"
                break

        traces = env.episode_traces()
        utilities = episode_utilities(traces)
        burdens = episode_burdens(traces, dt=env.dt())
        u_values = list(utilities.values())
        min_vid = min(utilities, key=lambda v: utilities[v])

        row = {
            "run_id": run_id, "condition": condition, "seed": seed, "bank": bank_name,
            "scenario_id": scenario.scenario_id, "traffic_type": scenario.traffic_type,
            "term_reason": term_reason,
            "completion": int(term_reason == "success"), "collision": int(term_reason == "collision"),
            "timeout": int(term_reason == "truncation"),
            "mean_U": sum(u_values) / len(u_values), "min_U": utilities[min_vid],
            "min_U_vehicle": min_vid, "min_U_role": scenario.vehicles[min_vid].role,
            "min_U_speed_class": scenario.vehicles[min_vid].speed_class,
            "ggi": generalized_gini_welfare(u_values), "gini": gini_coefficient(u_values),
            "C_max": max(burdens.values()), "C_mean": sum(burdens.values()) / len(burdens),
            "episode_length": steps_taken,
        }
        for vid in ("V0", "V1", "V2", "V3"):
            row[f"role_{vid}"] = scenario.vehicles[vid].role
            row[f"speed_class_{vid}"] = scenario.vehicles[vid].speed_class
            row[f"U_{vid}"] = utilities[vid]
            row[f"C_{vid}"] = burdens[vid]
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
        print(f"[evaluate_formal_welfare shard {args.shard_index}] {run_id} on {bank} ...", flush=True)
        rows = run_one(run_id, condition, seed, bank)
        all_rows.extend(rows)
        n = len(rows)
        comp = sum(r["completion"] for r in rows) / n
        coll = sum(r["collision"] for r in rows) / n
        print(f"  n={n} completion={comp:.3f} collision={coll:.3f}", flush=True)

    out_csv = OUT_ROOT / f"formal_welfare_evaluation_shard{args.shard_index}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nwrote {len(all_rows)} rows -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
