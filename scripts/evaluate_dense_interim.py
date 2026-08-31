"""INTERIM (partial-seed) WSC+Dense evaluation, adapted verbatim in logic
from F:\\正式训练_seed_replication_v1\\analysis_scripts\\ch5_baseline\\evaluate_wsc_formal_v2.py
(the script used to produce the existing wsc_v2_formal_seed_level.csv baseline
numbers), pointed at this bundle's maximin_wsc_dense checkpoints instead of
wsc_formal_runs_v2.

IMPORTANT CAVEAT, inherited unchanged from the template this was copied from:
any run of this script against fewer than all 12 formal seeds is an
INTERIM/PARTIAL, INFORMAL read, not a formal result. This project's own
frozen-protocol discipline treats early looks at accumulating seeds as
unreliable given known seed-level heterogeneity. This script and its output
must NOT be used for any thesis table/figure/formal conclusion.

Read-only: loads frozen checkpoints, writes one CSV, never trains, never
touches the still-running training processes' checkpoints/logs.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BUNDLE_ROOT / "src"))

from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig  # noqa: E402
from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv, StudyBHighwayWrapperConfig  # noqa: E402
from thesis.study_b.local_observation import LOCAL_OBS_DIM_WSC  # noqa: E402
from thesis.study_b.q_ensemble import ensemble_window_for_stage_end, load_ensemble_agents, select_ensemble_actions  # noqa: E402
from thesis.study_b.training_common import load_scenario_bank  # noqa: E402
from thesis.study_b.utility import episode_burdens, episode_utilities, generalized_gini_welfare, gini_coefficient  # noqa: E402

DENSE_ROOT = BUNDLE_ROOT / "checkpoints" / "maximin_wsc_dense"
WINDOW = ensemble_window_for_stage_end(2_000_000)
COND = "maximin"
OUT_DIR = BUNDLE_ROOT / "outputs" / "welfare_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FIELDS = [
    "run_id", "condition", "seed", "bank", "scenario_id", "traffic_type", "term_reason",
    "completion", "collision", "timeout", "mean_U", "min_U", "min_U_vehicle", "min_U_role",
    "min_U_speed_class", "ggi", "gini", "C_max", "C_mean", "episode_length",
] + [f"{p}_{v}" for v in ("V0", "V1", "V2", "V3") for p in ("role", "speed_class", "U", "C")]


def ckpt_dir_for(seed: int) -> Path:
    return DENSE_ROOT / f"maximin_wsc_dense_{seed}" / f"seed_{seed}_Dense_maximin_wsc_dense"


def checkpoint_paths_for(seed: int) -> dict[int, Path]:
    d = ckpt_dir_for(seed)
    return {s: d / f"ckpt_step_{s}.pt" for s in WINDOW}


def all_checkpoints_exist(seed: int) -> bool:
    paths = checkpoint_paths_for(seed)
    return all(p.exists() for p in paths.values())


def run_one(seed: int, bank_name: str = "H1") -> list[dict]:
    ckpt_paths = checkpoint_paths_for(seed)
    agents = load_ensemble_agents(
        seed=seed, checkpoint_paths=ckpt_paths, expected_steps=WINDOW,
        expected_stage_by_step=dict.fromkeys(WINDOW, "Dense_maximin_wsc_dense"),
        obs_dim=LOCAL_OBS_DIM_WSC,
    )
    scenarios = load_scenario_bank(BUNDLE_ROOT / "scenario_banks" / f"{bank_name}.json")
    env_config = ThesisHighwayMergeEnvConfig(episode_max_steps=200, action_representation="meta_speed")
    env = StudyBHeterogeneousHighwayEnv(
        StudyBHighwayWrapperConfig(env_config=env_config, local_sensing_range_m=50.0, include_welfare_state=True)
    )

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
            "run_id": f"maximin_wsc_dense_{seed}", "condition": COND, "seed": seed, "bank": bank_name,
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


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out-suffix", type=str, default="")
    args = ap.parse_args()

    print("=" * 78)
    print("INTERIM WSC+DENSE (maximin) EVALUATION -- CAVEAT: covers only the seeds")
    print("listed below, NOT the full 12-seed formal sample (6 seeds still training).")
    print("NOT a formal result -- informal direction-check only.")
    print(f"Seeds included: {args.seeds}")
    print("=" * 78)

    all_rows = []
    for seed in args.seeds:
        if not all_checkpoints_exist(seed):
            print(f"[SKIP] maximin_wsc_dense_{seed}: missing one or more ensemble-window checkpoints")
            continue
        print(f"[evaluate_dense_interim] maximin_wsc_dense_{seed} on H1 ...", flush=True)
        rows = run_one(seed)
        all_rows.extend(rows)
        n = len(rows)
        comp = sum(r["completion"] for r in rows) / n
        coll = sum(r["collision"] for r in rows) / n
        print(f"  n={n} completion={comp:.3f} collision={coll:.3f}", flush=True)

    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    out_csv = OUT_DIR / f"dense_interim_evaluation{suffix}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nwrote {len(all_rows)} rows -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
