"""Conflict-exposure diagnostic (additive, read-only against frozen
checkpoints/env/wrapper/agent code). Re-evaluates exactly the 12 Baseline
seeds x 256 frozen H1 held-out scenarios, epsilon_eval=0, the identical
checkpoint-Q ensemble already used for the formal Baseline result in
Chapter 5 (taskonly_arm checkpoints, window K(2,000,000)).

Audit findings this script encodes (see conflict_exposure_report.md sec 1
for the full writeup):
  - x=220 (before_merge_length), x=300 (+converge_merge_length=80),
    x=380 (+parallel_merge_length=80) -- verified directly from
    ThesisHighwayMergeEnvConfig, not assumed from prose documentation.
  - role ("ramp"/"mainline") and ttc_slot ("front"/"rear") are explicit
    per-vehicle fields in the H1 scenario bank; read dynamically per
    scenario (never hardcoded), even though empirically V0/V1=ramp,
    V2/V3=mainline in every H1 scenario.
  - No pre-existing continuous (sub-step) crossing-time interpolation was
    found anywhere in the codebase -- existing merge-order/window logic
    (evaluate_behavioral_window.py, illustrative_episode.py) only records
    the discrete policy-step index of first crossing. This script adds a
    simple linear interpolation between the last pre-threshold and first
    post-threshold step positions -- a NEW addition for this diagnostic,
    not a reuse of an existing "continuous" method.
  - Vehicle physical length (5.0 m) is stated directly in
    highwayenv_vehicle.py's own comments (Vehicle.LENGTH=5.0), so the
    optional clearance calc is sourced, not guessed.
"""
from __future__ import annotations
import os

import csv
import math
import sys
from pathlib import Path

FINAL_NEW = Path(os.environ.get("FINAL_NEW_BUNDLE", ""))  # raw checkpoints/logs not distributed with this repo; set env var
PROJECT_ROOT = Path(os.environ.get("SEED_REPL_PROJECT", str(Path(__file__).resolve().parent.parent.parent)))  # this repo restructured project/ into repo root
SB_SCRIPTS = PROJECT_ROOT / "experiments" / "pilots" / "study_b_fairness_mappo" / "scripts"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SB_SCRIPTS))

from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig  # noqa: E402
from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv, StudyBHighwayWrapperConfig  # noqa: E402
from thesis.study_b.q_ensemble import ensemble_window_for_stage_end, load_ensemble_agents, select_ensemble_actions  # noqa: E402
from thesis.study_b.training_common import load_scenario_bank  # noqa: E402
from thesis.study_b.utility import episode_burdens, episode_utilities, gini_coefficient, utility_range  # noqa: E402

SEEDS12 = [900101, 900102, 900103, 900104, 910101, 910102,
           920101, 920102, 920103, 920104, 920105, 920106]
STAGE_END = 2_000_000
WINDOW = ensemble_window_for_stage_end(STAGE_END)  # (1850000, 1900000, 1950000, 2000000)
BANK_ROOT = FINAL_NEW / "scenario_banks"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "conflict_exposure"
OUT_DIR.mkdir(parents=True, exist_ok=True)

X_CONVERGE_START = 220.0   # verified: before_merge_length
X_PARALLEL_START = 300.0   # verified: before_merge_length + converge_merge_length (220+80)
X_MERGE_END = 380.0        # verified: + parallel_merge_length (300+80)
HARD_BRAKE_THRESH = -3.0   # existing frozen definition, reused unchanged
DT = 0.2                   # 5 Hz policy decision interval
VEHICLE_LENGTH_M = 5.0     # highway_env default, stated in highwayenv_vehicle.py's own comments
VIDS = ["V0", "V1", "V2", "V3"]

CSV_FIELDS = [
    "seed", "scenario", "outcome", "completion", "collision", "timeout",
    "cross_road_overlap_any", "cross_road_overlap_steps", "cross_road_overlap_duration_s",
    "max_simultaneous_merge_vehicles",
    "min_crossroad_x_gap", "min_crossroad_euclidean_gap", "min_crossroad_clearance_m",
    "min_crossroad_crossing_gap_x300", "min_crossroad_crossing_gap_x380",
    "front_pair_crossing_gap_x300", "front_pair_crossing_gap_x380",
    "rear_pair_crossing_gap_x300", "rear_pair_crossing_gap_x380",
    "min_projected_ttc_gap_s",
    "any_BRAKE_action", "total_BRAKE_actions",
    "any_negative_acceleration",
    "any_hard_brake_event", "total_hard_brake_events",
    "C_brake",
    "any_below_target_burden", "C_mean",
    "U_mean", "U_min", "Utility_Gini", "utility_range", "worst_off_identity",
] + [f"role_{v}" for v in VIDS] + [f"ttc_slot_{v}" for v in VIDS] + [f"U_{v}" for v in VIDS] + [f"C_{v}" for v in VIDS]


def checkpoint_paths_for(seed: int) -> dict[int, Path]:
    d = FINAL_NEW / "checkpoints" / "taskonly_arm" / str(seed) / f"seed_{seed}_Formal_taskonly"
    return {s: d / f"ckpt_step_{s}.pt" for s in WINDOW}


def interp_crossing_time(steps: list[int], xs: list[float], threshold: float) -> float | None:
    """Linear interpolation between the last pre-threshold and first
    post-threshold recorded step position. New for this diagnostic (see
    module docstring) -- no equivalent existed in the codebase."""
    for i in range(1, len(xs)):
        if xs[i - 1] < threshold <= xs[i]:
            x0, x1 = xs[i - 1], xs[i]
            t0, t1 = steps[i - 1] * DT, steps[i] * DT
            if x1 == x0:
                return float(t1)
            frac = (threshold - x0) / (x1 - x0)
            return float(t0 + frac * (t1 - t0))
    return None  # vehicle never reached this threshold (collided/timed out first, or DNF)


def run_one_seed(seed: int) -> list[dict]:
    ckpt_paths = checkpoint_paths_for(seed)
    agents = load_ensemble_agents(
        seed=seed, checkpoint_paths=ckpt_paths, expected_steps=WINDOW,
        expected_stage_by_step=dict.fromkeys(WINDOW, "Formal_taskonly"),
    )
    scenarios = load_scenario_bank(BANK_ROOT / "H1.json")
    env_config = ThesisHighwayMergeEnvConfig(episode_max_steps=200, action_representation="meta_speed")
    env = StudyBHeterogeneousHighwayEnv(StudyBHighwayWrapperConfig(env_config=env_config, local_sensing_range_m=50.0))

    rows = []
    for scenario in scenarios:
        obs, _info = env.reset(seed=0, scenario=scenario)
        role = {v: scenario.vehicles[v].role for v in VIDS}
        ttc_slot = {v: scenario.vehicles[v].ttc_slot for v in VIDS}
        ramp_ids = [v for v in VIDS if role[v] == "ramp"]
        main_ids = [v for v in VIDS if role[v] == "mainline"]

        step_list: dict[str, list[int]] = {v: [] for v in VIDS}
        x_list: dict[str, list[float]] = {v: [] for v in VIDS}
        y_list: dict[str, list[float]] = {v: [] for v in VIDS}
        speed_list: dict[str, list[float]] = {v: [] for v in VIDS}
        accel_list: dict[str, list[float]] = {v: [] for v in VIDS}
        action_list: dict[str, list[int]] = {v: [] for v in VIDS}
        pre_active = {v: True for v in VIDS}
        term_reason = "truncation"

        for t in range(200):
            actions = select_ensemble_actions(agents, obs)
            obs, _r, terminated, truncated, step_info = env.step(actions)
            for v in VIDS:
                if not pre_active[v]:
                    continue
                vehicle = env._env._vehicle_by_id[v]  # noqa: SLF001
                x, y = env._env.world_xy(vehicle)  # noqa: SLF001
                step_list[v].append(t)
                x_list[v].append(float(x))
                y_list[v].append(float(y))
                speed_list[v].append(float(vehicle.speed))
                accel_list[v].append(float(vehicle.action["acceleration"]))
                action_list[v].append(int(actions[v]))
            pre_active = dict(step_info["active"])
            if terminated:
                term_reason = "collision" if step_info["collision_event"] else "success"
                break
            if truncated:
                term_reason = "truncation"
                break

        n_steps = max(len(step_list[v]) for v in VIDS)

        # ---- D1: cross-road merge-zone overlap (per policy step) ----
        overlap_steps = 0
        max_simul = 0
        for t in range(n_steps):
            in_region = {}
            for v in VIDS:
                if t < len(x_list[v]):
                    in_region[v] = X_CONVERGE_START <= x_list[v][t] < X_MERGE_END
                else:
                    in_region[v] = False
            n_in = sum(in_region.values())
            max_simul = max(max_simul, n_in)
            ramp_in = any(in_region[v] for v in ramp_ids)
            main_in = any(in_region[v] for v in main_ids)
            if ramp_in and main_in:
                overlap_steps += 1
        overlap_any = overlap_steps > 0
        overlap_duration_s = overlap_steps * DT

        # ---- D2: crossing-time separation (x=300, x=380) ----
        cross300 = {v: interp_crossing_time(step_list[v], x_list[v], X_PARALLEL_START) for v in VIDS}
        cross380 = {v: interp_crossing_time(step_list[v], x_list[v], X_MERGE_END) for v in VIDS}

        def all_gaps(cross: dict[str, float | None]) -> list[float]:
            gaps = []
            for r in ramp_ids:
                for m in main_ids:
                    if cross[r] is not None and cross[m] is not None:
                        gaps.append(abs(cross[r] - cross[m]))
            return gaps

        gaps300 = all_gaps(cross300)
        gaps380 = all_gaps(cross380)
        min_gap300 = min(gaps300) if gaps300 else None
        min_gap380 = min(gaps380) if gaps380 else None

        def pair_gap(cross: dict[str, float | None], slot: str) -> float | None:
            r = next((v for v in ramp_ids if ttc_slot[v] == slot), None)
            m = next((v for v in main_ids if ttc_slot[v] == slot), None)
            if r is None or m is None or cross[r] is None or cross[m] is None:
                return None
            return abs(cross[r] - cross[m])

        front_gap300 = pair_gap(cross300, "front")
        front_gap380 = pair_gap(cross380, "front")
        rear_gap300 = pair_gap(cross300, "rear")
        rear_gap380 = pair_gap(cross380, "rear")

        # ---- D3: minimum physical separation while both in-region ----
        min_x_gap, min_eucl_gap = None, None
        for t in range(n_steps):
            in_region = {v: (t < len(x_list[v]) and X_CONVERGE_START <= x_list[v][t] < X_MERGE_END) for v in VIDS}
            for r in ramp_ids:
                if not in_region[r]:
                    continue
                for m in main_ids:
                    if not in_region[m]:
                        continue
                    dx = abs(x_list[r][t] - x_list[m][t])
                    dy = y_list[r][t] - y_list[m][t]
                    deucl = math.sqrt(dx * dx + dy * dy)
                    if min_x_gap is None or dx < min_x_gap:
                        min_x_gap = dx
                    if min_eucl_gap is None or deucl < min_eucl_gap:
                        min_eucl_gap = deucl
        min_clearance = (min_x_gap - VEHICLE_LENGTH_M) if min_x_gap is not None else None

        # ---- D4 (secondary, offline diagnostic only): projected arrival-time gap ----
        min_proj_gap = None
        for t in range(n_steps):
            in_region = {v: (t < len(x_list[v]) and X_CONVERGE_START <= x_list[v][t] < X_MERGE_END) for v in VIDS}
            for r in ramp_ids:
                if not in_region[r] or speed_list[r][t] <= 0:
                    continue
                proj_r = (X_MERGE_END - x_list[r][t]) / speed_list[r][t]
                for m in main_ids:
                    if not in_region[m] or speed_list[m][t] <= 0:
                        continue
                    proj_m = (X_MERGE_END - x_list[m][t]) / speed_list[m][t]
                    gap = abs(proj_r - proj_m)
                    if min_proj_gap is None or gap < min_proj_gap:
                        min_proj_gap = gap

        # ---- E: behavioural response in [220,380) window ----
        brake_count = {v: 0 for v in VIDS}
        hb_events = {v: 0 for v in VIDS}
        c_brake = {v: 0.0 for v in VIDS}
        any_neg_accel = {v: False for v in VIDS}
        hb_in_run = {v: False for v in VIDS}
        for v in VIDS:
            for i, t in enumerate(step_list[v]):
                x = x_list[v][i]
                accel = accel_list[v][i]
                if accel < 0:
                    any_neg_accel[v] = True
                if X_CONVERGE_START <= x < X_MERGE_END:
                    if action_list[v][i] == 2:  # BRAKE
                        brake_count[v] += 1
                    c_brake[v] += DT * max(0.0, -accel) / 3.0
                    if accel <= HARD_BRAKE_THRESH:
                        if not hb_in_run[v]:
                            hb_events[v] += 1
                        hb_in_run[v] = True
                    else:
                        hb_in_run[v] = False
                else:
                    hb_in_run[v] = False

        # ---- F: fairness outcomes (existing, unmodified utility.py) ----
        traces = env.episode_traces()
        utilities = episode_utilities(traces)
        burdens = episode_burdens(traces, dt=env.dt())
        u_values = list(utilities.values())
        min_vid = min(utilities, key=lambda v: utilities[v])
        gini = gini_coefficient(u_values)
        u_range = utility_range(u_values)

        row = {
            "seed": seed, "scenario": scenario.scenario_id, "outcome": term_reason,
            "completion": int(term_reason == "success"), "collision": int(term_reason == "collision"),
            "timeout": int(term_reason == "truncation"),
            "cross_road_overlap_any": int(overlap_any),
            "cross_road_overlap_steps": overlap_steps,
            "cross_road_overlap_duration_s": round(overlap_duration_s, 4),
            "max_simultaneous_merge_vehicles": max_simul,
            "min_crossroad_x_gap": round(min_x_gap, 4) if min_x_gap is not None else None,
            "min_crossroad_euclidean_gap": round(min_eucl_gap, 4) if min_eucl_gap is not None else None,
            "min_crossroad_clearance_m": round(min_clearance, 4) if min_clearance is not None else None,
            "min_crossroad_crossing_gap_x300": round(min_gap300, 4) if min_gap300 is not None else None,
            "min_crossroad_crossing_gap_x380": round(min_gap380, 4) if min_gap380 is not None else None,
            "front_pair_crossing_gap_x300": round(front_gap300, 4) if front_gap300 is not None else None,
            "front_pair_crossing_gap_x380": round(front_gap380, 4) if front_gap380 is not None else None,
            "rear_pair_crossing_gap_x300": round(rear_gap300, 4) if rear_gap300 is not None else None,
            "rear_pair_crossing_gap_x380": round(rear_gap380, 4) if rear_gap380 is not None else None,
            "min_projected_ttc_gap_s": round(min_proj_gap, 4) if min_proj_gap is not None else None,
            "any_BRAKE_action": int(any(brake_count[v] > 0 for v in VIDS)),
            "total_BRAKE_actions": sum(brake_count.values()),
            "any_negative_acceleration": int(any(any_neg_accel.values())),
            "any_hard_brake_event": int(any(hb_events[v] > 0 for v in VIDS)),
            "total_hard_brake_events": sum(hb_events.values()),
            "C_brake": round(sum(c_brake.values()) / 4.0, 6),
            "any_below_target_burden": int(any(burdens[v] > 0 for v in VIDS)),
            "C_mean": round(sum(burdens.values()) / len(burdens), 6),
            "U_mean": round(sum(u_values) / len(u_values), 6),
            "U_min": round(utilities[min_vid], 6),
            "Utility_Gini": round(gini, 6) if gini is not None else None,
            "utility_range": round(u_range, 6),
            "worst_off_identity": min_vid,
        }
        for v in VIDS:
            row[f"role_{v}"] = role[v]
            row[f"ttc_slot_{v}"] = ttc_slot[v]
            row[f"U_{v}"] = round(utilities[v], 6)
            row[f"C_{v}"] = round(burdens[v], 6)
        rows.append(row)
    return rows


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    my_seeds = [s for i, s in enumerate(SEEDS12) if i % args.num_shards == args.shard_index]
    all_rows = []
    for seed in my_seeds:
        print(f"[conflict_exposure shard {args.shard_index}] seed {seed} ...", flush=True)
        rows = run_one_seed(seed)
        all_rows.extend(rows)
        n = len(rows)
        ov = sum(r["cross_road_overlap_any"] for r in rows) / n
        print(f"  n={n} overlap_rate={ov:.3f}", flush=True)

    out_csv = OUT_DIR / f"conflict_exposure_shard{args.shard_index}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {len(all_rows)} rows -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
