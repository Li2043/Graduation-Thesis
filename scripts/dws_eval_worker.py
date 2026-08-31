"""DWS final re-evaluation — single (cell, seed) rollout worker.

Runs 256 H1-bank held-out episodes for ONE checkpoint ensemble (one cell x
one seed), deterministic greedy Q-ensemble action selection (frozen
final-four window {1850000,1900000,1950000,2000000}), and writes:

  1. an episode-level CSV shard (one row per scenario_id) matching the
     required column list in Section 4 of the re-evaluation prompt;
  2. a per-step trajectory shard (gzipped JSONL, one line per episode) with
     enough raw per-step data to reconstruct every later mechanism analysis
     (yielding, merge priority, hard-brake/burden, worst-off recovery,
     Dense signal reconstruction) without re-running any rollout.

Reuses, unmodified, the exact source functions this project already uses
for these quantities:
  - thesis.study_b.q_ensemble (load_ensemble_agents, select_ensemble_actions,
    ensemble_window_for_stage_end) -- same ensemble/greedy-action machinery
    as evaluate_dense_interim.py / evaluate_wsc_formal_v2.py.
  - thesis.study_b.utility (episode_utilities, episode_burdens,
    gini_coefficient, generalized_gini_welfare, running_active_attainment,
    EpisodeVehicleTrace) -- same welfare/burden/Gini formulas as every other
    evaluation script in this project.
  - thesis.study_b.dense_shaping (welfare_objective_snapshot,
    dense_shaping_term, DenseShapingConfig, NEUTRAL_PHI) -- the ACTUAL
    frozen DWS formula implementation, for offline Phi_t/DeltaPhi_t/F_t
    reconstruction (not hand-rolled).
  - thesis.study_b.welfare_reward.condition_by_name -- for the Maximin
    welfare_fn used inside welfare_objective_snapshot.
  - Per-step introspection logic (xs, Ms via incremental
    target_speed_attainment accumulation, hard-brake classification,
    merge-priority pair bookkeeping, worst-off sampling with the frozen
    1e-9 tie tolerance) copied verbatim in shape from
    F:\\正式训练_seed_replication_v1\\analysis_scripts\\wsc_v2_behavioural\\wsc_v2_behavioural_run.py
    (X_CONVERGE_START=220.0, X_MERGE_END=380.0, R_OBS=50.0,
    HARD_BRAKE_THRESH=-3.0, DECELERATE=2, TIE_TOL=1e-9) -- this worker
    additionally DUMPS the raw per-step values instead of only aggregating
    them online, since the re-evaluation prompt requires trajectory-level
    reuse across many mechanism analyses, not just one aggregated counter
    set.

Read-only w.r.t. training: loads frozen checkpoints, writes new files only
under the caller-specified output paths. Does not modify any policy.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

X_CONVERGE_START = 220.0
X_MERGE_END = 380.0
R_OBS = 50.0
HARD_BRAKE_THRESH = -3.0
DECELERATE = 2
TIE_TOL = 1e-9
VIDS = ("V0", "V1", "V2", "V3")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--scenario-bank", type=Path, required=True)
    ap.add_argument("--ckpt-dir", type=Path, required=True, help="directory containing ckpt_step_<N>.pt for the frozen final-four window")
    ap.add_argument("--stage-name", type=str, required=True, help="expected stage-name recorded in each checkpoint, for load_ensemble_agents' own validation")
    ap.add_argument("--obs-dim", type=int, required=True, choices=[18, 22])
    ap.add_argument("--include-welfare-state", action="store_true")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cell", type=str, required=True, choices=["cell1", "cell2", "cell3", "cell4"])
    ap.add_argument("--condition", type=str, default="maximin")
    ap.add_argument("--dws-on", action="store_true", help="whether THIS cell's training used dense shaping (for the episode-level DWS-on/off column and for choosing whether the reconstructed F_t/Phi are 'training-consistent' or purely counterfactual)")
    ap.add_argument("--dense-magnitude", type=float, default=0.0005)
    ap.add_argument("--dense-epsilon", type=float, default=1e-6)
    ap.add_argument("--out-episode-csv", type=Path, required=True)
    ap.add_argument("--out-trajectory-gz", type=Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(args.project_root / "src"))
    sb_scripts = args.project_root / "experiments" / "pilots" / "study_b_fairness_mappo" / "scripts"
    sys.path.insert(0, str(sb_scripts))

    from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig
    from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv, StudyBHighwayWrapperConfig
    from thesis.study_b.q_ensemble import ensemble_window_for_stage_end, load_ensemble_agents, select_ensemble_actions
    from thesis.study_b.training_common import load_scenario_bank
    from thesis.study_b.utility import (
        episode_burdens, episode_utilities, gini_coefficient, generalized_gini_welfare,
    )
    from thesis.study_b.welfare_reward import condition_by_name
    from thesis.pilots.stage11_welfare import target_speed_attainment

    # NEUTRAL_PHI / dense_shaping_term inlined verbatim from
    # thesis.study_b.dense_shaping (that module only exists in the dense-reward
    # bundle's own project/src, not in the pre-existing F:\正式训练_seed_replication_v1
    # project used by Cells 1/3 -- inlining avoids a cross-bundle import dependency
    # while reconstructing the EXACT SAME frozen formula for all four cells,
    # including the terminal-only cells where this is a counterfactual signal
    # per Section 14 of the re-evaluation prompt).
    NEUTRAL_PHI = 1.0

    def dense_shaping_term(delta_phi: float, magnitude: float, epsilon: float) -> float:
        if delta_phi > epsilon:
            return float(magnitude)
        if delta_phi < -epsilon:
            return float(-magnitude)
        return 0.0

    WINDOW = ensemble_window_for_stage_end(2_000_000)
    ckpt_paths = {s: args.ckpt_dir / f"ckpt_step_{s}.pt" for s in WINDOW}
    missing = [str(p) for p in ckpt_paths.values() if not p.exists()]
    if missing:
        raise SystemExit(f"[dws_eval_worker] missing ensemble checkpoint(s) for seed={args.seed} cell={args.cell}: {missing}")

    agents = load_ensemble_agents(
        seed=args.seed, checkpoint_paths=ckpt_paths, expected_steps=WINDOW,
        expected_stage_by_step=dict.fromkeys(WINDOW, args.stage_name),
        obs_dim=args.obs_dim,
    )
    scenarios = load_scenario_bank(args.scenario_bank)
    if len(scenarios) != 256:
        raise SystemExit(f"[dws_eval_worker] expected 256 H1 scenarios, got {len(scenarios)}: {args.scenario_bank}")

    env_config = ThesisHighwayMergeEnvConfig(episode_max_steps=200, action_representation="meta_speed")
    env = StudyBHeterogeneousHighwayEnv(
        StudyBHighwayWrapperConfig(env_config=env_config, local_sensing_range_m=R_OBS, include_welfare_state=args.include_welfare_state)
    )
    condition = condition_by_name(args.condition)

    episode_rows: list[dict] = []
    args.out_trajectory_gz.parent.mkdir(parents=True, exist_ok=True)
    args.out_episode_csv.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(args.out_trajectory_gz, "wt", encoding="utf-8") as traj_f:
        for scenario in scenarios:
            obs, _info = env.reset(seed=0, scenario=scenario)
            pre_active = {v: True for v in VIDS}
            groups = {v: f"{scenario.vehicles[v].role}-{scenario.vehicles[v].speed_class}" for v in VIDS}
            roles = {v: scenario.vehicles[v].role for v in VIDS}
            speed_classes = {v: scenario.vehicles[v].speed_class for v in VIDS}
            target_speeds = {v: scenario.vehicles[v].target_speed for v in VIDS}
            running_sum = {v: 0.0 for v in VIDS}
            running_n = {v: 0 for v in VIDS}
            exit_step: dict[str, int | None] = {v: None for v in VIDS}
            hb_in_run: dict[str, bool] = {v: False for v in VIDS}
            pair_first_state: dict[str, tuple] = {}

            steps_log: list[dict] = []
            prev_phi = NEUTRAL_PHI
            term_reason = "truncation"
            steps_taken = 0

            for t in range(200):
                actions = select_ensemble_actions(agents, obs)

                xs: dict[str, float] = {}
                ms_now: dict[str, float] = {}
                accels: dict[str, float] = {}
                for v in VIDS:
                    if not pre_active[v]:
                        continue
                    vehicle = env._env._vehicle_by_id[v]  # noqa: SLF001
                    x, _y = env._env.world_xy(vehicle)  # noqa: SLF001
                    xs[v] = x
                    ms_now[v] = (running_sum[v] / running_n[v]) if running_n[v] > 0 else 1.0
                    running_sum[v] += target_speed_attainment(float(vehicle.speed), target_speeds[v])
                    running_n[v] += 1

                # Phi_t over the FIXED four-vehicle cohort (frozen active-set rule) --
                # matches welfare_objective_snapshot's own contract: uses env.episode_traces()
                # style M_i, but we already have the equivalent incremental value in ms_now for
                # ACTIVE vehicles; for exited vehicles reuse their last known M (frozen), matching
                # running_active_attainment's own freeze-on-exit behaviour.
                m_all = {v: ms_now.get(v, steps_log[-1]["M"][v] if steps_log else NEUTRAL_PHI) for v in VIDS}
                phi_t = condition.welfare_fn([m_all[v] for v in VIDS])
                delta_phi = phi_t - prev_phi
                f_t = dense_shaping_term(delta_phi, args.dense_magnitude, args.dense_epsilon)
                prev_phi = phi_t

                # merge-priority: record first co-occurrence welfare state per pair
                for i in xs:
                    for j in xs:
                        if j <= i:
                            continue
                        if abs(xs[i] - xs[j]) > R_OBS:
                            continue
                        pair_key = f"{i}-{j}"
                        if pair_key not in pair_first_state:
                            pair_first_state[pair_key] = (ms_now[i], ms_now[j], t)

                # hard-brake event starts (contiguous run collapsed to its first step)
                hb_start = {}
                for v in xs:
                    vehicle = env._env._vehicle_by_id[v]  # noqa: SLF001
                    accel = float(vehicle.action["acceleration"])
                    accels[v] = accel
                    is_hb = accel <= HARD_BRAKE_THRESH
                    hb_start[v] = bool(is_hb and not hb_in_run[v])
                    hb_in_run[v] = is_hb

                steps_log.append({
                    "t": t,
                    "active": {v: (v in xs) for v in VIDS},
                    "x": {v: xs.get(v) for v in VIDS},
                    "M": {v: m_all[v] for v in VIDS},
                    "action": {v: int(actions[v]) for v in VIDS if v in actions},
                    "accel": {v: accels.get(v) for v in VIDS},
                    "hard_brake_start": {v: hb_start.get(v, False) for v in VIDS},
                    "Phi": phi_t, "DeltaPhi": delta_phi, "F_t": f_t,
                })

                obs, _reward, terminated, truncated, step_info = env.step(actions)
                steps_taken += 1

                for v in VIDS:
                    if not pre_active[v]:
                        continue
                    vehicle = env._env._vehicle_by_id[v]  # noqa: SLF001
                    x, _y = env._env.world_xy(vehicle)  # noqa: SLF001
                    if exit_step[v] is None and x >= X_MERGE_END:
                        exit_step[v] = t

                if terminated:
                    term_reason = "collision" if step_info["collision_event"] else "success"
                elif truncated:
                    term_reason = "truncation"
                pre_active = dict(step_info["active"])
                if terminated or truncated:
                    break

            traces = env.episode_traces()
            utilities = episode_utilities(traces)
            burdens = episode_burdens(traces, dt=env.dt())
            u_values = [utilities[v] for v in VIDS]
            u_min_val = min(u_values)
            tied_worst = [v for v in VIDS if abs(utilities[v] - u_min_val) < TIE_TOL]

            row = {
                "cell": args.cell, "condition": args.condition,
                "observation_design": "WSC_22D" if args.include_welfare_state else "Original_18D",
                "dws_on": int(args.dws_on),
                "seed": args.seed, "scenario_id": scenario.scenario_id, "traffic_type": scenario.traffic_type,
                "term_reason": term_reason,
                "completion": int(term_reason == "success"), "collision": int(term_reason == "collision"),
                "timeout": int(term_reason == "truncation"), "episode_length": steps_taken,
                "mean_U": sum(u_values) / len(u_values), "min_U": u_min_val,
                "min_U_vehicles_tied": "|".join(sorted(tied_worst)), "min_U_n_tied": len(tied_worst),
                "ggi": generalized_gini_welfare(u_values), "gini": gini_coefficient(u_values),
                "C_max": max(burdens.values()), "C_mean": sum(burdens.values()) / len(burdens),
                "C_total": sum(burdens.values()),
            }
            for v in VIDS:
                row[f"role_{v}"] = roles[v]
                row[f"speed_class_{v}"] = speed_classes[v]
                row[f"U_{v}"] = utilities[v]
                row[f"C_{v}"] = burdens[v]
            episode_rows.append(row)

            traj_f.write(json.dumps({
                "cell": args.cell, "seed": args.seed, "scenario_id": scenario.scenario_id,
                "term_reason": term_reason, "roles": roles, "speed_classes": speed_classes,
                "target_speeds": target_speeds, "exit_step": exit_step,
                "pair_first_state": pair_first_state,
                "steps": steps_log,
            }) + "\n")

    fieldnames = list(episode_rows[0].keys())
    with open(args.out_episode_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(episode_rows)

    n = len(episode_rows)
    comp = sum(r["completion"] for r in episode_rows) / n
    coll = sum(r["collision"] for r in episode_rows) / n
    print(f"[dws_eval_worker] cell={args.cell} seed={args.seed} n={n} completion={comp:.3f} collision={coll:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
