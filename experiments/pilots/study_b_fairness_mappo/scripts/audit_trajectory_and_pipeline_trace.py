#!/usr/bin/env python3
"""Pre-formal end-to-end audit, Gates D + N: a deterministic, SCRIPTED
(not learned) action sequence run on one oracle-solvable scenario
(Q_00000), with per-physics-substep instrumentation, producing both
``AUDIT_ACTION_PIPELINE_TRACE.csv`` (Gate D) and
``AUDIT_SINGLE_TRAJECTORY_TRACE.csv`` (Gate N).

Per-substep granularity is obtained by monkey-patching
``MetaSpeedControlledVehicle.act`` for the duration of this script only
(never modifies the source file) to append every call's requested/
realized acceleration to a log, since the class itself only keeps the
LAST call's values as instance attributes."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(REPO_SRC))

import numpy as np  # noqa: E402

from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig  # noqa: E402
from thesis.study_b.envs.highwayenv_vehicle import MetaSpeedControlledVehicle  # noqa: E402
from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv, StudyBHighwayWrapperConfig  # noqa: E402
from thesis.study_b.oracle_controller import ACCELERATE, DECELERATE as BRAKE, MAINTAIN as HOLD, oracle_actions  # noqa: E402
from thesis.study_b.training_common import load_scenario_bank  # noqa: E402
from thesis.study_b.utility import episode_burdens, episode_utilities  # noqa: E402

ACTION_LABELS = {HOLD: "HOLD", ACCELERATE: "ACCELERATE", BRAKE: "BRAKE"}

OUT_DIR = REPO_SRC.parent / "output" / "pre_formal_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _debounce(raw, vehicle):
    if raw == ACCELERATE:
        return HOLD if vehicle.target_speed > vehicle.speed else ACCELERATE
    if raw == BRAKE:
        return HOLD if vehicle.target_speed < vehicle.speed else BRAKE
    return HOLD


def main() -> int:
    cfg = ThesisHighwayMergeEnvConfig(action_representation="meta_speed")
    merge_start = cfg.before_merge_length
    merge_end = cfg.before_merge_length + cfg.converge_merge_length

    scenarios = {s.scenario_id: s for s in load_scenario_bank(
        REPO_SRC.parent / "experiments/pilots/study_b_fairness_mappo/scenario_banks/Q.json"
    )}
    scenario = scenarios["Q_00000"]

    env = StudyBHeterogeneousHighwayEnv(StudyBHighwayWrapperConfig(env_config=cfg))
    obs, info = env.reset(seed=0, scenario=scenario)
    roles = info["roles"]
    role_by_vid = {vid: r for r, vids in roles.items() for vid in vids}
    speed_class_by_vid = {vid: scenario.vehicles[vid].speed_class for vid in env.active_vehicle_ids}

    # Monkey-patch (this script's process only) to capture EVERY act()
    # call, not just the last one, so every physics substep is visible.
    substep_log: list[dict] = []
    orig_act = MetaSpeedControlledVehicle.act
    vid_by_vehicle_obj = {}  # populated after reset, below

    def patched_act(self, action=None):
        orig_act(self, action)
        vid = vid_by_vehicle_obj.get(id(self), "?")
        substep_log.append({
            "vehicle_id": vid,
            "requested_accel": self.last_requested_acceleration,
            "realized_accel": self.last_realized_acceleration,
            "physics_accel_used": self.action["acceleration"],
            "target_speed": self.target_speed,
            "speed": self.speed,
            "frozen": self.frozen,
        })

    MetaSpeedControlledVehicle.act = patched_act
    for vid in env.active_vehicle_ids:
        vid_by_vehicle_obj[id(env._env._vehicle_by_id[vid])] = vid  # noqa: SLF001

    pipeline_rows = []
    trajectory_rows = []
    cumulative_utility_samples: dict[str, list[float]] = {vid: [] for vid in env.active_vehicle_ids}

    policy_step = 0
    terminated = truncated = False
    prev_active = {vid: True for vid in env.active_vehicle_ids}
    while not (terminated or truncated) and policy_step < 200:
        substep_log.clear()
        # Scripted policy: the (fixed, non-learned) oracle + windup-safe
        # debounce -- deterministic given the scenario, exercises a real
        # coordination trajectory rather than a trivial constant action.
        positions = {vid: env._env.world_xy(env._env._vehicle_by_id[vid])[0] for vid in env.active_vehicle_ids}  # noqa: SLF001
        lateral_positions = {vid: env._env.world_xy(env._env._vehicle_by_id[vid])[1] for vid in env.active_vehicle_ids}  # noqa: SLF001
        raw_actions = oracle_actions(
            scenario=scenario, positions=positions, merge_start=merge_start, merge_end=merge_end,
            active_vehicle_ids={vid: True for vid in env.active_vehicle_ids}, lateral_positions=lateral_positions,
        )
        actions = {vid: _debounce(a, env._env._vehicle_by_id[vid]) for vid, a in raw_actions.items()}  # noqa: SLF001

        pre_state = {
            vid: {
                "speed": float(env._env._vehicle_by_id[vid].speed),  # noqa: SLF001
                "target_speed": float(env._env._vehicle_by_id[vid].target_speed),  # noqa: SLF001
                "x": float(env._env.world_xy(env._env._vehicle_by_id[vid])[0]),  # noqa: SLF001
            }
            for vid in env.active_vehicle_ids
        }

        next_obs, reward, terminated, truncated, step_info = env.step(actions)

        # Group the substep log by vehicle in call order -> substep index.
        per_vehicle_substeps: dict[str, list[dict]] = {vid: [] for vid in env.active_vehicle_ids}
        for entry in substep_log:
            if entry["vehicle_id"] in per_vehicle_substeps:
                per_vehicle_substeps[entry["vehicle_id"]].append(entry)

        for vid in env.active_vehicle_ids:
            active_now = step_info["active"].get(vid, False)
            completed_now = env._env._completed.get(vid, False)  # noqa: SLF001
            veh = env._env._vehicle_by_id[vid]  # noqa: SLF001
            entries = per_vehicle_substeps[vid]
            for substep_idx, entry in enumerate(entries):
                pipeline_rows.append({
                    "policy_step": policy_step, "physics_substep": substep_idx, "vehicle_id": vid,
                    "active": int(prev_active.get(vid, True)), "completed": int(completed_now),
                    "action_index": actions[vid], "action_label": ACTION_LABELS[actions[vid]],
                    "desired_speed_before": pre_state[vid]["target_speed"] if substep_idx == 0 else entries[substep_idx - 1]["target_speed"],
                    "desired_speed_after": entry["target_speed"],
                    "controller_requested_accel": entry["requested_accel"],
                    "realized_physics_accel": entry["realized_accel"],
                    "speed_before": pre_state[vid]["speed"] if substep_idx == 0 else entries[substep_idx - 1]["speed"],
                    "speed_after": entry["speed"],
                    "path_position_before": "n/a_substep_granularity_not_tracked_for_position",
                    "path_position_after": "n/a_substep_granularity_not_tracked_for_position",
                })

            x, y = env._env.world_xy(veh)  # noqa: SLF001
            if active_now:
                cumulative_utility_samples[vid].append(min(1.0, max(0.0, veh.speed / scenario.vehicles[vid].target_speed)))

            trajectory_rows.append({
                "time": round(policy_step * env.dt(), 3), "policy_step": policy_step, "physics_substep": "final",
                "vehicle_id": vid, "role": role_by_vid[vid], "speed_class": speed_class_by_vid[vid],
                "active": int(active_now), "completed": int(completed_now),
                "lane_index": str(veh.lane_index), "path_longitudinal": "see world_x (shared axis, see scenario_adapter.py)",
                "world_x": round(x, 4), "world_y": round(y, 4), "heading": round(float(veh.heading), 5),
                "speed": round(float(veh.speed), 5), "desired_speed": round(float(veh.target_speed), 5),
                "action_index": actions[vid], "action_label": ACTION_LABELS[actions[vid]],
                "controller_requested_accel": round(entries[-1]["requested_accel"], 5) if entries else "",
                "realized_accel": round(entries[-1]["realized_accel"], 5) if entries else "",
                "observation_vector": ";".join(f"{v:.4f}" for v in next_obs.get(vid, np.array([]))),
                "reward_components": round(reward.get(vid, 0.0), 6),
                "instantaneous_utility": round(cumulative_utility_samples[vid][-1], 5) if (active_now and cumulative_utility_samples[vid]) else "",
                "cumulative_burden": round(env.dt() * sum(1.0 - u for u in cumulative_utility_samples[vid]), 5),
                "collision": int(step_info.get("collision_event", False)),
                "terminated": int(terminated), "truncated": int(truncated),
            })

        prev_active = dict(step_info["active"])
        policy_step += 1

    MetaSpeedControlledVehicle.act = orig_act  # restore

    with (OUT_DIR / "AUDIT_ACTION_PIPELINE_TRACE.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pipeline_rows[0].keys()))
        w.writeheader()
        w.writerows(pipeline_rows)

    with (OUT_DIR / "AUDIT_SINGLE_TRAJECTORY_TRACE.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(trajectory_rows[0].keys()))
        w.writeheader()
        w.writerows(trajectory_rows)

    traces = env.episode_traces()
    utilities = episode_utilities(traces)
    burdens = episode_burdens(traces, dt=env.dt())

    print(f"scenario=Q_00000 policy_steps={policy_step} terminated={terminated} truncated={truncated}")
    print(f"final utilities={utilities}")
    print(f"final burdens={burdens}")
    print(f"pipeline rows: {len(pipeline_rows)} -> {OUT_DIR / 'AUDIT_ACTION_PIPELINE_TRACE.csv'}")
    print(f"trajectory rows: {len(trajectory_rows)} -> {OUT_DIR / 'AUDIT_SINGLE_TRAJECTORY_TRACE.csv'}")

    # Hard bound assertion, per audit.md sec 1/7-D4.
    accels = [row["realized_physics_accel"] for row in pipeline_rows]
    assert min(accels) >= -3.0 - 1e-9, f"realized accel violates floor: {min(accels)}"
    assert max(accels) <= 2.0 + 1e-9, f"realized accel violates ceiling: {max(accels)}"
    print(f"D4 hard bound check: min={min(accels):.4f} max={max(accels):.4f} -- PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
