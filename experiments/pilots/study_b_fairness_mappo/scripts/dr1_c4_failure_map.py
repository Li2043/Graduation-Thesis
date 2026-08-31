#!/usr/bin/env python3
"""DR1 -- C4 four-seed failure map (RUNBOOK C4_STATUS=NOT_YET_QUALIFIED
diagnostic freeze, 2026-08-16).

For each of the four C4 600K checkpoints, runs a large batch of
diagnostic episodes reproducing the SAME execution policy that produced
the reported 600K window statistics during training (epsilon-greedy at
epsilon_at_step_v12(600000, decay_steps=640000), scenario sampled
uniformly among the 4 frozen C4 scenarios each episode, env reset with
seed=0+scenario per the training script's own convention) -- but with
full per-episode logging instead of a terse rolling-window print, so
failure structure can be inspected.

No training occurs: the checkpoint's network weights are loaded once
and never updated (no optimiser, no replay, no backward pass).

Collision-type classification rule (documented explicitly, not assumed):
using the frozen ThesisHighwayMergeEnvConfig geometry (before_merge_length
= merge_start_x, before_merge_length+converge_merge_length = merge_end_x):
  - both colliding vehicles' x >= merge_end_x   -> "same-lane rear-end"
  - both colliding vehicles' x >= merge_start_x -> "merge/conflict-zone"
  - otherwise                                   -> "cross-lane/geometry"
(no "other/unclassified" case has been observed to date; the label
exists in the schema for completeness.)

Dynamic same-lane TTC proxy (the scenario bank's own `nominal_ttc` is a
static spawn-time value, not tracked per-step by the env): at every
step, for every pair of vehicles with |y_a - y_b| < DEFAULT_LATERAL_
SAME_LANE_THRESHOLD_M (2.0m, same constant oracle_controller.py uses),
compute longitudinal gap and closing speed; TTC = gap / closing_speed
when closing_speed > 0, else +inf. The episode's min_ttc is the minimum
finite TTC observed over any such pair across the whole episode.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(REPO_SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig  # noqa: E402
from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv, StudyBHighwayWrapperConfig  # noqa: E402
from thesis.study_b.oracle_controller import DEFAULT_LATERAL_SAME_LANE_THRESHOLD_M  # noqa: E402
from thesis.study_b.shared_local_dqn import SharedLocalDQNAgent, build_study_b_dqn_config, epsilon_at_step_v12  # noqa: E402
from thesis.study_b.training_common import load_scenario_bank  # noqa: E402

_ENV_CONFIG = ThesisHighwayMergeEnvConfig()
_MERGE_START_X = _ENV_CONFIG.before_merge_length
_MERGE_END_X = _ENV_CONFIG.before_merge_length + _ENV_CONFIG.converge_merge_length

EPISODE_CSV_FIELDS = [
    "seed", "episode_index", "scenario_id", "term_reason", "episode_length",
    "min_ttc", "failure_timestep", "collision_type",
    "collision_vehicle_ids", "collision_vehicle_roles", "collision_vehicle_speed_classes",
    "collision_vehicle_ttc_slots", "merge_order",
]


def classify_collision_type(x_positions: dict[str, float], pair: tuple[str, str]) -> str:
    xa, xb = x_positions[pair[0]], x_positions[pair[1]]
    if xa >= _MERGE_END_X and xb >= _MERGE_END_X:
        return "same-lane rear-end"
    if xa >= _MERGE_START_X and xb >= _MERGE_START_X:
        return "merge/conflict-zone"
    return "cross-lane/geometry"


def min_pairwise_ttc(
    x_positions: dict[str, float], y_positions: dict[str, float], speeds: dict[str, float], vids: tuple[str, ...]
) -> float:
    best = float("inf")
    for i, a in enumerate(vids):
        for b in vids[i + 1:]:
            if abs(y_positions[a] - y_positions[b]) >= DEFAULT_LATERAL_SAME_LANE_THRESHOLD_M:
                continue
            gap = abs(x_positions[a] - x_positions[b])
            # closing speed: positive when the rear vehicle (smaller x) is faster
            if x_positions[a] <= x_positions[b]:
                rear, front = a, b
            else:
                rear, front = b, a
            closing = speeds[rear] - speeds[front]
            if closing > 1e-6:
                ttc = gap / closing
                if ttc < best:
                    best = ttc
    return best


def run_dr1_for_seed(
    *, seed: int, checkpoint: Path, scenario_bank_path: Path, scenario_ids: list[str],
    absolute_step: int, eps_decay_steps: int, n_episodes: int, episode_max_steps: int, device: str,
) -> list[dict]:
    scenarios = load_scenario_bank(scenario_bank_path)
    by_id = {s.scenario_id: s for s in scenarios}
    stage_scenarios = [by_id[sid] for sid in scenario_ids]

    env_config = ThesisHighwayMergeEnvConfig(episode_max_steps=episode_max_steps, action_representation="meta_speed")
    env = StudyBHeterogeneousHighwayEnv(StudyBHighwayWrapperConfig(env_config=env_config))

    dqn_config = build_study_b_dqn_config(device=device)
    # Salted, independent of the training RNG stream -- a fresh diagnostic
    # exploration sequence, not a replay of what happened during training.
    agent = SharedLocalDQNAgent(dqn_config, seed=seed * 1_000_003 + 555_001)
    ckpt = torch.load(checkpoint, map_location=device)
    agent.learner.online.load_state_dict(ckpt["online"])
    # No optimiser/target load, no maybe_update() call anywhere below --
    # this is read-only against the frozen checkpoint.

    eps = epsilon_at_step_v12(absolute_step, decay_steps=eps_decay_steps)
    scenario_rng = np.random.default_rng(seed * 1_000_003 + 555_002)

    rows: list[dict] = []
    for episode_index in range(n_episodes):
        scenario = stage_scenarios[int(scenario_rng.integers(0, len(stage_scenarios)))]
        obs, _info = env.reset(seed=0, scenario=scenario)
        episode_min_ttc = float("inf")
        term_reason = "ongoing"
        failure_timestep = None
        collision_pairs: list[tuple[str, str]] = []
        completion_step: dict[str, int | None] = dict.fromkeys(env.active_vehicle_ids)
        steps_taken = 0

        for t in range(episode_max_steps):
            vids = env.active_vehicle_ids
            x_positions = {vid: env._env.world_xy(env._env._vehicle_by_id[vid])[0] for vid in vids}  # noqa: SLF001
            y_positions = {vid: env._env.world_xy(env._env._vehicle_by_id[vid])[1] for vid in vids}  # noqa: SLF001
            speeds = {vid: float(env._env._vehicle_by_id[vid].speed) for vid in vids}  # noqa: SLF001
            step_ttc = min_pairwise_ttc(x_positions, y_positions, speeds, vids)
            if step_ttc < episode_min_ttc:
                episode_min_ttc = step_ttc

            actions = agent.select_actions(obs, epsilon=eps)
            obs, _reward, terminated, truncated, step_info = env.step(actions)
            steps_taken += 1

            for vid, done in step_info["exit_event"].items():
                if done and completion_step[vid] is None:
                    completion_step[vid] = t + 1

            if step_info["collision_event"]:
                term_reason = "collision"
                failure_timestep = t + 1
                collision_pairs = step_info["collision_pairs"]
                break
            if terminated:
                term_reason = "success"
                break
            if truncated:
                term_reason = "truncation"
                break

        collision_vehicle_ids: list[str] = []
        collision_vehicle_roles: list[str] = []
        collision_vehicle_speed_classes: list[str] = []
        collision_vehicle_ttc_slots: list[str] = []
        collision_type = ""
        if term_reason == "collision" and collision_pairs:
            pair = collision_pairs[0]
            x_positions_final = {vid: env._env.world_xy(env._env._vehicle_by_id[vid])[0] for vid in env.active_vehicle_ids}  # noqa: SLF001
            collision_type = classify_collision_type(x_positions_final, pair)
            colliding_ids = sorted({vid for p in collision_pairs for vid in p})
            collision_vehicle_ids = colliding_ids
            collision_vehicle_roles = [scenario.vehicles[vid].role for vid in colliding_ids]
            collision_vehicle_speed_classes = [scenario.vehicles[vid].speed_class for vid in colliding_ids]
            collision_vehicle_ttc_slots = [scenario.vehicles[vid].ttc_slot for vid in colliding_ids]

        merge_order = sorted(
            (vid for vid in completion_step if completion_step[vid] is not None),
            key=lambda vid: completion_step[vid],
        )
        merge_order_str = ">".join(merge_order) if merge_order else "DNF"

        rows.append({
            "seed": seed, "episode_index": episode_index, "scenario_id": scenario.scenario_id,
            "term_reason": term_reason, "episode_length": steps_taken,
            "min_ttc": (None if episode_min_ttc == float("inf") else round(episode_min_ttc, 4)),
            "failure_timestep": failure_timestep, "collision_type": collision_type,
            "collision_vehicle_ids": ";".join(collision_vehicle_ids),
            "collision_vehicle_roles": ";".join(collision_vehicle_roles),
            "collision_vehicle_speed_classes": ";".join(collision_vehicle_speed_classes),
            "collision_vehicle_ttc_slots": ";".join(collision_vehicle_ttc_slots),
            "merge_order": merge_order_str,
        })

    return rows


def summarize(all_rows: list[dict]) -> dict:
    by_seed: dict[int, list[dict]] = defaultdict(list)
    for r in all_rows:
        by_seed[r["seed"]].append(r)

    def rate(rows: list[dict], key: str, value: str) -> float:
        return sum(1 for r in rows if r[key] == value) / len(rows) if rows else 0.0

    summary: dict = {"pooled": {}, "per_seed": {}, "per_seed_per_scenario": {}}
    summary["pooled"]["n_episodes"] = len(all_rows)
    summary["pooled"]["completion_rate"] = rate(all_rows, "term_reason", "success")
    summary["pooled"]["collision_rate"] = rate(all_rows, "term_reason", "collision")
    summary["pooled"]["timeout_rate"] = rate(all_rows, "term_reason", "truncation")
    summary["pooled"]["collision_type_breakdown"] = dict(
        Counter(r["collision_type"] for r in all_rows if r["term_reason"] == "collision")
    )
    summary["pooled"]["collision_role_breakdown"] = dict(
        Counter(role for r in all_rows if r["term_reason"] == "collision" for role in r["collision_vehicle_roles"].split(";") if role)
    )
    summary["pooled"]["collision_speed_class_breakdown"] = dict(
        Counter(sc for r in all_rows if r["term_reason"] == "collision" for sc in r["collision_vehicle_speed_classes"].split(";") if sc)
    )
    summary["pooled"]["collision_ttc_slot_breakdown"] = dict(
        Counter(s for r in all_rows if r["term_reason"] == "collision" for s in r["collision_vehicle_ttc_slots"].split(";") if s)
    )
    summary["pooled"]["collision_by_scenario"] = dict(
        Counter(r["scenario_id"] for r in all_rows if r["term_reason"] == "collision")
    )

    for seed, rows in by_seed.items():
        s = {
            "n_episodes": len(rows),
            "completion_rate": rate(rows, "term_reason", "success"),
            "collision_rate": rate(rows, "term_reason", "collision"),
            "timeout_rate": rate(rows, "term_reason", "truncation"),
            "collision_type_breakdown": dict(Counter(r["collision_type"] for r in rows if r["term_reason"] == "collision")),
            "collision_by_scenario": dict(Counter(r["scenario_id"] for r in rows if r["term_reason"] == "collision")),
        }
        summary["per_seed"][str(seed)] = s

        by_scenario: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_scenario[r["scenario_id"]].append(r)
        summary["per_seed_per_scenario"][str(seed)] = {
            sid: {
                "n_episodes": len(srows),
                "completion_rate": rate(srows, "term_reason", "success"),
                "collision_rate": rate(srows, "term_reason", "collision"),
                "timeout_rate": rate(srows, "term_reason", "truncation"),
            }
            for sid, srows in by_scenario.items()
        }

    return summary


def write_markdown(summary: dict, output: Path) -> None:
    lines = ["# DR1 -- C4 four-seed failure map\n"]
    p = summary["pooled"]
    lines.append(f"Pooled over all seeds: n={p['n_episodes']}, "
                 f"completion={p['completion_rate']:.3f}, collision={p['collision_rate']:.3f}, timeout={p['timeout_rate']:.3f}\n")
    lines.append(f"Collision type breakdown (pooled): {p['collision_type_breakdown']}\n")
    lines.append(f"Collision role breakdown (pooled): {p['collision_role_breakdown']}\n")
    lines.append(f"Collision speed_class breakdown (pooled): {p['collision_speed_class_breakdown']}\n")
    lines.append(f"Collision ttc_slot breakdown (pooled): {p['collision_ttc_slot_breakdown']}\n")
    lines.append(f"Collision by scenario (pooled): {p['collision_by_scenario']}\n")
    lines.append("\n## Per-seed\n")
    lines.append("| seed | n | completion | collision | timeout | collision_types |")
    lines.append("|---|---|---|---|---|---|")
    for seed, s in summary["per_seed"].items():
        lines.append(
            f"| {seed} | {s['n_episodes']} | {s['completion_rate']:.3f} | {s['collision_rate']:.3f} | "
            f"{s['timeout_rate']:.3f} | {s['collision_type_breakdown']} |"
        )
    lines.append("\n## Per-seed per-scenario collision rate\n")
    lines.append("| seed | scenario_id | n | completion | collision | timeout |")
    lines.append("|---|---|---|---|---|---|")
    for seed, by_scen in summary["per_seed_per_scenario"].items():
        for sid, s in sorted(by_scen.items()):
            lines.append(f"| {seed} | {sid} | {s['n_episodes']} | {s['completion_rate']:.3f} | {s['collision_rate']:.3f} | {s['timeout_rate']:.3f} |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario-bank", type=Path, required=True)
    p.add_argument("--scenario-ids", type=str, nargs="+", required=True)
    p.add_argument("--checkpoints", type=str, nargs="+", required=True, help="seed:path pairs")
    p.add_argument("--absolute-step", type=int, default=600_000)
    p.add_argument("--eps-decay-steps", type=int, default=640_000)
    p.add_argument("--n-episodes-per-seed", type=int, default=600)
    p.add_argument("--episode-max-steps", type=int, default=200)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)

    all_rows: list[dict] = []
    for pair in args.checkpoints:
        seed_str, ckpt_str = pair.split(":", 1)
        seed = int(seed_str)
        rows = run_dr1_for_seed(
            seed=seed, checkpoint=Path(ckpt_str), scenario_bank_path=args.scenario_bank,
            scenario_ids=args.scenario_ids, absolute_step=args.absolute_step, eps_decay_steps=args.eps_decay_steps,
            n_episodes=args.n_episodes_per_seed, episode_max_steps=args.episode_max_steps, device=args.device,
        )
        all_rows.extend(rows)
        n = len(rows)
        print(f"[DR1] seed={seed} n={n} completion={sum(r['term_reason']=='success' for r in rows)/n:.3f} "
              f"collision={sum(r['term_reason']=='collision' for r in rows)/n:.3f} "
              f"timeout={sum(r['term_reason']=='truncation' for r in rows)/n:.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "C4_DR1_FAILURE_MAP.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EPISODE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    summary = summarize(all_rows)
    json_path = args.output_dir / "C4_DR1_FAILURE_MAP.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_path = args.output_dir / "C4_DR1_FAILURE_MAP.md"
    write_markdown(summary, md_path)

    print(f"wrote {csv_path}, {json_path}, {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
