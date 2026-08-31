#!/usr/bin/env python3
"""HighwayEnv-backend held-out evaluation (Gate K, pre-formal audit):
replays a FROZEN scenario bank against a trained DQN checkpoint,
strictly GREEDY (epsilon=0.0, argmax -- no training exploration),
using a dedicated evaluation env instance that never touches the
training replay buffer, and reports outcomes using the THESIS
term_reason (collision_event / completed_this_step / truncated from
``ThesisHighwayMergeEnv``), never any HighwayEnv-native metric.

Mirrors ``evaluate_policy.py``'s (legacy backend) shape/CSV columns so
downstream analysis code can consume either without modification."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(REPO_SRC))

import torch  # noqa: E402

from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig  # noqa: E402
from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv, StudyBHighwayWrapperConfig  # noqa: E402
from thesis.study_b.shared_local_dqn import SharedLocalDQNAgent, build_study_b_dqn_config  # noqa: E402
from thesis.study_b.training_common import load_scenario_bank  # noqa: E402
from thesis.study_b.utility import episode_burdens, episode_utilities, generalized_gini_welfare, gini_coefficient  # noqa: E402

CSV_FIELDS = [
    "scenario_id", "traffic_type", "term_reason", "completion", "collision", "timeout",
    "mean_U", "min_U", "min_U_vehicle", "min_U_role", "min_U_speed_class", "ggi", "gini",
    "C_max", "C_mean", "episode_length", "mean_undiscounted_return",
] + [f"{prefix}_{vid}" for vid in ("V0", "V1", "V2", "V3") for prefix in ("role", "speed_class", "U", "C")]


def load_policy_highwayenv(*, checkpoint: Path, device: str = "cpu"):
    """Loads ONLY the online network's weights (no optimizer/target-net
    state, no replay) -- a strictly greedy, evaluation-only policy
    closure, structurally incapable of writing back to any training
    state."""
    dqn_config = build_study_b_dqn_config(device=device)
    agent = SharedLocalDQNAgent(dqn_config, seed=0)
    ckpt = torch.load(checkpoint, map_location=device)
    agent.learner.online.load_state_dict(ckpt["online"])
    return lambda obs: agent.select_actions(obs, epsilon=0.0, greedy=True)


def run_eval_highwayenv(
    *, checkpoint: Path, scenario_bank: Path, action_representation: str = "meta_speed",
    episode_max_steps: int = 200, device: str = "cpu",
) -> list[dict]:
    scenarios = load_scenario_bank(scenario_bank)
    env_config = ThesisHighwayMergeEnvConfig(episode_max_steps=episode_max_steps, action_representation=action_representation)
    # Dedicated evaluation env instance -- never the one a training loop
    # (if any were concurrently running) uses, and this function never
    # calls agent.store_transition()/maybe_update() anywhere, so no
    # evaluation trajectory can ever enter replay.
    env = StudyBHeterogeneousHighwayEnv(StudyBHighwayWrapperConfig(env_config=env_config))
    select = load_policy_highwayenv(checkpoint=checkpoint, device=device)

    rows = []
    for scenario in scenarios:
        obs, _info = env.reset(seed=0, scenario=scenario)
        episode_return: dict[str, float] = dict.fromkeys(env.active_vehicle_ids, 0.0)
        steps_taken = 0
        term_reason = "truncation"
        for _t in range(episode_max_steps):
            actions = select(obs)
            obs, reward, terminated, truncated, step_info = env.step(actions)
            for vid, r in reward.items():
                episode_return[vid] = episode_return.get(vid, 0.0) + r
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
            "scenario_id": scenario.scenario_id,
            "traffic_type": scenario.traffic_type,
            "term_reason": term_reason,
            "completion": int(term_reason == "success"),
            "collision": int(term_reason == "collision"),
            "timeout": int(term_reason == "truncation"),
            "mean_U": sum(u_values) / len(u_values),
            "min_U": utilities[min_vid],
            "min_U_vehicle": min_vid,
            "min_U_role": scenario.vehicles[min_vid].role,
            "min_U_speed_class": scenario.vehicles[min_vid].speed_class,
            "ggi": generalized_gini_welfare(u_values),
            "gini": gini_coefficient(u_values),
            "C_max": max(burdens.values()),
            "C_mean": sum(burdens.values()) / len(burdens),
            "episode_length": steps_taken,
            "mean_undiscounted_return": sum(episode_return.values()) / len(episode_return),
        }
        for vid in ("V0", "V1", "V2", "V3"):
            row[f"role_{vid}"] = scenario.vehicles[vid].role
            row[f"speed_class_{vid}"] = scenario.vehicles[vid].speed_class
            row[f"U_{vid}"] = utilities[vid]
            row[f"C_{vid}"] = burdens[vid]
        rows.append(row)

    # Mutually exclusive outcome check (Gate K).
    for row in rows:
        assert row["completion"] + row["collision"] + row["timeout"] == 1, row
    return rows


def write_csv(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--scenario-bank", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--action-representation", type=str, default="meta_speed", choices=["direct_accel", "meta_speed"])
    p.add_argument("--episode-max-steps", type=int, default=200)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args(argv)

    rows = run_eval_highwayenv(
        checkpoint=args.checkpoint, scenario_bank=args.scenario_bank,
        action_representation=args.action_representation, episode_max_steps=args.episode_max_steps, device=args.device,
    )
    write_csv(rows, args.output)

    n = len(rows)
    completion_rate = sum(r["completion"] for r in rows) / n
    collision_rate = sum(r["collision"] for r in rows) / n
    timeout_rate = sum(r["timeout"] for r in rows) / n
    print(f"evaluated {n} scenarios -> {args.output}")
    print(f"completion_rate={completion_rate:.4f} collision_rate={collision_rate:.4f} timeout_rate={timeout_rate:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
