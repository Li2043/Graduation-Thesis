#!/usr/bin/env python3
"""Study B held-out evaluation: replays a FROZEN scenario bank against a
trained checkpoint (greedy/deterministic policy, epsilon=0 / argmax) and
writes one row per scenario to a CSV -- the raw per-seed input
``analysis/*.py`` (bootstrap/Holm/behavioural) then consumes.

Works for both algorithms (``--algorithm mappo`` or ``--algorithm dqn``)
so downstream analysis code never needs to know which one produced a given
run's checkpoints."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(REPO_SRC))

import torch  # noqa: E402

from thesis.study_b.heterogeneous_env import StudyBEnvConfig, StudyBHeterogeneousEnv  # noqa: E402
from thesis.study_b.mappo import MAPPOConfig, MAPPOLearner  # noqa: E402
from thesis.study_b.shared_local_dqn import ALWAYS_LEGAL_ACTION_MASK, SharedLocalDQNAgent, build_study_b_dqn_config  # noqa: E402
from thesis.study_b.training_common import load_scenario_bank  # noqa: E402
from thesis.study_b.utility import (  # noqa: E402
    episode_burdens,
    episode_utilities,
    generalized_gini_welfare,
    gini_coefficient,
)

CSV_FIELDS = [
    "scenario_id", "traffic_type", "term_reason", "completion", "collision", "timeout",
    "mean_U", "min_U", "min_U_vehicle", "min_U_role", "min_U_speed_class", "ggi", "gini",
    "C_max", "C_mean", "hard_brake_total", "episode_length", "mean_undiscounted_return",
] + [f"{prefix}_{vid}" for vid in ("V0", "V1", "V2", "V3") for prefix in ("role", "speed_class", "U", "C", "hard_brake")]


def _select_actions_mappo(learner: MAPPOLearner, obs) -> dict[str, int]:
    actions, _log_probs = learner.select_actions(obs, deterministic=True)
    return actions


def _select_actions_dqn(agent: SharedLocalDQNAgent, obs) -> dict[str, int]:
    return agent.select_actions(obs, epsilon=0.0, greedy=True)


def load_policy(*, algorithm: str, checkpoint: Path, env: StudyBHeterogeneousEnv, device: str = "cpu", hidden_size: int = 128):
    """Loads a checkpoint and returns a ``select(obs) -> dict[str, int]``
    closure -- shared by ``main()`` and any other script that needs to run
    a trained checkpoint against the env (e.g. multi-checkpoint diagnostic
    tooling) without duplicating the load logic."""
    if algorithm == "mappo":
        config = MAPPOConfig(
            obs_dim=env.observation_dim, global_state_dim=env.global_state_dim,
            hidden_sizes=(hidden_size, hidden_size), device=device,
        )
        learner = MAPPOLearner(config, seed=0)
        ckpt = torch.load(checkpoint, map_location=device)
        learner.load_state_dict(ckpt["learner_state"])
        return lambda obs: _select_actions_mappo(learner, obs)
    dqn_config = build_study_b_dqn_config(device=device)
    agent = SharedLocalDQNAgent(dqn_config, seed=0)
    ckpt = torch.load(checkpoint, map_location=device)
    agent.learner.online.load_state_dict(ckpt["online"])
    return lambda obs: _select_actions_dqn(agent, obs)


def run_eval(
    *, algorithm: str, checkpoint: Path, scenario_bank: Path,
    episode_max_steps: int = 200, device: str = "cpu", hidden_size: int = 128,
) -> list[dict]:
    """Runs one checkpoint against one frozen scenario bank, greedily, and
    returns one summary row per scenario (the same shape written to
    ``evaluate_policy.py``'s CSV). Factored out of ``main()`` so other
    scripts (multi-checkpoint diagnostics, etc.) can call it directly
    instead of shelling out to this script once per checkpoint."""
    scenarios = load_scenario_bank(scenario_bank)
    env = StudyBHeterogeneousEnv(StudyBEnvConfig(episode_max_steps=episode_max_steps))
    select = load_policy(algorithm=algorithm, checkpoint=checkpoint, env=env, device=device, hidden_size=hidden_size)

    rows = []
    for scenario in scenarios:
        obs, _info = env.reset(seed=0, scenario=scenario)
        episode_return: dict[str, float] = dict.fromkeys(env.active_vehicle_ids, 0.0)
        steps_taken = 0
        for _t in range(episode_max_steps):
            actions = select(obs)
            obs, base_reward, terminated, truncated, step_info = env.step(actions)
            for vid, r in base_reward.items():
                episode_return[vid] = episode_return.get(vid, 0.0) + r
            steps_taken += 1
            if terminated or truncated:
                break
        else:
            step_info = {"term_reason": "truncation"}  # safety net if loop exhausts without a flag

        traces = env.episode_traces()
        utilities = episode_utilities(traces)
        burdens = episode_burdens(traces, dt=env.dt())
        u_values = list(utilities.values())
        min_vid = min(utilities, key=lambda v: utilities[v])

        row = {
            "scenario_id": scenario.scenario_id,
            "traffic_type": scenario.traffic_type,
            "term_reason": step_info["term_reason"],
            "completion": int(step_info["term_reason"] == "success"),
            "collision": int(step_info["term_reason"] == "collision"),
            "timeout": int(step_info["term_reason"] == "truncation"),
            "mean_U": sum(u_values) / len(u_values),
            "min_U": utilities[min_vid],
            "min_U_vehicle": min_vid,
            "min_U_role": scenario.vehicles[min_vid].role,
            "min_U_speed_class": scenario.vehicles[min_vid].speed_class,
            "ggi": generalized_gini_welfare(u_values),
            "gini": gini_coefficient(u_values),
            "C_max": max(burdens.values()),
            "C_mean": sum(burdens.values()) / len(burdens),
            "hard_brake_total": sum(traces[vid].hard_brake_count() for vid in traces),
            "episode_length": steps_taken,
            "mean_undiscounted_return": sum(episode_return.values()) / len(episode_return),
        }
        for vid in ("V0", "V1", "V2", "V3"):
            row[f"role_{vid}"] = scenario.vehicles[vid].role
            row[f"speed_class_{vid}"] = scenario.vehicles[vid].speed_class
            row[f"U_{vid}"] = utilities[vid]
            row[f"C_{vid}"] = burdens[vid]
            row[f"hard_brake_{vid}"] = traces[vid].hard_brake_count()
        rows.append(row)
    return rows


def write_csv(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", required=True, choices=["mappo", "dqn"])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--scenario-bank", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episode-max-steps", type=int, default=200)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--hidden-size", type=int, default=128, help="MAPPO only -- must match the training run")
    args = p.parse_args(argv)

    rows = run_eval(
        algorithm=args.algorithm, checkpoint=args.checkpoint, scenario_bank=args.scenario_bank,
        episode_max_steps=args.episode_max_steps, device=args.device, hidden_size=args.hidden_size,
    )
    write_csv(rows, args.output)

    n = len(rows)
    completion_rate = sum(r["completion"] for r in rows) / n
    collision_rate = sum(r["collision"] for r in rows) / n
    print(f"evaluated {n} scenarios -> {args.output}")
    print(f"completion_rate={completion_rate:.4f} collision_rate={collision_rate:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
