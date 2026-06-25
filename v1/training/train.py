"""V1 training and evaluation entrypoint.

Runs the two V1 conditions through a single, identical pipeline. The *only*
difference between conditions is which ``RewardFunction`` is injected:

    egoistic  -> EgoisticReward   (environment-provided individual reward)
    rawlsian  -> RawlsianReward   (R_rawls = min_i E_i from the experience fn)

The policy layer is decoupled from the environment: the training loop computes
the scalar reward via the reward function and stores it through the standardised
``policy.remember(state, action, reward, next_state, done)`` interface.

This module uses only the standard library, NumPy, and Torch (already required
by the project). It does not import any legacy/prototype code, does not modify
the experience-function mathematics, and does not introduce new RL algorithms.

Usage:
    python -m v1.training.train --mode egoistic --episodes 50 --seed 0
    python -m v1.training.train --mode rawlsian --episodes 50 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v1.env.highway_merge_env import HighwayMergeEnv  # noqa: E402
from v1.experience.experience import (  # noqa: E402
    ExperienceFunction,
    compute_all_experiences,
)
from v1.policies.egoistic_dqn import EgoisticDQN  # noqa: E402
from v1.policies.rawlsian_dqn import RawlsianDQN  # noqa: E402
from v1.rewards.base_reward import RewardFunction  # noqa: E402
from v1.rewards.egoistic_reward import EgoisticReward  # noqa: E402
from v1.rewards.rawlsian_reward import RawlsianReward  # noqa: E402

EXPERIMENTS_DIR = ROOT / "experiments"
LOGS_DIR = EXPERIMENTS_DIR / "logs"
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"
RESULTS_CSV = EXPERIMENTS_DIR / "results.csv"

# Held-out evaluation seeds. Fixed and disjoint from typical training seeds so
# there is no train/eval leakage (see docs/V1_SYSTEM_SPEC.md, Section 5).
EVAL_SEEDS = [9001, 9002, 9003, 9004, 9005, 9006, 9007, 9008, 9009, 9010]

NEAR_COLLISION_TTC = 2.0


@dataclass
class RunConfig:
    mode: str
    seed: int = 0
    episodes: int = 50
    max_steps: int = 60
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 40
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 64
    buffer_capacity: int = 50000
    target_update_interval: int = 500
    hidden_dim: int = 64
    ttc_horizon: float = 10.0
    w_mobility: float = 1.0
    w_safety: float = 1.0
    w_waiting: float = 1.0
    eval_seeds: list = field(default_factory=lambda: list(EVAL_SEEDS))


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _gini(values: list[float]) -> float:
    """Gini coefficient of a list of values (shifted to be non-negative)."""
    vals = list(values)
    if not vals:
        return 0.0
    minimum = min(vals)
    if minimum < 0:
        vals = [v - minimum for v in vals]
    total = sum(vals)
    if total <= 1e-12:
        return 0.0
    vals.sort()
    n = len(vals)
    cumulative = sum((i + 1) * v for i, v in enumerate(vals))
    return float((2.0 * cumulative) / (n * total) - (n + 1) / n)


def build_experience_function(config: RunConfig) -> ExperienceFunction:
    from v1.experience.experience import ExperienceWeights

    weights = ExperienceWeights(
        w_mobility=config.w_mobility,
        w_safety=config.w_safety,
        w_waiting=config.w_waiting,
    )
    return ExperienceFunction(weights=weights, ttc_horizon=config.ttc_horizon)


def build_policy(config: RunConfig, action_dim: int, obs_dim: int):
    common = dict(
        state_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=config.hidden_dim,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        buffer_capacity=config.buffer_capacity,
        batch_size=config.batch_size,
        target_update_interval=config.target_update_interval,
        seed=config.seed,
    )
    if config.mode == "egoistic":
        return EgoisticDQN(**common)
    if config.mode == "rawlsian":
        return RawlsianDQN(**common)
    raise ValueError(f"Unknown mode '{config.mode}'. Use 'egoistic' or 'rawlsian'.")


def build_reward_function(
    config: RunConfig,
    ego_agent: Any,
    experience_function: ExperienceFunction,
) -> RewardFunction:
    if config.mode == "egoistic":
        return EgoisticReward(agent_id=ego_agent)
    if config.mode == "rawlsian":
        return RawlsianReward(experience_function=experience_function)
    raise ValueError(f"Unknown mode '{config.mode}'. Use 'egoistic' or 'rawlsian'.")


def _epsilon_for_episode(config: RunConfig, episode: int) -> float:
    if config.epsilon_decay_episodes <= 0:
        return config.epsilon_end
    frac = min(1.0, episode / config.epsilon_decay_episodes)
    return config.epsilon_start + frac * (config.epsilon_end - config.epsilon_start)


def run_episode(
    env: HighwayMergeEnv,
    policy,
    reward_fn: RewardFunction,
    metrics_exp_fn: ExperienceFunction,
    seed: int,
    epsilon: float,
    max_steps: int,
    learn: bool,
) -> dict:
    """Run one episode and return its aggregated metrics."""
    obs, env_state = env.reset(seed=seed)

    episode_reward = 0.0
    sum_min_exp = 0.0
    sum_mean_exp = 0.0
    near_collision_steps = 0
    collision = False
    steps = 0

    for _ in range(max_steps):
        action = policy.act(obs, epsilon)
        next_obs, terminated, truncated, info, next_env_state = env.step(action)
        done = bool(terminated or truncated)

        reward = reward_fn.compute(obs, next_obs, next_env_state, env_state)
        episode_reward += float(reward)

        if learn:
            policy.remember(obs, action, reward, next_obs, done)
            policy.train_step()

        experiences = compute_all_experiences(next_env_state, env_state, metrics_exp_fn)
        exp_values = list(experiences.values())
        sum_min_exp += min(exp_values)
        sum_mean_exp += sum(exp_values) / len(exp_values)

        for agent_state in next_env_state.values():
            ttc = agent_state.get("ttc")
            if ttc is not None and not np.isinf(ttc) and ttc < NEAR_COLLISION_TTC:
                near_collision_steps += 1
                break

        if info.get("collision"):
            collision = True

        steps += 1
        obs, env_state = next_obs, next_env_state
        if done:
            break

    denom = steps if steps > 0 else 1
    final_experiences = compute_all_experiences(env_state, None, metrics_exp_fn)
    final_values = list(final_experiences.values())

    return {
        "episode_reward": episode_reward,
        "steps": steps,
        "collision": int(collision),
        "mean_experience": sum_mean_exp / denom,
        "min_experience": sum_min_exp / denom,
        "final_min_experience": min(final_values),
        "final_gini_experience": _gini(final_values),
        "near_collision_steps": near_collision_steps,
        "time_to_merge": steps if not collision else max_steps,
    }


def evaluate(
    env: HighwayMergeEnv,
    policy,
    reward_fn: RewardFunction,
    metrics_exp_fn: ExperienceFunction,
    config: RunConfig,
) -> dict:
    """Run the held-out evaluation seeds with epsilon=0 and no learning."""
    per_seed = []
    for eval_seed in config.eval_seeds:
        metrics = run_episode(
            env,
            policy,
            reward_fn,
            metrics_exp_fn,
            seed=eval_seed,
            epsilon=0.0,
            max_steps=config.max_steps,
            learn=False,
        )
        per_seed.append(metrics)

    def _mean(key: str) -> float:
        return float(np.mean([m[key] for m in per_seed]))

    return {
        # Fairness
        "eval_min_experience": _mean("min_experience"),
        "eval_final_min_experience": _mean("final_min_experience"),
        "eval_gini_experience": _mean("final_gini_experience"),
        # Safety
        "eval_collision_rate": _mean("collision"),
        "eval_near_collision_steps": _mean("near_collision_steps"),
        # Efficiency
        "eval_mean_experience": _mean("mean_experience"),
        "eval_time_to_merge": _mean("time_to_merge"),
        "eval_episode_reward": _mean("episode_reward"),
        "n_eval_seeds": len(per_seed),
    }


def _ensure_dirs() -> None:
    for directory in (EXPERIMENTS_DIR, LOGS_DIR, CONFIGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _write_config(run_id: str, config: RunConfig) -> None:
    path = CONFIGS_DIR / f"{run_id}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)


def _write_episode_log(run_id: str, rows: list[dict]) -> None:
    if not rows:
        return
    path = LOGS_DIR / f"{run_id}.csv"
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_results(run_id: str, config: RunConfig, eval_metrics: dict) -> None:
    row = {"run_id": run_id, "mode": config.mode, "seed": config.seed, **eval_metrics}
    write_header = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_experiment(config: RunConfig) -> dict:
    """Train then evaluate a single condition; persist config, logs, results."""
    _ensure_dirs()
    seed_everything(config.seed)

    env = HighwayMergeEnv(max_steps=config.max_steps)
    metrics_exp_fn = build_experience_function(config)
    policy = build_policy(config, action_dim=env.action_dim, obs_dim=env.obs_dim)
    reward_fn = build_reward_function(config, env.ego_agent, metrics_exp_fn)

    run_id = f"{config.mode}_seed{config.seed}_{int(time.time())}"
    _write_config(run_id, config)

    episode_rows = []
    for episode in range(config.episodes):
        epsilon = _epsilon_for_episode(config, episode)
        train_seed = config.seed * 100000 + episode  # disjoint from EVAL_SEEDS
        metrics = run_episode(
            env,
            policy,
            reward_fn,
            metrics_exp_fn,
            seed=train_seed,
            epsilon=epsilon,
            max_steps=config.max_steps,
            learn=True,
        )
        row = {"episode": episode, "epsilon": round(epsilon, 4), **metrics}
        episode_rows.append(row)
        print(
            f"[{config.mode}] ep {episode:03d} "
            f"reward={metrics['episode_reward']:.3f} "
            f"min_exp={metrics['min_experience']:.3f} "
            f"collision={metrics['collision']} steps={metrics['steps']}"
        )

    _write_episode_log(run_id, episode_rows)

    eval_metrics = evaluate(env, policy, reward_fn, metrics_exp_fn, config)
    _append_results(run_id, config, eval_metrics)

    print(f"\n=== {config.mode} evaluation (held-out seeds) ===")
    for key, value in eval_metrics.items():
        print(f"  {key}: {value}")
    print(f"\nrun_id={run_id}")
    print(f"config -> {CONFIGS_DIR / (run_id + '.json')}")
    print(f"episode log -> {LOGS_DIR / (run_id + '.csv')}")
    print(f"results -> {RESULTS_CSV}")

    return eval_metrics


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V1 training/evaluation entrypoint.")
    parser.add_argument("--mode", choices=["egoistic", "rawlsian"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=60)
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    config = RunConfig(
        mode=args.mode,
        seed=args.seed,
        episodes=args.episodes,
        max_steps=args.max_steps,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
