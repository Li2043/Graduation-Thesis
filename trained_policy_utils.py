"""Shared utilities for evaluating trained DQN policies."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from metrics import collect_step_metrics

COMMON_METRICS = [
    "total_reward",
    "mean_reward",
    "mean_min_experience",
    "final_min_experience",
    "mean_vehicle_experience",
    "mean_gini_experience",
    "final_gini_experience",
    "total_collision_count",
    "final_collision_count",
    "mean_n_vehicles",
    "steps",
]

RAWLSIAN_EXTRA_METRICS = [
    "mean_original_reward",
    "mean_rawlsian_reward",
    "rawlsian_reward_sum",
]


def evaluate_model_on_env(
    model,
    env,
    n_episodes: int,
    max_steps: int,
    speed_normalizer: float,
    base_seed: int,
    is_rawlsian: bool = False,
) -> pd.DataFrame:
    """
    Run deterministic policy evaluation and return per-episode metrics.

    Does not save CSV or train the model.
    """
    rows = []

    for episode_id in range(n_episodes):
        episode_seed = base_seed + episode_id
        env.action_space.seed(episode_seed)
        obs, info = env.reset(seed=episode_seed)

        total_reward = 0.0
        sum_original = 0.0
        sum_rawlsian = 0.0
        rawlsian_reward_sum = 0.0
        sum_min_exp = 0.0
        sum_mean_vehicle_exp = 0.0
        sum_gini = 0.0
        sum_n_vehicles = 0.0
        total_collision_count = 0
        final_min_exp = 0.0
        final_gini = 0.0
        final_collision_count = 0
        steps = 0
        terminated = False
        truncated = False

        for _ in range(max_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step_metrics = collect_step_metrics(env, speed_normalizer)

            total_reward += float(reward)
            sum_min_exp += step_metrics["min_experience"]
            sum_mean_vehicle_exp += step_metrics["mean_experience"]
            sum_gini += step_metrics["gini_experience"]
            sum_n_vehicles += step_metrics["n_vehicles"]
            total_collision_count += step_metrics["collision_count"]

            if is_rawlsian:
                sum_original += float(info.get("original_reward", 0.0))
                rawlsian_step = float(info.get("rawlsian_reward", 0.0))
                sum_rawlsian += rawlsian_step
                rawlsian_reward_sum += rawlsian_step

            final_min_exp = step_metrics["min_experience"]
            final_gini = step_metrics["gini_experience"]
            final_collision_count = step_metrics["collision_count"]
            steps += 1

            if terminated or truncated:
                break

        denom = steps if steps > 0 else 1
        row = {
            "episode": episode_id,
            "total_reward": total_reward,
            "mean_reward": total_reward / denom,
            "mean_min_experience": sum_min_exp / denom,
            "final_min_experience": final_min_exp,
            "mean_vehicle_experience": sum_mean_vehicle_exp / denom,
            "mean_gini_experience": sum_gini / denom,
            "final_gini_experience": final_gini,
            "total_collision_count": total_collision_count,
            "final_collision_count": final_collision_count,
            "mean_n_vehicles": sum_n_vehicles / denom,
            "steps": steps,
            "terminated": terminated,
            "truncated": truncated,
            "mean_original_reward": sum_original / denom if is_rawlsian else 0.0,
            "mean_rawlsian_reward": sum_rawlsian / denom if is_rawlsian else 0.0,
            "rawlsian_reward_sum": rawlsian_reward_sum if is_rawlsian else 0.0,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def make_trained_summary(baseline_df: pd.DataFrame, rawlsian_df: pd.DataFrame) -> pd.DataFrame:
    """Build summary table comparing baseline and Rawlsian evaluation runs."""
    rows = []

    for metric in COMMON_METRICS:
        if metric not in baseline_df.columns or metric not in rawlsian_df.columns:
            continue
        b_mean = baseline_df[metric].mean()
        b_std = baseline_df[metric].std()
        r_mean = rawlsian_df[metric].mean()
        r_std = rawlsian_df[metric].std()
        rows.append(
            {
                "metric": metric,
                "baseline_mean": b_mean,
                "baseline_std": b_std,
                "rawlsian_mean": r_mean,
                "rawlsian_std": r_std,
                "difference_rawlsian_minus_baseline": r_mean - b_mean,
            }
        )

    # Rawlsian-only diagnostics: baseline columns set to 0.0 (not applicable to baseline env).
    for metric in RAWLSIAN_EXTRA_METRICS:
        if metric not in rawlsian_df.columns:
            continue
        rows.append(
            {
                "metric": metric,
                "baseline_mean": 0.0,
                "baseline_std": 0.0,
                "rawlsian_mean": rawlsian_df[metric].mean(),
                "rawlsian_std": rawlsian_df[metric].std(),
                "difference_rawlsian_minus_baseline": rawlsian_df[metric].mean(),
            }
        )

    return pd.DataFrame(rows)


def save_bar_plot(
    metric: str,
    title: str,
    baseline_mean: float,
    rawlsian_mean: float,
    path: Path,
) -> None:
    """Save a single bar chart comparing baseline vs Rawlsian means."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Baseline", "Rawlsian"], [baseline_mean, rawlsian_mean])
    ax.set_ylabel(metric)
    ax.set_title(f"Trained DQN: {title}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
