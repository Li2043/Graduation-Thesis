"""Evaluate multi-seed baseline and Rawlsian DQN models; aggregate robustness metrics."""

import sys
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import DQN

from config import (
    EGO_NEIGHBOURHOOD_RADIUS,
    ENV_ID,
    EVAL_EPISODES,
    EXPERIENCE_MODE,
    LOW_SPEED_THRESHOLD,
    MAX_STEPS,
    MULTISEED_AGGREGATE_CSV,
    MULTISEED_MODEL_DIR,
    MULTISEED_RESULTS_DIR,
    MULTISEED_TRAINED_EPISODE_CSV,
    RAWLSIAN_SCOPE,
    RAWLSIAN_XI,
    RISK_DISTANCE_NORMALIZER,
    SEEDS,
    SPEED_NORMALIZER,
    TARGET_SPEED,
    W_COLLISION,
    W_LOW_SPEED,
    W_MOBILITY,
    W_RISK,
)
from rawlsian_wrapper import RawlsianRewardWrapper
from trained_policy_utils import evaluate_model_on_env

PROJECT_ROOT = Path(__file__).resolve().parent

EXPERIENCE_KWARGS = {
    "experience_mode": EXPERIENCE_MODE,
    "target_speed": TARGET_SPEED,
    "low_speed_threshold": LOW_SPEED_THRESHOLD,
    "w_mobility": W_MOBILITY,
    "w_collision": W_COLLISION,
    "w_low_speed": W_LOW_SPEED,
    "w_risk": W_RISK,
    "risk_distance_normalizer": RISK_DISTANCE_NORMALIZER,
}

AGGREGATE_METRICS = [
    "total_reward",
    "mean_reward",
    "mean_min_experience",
    "final_min_experience",
    "mean_vehicle_experience",
    "mean_gini_experience",
    "final_gini_experience",
    "total_collision_count",
    "final_collision_count",
    "least_advantaged_ego_ratio",
    "mean_risk_penalty",
    "mean_mobility_score",
    "mean_low_speed_penalty",
    "mean_collision_penalty",
    "reason_collision_steps",
    "reason_low_mobility_steps",
    "reason_low_speed_steps",
    "reason_risk_steps",
    "steps",
    "mean_scoped_vehicle_count",
]

HIGHER_IS_BETTER = {
    "mean_min_experience",
    "final_min_experience",
    "mean_vehicle_experience",
    "least_advantaged_ego_ratio",
    "mean_mobility_score",
    "total_reward",
    "mean_reward",
    "steps",
}

LOWER_IS_BETTER = {
    "mean_gini_experience",
    "final_gini_experience",
    "total_collision_count",
    "final_collision_count",
    "mean_risk_penalty",
    "mean_low_speed_penalty",
    "mean_collision_penalty",
    "reason_collision_steps",
    "reason_risk_steps",
}

PLOT_CONFIG = [
    ("mean_min_experience", "multiseed_mean_min_experience.png", "Mean min experience"),
    ("final_min_experience", "multiseed_final_min_experience.png", "Final min experience"),
    ("total_collision_count", "multiseed_collision_count.png", "Total collision count"),
    ("mean_gini_experience", "multiseed_gini_experience.png", "Mean Gini (experience)"),
    (
        "least_advantaged_ego_ratio",
        "multiseed_least_advantaged_ego_ratio.png",
        "Least advantaged vehicle is ego ratio",
    ),
    ("mean_risk_penalty", "multiseed_risk_penalty.png", "Mean risk penalty"),
    ("mean_mobility_score", "multiseed_mobility_score.png", "Mean mobility score"),
]


def _model_paths(seed: int) -> tuple[Path, Path]:
    model_dir = PROJECT_ROOT / MULTISEED_MODEL_DIR
    baseline_path = model_dir / f"baseline_seed_{seed}.zip"
    rawlsian_path = model_dir / f"rawlsian_seed_{seed}.zip"
    return baseline_path, rawlsian_path


def _check_models_exist() -> bool:
    missing = []
    for seed in SEEDS:
        baseline_path, rawlsian_path = _model_paths(seed)
        if not baseline_path.exists():
            missing.append(str(baseline_path))
        if not rawlsian_path.exists():
            missing.append(str(rawlsian_path))
    if missing:
        print("Error: missing model file(s):")
        for path in missing:
            print(f"  - {path}")
        print("\nPlease run: python train_multiseed.py")
        return False
    return True


def _make_rawlsian_env() -> gym.Env:
    base_env = gym.make(ENV_ID)
    return RawlsianRewardWrapper(
        base_env,
        xi=RAWLSIAN_XI,
        speed_normalizer=SPEED_NORMALIZER,
        scope=RAWLSIAN_SCOPE,
        radius=EGO_NEIGHBOURHOOD_RADIUS,
        mode=EXPERIENCE_MODE,
        target_speed=TARGET_SPEED,
        low_speed_threshold=LOW_SPEED_THRESHOLD,
        w_mobility=W_MOBILITY,
        w_collision=W_COLLISION,
        w_low_speed=W_LOW_SPEED,
        w_risk=W_RISK,
        risk_distance_normalizer=RISK_DISTANCE_NORMALIZER,
    )


def _evaluate_seed(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_path, rawlsian_path = _model_paths(seed)
    eval_kwargs = {
        "n_episodes": EVAL_EPISODES,
        "max_steps": MAX_STEPS,
        "speed_normalizer": SPEED_NORMALIZER,
        "base_seed": seed,
        "metric_scope": RAWLSIAN_SCOPE,
        "radius": EGO_NEIGHBOURHOOD_RADIUS,
        **EXPERIENCE_KWARGS,
    }

    baseline_env = gym.make(ENV_ID)
    baseline_model = DQN.load(str(baseline_path), env=baseline_env)
    baseline_df = evaluate_model_on_env(
        baseline_model,
        baseline_env,
        is_rawlsian=False,
        **eval_kwargs,
    )
    baseline_df["seed"] = seed
    baseline_df["model_type"] = "baseline"
    baseline_env.close()

    rawlsian_env = _make_rawlsian_env()
    rawlsian_model = DQN.load(str(rawlsian_path), env=rawlsian_env)
    rawlsian_df = evaluate_model_on_env(
        rawlsian_model,
        rawlsian_env,
        is_rawlsian=True,
        **eval_kwargs,
    )
    rawlsian_df["seed"] = seed
    rawlsian_df["model_type"] = "rawlsian"
    rawlsian_env.close()

    return baseline_df, rawlsian_df


def build_aggregate_summary(episode_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in AGGREGATE_METRICS if c in episode_df.columns]
    per_seed = episode_df.groupby(["seed", "model_type"])[numeric_cols].mean()

    rows = []
    for metric in AGGREGATE_METRICS:
        if metric not in numeric_cols:
            continue

        baseline_vals = [per_seed.loc[(seed, "baseline"), metric] for seed in SEEDS]
        rawlsian_vals = [per_seed.loc[(seed, "rawlsian"), metric] for seed in SEEDS]

        baseline_mean = float(pd.Series(baseline_vals).mean())
        baseline_std = float(pd.Series(baseline_vals).std(ddof=0))
        rawlsian_mean = float(pd.Series(rawlsian_vals).mean())
        rawlsian_std = float(pd.Series(rawlsian_vals).std(ddof=0))

        seed_means = per_seed[metric].unstack("model_type")
        better_count = float("nan")
        if metric in HIGHER_IS_BETTER:
            better_count = float((seed_means["rawlsian"] > seed_means["baseline"]).sum())
        elif metric in LOWER_IS_BETTER:
            better_count = float((seed_means["rawlsian"] < seed_means["baseline"]).sum())

        rows.append(
            {
                "metric": metric,
                "baseline_mean_across_seeds": baseline_mean,
                "baseline_std_across_seeds": baseline_std,
                "rawlsian_mean_across_seeds": rawlsian_mean,
                "rawlsian_std_across_seeds": rawlsian_std,
                "difference_rawlsian_minus_baseline": rawlsian_mean - baseline_mean,
                "rawlsian_better_seed_count": better_count,
                "n_seeds": len(SEEDS),
            }
        )

    return pd.DataFrame(rows)


def save_multiseed_plot(
    metric: str,
    title: str,
    aggregate_df: pd.DataFrame,
    path: Path,
) -> None:
    row = aggregate_df.loc[aggregate_df["metric"] == metric].iloc[0]
    baseline_mean = row["baseline_mean_across_seeds"]
    rawlsian_mean = row["rawlsian_mean_across_seeds"]
    baseline_std = row["baseline_std_across_seeds"]
    rawlsian_std = row["rawlsian_std_across_seeds"]

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Baseline", "Rawlsian"]
    means = [baseline_mean, rawlsian_mean]
    stds = [baseline_std, rawlsian_std]
    ax.bar(labels, means, yerr=stds, capsize=5)
    ax.set_ylabel(metric)
    ax.set_title(f"v0.6.1 Multi-seed: {title}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not _check_models_exist():
        sys.exit(1)

    results_dir = PROJECT_ROOT / MULTISEED_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    episode_frames = []
    for seed in SEEDS:
        print(f"Evaluating seed={seed} ...")
        baseline_df, rawlsian_df = _evaluate_seed(seed)
        episode_frames.extend([baseline_df, rawlsian_df])

    episode_df = pd.concat(episode_frames, ignore_index=True)
    episode_csv = PROJECT_ROOT / MULTISEED_TRAINED_EPISODE_CSV
    episode_df.to_csv(episode_csv, index=False)
    print(f"Saved {episode_csv}")

    aggregate_df = build_aggregate_summary(episode_df)
    aggregate_csv = PROJECT_ROOT / MULTISEED_AGGREGATE_CSV
    aggregate_df.to_csv(aggregate_csv, index=False)

    print("\n=== v0.6.1 Multi-seed aggregate summary ===")
    print(aggregate_df.to_string(index=False))
    print(f"\nSaved {aggregate_csv}")

    saved_plots = []
    for metric, filename, title in PLOT_CONFIG:
        if metric not in aggregate_df["metric"].values:
            continue
        out_path = results_dir / filename
        save_multiseed_plot(metric, title, aggregate_df, out_path)
        saved_plots.append(out_path)

    for path in saved_plots:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
