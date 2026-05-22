"""Evaluate trained baseline and Rawlsian DQN policies on shared fairness metrics."""

import sys
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import pandas as pd
from stable_baselines3 import DQN

from config import (
    BASE_SEED,
    BASELINE_MODEL_PATH,
    EVAL_EPISODES,
    ENV_ID,
    MAX_STEPS,
    RAWLSIAN_MODEL_PATH,
    RAWLSIAN_XI,
    SPEED_NORMALIZER,
    TRAINED_BASELINE_CSV,
    TRAINED_RAWLSIAN_CSV,
    TRAINED_SUMMARY_CSV,
)
from rawlsian_wrapper import RawlsianRewardWrapper
from trained_policy_utils import (
    evaluate_model_on_env,
    make_trained_summary,
    save_bar_plot,
)

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"

PLOT_CONFIG = [
    ("total_reward", "trained_comparison_total_reward.png", "Average total reward"),
    ("mean_min_experience", "trained_comparison_min_experience.png", "Average mean min experience"),
    ("mean_gini_experience", "trained_comparison_gini.png", "Average mean Gini (experience)"),
    ("total_collision_count", "trained_comparison_collision.png", "Average total collision count"),
    ("steps", "trained_comparison_steps.png", "Average steps per episode"),
    (
        "least_advantaged_ego_ratio",
        "trained_comparison_least_advantaged_ego_ratio.png",
        "least advantaged vehicle is ego ratio",
    ),
]


def check_models_exist() -> bool:
    baseline_path = PROJECT_ROOT / BASELINE_MODEL_PATH
    rawlsian_path = PROJECT_ROOT / RAWLSIAN_MODEL_PATH
    missing = []
    if not baseline_path.exists():
        missing.append(str(baseline_path))
    if not rawlsian_path.exists():
        missing.append(str(rawlsian_path))
    if missing:
        print("Error: trained model file(s) not found:")
        for p in missing:
            print(f"  - {p}")
        print("\nPlease run training first:")
        print("  python train_dqn_baseline.py")
        print("  python train_dqn_rawlsian.py")
        return False
    return True


def main() -> None:
    if not check_models_exist():
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline_env = gym.make(ENV_ID)
    rawlsian_base_env = gym.make(ENV_ID)
    rawlsian_env = RawlsianRewardWrapper(
        rawlsian_base_env,
        xi=RAWLSIAN_XI,
        speed_normalizer=SPEED_NORMALIZER,
    )

    baseline_model = DQN.load(
        str(PROJECT_ROOT / BASELINE_MODEL_PATH),
        env=baseline_env,
    )
    rawlsian_model = DQN.load(
        str(PROJECT_ROOT / RAWLSIAN_MODEL_PATH),
        env=rawlsian_env,
    )

    baseline_df = evaluate_model_on_env(
        baseline_model,
        baseline_env,
        n_episodes=EVAL_EPISODES,
        max_steps=MAX_STEPS,
        speed_normalizer=SPEED_NORMALIZER,
        base_seed=BASE_SEED,
        is_rawlsian=False,
    )

    rawlsian_df = evaluate_model_on_env(
        rawlsian_model,
        rawlsian_env,
        n_episodes=EVAL_EPISODES,
        max_steps=MAX_STEPS,
        speed_normalizer=SPEED_NORMALIZER,
        base_seed=BASE_SEED,
        is_rawlsian=True,
    )

    baseline_csv = PROJECT_ROOT / TRAINED_BASELINE_CSV
    rawlsian_csv = PROJECT_ROOT / TRAINED_RAWLSIAN_CSV
    summary_csv = PROJECT_ROOT / TRAINED_SUMMARY_CSV

    baseline_df.to_csv(baseline_csv, index=False)
    rawlsian_df.to_csv(rawlsian_csv, index=False)

    summary_df = make_trained_summary(baseline_df, rawlsian_df)
    summary_df.to_csv(summary_csv, index=False)

    print("\n=== Trained policy summary ===")
    print(summary_df.to_string(index=False))

    saved_plots = []
    for metric, filename, title in PLOT_CONFIG:
        if metric not in baseline_df.columns or metric not in rawlsian_df.columns:
            continue
        out_path = RESULTS_DIR / filename
        save_bar_plot(
            metric,
            title,
            baseline_df[metric].mean(),
            rawlsian_df[metric].mean(),
            out_path,
        )
        saved_plots.append(out_path)

    baseline_env.close()
    rawlsian_env.close()

    print(f"\nSaved {baseline_csv}")
    print(f"Saved {rawlsian_csv}")
    print(f"Saved {summary_csv}")
    for p in saved_plots:
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
