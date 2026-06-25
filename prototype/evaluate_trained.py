"""Evaluate trained baseline and Rawlsian DQN policies on scoped fairness metrics."""

import sys
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import DQN

from config import (
    BASE_SEED,
    BASELINE_MODEL_PATH,
    EGO_NEIGHBOURHOOD_RADIUS,
    EVAL_EPISODES,
    ENV_ID,
    EXPERIENCE_MODE,
    LOW_SPEED_THRESHOLD,
    MAX_STEPS,
    RAWLSIAN_MODEL_PATH,
    RAWLSIAN_SCOPE,
    RAWLSIAN_XI,
    RISK_DISTANCE_NORMALIZER,
    SPEED_NORMALIZER,
    TARGET_SPEED,
    TRAINED_BASELINE_CSV,
    TRAINED_RAWLSIAN_CSV,
    TRAINED_SUMMARY_CSV,
    W_COLLISION,
    W_LOW_SPEED,
    W_MOBILITY,
    W_RISK,
)
from rawlsian_wrapper import RawlsianRewardWrapper
from trained_policy_utils import (
    evaluate_model_on_env,
    make_trained_summary,
    save_bar_plot,
)

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = Path(TRAINED_SUMMARY_CSV).resolve().parent

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
    (
        "mean_scoped_vehicle_count",
        "trained_comparison_scoped_vehicle_count.png",
        "Average scoped vehicle count",
    ),
    (
        "mean_mobility_score",
        "trained_comparison_mean_mobility_score.png",
        "Average mean mobility score",
    ),
    (
        "mean_risk_penalty",
        "trained_comparison_mean_risk_penalty.png",
        "Average mean risk penalty",
    ),
    (
        "reason_risk_steps",
        "trained_comparison_reason_risk_steps.png",
        "Average reason_risk steps",
    ),
    (
        "reason_low_mobility_steps",
        "trained_comparison_reason_low_mobility_steps.png",
        "Average reason_low_mobility steps",
    ),
    (
        "reason_collision_steps",
        "trained_comparison_reason_collision_steps.png",
        "Average reason_collision steps",
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
    print(
        f"Evaluating trained policies with metric_scope={RAWLSIAN_SCOPE} "
        f"experience_mode={EXPERIENCE_MODE}"
    )

    baseline_env = gym.make(ENV_ID)
    rawlsian_base_env = gym.make(ENV_ID)
    rawlsian_env = RawlsianRewardWrapper(
        rawlsian_base_env,
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

    baseline_model = DQN.load(
        str(PROJECT_ROOT / BASELINE_MODEL_PATH),
        env=baseline_env,
    )
    rawlsian_model = DQN.load(
        str(PROJECT_ROOT / RAWLSIAN_MODEL_PATH),
        env=rawlsian_env,
    )

    eval_kwargs = {
        "n_episodes": EVAL_EPISODES,
        "max_steps": MAX_STEPS,
        "speed_normalizer": SPEED_NORMALIZER,
        "base_seed": BASE_SEED,
        "metric_scope": RAWLSIAN_SCOPE,
        "radius": EGO_NEIGHBOURHOOD_RADIUS,
        **EXPERIENCE_KWARGS,
    }

    baseline_df = evaluate_model_on_env(
        baseline_model,
        baseline_env,
        is_rawlsian=False,
        **eval_kwargs,
    )

    rawlsian_df = evaluate_model_on_env(
        rawlsian_model,
        rawlsian_env,
        is_rawlsian=True,
        **eval_kwargs,
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
