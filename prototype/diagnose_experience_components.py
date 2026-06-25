"""Quick diagnostic: why is the least advantaged vehicle least advantaged? (v0.5)"""

from collections import Counter
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import pandas as pd

from config import (
    BASE_SEED,
    EGO_NEIGHBOURHOOD_RADIUS,
    ENV_ID,
    EXPERIENCE_COMPONENTS_DIAGNOSTIC_CSV,
    EXPERIENCE_MODE,
    LOW_SPEED_THRESHOLD,
    RISK_DISTANCE_NORMALIZER,
    SPEED_NORMALIZER,
    TARGET_SPEED,
    W_COLLISION,
    W_LOW_SPEED,
    W_MOBILITY,
    W_RISK,
)
from metrics import collect_step_metrics

OUTPUT_CSV = Path(EXPERIENCE_COMPONENTS_DIAGNOSTIC_CSV)

N_EPISODES = 5
MAX_STEPS = 50

METRIC_KWARGS = {
    "mode": EXPERIENCE_MODE,
    "target_speed": TARGET_SPEED,
    "low_speed_threshold": LOW_SPEED_THRESHOLD,
    "w_mobility": W_MOBILITY,
    "w_collision": W_COLLISION,
    "w_low_speed": W_LOW_SPEED,
    "w_risk": W_RISK,
    "risk_distance_normalizer": RISK_DISTANCE_NORMALIZER,
}


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    env = gym.make(ENV_ID)
    rows = []

    for episode_id in range(N_EPISODES):
        episode_seed = BASE_SEED + episode_id
        env.action_space.seed(episode_seed)
        obs, info = env.reset(seed=episode_seed)

        reason_counter: Counter[str] = Counter()
        steps = 0

        for step in range(MAX_STEPS):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            step_metrics = collect_step_metrics(
                env,
                SPEED_NORMALIZER,
                scope="ego_neighbourhood",
                radius=EGO_NEIGHBOURHOOD_RADIUS,
                **METRIC_KWARGS,
            )
            reason = step_metrics["least_advantaged_reason"]
            reason_counter[reason] += 1
            steps += 1

            print(
                f"episode={episode_id} step={step} "
                f"min_experience={step_metrics['min_experience']:.4f} "
                f"least_advantaged_is_ego={step_metrics['least_advantaged_is_ego']} "
                f"least_advantaged_reason={reason} "
                f"mobility={step_metrics['least_advantaged_mobility_score']:.4f} "
                f"risk={step_metrics['least_advantaged_risk_penalty']:.4f} "
                f"scoped_vehicle_count={step_metrics['scoped_vehicle_count']} "
                f"n_vehicles={step_metrics['n_vehicles']}"
            )

            rows.append(
                {
                    "episode": episode_id,
                    "step": step,
                    "min_experience": step_metrics["min_experience"],
                    "least_advantaged_is_ego": step_metrics["least_advantaged_is_ego"],
                    "least_advantaged_index": step_metrics["least_advantaged_index"],
                    "least_advantaged_reason": reason,
                    "least_advantaged_speed": step_metrics["least_advantaged_speed"],
                    "least_advantaged_mobility_score": step_metrics[
                        "least_advantaged_mobility_score"
                    ],
                    "least_advantaged_collision_penalty": step_metrics[
                        "least_advantaged_collision_penalty"
                    ],
                    "least_advantaged_low_speed_penalty": step_metrics[
                        "least_advantaged_low_speed_penalty"
                    ],
                    "least_advantaged_risk_penalty": step_metrics[
                        "least_advantaged_risk_penalty"
                    ],
                    "least_advantaged_nearest_distance": step_metrics[
                        "least_advantaged_nearest_distance"
                    ],
                    "scoped_vehicle_count": step_metrics["scoped_vehicle_count"],
                    "n_vehicles": step_metrics["n_vehicles"],
                }
            )

            if terminated or truncated:
                break

        print(f"episode={episode_id} reason distribution: {dict(reason_counter)}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    env.close()
    print(f"\nSaved {OUTPUT_CSV} ({len(df)} rows)")


if __name__ == "__main__":
    main()
