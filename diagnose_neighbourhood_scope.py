"""Compare global vs ego-neighbourhood least advantaged vehicle identity."""

from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import pandas as pd

from config import (
    BASE_SEED,
    EGO_NEIGHBOURHOOD_RADIUS,
    ENV_ID,
    NEIGHBOURHOOD_SCOPE_DIAGNOSTIC_CSV,
    SPEED_NORMALIZER,
)
from metrics import collect_step_metrics

OUTPUT_CSV = Path(NEIGHBOURHOOD_SCOPE_DIAGNOSTIC_CSV)

N_EPISODES = 5
MAX_STEPS = 50


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    env = gym.make(ENV_ID)
    rows = []

    for episode_id in range(N_EPISODES):
        episode_seed = BASE_SEED + episode_id
        env.action_space.seed(episode_seed)
        obs, info = env.reset(seed=episode_seed)

        global_ego_steps = 0
        neighbourhood_ego_steps = 0
        sum_nb_scoped_count = 0
        steps = 0

        for step in range(MAX_STEPS):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            global_metrics = collect_step_metrics(env, SPEED_NORMALIZER, scope="global")
            neighbourhood_metrics = collect_step_metrics(
                env,
                SPEED_NORMALIZER,
                scope="ego_neighbourhood",
                radius=EGO_NEIGHBOURHOOD_RADIUS,
            )

            if global_metrics["least_advantaged_is_ego"]:
                global_ego_steps += 1
            if neighbourhood_metrics["least_advantaged_is_ego"]:
                neighbourhood_ego_steps += 1
            sum_nb_scoped_count += neighbourhood_metrics["scoped_vehicle_count"]
            steps += 1

            rows.append(
                {
                    "episode": episode_id,
                    "step": step,
                    "global_min_experience": global_metrics["min_experience"],
                    "global_least_advantaged_is_ego": global_metrics["least_advantaged_is_ego"],
                    "global_least_advantaged_index": global_metrics["least_advantaged_index"],
                    "global_scoped_vehicle_count": global_metrics["scoped_vehicle_count"],
                    "neighbourhood_min_experience": neighbourhood_metrics["min_experience"],
                    "neighbourhood_least_advantaged_is_ego": neighbourhood_metrics[
                        "least_advantaged_is_ego"
                    ],
                    "neighbourhood_least_advantaged_index": neighbourhood_metrics[
                        "least_advantaged_index"
                    ],
                    "neighbourhood_scoped_vehicle_count": neighbourhood_metrics[
                        "scoped_vehicle_count"
                    ],
                    "neighbourhood_radius": EGO_NEIGHBOURHOOD_RADIUS,
                    "n_vehicles": global_metrics["n_vehicles"],
                }
            )

            if terminated or truncated:
                break

        denom = steps if steps > 0 else 1
        print(
            f"episode={episode_id} summary: "
            f"global_ego_ratio={global_ego_steps / denom:.4f} "
            f"neighbourhood_ego_ratio={neighbourhood_ego_steps / denom:.4f} "
            f"avg_neighbourhood_scoped_count={sum_nb_scoped_count / denom:.4f}"
        )

    env.close()

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
