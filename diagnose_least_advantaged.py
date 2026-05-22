"""Quick diagnostic: who is the least advantaged vehicle at each step?"""

from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import pandas as pd

from config import BASE_SEED, ENV_ID, SPEED_NORMALIZER

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_CSV = RESULTS_DIR / "least_advantaged_diagnostics.csv"

N_EPISODES = 5
MAX_STEPS = 50


def main() -> None:
    from metrics import collect_step_metrics

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    env = gym.make(ENV_ID)
    rows = []

    for episode_id in range(N_EPISODES):
        episode_seed = BASE_SEED + episode_id
        env.action_space.seed(episode_seed)
        obs, info = env.reset(seed=episode_seed)

        ego_steps = 0
        non_ego_steps = 0
        steps = 0

        for step in range(MAX_STEPS):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            step_metrics = collect_step_metrics(env, SPEED_NORMALIZER)

            is_ego = bool(step_metrics["least_advantaged_is_ego"])
            if is_ego:
                ego_steps += 1
            else:
                non_ego_steps += 1
            steps += 1

            print(
                f"episode={episode_id} step={step} "
                f"min_experience={step_metrics['min_experience']:.4f} "
                f"least_advantaged_index={step_metrics['least_advantaged_index']} "
                f"least_advantaged_is_ego={is_ego} "
                f"least_advantaged_speed={step_metrics['least_advantaged_speed']:.4f} "
                f"least_advantaged_crashed={step_metrics['least_advantaged_crashed']} "
                f"n_vehicles={step_metrics['n_vehicles']}"
            )

            rows.append(
                {
                    "episode": episode_id,
                    "step": step,
                    "min_experience": step_metrics["min_experience"],
                    "least_advantaged_index": step_metrics["least_advantaged_index"],
                    "least_advantaged_is_ego": is_ego,
                    "least_advantaged_speed": step_metrics["least_advantaged_speed"],
                    "least_advantaged_crashed": step_metrics["least_advantaged_crashed"],
                    "n_vehicles": step_metrics["n_vehicles"],
                }
            )

            if terminated or truncated:
                break

        denom = steps if steps > 0 else 1
        ego_ratio = ego_steps / denom
        non_ego_ratio = non_ego_steps / denom
        print(
            f"episode={episode_id} summary: "
            f"ego_ratio={ego_ratio:.4f} non_ego_ratio={non_ego_ratio:.4f}"
        )

    env.close()

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved step-level diagnostics to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
