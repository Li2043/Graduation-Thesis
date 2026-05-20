"""Random policy with RawlsianRewardWrapper; same fairness metrics as baseline."""

from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import pandas as pd

from config import (
    BASE_SEED,
    ENV_ID,
    MAX_STEPS,
    N_EPISODES,
    RAWLSIAN_XI,
    SPEED_NORMALIZER,
)
from metrics import collect_step_metrics
from rawlsian_wrapper import RawlsianRewardWrapper

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_CSV = RESULTS_DIR / "random_rawlsian.csv"


def run_episode(env: gym.Env, episode_id: int) -> dict:
    episode_seed = BASE_SEED + episode_id
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

    for _ in range(MAX_STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        step_metrics = collect_step_metrics(env, SPEED_NORMALIZER)

        total_reward += float(reward)
        sum_original += float(info.get("original_reward", 0.0))
        rawlsian_step = float(info.get("rawlsian_reward", 0.0))
        sum_rawlsian += rawlsian_step
        rawlsian_reward_sum += rawlsian_step

        sum_min_exp += step_metrics["min_experience"]
        sum_mean_vehicle_exp += step_metrics["mean_experience"]
        sum_gini += step_metrics["gini_experience"]
        sum_n_vehicles += step_metrics["n_vehicles"]
        total_collision_count += step_metrics["collision_count"]

        final_min_exp = step_metrics["min_experience"]
        final_gini = step_metrics["gini_experience"]
        final_collision_count = step_metrics["collision_count"]
        steps += 1

        if terminated or truncated:
            break

    denom = steps if steps > 0 else 1
    return {
        "episode": episode_id,
        "total_reward": total_reward,
        "mean_reward": total_reward / denom,
        "mean_original_reward": sum_original / denom,
        "mean_rawlsian_reward": sum_rawlsian / denom,
        "rawlsian_reward_sum": rawlsian_reward_sum,
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
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base_env = gym.make(ENV_ID)
    env = RawlsianRewardWrapper(
        base_env,
        xi=RAWLSIAN_XI,
        speed_normalizer=SPEED_NORMALIZER,
    )

    rows = [run_episode(env, ep) for ep in range(N_EPISODES)]
    env.close()

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV}")
    print(df)


if __name__ == "__main__":
    main()
