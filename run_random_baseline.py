"""Random policy on merge-v0 without Rawlsian wrapper; log episode metrics."""

from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import pandas as pd

from config import ENV_ID, MAX_STEPS, N_EPISODES

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_CSV = RESULTS_DIR / "random_baseline.csv"


def run_episode(env: gym.Env, episode_id: int) -> dict:
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    for _ in range(MAX_STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break

    mean_reward = total_reward / steps if steps > 0 else 0.0
    return {
        "episode": episode_id,
        "total_reward": total_reward,
        "mean_reward": mean_reward,
        "steps": steps,
        "terminated": terminated,
        "truncated": truncated,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    env = gym.make(ENV_ID)

    rows = [run_episode(env, ep) for ep in range(N_EPISODES)]
    env.close()

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV}")
    print(df)


if __name__ == "__main__":
    main()
