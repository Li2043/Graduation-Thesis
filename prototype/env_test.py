"""Minimal smoke test: merge-v0 with random actions for 20 steps."""

import gymnasium as gym
import highway_env  # noqa: F401 — registers highway-env environments


def main() -> None:
    env = gym.make("merge-v0")
    obs, info = env.reset()

    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"step={step} reward={reward:.4f} "
            f"terminated={terminated} truncated={truncated}"
        )
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print("env_test.py finished successfully.")


if __name__ == "__main__":
    main()
