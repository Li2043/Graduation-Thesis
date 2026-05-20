"""Train baseline DQN on merge-v0 (original highway-env reward)."""

from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import DQN

from config import (
    BASE_SEED,
    BASELINE_MODEL_PATH,
    DQN_BATCH_SIZE,
    DQN_BUFFER_SIZE,
    DQN_GAMMA,
    DQN_LEARNING_RATE,
    DQN_LEARNING_STARTS,
    DQN_TARGET_UPDATE_INTERVAL,
    DQN_TOTAL_TIMESTEPS,
    DQN_TRAIN_FREQ,
    ENV_ID,
    LOG_DIR,
    MODEL_DIR,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def _tensorboard_log() -> str | None:
    """Use TensorBoard logs if installed; otherwise train without logging."""
    try:
        import tensorboard  # noqa: F401
        return str(PROJECT_ROOT / LOG_DIR)
    except ImportError:
        print("Warning: tensorboard not installed. Run: pip install tensorboard")
        print("Continuing training without tensorboard_log.")
        return None


def main() -> None:
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    env = gym.make(ENV_ID)
    env.reset(seed=BASE_SEED)
    env.action_space.seed(BASE_SEED)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=DQN_LEARNING_RATE,
        buffer_size=DQN_BUFFER_SIZE,
        learning_starts=DQN_LEARNING_STARTS,
        batch_size=DQN_BATCH_SIZE,
        gamma=DQN_GAMMA,
        train_freq=DQN_TRAIN_FREQ,
        target_update_interval=DQN_TARGET_UPDATE_INTERVAL,
        verbose=1,
        tensorboard_log=_tensorboard_log(),
        seed=BASE_SEED,
    )

    model.learn(total_timesteps=DQN_TOTAL_TIMESTEPS)

    save_path = PROJECT_ROOT / BASELINE_MODEL_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    env.close()

    print(f"Baseline DQN saved to {save_path}")


if __name__ == "__main__":
    main()
