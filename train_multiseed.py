"""Train baseline and Rawlsian DQN for each seed (v0.6.1 multi-seed robustness)."""

from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import DQN

from config import (
    DQN_BATCH_SIZE,
    DQN_BUFFER_SIZE,
    DQN_GAMMA,
    DQN_LEARNING_RATE,
    DQN_LEARNING_STARTS,
    DQN_TARGET_UPDATE_INTERVAL,
    DQN_TOTAL_TIMESTEPS,
    DQN_TRAIN_FREQ,
    EGO_NEIGHBOURHOOD_RADIUS,
    ENV_ID,
    EXPERIENCE_MODE,
    LOW_SPEED_THRESHOLD,
    MULTISEED_LOG_DIR,
    MULTISEED_MODEL_DIR,
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

PROJECT_ROOT = Path(__file__).resolve().parent


def _tensorboard_log(log_subdir: str) -> str | None:
    """Use TensorBoard logs if installed; otherwise train without logging."""
    try:
        import tensorboard  # noqa: F401
        return str(PROJECT_ROOT / MULTISEED_LOG_DIR / log_subdir)
    except ImportError:
        print("Warning: tensorboard not installed. Run: pip install tensorboard")
        print("Continuing training without tensorboard_log.")
        return None


def _build_dqn(env: gym.Env, seed: int, log_subdir: str) -> DQN:
    return DQN(
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
        tensorboard_log=_tensorboard_log(log_subdir),
        seed=seed,
    )


def train_baseline(seed: int, model_dir: Path) -> Path:
    print(f"\n=== seed={seed} model_type=baseline ===")
    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    env.action_space.seed(seed)

    model = _build_dqn(env, seed, f"baseline_seed_{seed}")
    model.learn(total_timesteps=DQN_TOTAL_TIMESTEPS)

    save_path = model_dir / f"baseline_seed_{seed}.zip"
    model.save(str(save_path))
    env.close()
    print(f"Saved baseline DQN to {save_path}")
    return save_path


def train_rawlsian(seed: int, model_dir: Path) -> Path:
    print(f"\n=== seed={seed} model_type=rawlsian ===")
    base_env = gym.make(ENV_ID)
    env = RawlsianRewardWrapper(
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
    env.reset(seed=seed)
    env.action_space.seed(seed)

    model = _build_dqn(env, seed, f"rawlsian_seed_{seed}")
    model.learn(total_timesteps=DQN_TOTAL_TIMESTEPS)

    save_path = model_dir / f"rawlsian_seed_{seed}.zip"
    model.save(str(save_path))
    env.close()
    print(f"Saved Rawlsian DQN to {save_path}")
    return save_path


def main() -> None:
    model_dir = PROJECT_ROOT / MULTISEED_MODEL_DIR
    log_dir = PROJECT_ROOT / MULTISEED_LOG_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"v0.6.1 multi-seed training: SEEDS={SEEDS}")
    print(f"Models -> {model_dir}")
    print(f"Logs   -> {log_dir}")

    for seed in SEEDS:
        train_baseline(seed, model_dir)
        train_rawlsian(seed, model_dir)

    print("\nMulti-seed training complete.")


if __name__ == "__main__":
    main()
