"""Train Rawlsian DQN on merge-v0 with RawlsianRewardWrapper."""

from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401

# Import DQN from stable_baselines3
from stable_baselines3 import DQN

# Import config
from config import (
    BASE_SEED, #random seed
    DQN_BATCH_SIZE,
    DQN_BUFFER_SIZE,
    DQN_GAMMA, #discount factor:control reward importance
    DQN_LEARNING_RATE, #learning rate:how much to update the model
    DQN_LEARNING_STARTS, #number of randomsteps before learning starts
    DQN_TARGET_UPDATE_INTERVAL, #how often update the target network
    DQN_TOTAL_TIMESTEPS, #total number of training steps
    DQN_TRAIN_FREQ, #training frequency:training steps between updates
    EGO_NEIGHBOURHOOD_RADIUS,
    ENV_ID,#environment id:merge-v0
    LOG_DIR, #log directory
    MODEL_DIR, #model directory
    RAWLSIAN_MODEL_PATH, #rawlsian model path
    RAWLSIAN_SCOPE,
    RAWLSIAN_XI, #rawlsian reward factor
    SPEED_NORMALIZER,
)
from rawlsian_wrapper import RawlsianRewardWrapper

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
    #model directory
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    #log directory
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    #create environment: make merge-v0 environment
    base_env = gym.make(ENV_ID)
    #apply rawlsian reward wrapper
    env = RawlsianRewardWrapper(
        base_env,
        xi=RAWLSIAN_XI,
        speed_normalizer=SPEED_NORMALIZER,
        scope=RAWLSIAN_SCOPE,
        radius=EGO_NEIGHBOURHOOD_RADIUS,
    )
    print(
        f"Training Rawlsian DQN: ENV_ID={ENV_ID} "
        f"RAWLSIAN_SCOPE={RAWLSIAN_SCOPE} "
        f"EGO_NEIGHBOURHOOD_RADIUS={EGO_NEIGHBOURHOOD_RADIUS} "
        f"RAWLSIAN_XI={RAWLSIAN_XI} "
        f"DQN_TOTAL_TIMESTEPS={DQN_TOTAL_TIMESTEPS}"
    )
    #set random seed
    env.reset(seed=BASE_SEED)
    env.action_space.seed(BASE_SEED)

    #create DQN model
    model = DQN(
        "MlpPolicy",
        env,
        # learning rate: how much to update the arg
        learning_rate=DQN_LEARNING_RATE,
        buffer_size=DQN_BUFFER_SIZE, # store past experiences
        learning_starts=DQN_LEARNING_STARTS,
        batch_size=DQN_BATCH_SIZE,
        gamma=DQN_GAMMA, # how much value future reward
        train_freq=DQN_TRAIN_FREQ,
        target_update_interval=DQN_TARGET_UPDATE_INTERVAL,
        verbose=1,
        tensorboard_log=_tensorboard_log(),
        seed=BASE_SEED,
    )

    model.learn(total_timesteps=DQN_TOTAL_TIMESTEPS)

    save_path = PROJECT_ROOT / RAWLSIAN_MODEL_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    env.close()

    print(f"Rawlsian DQN saved to {save_path}")


if __name__ == "__main__":
    main()
