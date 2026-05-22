"""Train Rawlsian DQN on merge-v0 with RawlsianRewardWrapper."""



from pathlib import Path



import gymnasium as gym

import highway_env  # noqa: F401

from stable_baselines3 import DQN



from config import (

    BASE_SEED,

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

    LOG_DIR,

    LOW_SPEED_THRESHOLD,

    MODEL_DIR,

    RAWLSIAN_MODEL_PATH,

    RAWLSIAN_SCOPE,

    RAWLSIAN_XI,

    RISK_DISTANCE_NORMALIZER,

    SPEED_NORMALIZER,

    TARGET_SPEED,

    W_COLLISION,

    W_LOW_SPEED,

    W_MOBILITY,

    W_RISK,

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

    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)



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

    print(

        f"Training Rawlsian DQN: ENV_ID={ENV_ID} "

        f"RAWLSIAN_SCOPE={RAWLSIAN_SCOPE} "

        f"EGO_NEIGHBOURHOOD_RADIUS={EGO_NEIGHBOURHOOD_RADIUS} "

        f"EXPERIENCE_MODE={EXPERIENCE_MODE} "

        f"TARGET_SPEED={TARGET_SPEED} "

        f"LOW_SPEED_THRESHOLD={LOW_SPEED_THRESHOLD} "

        f"W_MOBILITY={W_MOBILITY} W_COLLISION={W_COLLISION} "

        f"W_LOW_SPEED={W_LOW_SPEED} W_RISK={W_RISK} "

        f"RISK_DISTANCE_NORMALIZER={RISK_DISTANCE_NORMALIZER} "

        f"RAWLSIAN_XI={RAWLSIAN_XI} "

        f"DQN_TOTAL_TIMESTEPS={DQN_TOTAL_TIMESTEPS}"

    )



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



    save_path = PROJECT_ROOT / RAWLSIAN_MODEL_PATH

    save_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(str(save_path))

    env.close()



    print(f"Rawlsian DQN saved to {save_path}")





if __name__ == "__main__":

    main()

