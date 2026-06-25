"""Random policy with RawlsianRewardWrapper using configurable scope (v0.4 default: ego_neighbourhood)."""

from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import pandas as pd

from config import (
    BASE_SEED,
    EGO_NEIGHBOURHOOD_RADIUS,
    ENV_ID,
    EXPERIENCE_MODE,
    LOW_SPEED_THRESHOLD,
    MAX_STEPS,
    N_EPISODES,
    RANDOM_RAWLSIAN_CSV,
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
from metrics import (
    accumulate_least_advantaged_step,
    accumulate_reason_step,
    collect_step_metrics,
    finalize_least_advantaged_episode,
    finalize_reason_episode,
    new_least_advantaged_counters,
    new_reason_counters,
)
from rawlsian_wrapper import RawlsianRewardWrapper

RESULTS_DIR = Path(RANDOM_RAWLSIAN_CSV).resolve().parent
OUTPUT_CSV = Path(RANDOM_RAWLSIAN_CSV)

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
    sum_scoped_vehicle_count = 0.0
    sum_mobility = 0.0
    sum_risk = 0.0
    sum_low_speed = 0.0
    sum_collision_penalty = 0.0
    total_collision_count = 0
    final_min_exp = 0.0
    final_gini = 0.0
    final_collision_count = 0
    la_counters = new_least_advantaged_counters()
    reason_counters = new_reason_counters()
    steps = 0
    terminated = False
    truncated = False

    for _ in range(MAX_STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        step_metrics = collect_step_metrics(
            env,
            SPEED_NORMALIZER,
            scope=RAWLSIAN_SCOPE,
            radius=EGO_NEIGHBOURHOOD_RADIUS,
            **METRIC_KWARGS,
        )

        total_reward += float(reward)
        sum_original += float(info.get("original_reward", 0.0))
        rawlsian_step = float(info.get("rawlsian_reward", 0.0))
        sum_rawlsian += rawlsian_step
        rawlsian_reward_sum += rawlsian_step

        sum_min_exp += step_metrics["min_experience"]
        sum_mean_vehicle_exp += step_metrics["mean_experience"]
        sum_gini += step_metrics["gini_experience"]
        sum_n_vehicles += step_metrics["n_vehicles"]
        sum_scoped_vehicle_count += step_metrics["scoped_vehicle_count"]
        sum_mobility += step_metrics["mean_mobility_score"]
        sum_risk += step_metrics["mean_risk_penalty"]
        sum_low_speed += step_metrics["mean_low_speed_penalty"]
        sum_collision_penalty += step_metrics["mean_collision_penalty"]
        total_collision_count += step_metrics["collision_count"]
        accumulate_least_advantaged_step(step_metrics, la_counters)
        accumulate_reason_step(step_metrics, reason_counters)

        final_min_exp = step_metrics["min_experience"]
        final_gini = step_metrics["gini_experience"]
        final_collision_count = step_metrics["collision_count"]
        steps += 1

        if terminated or truncated:
            break

    denom = steps if steps > 0 else 1
    la_episode = finalize_least_advantaged_episode(la_counters, steps)
    reason_episode = finalize_reason_episode(reason_counters)
    return {
        "episode": episode_id,
        "total_reward": total_reward,
        "mean_reward": total_reward / denom,
        "mean_original_reward": sum_original / denom,
        "mean_rawlsian_reward": sum_rawlsian / denom,
        "rawlsian_reward_sum": rawlsian_reward_sum,
        "rawlsian_scope": RAWLSIAN_SCOPE,
        "ego_neighbourhood_radius": EGO_NEIGHBOURHOOD_RADIUS,
        "experience_mode": EXPERIENCE_MODE,
        "mean_scoped_vehicle_count": sum_scoped_vehicle_count / denom,
        "mean_min_experience": sum_min_exp / denom,
        "final_min_experience": final_min_exp,
        "mean_vehicle_experience": sum_mean_vehicle_exp / denom,
        "mean_gini_experience": sum_gini / denom,
        "final_gini_experience": final_gini,
        "total_collision_count": total_collision_count,
        "final_collision_count": final_collision_count,
        "mean_n_vehicles": sum_n_vehicles / denom,
        "mean_mobility_score": sum_mobility / denom,
        "mean_risk_penalty": sum_risk / denom,
        "mean_low_speed_penalty": sum_low_speed / denom,
        "mean_collision_penalty": sum_collision_penalty / denom,
        **reason_episode,
        **la_episode,
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

    rows = [run_episode(env, ep) for ep in range(N_EPISODES)]
    env.close()

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV}")
    print(df)


if __name__ == "__main__":
    main()
