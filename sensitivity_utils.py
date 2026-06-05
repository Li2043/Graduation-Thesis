"""Shared helpers for v0.6.3 Rawlsian sensitivity analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import highway_env  # noqa: F401
import pandas as pd
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
from trained_policy_utils import evaluate_model_on_env

RESULT_ROW_METRICS = [
    "mean_min_experience",
    "final_min_experience",
    "mean_vehicle_experience",
    "mean_gini_experience",
    "total_collision_count",
    "mean_risk_penalty",
    "mean_mobility_score",
    "mean_collision_penalty",
    "least_advantaged_ego_ratio",
    "reason_risk_steps",
    "reason_low_mobility_steps",
    "steps",
    "total_reward",
    "mean_reward",
]

DEFAULT_PARAMETER_VALUES = {
    "xi": RAWLSIAN_XI,
    "radius": EGO_NEIGHBOURHOOD_RADIUS,
    "W_RISK": W_RISK,
}


@dataclass(frozen=True)
class RawlsianRunConfig:
    """Effective Rawlsian DQN hyperparameters for one sensitivity condition."""

    xi: float
    ego_neighbourhood_radius: float
    w_risk: float
    rawlsian_scope: str = RAWLSIAN_SCOPE

    def experience_kwargs(self) -> dict[str, Any]:
        return {
            "experience_mode": EXPERIENCE_MODE,
            "target_speed": TARGET_SPEED,
            "low_speed_threshold": LOW_SPEED_THRESHOLD,
            "w_mobility": W_MOBILITY,
            "w_collision": W_COLLISION,
            "w_low_speed": W_LOW_SPEED,
            "w_risk": self.w_risk,
            "risk_distance_normalizer": RISK_DISTANCE_NORMALIZER,
        }


@dataclass(frozen=True)
class SensitivityVariant:
    variant_id: str
    parameter_changed: str
    parameter_value: float
    config: RawlsianRunConfig


def default_rawlsian_config() -> RawlsianRunConfig:
    return RawlsianRunConfig(
        xi=RAWLSIAN_XI,
        ego_neighbourhood_radius=EGO_NEIGHBOURHOOD_RADIUS,
        w_risk=W_RISK,
        rawlsian_scope=RAWLSIAN_SCOPE,
    )


def build_sensitivity_variants() -> list[SensitivityVariant]:
    """One-factor-at-a-time grid: only one parameter differs from defaults per sweep."""
    base = default_rawlsian_config()
    variants: list[SensitivityVariant] = []

    for xi in (0.05, 0.10, 0.20):
        variants.append(
            SensitivityVariant(
                variant_id=f"xi_{xi:g}",
                parameter_changed="xi",
                parameter_value=xi,
                config=RawlsianRunConfig(
                    xi=xi,
                    ego_neighbourhood_radius=base.ego_neighbourhood_radius,
                    w_risk=base.w_risk,
                    rawlsian_scope=base.rawlsian_scope,
                ),
            )
        )

    for radius in (30.0, 50.0, 70.0):
        variants.append(
            SensitivityVariant(
                variant_id=f"radius_{radius:g}",
                parameter_changed="radius",
                parameter_value=radius,
                config=RawlsianRunConfig(
                    xi=base.xi,
                    ego_neighbourhood_radius=radius,
                    w_risk=base.w_risk,
                    rawlsian_scope=base.rawlsian_scope,
                ),
            )
        )

    for w_risk in (0.0, 0.5, 1.0):
        variants.append(
            SensitivityVariant(
                variant_id=f"w_risk_{w_risk:g}",
                parameter_changed="W_RISK",
                parameter_value=w_risk,
                config=RawlsianRunConfig(
                    xi=base.xi,
                    ego_neighbourhood_radius=base.ego_neighbourhood_radius,
                    w_risk=w_risk,
                    rawlsian_scope=base.rawlsian_scope,
                ),
            )
        )

    return variants


def model_path_for_variant(
    project_root: Path,
    model_dir: str,
    variant_id: str,
    seed: int,
) -> Path:
    return (
        project_root
        / model_dir
        / variant_id
        / f"seed_{seed}"
        / "rawlsian_dqn.zip"
    )


def tensorboard_log_path(project_root: Path, log_dir: str, variant_id: str, seed: int) -> str | None:
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        print("Warning: tensorboard not installed. Training without tensorboard_log.")
        return None
    return str(project_root / log_dir / variant_id / f"seed_{seed}")


def make_rawlsian_env(config: RawlsianRunConfig) -> gym.Env:
    base_env = gym.make(ENV_ID)
    return RawlsianRewardWrapper(
        base_env,
        xi=config.xi,
        speed_normalizer=SPEED_NORMALIZER,
        scope=config.rawlsian_scope,
        radius=config.ego_neighbourhood_radius,
        mode=EXPERIENCE_MODE,
        target_speed=TARGET_SPEED,
        low_speed_threshold=LOW_SPEED_THRESHOLD,
        w_mobility=W_MOBILITY,
        w_collision=W_COLLISION,
        w_low_speed=W_LOW_SPEED,
        w_risk=config.w_risk,
        risk_distance_normalizer=RISK_DISTANCE_NORMALIZER,
    )


def build_dqn(
    env: gym.Env,
    seed: int,
    total_timesteps: int,
    tensorboard_log: str | None,
) -> DQN:
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
        tensorboard_log=tensorboard_log,
        seed=seed,
    )


def train_rawlsian_variant(
    variant: SensitivityVariant,
    seed: int,
    project_root: Path,
    model_dir: str,
    log_dir: str,
    total_timesteps: int,
) -> Path:
    save_path = model_path_for_variant(project_root, model_dir, variant.variant_id, seed)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_rawlsian_env(variant.config)
    env.reset(seed=seed)
    env.action_space.seed(seed)

    print(
        f"\n=== Train {variant.variant_id} seed={seed} "
        f"({variant.parameter_changed}={variant.parameter_value}) ==="
    )
    print(
        f"xi={variant.config.xi} radius={variant.config.ego_neighbourhood_radius} "
        f"W_RISK={variant.config.w_risk} scope={variant.config.rawlsian_scope} "
        f"timesteps={total_timesteps}"
    )

    model = build_dqn(
        env,
        seed=seed,
        total_timesteps=total_timesteps,
        tensorboard_log=tensorboard_log_path(project_root, log_dir, variant.variant_id, seed),
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(str(save_path))
    env.close()
    print(f"Saved model to {save_path}")
    return save_path


def evaluate_rawlsian_variant(
    variant: SensitivityVariant,
    seed: int,
    project_root: Path,
    model_dir: str,
    n_episodes: int,
    max_steps: int,
) -> dict[str, Any]:
    model_path = model_path_for_variant(project_root, model_dir, variant.variant_id, seed)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model: {model_path}")

    env = make_rawlsian_env(variant.config)
    model = DQN.load(str(model_path), env=env)

    episode_df = evaluate_model_on_env(
        model,
        env,
        is_rawlsian=True,
        n_episodes=n_episodes,
        max_steps=max_steps,
        speed_normalizer=SPEED_NORMALIZER,
        base_seed=seed,
        metric_scope=variant.config.rawlsian_scope,
        radius=variant.config.ego_neighbourhood_radius,
        **variant.config.experience_kwargs(),
    )
    env.close()

    metrics = {m: float(episode_df[m].mean()) for m in RESULT_ROW_METRICS if m in episode_df.columns}
    row = {
        "variant_id": variant.variant_id,
        "parameter_changed": variant.parameter_changed,
        "parameter_value": variant.parameter_value,
        "seed": seed,
        "xi": variant.config.xi,
        "rawlsian_scope": variant.config.rawlsian_scope,
        "ego_neighbourhood_radius": variant.config.ego_neighbourhood_radius,
        "W_RISK": variant.config.w_risk,
        **metrics,
    }
    return row


def append_result_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        key = (row["variant_id"], row["seed"])
        mask = (existing["variant_id"] == key[0]) & (existing["seed"] == key[1])
        existing = existing.loc[~mask]
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(csv_path, index=False)


def load_existing_keys(csv_path: Path) -> set[tuple[str, int]]:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    return {(str(r.variant_id), int(r.seed)) for r in df.itertuples()}

