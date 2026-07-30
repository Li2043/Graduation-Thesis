"""Authoritative final environment + comfort lock loading (Stage 5A-0)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from thesis.calibration.final_environment_trace_loader import (
    DEFAULT_LOCK_PATH,
    EnvironmentLockError,
    LoadedFinalEnvironment,
    load_final_environment_lock,
    sha256_file,
)
from thesis.envs.final_environment_config import (
    LearningDynamics,
    TimingConfig,
    VehicleGeometry,
)
from thesis.envs.merge_env_candidate_v3 import MergeEnvCandidateV3, MergeEnvCandidateV3Config
from thesis.rewards.base_reward_v2 import BaseRewardConfig

EXPECTED_ENVIRONMENT_LOCK_SHA256 = (
    "d2d82ac02feb5bb2f5217f8e399972b91bd56cce343f60861954f66d7f70bf12"
)
EXPECTED_COMFORT_LOCK_SHA256 = (
    "1d9439c211955f9a8a177e455b6b5ff34aa98f85f3ff0677bbd62abd6d29b061"
)

DEFAULT_COMFORT_LOCK_PATH = Path(
    "experiments/pre_impl/stage3b_r1_final_environment_comfort_calibration/"
    "artifacts/20260730T005639Z_c6992dd4/final_comfort_parameters.yaml"
)


class FinalLockBlockedError(RuntimeError):
    """Raised when a retained lock fails integrity checks (BLOCKED)."""


@dataclass(frozen=True)
class LoadedComfortParameters:
    lock_path: Path
    lock_sha256: str
    lock: dict[str, Any]
    a_comfort: float
    a_hard: float
    eta_H: float
    eta_hard_brake: float
    comfort_parameters_final: bool
    environment_parameters_final: bool
    policy_training_started: bool

    def to_base_reward_config(self) -> BaseRewardConfig:
        cfg = BaseRewardConfig(
            a_comfort=float(self.a_comfort),
            a_hard=float(self.a_hard),
            eta_hard_brake=float(self.eta_hard_brake),
        )
        cfg.validate()
        return cfg


@dataclass(frozen=True)
class FinalLockBundle:
    environment: LoadedFinalEnvironment
    comfort: LoadedComfortParameters
    environment_lock_sha256_before: str
    comfort_lock_sha256_before: str

    @property
    def candidate_id(self) -> str:
        return str(self.environment.candidate.candidate_id)

    @property
    def observation_dimension(self) -> int:
        return int(self.environment.lock["observation_dimension"])

    @property
    def route_geometry_version(self) -> str:
        return str(self.environment.lock["route_geometry_version"])

    def timing_from_lock(self) -> TimingConfig:
        lock = self.environment.lock
        return TimingConfig(
            physics_dt=float(lock["physics_dt"]),
            policy_interval=float(lock["policy_interval"]),
            physics_substeps_per_action=int(lock["physics_substeps_per_action"]),
        )

    def vehicle_from_lock(self) -> VehicleGeometry:
        vd = self.environment.lock["vehicle_dimensions"]
        return VehicleGeometry(length=float(vd["length"]), width=float(vd["width"]))

    def dynamics_from_lock(self) -> LearningDynamics:
        lock = self.environment.lock
        acc = lock["learning_action_accelerations"]
        limits = lock.get("speed_limits") or {"v_min": 0.0, "v_max": 30.0}
        return LearningDynamics(
            accel=float(acc["ACCELERATE"]),
            maintain=float(acc["MAINTAIN"]),
            decel=float(acc["DECELERATE"]),
            v_min=float(limits.get("v_min", 0.0)),
            v_max=float(limits.get("v_max", 30.0)),
        )

    def build_env(
        self,
        *,
        block_id: str | None = None,
        block_set: str = "calibration",
        max_policy_steps: int | None = None,
    ) -> MergeEnvCandidateV3:
        """Construct MergeEnvCandidateV3 exclusively from verified locks."""
        blocks = (
            self.environment.calibration_blocks
            if block_set == "calibration"
            else self.environment.validation_blocks
        )
        if block_id is None:
            block = blocks[0]
        else:
            matched = [b for b in blocks if b.block_id == block_id]
            if not matched:
                raise ValueError(f"block_id {block_id!r} not found in {block_set}")
            block = matched[0]
        cfg = MergeEnvCandidateV3Config(
            candidate=self.environment.candidate,
            block=block,
            timing=self.timing_from_lock(),
            vehicle=self.vehicle_from_lock(),
            dynamics=self.dynamics_from_lock(),
            comfort=self.comfort.to_base_reward_config(),
            max_policy_steps=int(
                max_policy_steps
                if max_policy_steps is not None
                else 400
            ),
        )
        return MergeEnvCandidateV3(cfg)


def load_comfort_lock(
    lock_path: Path | None = None,
    *,
    expected_sha256: str = EXPECTED_COMFORT_LOCK_SHA256,
) -> LoadedComfortParameters:
    path = Path(lock_path) if lock_path is not None else DEFAULT_COMFORT_LOCK_PATH
    if not path.is_file():
        raise FinalLockBlockedError(f"missing comfort lock: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise FinalLockBlockedError(
            f"comfort lock SHA-256 mismatch: expected={expected_sha256} actual={actual}"
        )
    with path.open("r", encoding="utf-8") as f:
        lock = yaml.safe_load(f)
    if not isinstance(lock, dict):
        raise FinalLockBlockedError("comfort lock root must be a mapping")
    required = ("a_comfort", "a_hard", "eta_H", "eta_hard_brake")
    missing = [k for k in required if k not in lock]
    if missing:
        raise FinalLockBlockedError(f"comfort lock missing keys: {missing}")
    if lock.get("comfort_parameters_final") is not True:
        raise FinalLockBlockedError("comfort_parameters_final must be true")
    if lock.get("policy_training_started") is True:
        raise FinalLockBlockedError("policy_training_started must remain false")
    return LoadedComfortParameters(
        lock_path=path.resolve(),
        lock_sha256=actual,
        lock=lock,
        a_comfort=float(lock["a_comfort"]),
        a_hard=float(lock["a_hard"]),
        eta_H=float(lock["eta_H"]),
        eta_hard_brake=float(lock["eta_hard_brake"]),
        comfort_parameters_final=bool(lock.get("comfort_parameters_final")),
        environment_parameters_final=bool(lock.get("environment_parameters_final")),
        policy_training_started=bool(lock.get("policy_training_started", False)),
    )


def load_final_locks(
    *,
    environment_lock_path: Path | None = None,
    comfort_lock_path: Path | None = None,
    expected_environment_sha256: str = EXPECTED_ENVIRONMENT_LOCK_SHA256,
    expected_comfort_sha256: str = EXPECTED_COMFORT_LOCK_SHA256,
) -> FinalLockBundle:
    """Load and verify both retained locks; abort with BLOCKED on mismatch."""
    try:
        env = load_final_environment_lock(environment_lock_path)
    except EnvironmentLockError as exc:
        raise FinalLockBlockedError(str(exc)) from exc
    if env.lock_sha256 != expected_environment_sha256:
        raise FinalLockBlockedError(
            "environment lock SHA-256 mismatch: "
            f"expected={expected_environment_sha256} actual={env.lock_sha256}"
        )
    comfort = load_comfort_lock(
        comfort_lock_path, expected_sha256=expected_comfort_sha256
    )

    # Cross-checks from locks
    if env.candidate.candidate_id != "G1-I1":
        raise FinalLockBlockedError(
            f"selected environment must be G1-I1, got {env.candidate.candidate_id!r}"
        )
    if int(env.lock.get("observation_dimension", -1)) != 27:
        raise FinalLockBlockedError("observation_dimension must be 27")
    if float(env.lock.get("physics_dt")) != 0.05:
        raise FinalLockBlockedError("physics_dt must be 0.05")
    if float(env.lock.get("policy_interval")) != 0.20:
        raise FinalLockBlockedError("policy_interval must be 0.20")
    if int(env.lock.get("physics_substeps_per_action")) != 4:
        raise FinalLockBlockedError("physics_substeps must be 4")
    if not (
        comfort.a_comfort == 1.5
        and comfort.a_hard == 3.5
        and comfort.eta_H == 0.015
        and comfort.eta_hard_brake == 0.015
    ):
        raise FinalLockBlockedError(
            f"unexpected comfort tuple: a_comfort={comfort.a_comfort}, "
            f"a_hard={comfort.a_hard}, eta_H={comfort.eta_H}"
        )
    if env.lock.get("environment_parameters_final") is not True:
        raise FinalLockBlockedError("environment_parameters_final must be true")
    if env.lock.get("policy_training_started") is True:
        raise FinalLockBlockedError("policy_training_started must remain false")

    return FinalLockBundle(
        environment=env,
        comfort=comfort,
        environment_lock_sha256_before=env.lock_sha256,
        comfort_lock_sha256_before=comfort.lock_sha256,
    )


__all__ = [
    "DEFAULT_COMFORT_LOCK_PATH",
    "DEFAULT_LOCK_PATH",
    "EXPECTED_COMFORT_LOCK_SHA256",
    "EXPECTED_ENVIRONMENT_LOCK_SHA256",
    "FinalLockBlockedError",
    "FinalLockBundle",
    "LoadedComfortParameters",
    "load_comfort_lock",
    "load_final_locks",
]
