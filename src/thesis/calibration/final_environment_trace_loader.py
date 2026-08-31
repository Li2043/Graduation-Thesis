"""Load and verify the Stage 4A-R1 final environment lock (Stage 3B-R1)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thesis.envs.final_environment_config import (
    EnvironmentCandidate,
    GeometryCandidate,
    IDMProfile,
    InitialConditionBlock,
    TargetSpeeds,
)

DEFAULT_LOCK_PATH = Path(
    "experiments/pre_impl/stage4a_r1_final_environment_reselection/artifacts/"
    "20260730T003122Z_aee2d425/final_environment_lock.yaml"
)
DEFAULT_LOCK_SHA256_PATH = DEFAULT_LOCK_PATH.with_suffix(".sha256")


class EnvironmentLockError(RuntimeError):
    """Raised when the retained environment lock cannot be used (BLOCKED)."""


@dataclass(frozen=True)
class LoadedFinalEnvironment:
    lock_path: Path
    lock_sha256: str
    lock: dict[str, Any]
    candidate: EnvironmentCandidate
    calibration_blocks: list[InitialConditionBlock]
    validation_blocks: list[InitialConditionBlock]

    def summary(self) -> dict[str, Any]:
        return {
            "lock_path": str(self.lock_path),
            "lock_sha256": self.lock_sha256,
            "candidate_id": self.candidate.candidate_id,
            "geometry": self.candidate.geometry.to_dict(),
            "idm": self.candidate.idm.to_dict(),
            "physics_dt": self.lock.get("physics_dt"),
            "policy_interval": self.lock.get("policy_interval"),
            "physics_substeps_per_action": self.lock.get("physics_substeps_per_action"),
            "vehicle_dimensions": self.lock.get("vehicle_dimensions"),
            "learning_action_accelerations": self.lock.get("learning_action_accelerations"),
            "observation_version": self.lock.get("observation_version"),
            "observation_dimension": self.lock.get("observation_dimension"),
            "route_geometry_version": self.lock.get("route_geometry_version"),
            "collision_model_version": self.lock.get("collision_model_version"),
            "n_calibration_blocks": len(self.calibration_blocks),
            "n_validation_blocks": len(self.validation_blocks),
            "source_hashes": self.lock.get("source_hashes"),
            "environment_parameters_final": self.lock.get("environment_parameters_final"),
            "comfort_parameters_final": self.lock.get("comfort_parameters_final"),
            "policy_training_started": self.lock.get("policy_training_started"),
        }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def read_recorded_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise EnvironmentLockError(f"empty sha256 file: {path}")
    return text.split()[0].strip()


def _block_from_dict(d: dict[str, Any]) -> InitialConditionBlock:
    ts = d.get("target_speeds") or {}
    return InitialConditionBlock(
        block_id=str(d["block_id"]),
        block_set=str(d["block_set"]),
        seed=int(d["seed"]),
        role_A=str(d["role_A"]),
        role_B=str(d["role_B"]),
        spawn_route_mainline=float(d["spawn_route_mainline"]),
        spawn_route_ramp=float(d["spawn_route_ramp"]),
        spawn_speed_mainline=float(d["spawn_speed_mainline"]),
        spawn_speed_ramp=float(d["spawn_speed_ramp"]),
        spawn_route_B_front=float(d["spawn_route_B_front"]),
        spawn_route_B_rear=float(d["spawn_route_B_rear"]),
        spawn_speed_B_front=float(d["spawn_speed_B_front"]),
        spawn_speed_B_rear=float(d["spawn_speed_B_rear"]),
        delta_arrival=float(d["delta_arrival"]),
        arrival_category=str(d["arrival_category"]),
        background_time_headway=float(d["background_time_headway"]),
        target_speeds=TargetSpeeds(
            A=float(ts.get("A", 20.0)),
            B=float(ts.get("B", 20.0)),
            B_front=float(ts.get("B_front", 20.0)),
            B_rear=float(ts.get("B_rear", 20.0)),
        ),
    )


def candidate_from_lock(lock: dict[str, Any]) -> EnvironmentCandidate:
    geom = GeometryCandidate(**lock["geometry"])
    idm = IDMProfile(**lock["idm_parameters"])
    return EnvironmentCandidate(
        candidate_id=str(lock["candidate_id"]),
        geometry=geom,
        idm=idm,
        priority_rank=int(lock["candidate_priority_rank"]),
    )


def load_final_environment_lock(
    lock_path: Path | None = None,
    *,
    sha256_path: Path | None = None,
) -> LoadedFinalEnvironment:
    """Load retained Stage 4A-R1 lock; abort with BLOCKED on hash/parse failure."""
    import yaml

    path = Path(lock_path) if lock_path is not None else DEFAULT_LOCK_PATH
    sha_path = Path(sha256_path) if sha256_path is not None else path.with_suffix(".sha256")
    if not path.is_file():
        raise EnvironmentLockError(f"missing final environment lock: {path}")
    if not sha_path.is_file():
        raise EnvironmentLockError(f"missing lock sha256 file: {sha_path}")

    recorded = read_recorded_sha256(sha_path)
    actual = sha256_file(path)
    if recorded != actual:
        raise EnvironmentLockError(
            f"environment lock SHA-256 mismatch: recorded={recorded} actual={actual}"
        )

    try:
        with path.open("r", encoding="utf-8") as f:
            lock = yaml.safe_load(f)
    except Exception as exc:  # noqa: BLE001
        raise EnvironmentLockError(f"failed to parse environment lock: {exc}") from exc

    if not isinstance(lock, dict):
        raise EnvironmentLockError("environment lock root must be a mapping")

    required = (
        "candidate_id",
        "geometry",
        "idm_parameters",
        "physics_dt",
        "policy_interval",
        "physics_substeps_per_action",
        "vehicle_dimensions",
        "learning_action_accelerations",
        "observation_version",
        "observation_dimension",
        "route_geometry_version",
        "collision_model_version",
        "calibration_block_definitions",
        "validation_block_definitions",
        "source_hashes",
    )
    missing = [k for k in required if k not in lock]
    if missing:
        raise EnvironmentLockError(f"environment lock missing keys: {missing}")

    if lock.get("candidate_id") != "G1-I1":
        raise EnvironmentLockError(
            f"expected candidate_id=G1-I1, got {lock.get('candidate_id')!r}"
        )
    if lock.get("environment_parameters_final") is not True:
        raise EnvironmentLockError("environment_parameters_final must be true")

    try:
        candidate = candidate_from_lock(lock)
        cal = [_block_from_dict(b) for b in lock["calibration_block_definitions"]]
        val = [_block_from_dict(b) for b in lock["validation_block_definitions"]]
    except Exception as exc:  # noqa: BLE001
        raise EnvironmentLockError(f"failed to materialise lock objects: {exc}") from exc

    if len(cal) != 12 or len(val) != 8:
        raise EnvironmentLockError(
            f"expected 12/8 cal/val blocks, got {len(cal)}/{len(val)}"
        )
    if {b.block_id for b in cal} & {b.block_id for b in val}:
        raise EnvironmentLockError("calibration and validation block IDs must be disjoint")

    return LoadedFinalEnvironment(
        lock_path=path.resolve(),
        lock_sha256=actual,
        lock=lock,
        candidate=candidate,
        calibration_blocks=cal,
        validation_blocks=val,
    )


__all__ = [
    "DEFAULT_LOCK_PATH",
    "DEFAULT_LOCK_SHA256_PATH",
    "EnvironmentLockError",
    "LoadedFinalEnvironment",
    "load_final_environment_lock",
    "sha256_file",
]
