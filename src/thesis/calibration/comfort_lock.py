"""Write Stage 3B-R1 final comfort parameter lock (only after full PASS)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def build_comfort_lock(
    *,
    a_comfort: float,
    a_hard: float,
    eta_h: float,
    candidate_grids: dict[str, Any],
    calibration_metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
    environment_lock_path: str,
    environment_lock_sha256: str,
    git_commit: str,
    source_hashes: dict[str, str],
    config_hash: str,
    confirmatory_errors: dict[str, float],
    stage4a_r1_run_id: str = "20260730T003122Z_aee2d425",
    original_stage3b_run_id: str = "20260729T225046Z_8624f03a",
) -> dict[str, Any]:
    return {
        "a_comfort": a_comfort,
        "a_hard": a_hard,
        "eta_H": eta_h,
        "eta_hard_brake": eta_h,
        "candidate_grids": candidate_grids,
        "tuple_selection_rule": [
            "smallest_eta_H",
            "largest_hard_minus_nominal_mean_H_separation",
            "lowest_median_nominal_safe_braking_share",
            "highest_a_comfort",
            "highest_a_hard",
        ],
        "share_definition": {
            "B_i": "sum_t gamma^t * eta_H * H_i,t",
            "D_i": "sum_t gamma^t * (|0.4 Δρ| + |0.6 exit| + |collision| + eta_H H)",
            "S_i": "B_i / max(D_i, 1e-12)",
            "team": "mean of available learner shares",
        },
        "policy_level_acceleration_definition": (
            "minimum realised acceleration across active physics substeps "
            "of the 0.20 s policy transition"
        ),
        "calibration_metrics": calibration_metrics,
        "validation_metrics": validation_metrics,
        "final_environment_lock_path": environment_lock_path,
        "final_environment_lock_sha256": environment_lock_sha256,
        "source_git_commit": git_commit,
        "source_hashes": source_hashes,
        "configuration_sha256": config_hash,
        "original_stage3b_failure_reference": {
            "run_id": original_stage3b_run_id,
            "development_calibration": "FAIL",
            "reason": (
                "threshold candidates existed; no eta in the original coarse "
                "preregistered grid was feasible; thresholds selected before eta; "
                "comfort and eta were not jointly selected"
            ),
        },
        "stage4a_r1_environment_reference": {
            "run_id": stage4a_r1_run_id,
            "candidate_id": "G1-I1",
        },
        "online_offline_maximum_errors": confirmatory_errors,
        "environment_parameters_final": True,
        "comfort_parameters_final": True,
        "policy_training_started": False,
    }


def write_comfort_lock(path: Path, lock: dict[str, Any]) -> str:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(lock, f, sort_keys=False)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    digest = h.hexdigest()
    path.with_suffix(".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


__all__ = ["build_comfort_lock", "write_comfort_lock"]
