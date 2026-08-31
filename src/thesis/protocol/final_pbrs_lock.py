"""Final PBRS parameter lock (Stage 5C-0)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from thesis.protocol.prerequisites import Stage5C0Prerequisites

EXPECTED_LAMBDA_MEAN = 0.2
EXPECTED_LAMBDA_MIN = 0.2
EXPECTED_GAMMA = 0.995


def build_final_pbrs_lock(
    prereq: Stage5C0Prerequisites,
    *,
    git_commit: str,
    source_hashes: dict[str, str],
    configuration_sha256: str,
) -> dict[str, Any]:
    """Build immutable PBRS lock. Lambda not selected from pilot performance."""
    return {
        "lock_type": "final_pbrs_parameters",
        "stage": "stage5c0",
        "lambda_mean": EXPECTED_LAMBDA_MEAN,
        "lambda_min": EXPECTED_LAMBDA_MIN,
        "baseline_lambda": 0.0,
        "gamma": EXPECTED_GAMMA,
        "shaping_gamma": EXPECTED_GAMMA,
        "learner_gamma": EXPECTED_GAMMA,
        "equal_scales_across_shaped_conditions": True,
        "conditions": {
            "baseline": {"lambda": 0.0},
            "mean_pbrs": {"lambda_mean": EXPECTED_LAMBDA_MEAN},
            "min_pbrs": {"lambda_min": EXPECTED_LAMBDA_MIN},
        },
        "stakeholder_set": ["A", "B", "B_front", "B_rear"],
        "potential_definitions": {
            "Phi_mean": "(E_A + E_B + E_B_front + E_B_rear) / 4",
            "Phi_min": "min(E_A, E_B, E_B_front, E_B_rear)",
            "E_i_active": "clip01(v_i / v_target_i)",
            "E_i_safely_exited": 1.0,
        },
        "terminal_semantics": {
            "true_terminal_successor_potential": 0.0,
            "true_terminal_includes": [
                "stakeholder_collision",
                "joint_learner_success",
            ],
            "truncation_successor_potential": "actual_successor_potential",
            "truncation_is_not_true_terminal": True,
        },
        "base_reward_retained": {
            "formula": (
                "0.4*delta_rho + 0.6*safe_exit - 1.0*collision - 0.015*H"
            ),
            "a_comfort": prereq.a_comfort,
            "a_hard": prereq.a_hard,
            "eta_H": prereq.eta_H,
        },
        "selection_justification": [
            "equal scales isolate the effect of potential aggregation",
            "0.2 exercised non-zero shaping paths in Stage 5A-0",
            "0.2 completed the engineering pilot without numerical failure",
            "value was not selected using comparative pilot outcomes",
            "no alternative lambda candidates were compared using pilot performance",
        ],
        "pilot_comparative_outcomes_used_for_selection": False,
        "pilot_behavioral_observations_read": False,
        "admissible_pilot_engineering_inputs_only": [
            "runs_completed",
            "checkpoint_and_resume_success",
            "finite_numerical_values",
            "absence_of_illegal_actions",
            "evaluation_isolation",
            "lock_integrity",
        ],
        "environment_lock_path": str(prereq.environment_lock_path),
        "environment_lock_sha256": prereq.environment_lock_sha256,
        "comfort_lock_path": str(prereq.comfort_lock_path),
        "comfort_lock_sha256": prereq.comfort_lock_sha256,
        "stage5a0_run_id": prereq.stage5a0_run_id,
        "stage5b0_run_id": prereq.stage5b0_run_id,
        "source_git_commit": git_commit,
        "source_hashes": source_hashes,
        "configuration_sha256": configuration_sha256,
        "pbrs_parameters_final": True,
        "formal_training_started": False,
    }


def write_final_pbrs_lock(path: Path, lock: dict[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(lock, f, sort_keys=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sha_path = path.parent / f"{path.stem}.sha256"
    sha_path.write_text(digest + "\n", encoding="utf-8")
    return digest


__all__ = [
    "EXPECTED_GAMMA",
    "EXPECTED_LAMBDA_MEAN",
    "EXPECTED_LAMBDA_MIN",
    "build_final_pbrs_lock",
    "write_final_pbrs_lock",
]
