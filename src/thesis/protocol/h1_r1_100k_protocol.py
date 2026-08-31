"""Stage 5C-0-H1-R1 — 100K formal protocol amendment.

Supersedes the original Stage 5C-0 training lock for future formal execution.
Does not start formal training.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import yaml

from thesis.envs.final_observation import OBSERVATION_DIM
from thesis.protocol.prerequisites import (
    Stage5C0Prerequisites,
    verify_stage5c0_prerequisites,
)
from thesis.training.final_lock_loader import EXPECTED_ENVIRONMENT_LOCK_SHA256

PROTOCOL_VERSION = "5C-0-H1-R1-100K"
SUPERSEDED_STAGE5C0_PROTOCOL_SHA256 = (
    "a7b0a34ce08541d8f9e5adf24856924429fbb8463bc6e271bba824affc806b4f"
)
SUPERSEDED_STAGE5C0_PBRS_SHA256 = (
    "8ce5025690e5f217296386af765a85f6c1afa7f3028c3eab5fb06a64d87fbf4b"
)
STAGE5C0_RUN_ID = "20260730T072103Z_94767983"

FORMAL_MASTER_SEEDS: tuple[int, ...] = tuple(range(61001, 61011))
FORMAL_CONDITIONS: tuple[str, ...] = ("baseline", "mean_pbrs", "min_pbrs")
FORMAL_STEPS_PER_RUN = 100_000
FORMAL_CHECKPOINT_STEPS: tuple[int, ...] = (10_000, 25_000, 50_000, 75_000, 100_000)
FORMAL_EVALUATION_STEPS: tuple[int, ...] = (0, 10_000, 25_000, 50_000, 75_000, 100_000)
PRIMARY_ENDPOINT_STEP = 100_000
EPSILON_DECAY_STEPS = 50_000
LAMBDA_MEAN = 0.2
LAMBDA_MIN = 0.2
GAMMA = 0.995


def derive_formal_seeds(master_seed: int) -> dict[str, int]:
    s = int(master_seed)
    return {
        "master_seed": s,
        "environment_seed": s,
        "learner_A_seed": s + 100_000,
        "learner_B_seed": s + 200_000,
        "replay_A_seed": s + 300_000,
        "replay_B_seed": s + 400_000,
        "evaluation_seed": s + 500_000,
        "schedule_seed": s + 600_000,
    }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_yaml_hashed(path: Path, payload: dict[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(payload, sort_keys=False)
    path.write_text(text, encoding="utf-8")
    digest = _sha256_bytes(path.read_bytes())
    (path.parent / f"{path.stem}.sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def build_h1_r1_pbrs_lock(prereq: Stage5C0Prerequisites, *, git_commit: str) -> dict[str, Any]:
    return {
        "lock_type": "final_pbrs_parameters",
        "protocol_version": PROTOCOL_VERSION,
        "stage": "stage5c0_h1_r1",
        "supersedes_stage5c0_pbrs_sha256": SUPERSEDED_STAGE5C0_PBRS_SHA256,
        "lambda_mean": LAMBDA_MEAN,
        "lambda_min": LAMBDA_MIN,
        "baseline_lambda": 0.0,
        "gamma": GAMMA,
        "comparison_type": "equal_coefficient",
        "magnitude_matched": False,
        "rms_matched": False,
        "equal_scales_across_shaped_conditions": True,
        "conditions": {
            "baseline": {"lambda": 0.0},
            "mean_pbrs": {"lambda_mean": LAMBDA_MEAN},
            "min_pbrs": {"lambda_min": LAMBDA_MIN},
        },
        "stakeholder_set": ["A", "B", "B_front", "B_rear"],
        "potential_definitions": {
            "Phi_mean": "(E_A + E_B + E_B_front + E_B_rear) / 4",
            "Phi_min": "min(E_A, E_B, E_B_front, E_B_rear)",
        },
        "terminal_semantics": {
            "true_terminal_successor_potential": 0.0,
            "truncation_successor_potential": "actual_successor_potential",
        },
        "base_reward_retained": {
            "a_comfort": prereq.a_comfort,
            "a_hard": prereq.a_hard,
            "eta_H": prereq.eta_H,
        },
        "pilot_comparative_outcomes_used_for_selection": False,
        "environment_lock_sha256": prereq.environment_lock_sha256,
        "comfort_lock_sha256": prereq.comfort_lock_sha256,
        "source_git_commit": git_commit,
        "pbrs_parameters_final": True,
        "formal_training_started": False,
        "pre_result_budget_amendment": True,
        "formal_results_existed_at_amendment": False,
    }


def build_formal_analysis_plan_100k() -> dict[str, Any]:
    return {
        "statistical_unit": "formal_training_seed",
        "pairing": "same master seeds across all three conditions",
        "pairwise_contrasts": [
            "mean_pbrs - baseline",
            "min_pbrs - baseline",
            "min_pbrs - mean_pbrs",
        ],
        "primary_endpoints_at_step_100000": [
            "evaluation_success_rate",
            "stakeholder_collision_rate",
            "mean_stakeholder_episode_utility",
            "minimum_stakeholder_episode_utility",
            "convention_consistency",
        ],
        "missing_convention_policy": {
            "zero_fill": False,
            "report_missing_count": True,
            "compare_only_complete_paired_seeds": True,
        },
        "bootstrap": {
            "resampling_unit": "paired_seed",
            "replicates": 10_000,
            "rng_seed": 91001,
        },
        "multiple_comparisons": {"method": "Holm"},
        "hypothesis_tests": {"sidedness": "two_sided", "alpha": 0.05},
        "best_checkpoint_selection": False,
        "primary_formal_endpoint_checkpoint": PRIMARY_ENDPOINT_STEP,
        "learning_curves": {
            "checkpoints": list(FORMAL_EVALUATION_STEPS),
            "auc_method": "trapezoidal_rule_over_fixed_checkpoints",
        },
    }


def build_h1_r1_training_protocol(
    prereq: Stage5C0Prerequisites,
    *,
    git_commit: str,
    pbrs_lock_sha256: str,
) -> dict[str, Any]:
    n_runs = len(FORMAL_CONDITIONS) * len(FORMAL_MASTER_SEEDS)
    return {
        "lock_type": "final_training_protocol",
        "protocol_version": PROTOCOL_VERSION,
        "stage": "stage5c0_h1_r1",
        "supersedes_stage5c0_protocol_sha256": SUPERSEDED_STAGE5C0_PROTOCOL_SHA256,
        "superseded_stage5c0_run_id": STAGE5C0_RUN_ID,
        "pre_result_budget_amendment": True,
        "formal_results_existed_at_amendment": False,
        "conditions": list(FORMAL_CONDITIONS),
        "only_condition_dependent_component": "pbrs_shaping_term",
        "environment": {
            "candidate_id": prereq.candidate_id,
            "observation_dimension": OBSERVATION_DIM,
            "environment_class": "MergeEnvCandidateV3",
            "environment_lock_sha256": prereq.environment_lock_sha256,
            "comfort_lock_sha256": prereq.comfort_lock_sha256,
            "num_parallel_training_envs_per_run": 1,
            "vectorized_training": False,
            "environment_execution_mode": "single_environment",
            "policy_interval_seconds": 0.20,
            "physics_dt_seconds": 0.05,
            "physics_substeps": 4,
            "maximum_episode_policy_steps": 400,
            "maximum_episode_simulated_seconds": 80,
        },
        "timestep_semantics": {
            "one_timestep": "one joint policy-level environment step / MergeEnvCandidateV3.step()",
            "does_not_increment_for": [
                "physics_substeps",
                "replay_rows",
                "controller_count",
                "optimiser_updates",
                "evaluation_steps",
                "checkpoint_operations",
                "resets",
            ],
        },
        "pbrs": {
            "lambda_mean": LAMBDA_MEAN,
            "lambda_min": LAMBDA_MIN,
            "baseline_lambda": 0.0,
            "gamma": GAMMA,
            "comparison_type": "equal_coefficient",
            "magnitude_matched": False,
            "rms_matched": False,
            "pbrs_lock_sha256": pbrs_lock_sha256,
            "pbrs_parameters_final": True,
        },
        "dqn": {
            "algorithm": "vanilla_independent_dqn",
            "separate_learners": ["A", "B"],
            "shared_weights": False,
            "observation_dimension": OBSERVATION_DIM,
            "action_count": 3,
            "hidden_sizes": [64, 64],
            "activation": "ReLU",
            "loss": "mean_squared_td_error",
            "gamma": GAMMA,
            "learning_rate": 0.0005,
            "optimiser": "Adam",
            "batch_size": 64,
            "replay_capacity_per_controller": 20_000,
            "replay_warmup_per_controller": 512,
            "target_sync_type": "hard",
            "target_sync_interval_updates": 250,
            "updates_per_active_controller_per_environment_step": 1,
            "device": "cpu",
            "explicit_replay_seed_injection": True,
        },
        "exploration": {
            "epsilon_start": 1.0,
            "epsilon_end": 0.10,
            "epsilon_schedule": "linear",
            "epsilon_decay_environment_steps": EPSILON_DECAY_STEPS,
            "epsilon_after_decay": 0.10,
        },
        "training_budget": {
            "formal_environment_steps_per_run": FORMAL_STEPS_PER_RUN,
            "early_stopping": False,
            "master_seeds": list(FORMAL_MASTER_SEEDS),
            "n_master_seeds": len(FORMAL_MASTER_SEEDS),
            "n_conditions": len(FORMAL_CONDITIONS),
            "n_formal_runs": n_runs,
            "total_planned_environment_steps": n_runs * FORMAL_STEPS_PER_RUN,
        },
        "seed_derivation": {
            "environment_seed": "S",
            "learner_A_seed": "S + 100000",
            "learner_B_seed": "S + 200000",
            "replay_A_seed": "S + 300000",
            "replay_B_seed": "S + 400000",
            "evaluation_seed": "S + 500000",
            "schedule_seed": "S + 600000",
            "condition_name_affects_derivation": False,
        },
        "parallelism": {
            "num_parallel_training_envs_per_run": 1,
            "vectorized_training": False,
            "independent_job_parallelism_allowed": True,
            "no_shared_state_across_jobs": True,
            "default_workers": 12,
            "default_threads_per_worker": 1,
            "multiprocessing_start_method": "spawn",
            "worker_count_is_execution_metadata_not_experimental_variable": True,
        },
        "training_initial_conditions": {
            "source": "12_retained_calibration_blocks_only",
            "validation_blocks_used_for_training": False,
            "cycle_episodes": 24,
            "scheduling": [
                "deterministic_seeded_shuffled_24_episode_cycles",
                "every_cycle_includes_each_block_assignment_pair_exactly_once",
                "schedule_cursor_checkpointed",
            ],
        },
        "checkpoint_schedule": {
            "steps": list(FORMAL_CHECKPOINT_STEPS),
            "primary_endpoint": PRIMARY_ENDPOINT_STEP,
            "best_checkpoint_selection": False,
            "retain_all_scheduled_checkpoints": True,
            "full_replay_checkpoints_local_only": True,
        },
        "evaluation_schedule": {
            "steps": list(FORMAL_EVALUATION_STEPS),
            "episodes_per_point": 16,
            "validation_blocks": 8,
            "controller_role_assignments": 2,
            "greedy": True,
            "epsilon": 0.0,
            "episode_seed_formula": (
                "evaluation_seed + 1000 * checkpoint_index + 2 * block_index + assignment_index"
            ),
            "results_alter_training": False,
            "results_alter_checkpoint_selection": False,
        },
        "failure_and_resume_policy": {
            "replace_failed_seeds_with_new_seeds": False,
            "silent_omit_failed_run": False,
            "terminal_statuses": [
                "COMPLETE",
                "FAILED_WITH_REASON",
                "INTERRUPTED_RESUMABLE",
            ],
        },
        "analysis_plan": build_formal_analysis_plan_100k(),
        "prerequisites": {
            "stage5a0_run_id": prereq.stage5a0_run_id,
            "stage5b0_run_id": prereq.stage5b0_run_id,
            "stage5c0_run_id": STAGE5C0_RUN_ID,
        },
        "source_git_commit": git_commit,
        "environment_parameters_final": True,
        "comfort_parameters_final": True,
        "pbrs_parameters_final": True,
        "training_protocol_final": True,
        "formal_training_started": False,
        "sustained_training_invoked_in_this_stage": False,
    }


def build_protocol_diff() -> dict[str, Any]:
    return {
        "from_protocol_sha256": SUPERSEDED_STAGE5C0_PROTOCOL_SHA256,
        "to_protocol_version": PROTOCOL_VERSION,
        "changes": {
            "formal_environment_steps_per_run": {"from": 20_000, "to": 100_000},
            "total_planned_environment_steps": {"from": 600_000, "to": 3_000_000},
            "epsilon_decay_environment_steps": {"from": 4_000, "to": 50_000},
            "checkpoint_steps": {
                "from": [5000, 10000, 15000, 20000],
                "to": list(FORMAL_CHECKPOINT_STEPS),
            },
            "evaluation_steps": {
                "from": [0, 5000, 10000, 15000, 20000],
                "to": list(FORMAL_EVALUATION_STEPS),
            },
            "primary_endpoint": {"from": 20_000, "to": 100_000},
            "num_parallel_training_envs_per_run": {"from": 1, "to": 1},
            "training_cycle": {"from": "12_block_alternating", "to": "24_pair_shuffled"},
        },
        "unchanged": [
            "conditions",
            "lambda_mean",
            "lambda_min",
            "dqn_architecture",
            "master_seeds",
            "seed_derivation",
            "statistical_unit",
            "paired_contrasts",
        ],
        "pre_result_budget_amendment": True,
    }


def build_formal_run_matrix_100k(
    *,
    protocol_sha256: str,
    environment_lock_sha256: str,
    comfort_lock_sha256: str,
    output_root_template: str = "<output-root>/jobs/{formal_job_id}",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in FORMAL_CONDITIONS:
        for master in FORMAL_MASTER_SEEDS:
            derived = derive_formal_seeds(master)
            job_id = f"{condition}__{master}"
            rows.append(
                {
                    "formal_job_id": job_id,
                    "condition": condition,
                    "master_seed": master,
                    "environment_seed": derived["environment_seed"],
                    "learner_A_seed": derived["learner_A_seed"],
                    "learner_B_seed": derived["learner_B_seed"],
                    "replay_A_seed": derived["replay_A_seed"],
                    "replay_B_seed": derived["replay_B_seed"],
                    "evaluation_seed": derived["evaluation_seed"],
                    "schedule_seed": derived["schedule_seed"],
                    "protocol_hash": protocol_sha256,
                    "environment_lock_hash": environment_lock_sha256,
                    "comfort_lock_hash": comfort_lock_sha256,
                    "expected_steps": FORMAL_STEPS_PER_RUN,
                    "expected_evaluation_points": ";".join(
                        str(x) for x in FORMAL_EVALUATION_STEPS
                    ),
                    "expected_checkpoint_points": ";".join(
                        str(x) for x in FORMAL_CHECKPOINT_STEPS
                    ),
                    "expected_output_directory": output_root_template.format(
                        formal_job_id=job_id
                    ),
                    "status_planned": "PENDING",
                }
            )
    if len(rows) != 30:
        raise RuntimeError(f"expected 30 formal slots, got {len(rows)}")
    return rows


def write_formal_run_matrix_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


def write_h1_r1_artifact_bundle(
    artifact_dir: Path,
    *,
    git_commit: str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Verify predecessors and write the authoritative H1-R1 lock bundle."""
    from thesis.calibration.final_environment_trace_loader import sha256_file

    root = Path(repo_root) if repo_root is not None else Path(".")
    prereq = verify_stage5c0_prerequisites(repo_root=root)

    # Verify Stage 5C-0 locks unchanged
    pbrs_path = (
        root
        / "experiments/formal/protocol/artifacts"
        / STAGE5C0_RUN_ID
        / "final_pbrs_parameters.yaml"
    )
    proto_path = (
        root
        / "experiments/formal/protocol/artifacts"
        / STAGE5C0_RUN_ID
        / "final_training_protocol.yaml"
    )
    if sha256_file(pbrs_path) != SUPERSEDED_STAGE5C0_PBRS_SHA256:
        raise RuntimeError("Stage 5C-0 PBRS lock hash mismatch (BLOCKED)")
    if sha256_file(proto_path) != SUPERSEDED_STAGE5C0_PROTOCOL_SHA256:
        raise RuntimeError("Stage 5C-0 protocol lock hash mismatch (BLOCKED)")
    if prereq.environment_lock_sha256 != EXPECTED_ENVIRONMENT_LOCK_SHA256:
        raise RuntimeError("environment lock mismatch (BLOCKED)")

    art = Path(artifact_dir)
    art.mkdir(parents=True, exist_ok=True)

    pbrs = build_h1_r1_pbrs_lock(prereq, git_commit=git_commit)
    pbrs_sha = _write_yaml_hashed(art / "final_pbrs_parameters.yaml", pbrs)
    protocol = build_h1_r1_training_protocol(
        prereq, git_commit=git_commit, pbrs_lock_sha256=pbrs_sha
    )
    protocol_sha = _write_yaml_hashed(art / "final_training_protocol.yaml", protocol)
    rows = build_formal_run_matrix_100k(
        protocol_sha256=protocol_sha,
        environment_lock_sha256=prereq.environment_lock_sha256,
        comfort_lock_sha256=prereq.comfort_lock_sha256,
    )
    write_formal_run_matrix_csv(art / "formal_run_matrix.csv", rows)
    with (art / "formal_analysis_plan.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(build_formal_analysis_plan_100k(), f, sort_keys=False)
    amendment = {
        "protocol_version": PROTOCOL_VERSION,
        "supersedes": SUPERSEDED_STAGE5C0_PROTOCOL_SHA256,
        "pre_result_budget_amendment": True,
        "formal_results_existed_at_amendment": False,
        "formal_environment_steps_per_run": FORMAL_STEPS_PER_RUN,
        "total_planned_environment_steps": 3_000_000,
        "epsilon_decay_environment_steps": EPSILON_DECAY_STEPS,
        "pbrs_lock_sha256": pbrs_sha,
        "training_protocol_sha256": protocol_sha,
        "stage5c0_run_id": STAGE5C0_RUN_ID,
        "stage5a0_run_id": prereq.stage5a0_run_id,
        "stage5b0_run_id": prereq.stage5b0_run_id,
        "formal_training_started": False,
    }
    with (art / "protocol_amendment_record.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(amendment, f, sort_keys=False)
    with (art / "protocol_diff_from_stage5c0.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(build_protocol_diff(), f, sort_keys=False)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "pbrs_lock_sha256": pbrs_sha,
        "training_protocol_sha256": protocol_sha,
        "n_formal_run_slots": 30,
        "formal_environment_steps_per_run": FORMAL_STEPS_PER_RUN,
        "total_planned_environment_steps": 3_000_000,
        "formal_training_started": False,
        "supersedes_stage5c0_protocol_sha256": SUPERSEDED_STAGE5C0_PROTOCOL_SHA256,
    }
    (art / "formal_experiment_manifest.json").write_text(
        __import__("json").dumps(manifest, indent=2), encoding="utf-8"
    )
    return {
        "pbrs_lock_sha256": pbrs_sha,
        "training_protocol_sha256": protocol_sha,
        "n_rows": len(rows),
        "artifact_dir": str(art),
        "manifest": manifest,
    }


__all__ = [
    "EPSILON_DECAY_STEPS",
    "FORMAL_CHECKPOINT_STEPS",
    "FORMAL_CONDITIONS",
    "FORMAL_EVALUATION_STEPS",
    "FORMAL_MASTER_SEEDS",
    "FORMAL_STEPS_PER_RUN",
    "PRIMARY_ENDPOINT_STEP",
    "PROTOCOL_VERSION",
    "SUPERSEDED_STAGE5C0_PROTOCOL_SHA256",
    "build_formal_run_matrix_100k",
    "derive_formal_seeds",
    "write_h1_r1_artifact_bundle",
]
