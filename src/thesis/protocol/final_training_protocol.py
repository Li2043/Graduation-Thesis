"""Final training protocol lock, run matrix, and analysis plan (Stage 5C-0)."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import yaml

from thesis.envs.final_observation import OBSERVATION_DIM
from thesis.protocol.final_pbrs_lock import EXPECTED_GAMMA, EXPECTED_LAMBDA_MEAN, EXPECTED_LAMBDA_MIN
from thesis.protocol.prerequisites import Stage5C0Prerequisites

FORMAL_MASTER_SEEDS: tuple[int, ...] = (
    61001,
    61002,
    61003,
    61004,
    61005,
    61006,
    61007,
    61008,
    61009,
    61010,
)
FORMAL_CONDITIONS: tuple[str, ...] = ("baseline", "mean_pbrs", "min_pbrs")
FORMAL_STEPS_PER_RUN = 20_000
FORMAL_CHECKPOINT_STEPS: tuple[int, ...] = (5000, 10000, 15000, 20000)
FORMAL_EVALUATION_STEPS: tuple[int, ...] = (0, 5000, 10000, 15000, 20000)
PRIMARY_ENDPOINT_STEP = 20_000


def derive_formal_seeds(master_seed: int) -> dict[str, int]:
    """Condition-independent seed derivation from master seed S."""
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


def build_formal_run_matrix() -> list[dict[str, Any]]:
    """Exactly 30 paired formal run slots."""
    rows: list[dict[str, Any]] = []
    slot = 0
    for condition in FORMAL_CONDITIONS:
        for master in FORMAL_MASTER_SEEDS:
            derived = derive_formal_seeds(master)
            slot += 1
            rows.append(
                {
                    "slot_id": slot,
                    "condition": condition,
                    "master_seed": master,
                    **{k: v for k, v in derived.items() if k != "master_seed"},
                    "environment_steps": FORMAL_STEPS_PER_RUN,
                    "primary_endpoint_checkpoint": PRIMARY_ENDPOINT_STEP,
                    "status_planned": "PENDING",
                    "terminal_status_allowed": "COMPLETE|FAILED_WITH_REASON",
                }
            )
    if len(rows) != 30:
        raise RuntimeError(f"expected 30 formal slots, got {len(rows)}")
    return rows


def write_formal_run_matrix(path: Path, rows: list[dict[str, Any]] | None = None) -> Path:
    rows = rows if rows is not None else build_formal_run_matrix()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


def build_formal_analysis_plan() -> dict[str, Any]:
    return {
        "statistical_unit": "formal_training_seed",
        "aggregation": "condition × seed × checkpoint before condition-level comparisons",
        "pairing": "same master seeds across all three conditions",
        "pairwise_contrasts": [
            "mean_pbrs - baseline",
            "min_pbrs - baseline",
            "min_pbrs - mean_pbrs",
        ],
        "primary_endpoints_at_step_20000": [
            "evaluation_success_rate",
            "stakeholder_collision_rate",
            "mean_stakeholder_episode_utility",
            "minimum_stakeholder_episode_utility",
            "convention_consistency",
        ],
        "convention_labels": ["mainline_first", "ramp_first", "simultaneous"],
        "convention_consistency_definition": (
            "proportion of successful evaluation episodes following the seed's "
            "modal non-simultaneous convention; missing (not zero) when no "
            "successful non-simultaneous episode exists"
        ),
        "missing_convention_policy": {
            "zero_fill": False,
            "report_missing_count": True,
            "compare_only_complete_paired_seeds": True,
            "include_success_availability_sensitivity": True,
        },
        "secondary_endpoints": [
            "episode_length",
            "learner_A_utility",
            "learner_B_utility",
            "B_front_utility",
            "B_rear_utility",
            "worst_off_stakeholder_identity",
            "discounted_base_return",
            "discounted_learner_reward",
            "collision_type",
            "minimum_bumper_gap",
            "minimum_TTC",
            "hard_braking_rate",
            "background_maximum_braking",
            "mainline_first_frequency",
            "ramp_first_frequency",
            "learning_curve_area_under_the_curve",
        ],
        "per_primary_endpoint_reports": [
            "per_condition_seed_level_values",
            "paired_seed_level_differences",
            "mean_difference",
            "median_difference",
            "95_percent_paired_bootstrap_ci",
            "paired_wilcoxon_signed_rank_when_defined",
            "paired_standardised_effect_size_when_defined",
        ],
        "bootstrap": {
            "resampling_unit": "paired_seed",
            "replicates": 10_000,
            "rng_seed": 91001,
        },
        "multiple_comparisons": {
            "method": "Holm",
            "within": "each primary endpoint across the three pairwise contrasts",
            "report_raw_and_adjusted_p": True,
            "no_significance_filtering_of_reported_effects": True,
        },
        "hypothesis_tests": {
            "sidedness": "two_sided",
            "alpha": 0.05,
            "no_effect_claim_from_p_gt_alpha_alone": True,
        },
        "learning_curves": {
            "checkpoints": list(FORMAL_EVALUATION_STEPS),
            "role": "descriptive_and_secondary",
            "auc_method": "trapezoidal_rule_over_fixed_checkpoints",
            "no_interpolation_of_missing_checkpoints": True,
            "no_smoothing_for_statistical_computation": True,
            "figure_smoothing_visual_only_if_used": True,
        },
        "best_checkpoint_selection": False,
        "primary_formal_endpoint_checkpoint": PRIMARY_ENDPOINT_STEP,
    }


def build_final_training_protocol(
    prereq: Stage5C0Prerequisites,
    *,
    git_commit: str,
    source_hashes: dict[str, str],
    configuration_sha256: str,
    pbrs_lock_sha256: str,
) -> dict[str, Any]:
    n_runs = len(FORMAL_CONDITIONS) * len(FORMAL_MASTER_SEEDS)
    return {
        "lock_type": "final_training_protocol",
        "stage": "stage5c0",
        "conditions": list(FORMAL_CONDITIONS),
        "only_condition_dependent_component": "pbrs_shaping_term",
        "environment": {
            "candidate_id": prereq.candidate_id,
            "observation_dimension": OBSERVATION_DIM,
            "environment_class": "MergeEnvCandidateV3",
            "environment_lock_sha256": prereq.environment_lock_sha256,
            "comfort_lock_sha256": prereq.comfort_lock_sha256,
            "a_comfort": prereq.a_comfort,
            "a_hard": prereq.a_hard,
            "eta_H": prereq.eta_H,
        },
        "pbrs": {
            "lambda_mean": EXPECTED_LAMBDA_MEAN,
            "lambda_min": EXPECTED_LAMBDA_MIN,
            "baseline_lambda": 0.0,
            "gamma": EXPECTED_GAMMA,
            "pbrs_lock_sha256": pbrs_lock_sha256,
            "pbrs_parameters_final": True,
            "selected_using_pilot_performance": False,
        },
        "dqn": {
            "algorithm": "vanilla_independent_dqn",
            "separate_learners": ["A", "B"],
            "shared_weights": False,
            "observation_dimension": OBSERVATION_DIM,
            "action_count": 3,
            "hidden_sizes": [64, 64],
            "activation": "ReLU",
            "output": "3_linear_Q_values",
            "loss": "mean_squared_td_error",
            "gamma": EXPECTED_GAMMA,
            "learning_rate": 0.0005,
            "optimiser": "Adam",
            "batch_size": 64,
            "replay_capacity_per_controller": 20_000,
            "replay_warmup_per_controller": 512,
            "sampling": "uniform_without_replacement_within_batch",
            "update_frequency": "every_environment_policy_step",
            "updates_per_active_controller": 1,
            "target_sync_type": "hard",
            "target_sync_interval_updates": 250,
            "device": "cpu",
            "promoted_unchanged_from_engineering_pilot": True,
            "selected_by_pilot_reward_performance": False,
        },
        "exploration": {
            "epsilon_start": 1.0,
            "epsilon_end": 0.10,
            "epsilon_decay_environment_steps": 4000,
            "epsilon_schedule": "linear",
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
        "training_initial_conditions": {
            "source": "12_retained_calibration_blocks_only",
            "validation_blocks_used_for_training": False,
            "scheduling": [
                "deterministic_seeded_shuffled_cycles",
                "every_cycle_includes_all_12_blocks",
                "both_controller_role_assignments_balanced",
                "schedule_cursor_checkpointed",
                "resume_continues_from_exact_cursor",
            ],
            "block_definitions_modified": False,
        },
        "checkpoint_schedule": {
            "steps": list(FORMAL_CHECKPOINT_STEPS),
            "primary_endpoint": PRIMARY_ENDPOINT_STEP,
            "best_checkpoint_selection": False,
            "retain_all_scheduled_checkpoints": True,
            "emergency_checkpoints_on_controlled_interruption": True,
        },
        "evaluation_schedule": {
            "steps": list(FORMAL_EVALUATION_STEPS),
            "episodes_per_point": 16,
            "validation_blocks": 8,
            "controller_role_assignments": 2,
            "greedy": True,
            "epsilon": 0.0,
            "no_replay_writes": True,
            "no_optimiser_updates": True,
            "no_target_sync": True,
            "no_epsilon_counter_changes": True,
            "no_training_rng_mutation": True,
            "no_network_or_optimiser_mutation": True,
            "results_alter_training": False,
            "results_alter_checkpoint_selection": False,
            "results_alter_run_termination": False,
        },
        "terminal_and_replay_semantics": {
            "learner_safe_exit": {
                "controller_terminal": True,
                "next_observation": None,
                "next_action_mask": None,
                "bootstrap": False,
                "store_exit_transition": True,
                "store_later_transitions": False,
            },
            "collision_or_joint_success": {
                "controller_terminal": True,
                "bootstrap": False,
            },
            "max_step_truncation_without_completion": {
                "controller_terminal": False,
                "successor_required": True,
                "bootstrap": True,
            },
            "placeholder_replay_rows_for_completed_learners": False,
        },
        "failure_and_resume_policy": {
            "infrastructure_failures": [
                "process_interruption",
                "machine_restart",
                "disk_or_checkpoint_io_failure",
                "verified_software_exception_unrelated_to_condition_behaviour",
            ],
            "infrastructure_resume": [
                "resume_from_most_recent_valid_checkpoint",
                "same_condition_and_seed",
                "retain_interruption_log",
                "do_not_restart_with_different_seed",
                "do_not_alter_protocol",
            ],
            "scientific_behavioral_outcomes_are_not_run_failures": True,
            "numerical_invalidity_is_formal_run_failure": [
                "nan_or_infinity",
                "illegal_action",
                "corrupted_replay_semantics",
                "reward_decomposition_failure",
                "lock_mismatch",
            ],
            "silent_omit_failed_run": False,
            "replace_failed_seeds_with_new_seeds": False,
        },
        "run_completion_rule": {
            "exact_environment_steps": FORMAL_STEPS_PER_RUN,
            "all_evaluation_points_exist": True,
            "all_scheduled_checkpoints_exist": True,
            "final_checkpoint_reloads": True,
            "lock_hashes_match": True,
            "no_numerical_integrity_failure": True,
            "required_logs_and_manifests_exist": True,
            "terminal_statuses": ["COMPLETE", "FAILED_WITH_REASON"],
            "all_30_slots_must_have_terminal_status": True,
        },
        "data_retention_plan": [
            "resolved_run_config",
            "source_and_lock_hashes",
            "episode_summaries",
            "evaluation_episode_records",
            "update_summaries",
            "checkpoint_manifests",
            "integrity_logs",
            "final_online_and_target_networks",
            "optimiser_state",
            "replay_metadata",
            "interruption_resume_records",
        ],
        "analysis_plan": build_formal_analysis_plan(),
        "prerequisites": {
            "stage5a0_run_id": prereq.stage5a0_run_id,
            "stage5b0_run_id": prereq.stage5b0_run_id,
            "pilot_comparative_outcomes_used": False,
        },
        "source_git_commit": git_commit,
        "source_hashes": source_hashes,
        "configuration_sha256": configuration_sha256,
        "environment_parameters_final": True,
        "comfort_parameters_final": True,
        "pbrs_parameters_final": True,
        "training_protocol_final": True,
        "pilot_training_started": True,
        "policy_training_started": True,
        "sustained_training_invoked": True,
        "formal_training_started": False,
        "sustained_training_invoked_during_stage5c0": False,
    }


def write_final_training_protocol(path: Path, lock: dict[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(lock, f, sort_keys=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sha_path = path.parent / f"{path.stem}.sha256"
    sha_path.write_text(digest + "\n", encoding="utf-8")
    return digest


def write_formal_analysis_plan(path: Path, plan: dict[str, Any] | None = None) -> Path:
    plan = plan if plan is not None else build_formal_analysis_plan()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(plan, f, sort_keys=False)
    return path


__all__ = [
    "FORMAL_CHECKPOINT_STEPS",
    "FORMAL_CONDITIONS",
    "FORMAL_EVALUATION_STEPS",
    "FORMAL_MASTER_SEEDS",
    "FORMAL_STEPS_PER_RUN",
    "PRIMARY_ENDPOINT_STEP",
    "build_final_training_protocol",
    "build_formal_analysis_plan",
    "build_formal_run_matrix",
    "derive_formal_seeds",
    "write_final_training_protocol",
    "write_formal_analysis_plan",
    "write_formal_run_matrix",
]
