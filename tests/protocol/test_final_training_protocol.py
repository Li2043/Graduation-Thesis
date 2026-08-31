"""Stage 5C-0 — final training protocol lock tests."""

from __future__ import annotations

from thesis.protocol.final_training_protocol import (
    FORMAL_CHECKPOINT_STEPS,
    FORMAL_EVALUATION_STEPS,
    FORMAL_STEPS_PER_RUN,
    PRIMARY_ENDPOINT_STEP,
    build_final_training_protocol,
)
from thesis.protocol.prerequisites import verify_stage5c0_prerequisites


def test_training_protocol_frozen_settings():
    prereq = verify_stage5c0_prerequisites()
    lock = build_final_training_protocol(
        prereq,
        git_commit="test",
        source_hashes={},
        configuration_sha256="abc",
        pbrs_lock_sha256="deadbeef",
    )
    assert lock["dqn"]["hidden_sizes"] == [64, 64]
    assert lock["dqn"]["observation_dimension"] == 27
    assert lock["dqn"]["learning_rate"] == 0.0005
    assert lock["dqn"]["batch_size"] == 64
    assert lock["dqn"]["replay_capacity_per_controller"] == 20_000
    assert lock["dqn"]["replay_warmup_per_controller"] == 512
    assert lock["dqn"]["target_sync_interval_updates"] == 250
    assert lock["dqn"]["selected_by_pilot_reward_performance"] is False
    assert lock["exploration"]["epsilon_start"] == 1.0
    assert lock["exploration"]["epsilon_end"] == 0.10
    assert lock["exploration"]["epsilon_decay_environment_steps"] == 4000
    assert lock["training_budget"]["formal_environment_steps_per_run"] == FORMAL_STEPS_PER_RUN
    assert lock["training_budget"]["early_stopping"] is False
    assert lock["training_budget"]["n_formal_runs"] == 30
    assert lock["training_budget"]["total_planned_environment_steps"] == 600_000
    assert lock["checkpoint_schedule"]["steps"] == list(FORMAL_CHECKPOINT_STEPS)
    assert lock["checkpoint_schedule"]["primary_endpoint"] == PRIMARY_ENDPOINT_STEP
    assert lock["checkpoint_schedule"]["best_checkpoint_selection"] is False
    assert lock["evaluation_schedule"]["steps"] == list(FORMAL_EVALUATION_STEPS)
    assert lock["evaluation_schedule"]["episodes_per_point"] == 16
    assert lock["training_initial_conditions"]["validation_blocks_used_for_training"] is False
    assert lock["failure_and_resume_policy"]["replace_failed_seeds_with_new_seeds"] is False
    assert lock["training_protocol_final"] is True
    assert lock["pbrs_parameters_final"] is True
    assert lock["formal_training_started"] is False
    assert lock["sustained_training_invoked_during_stage5c0"] is False
