"""Stage 5A-0 pipeline smoke / integrity."""

from __future__ import annotations

from pathlib import Path

from thesis.calibration.final_environment_trace_loader import sha256_file
from thesis.training.final_experiment_runtime import run_condition_suite
from thesis.training.final_lock_loader import (
    EXPECTED_COMFORT_LOCK_SHA256,
    EXPECTED_ENVIRONMENT_LOCK_SHA256,
    load_final_locks,
)
from thesis.training.final_v3_pipeline import (
    ENVIRONMENT_CLASS,
    assert_final_v3_runtime,
)

ENV_LOCK = Path(
    "experiments/pre_impl/stage4a_r1_final_environment_reselection/artifacts/"
    "20260730T003122Z_aee2d425/final_environment_lock.yaml"
)
COMFORT_LOCK = Path(
    "experiments/pre_impl/stage3b_r1_final_environment_comfort_calibration/artifacts/"
    "20260730T005639Z_c6992dd4/final_comfort_parameters.yaml"
)


def test_stage5a0_pipeline_suite_integrity():
    bundle = load_final_locks()
    env = bundle.build_env()
    meta = assert_final_v3_runtime(env, bundle)
    assert meta.environment_class == ENVIRONMENT_CLASS
    assert meta.observation_dimension == 27
    assert meta.policy_training_started is False
    assert meta.sustained_training_invoked is False
    assert meta.pbrs_parameters_final is False

    suite = run_condition_suite(bundle)
    assert suite["invariance"]["max_physical_diff"] == 0.0
    assert suite["invariance"]["max_base_reward_diff"] == 0.0
    assert suite["lambda0_max_reward_diff"] == 0.0
    for name, tele in suite["telescoping"].items():
        assert tele["mean_error"] < 1e-10, name
        assert tele["min_error"] < 1e-10, name
    assert suite["isolated_updates"]
    for upd in suite["isolated_updates"]:
        assert upd["finite_loss"]
        assert upd["target_unchanged_without_sync"]
        assert upd["sustained_training_invoked"] is False

    # Early exit continuation present
    exit_ep = suite["early_exit"]
    a_exit = [
        r
        for r in exit_ep["transitions"]
        if r["controller_id"] == "A" and r["exit_event"]["A"] >= 1.0
    ]
    assert a_exit
    b_after = [
        r
        for r in exit_ep["transitions"]
        if r["controller_id"] == "B" and r["policy_step"] > a_exit[0]["policy_step"]
    ]
    assert b_after

    assert sha256_file(ENV_LOCK) == EXPECTED_ENVIRONMENT_LOCK_SHA256
    assert sha256_file(COMFORT_LOCK) == EXPECTED_COMFORT_LOCK_SHA256
