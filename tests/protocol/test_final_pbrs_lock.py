"""Stage 5C-0 — final PBRS lock tests."""

from __future__ import annotations

from pathlib import Path

from thesis.calibration.final_environment_trace_loader import sha256_file
from thesis.protocol.final_pbrs_lock import (
    EXPECTED_LAMBDA_MEAN,
    EXPECTED_LAMBDA_MIN,
    build_final_pbrs_lock,
    write_final_pbrs_lock,
)
from thesis.protocol.prerequisites import verify_stage5c0_prerequisites
from thesis.training.final_lock_loader import (
    EXPECTED_COMFORT_LOCK_SHA256,
    EXPECTED_ENVIRONMENT_LOCK_SHA256,
)

ENV_LOCK = Path(
    "experiments/pre_impl/stage4a_r1_final_environment_reselection/artifacts/"
    "20260730T003122Z_aee2d425/final_environment_lock.yaml"
)
COMFORT_LOCK = Path(
    "experiments/pre_impl/stage3b_r1_final_environment_comfort_calibration/artifacts/"
    "20260730T005639Z_c6992dd4/final_comfort_parameters.yaml"
)


def test_input_lock_hashes():
    assert sha256_file(ENV_LOCK) == EXPECTED_ENVIRONMENT_LOCK_SHA256
    assert sha256_file(COMFORT_LOCK) == EXPECTED_COMFORT_LOCK_SHA256


def test_pbrs_lock_values_and_flags(tmp_path):
    prereq = verify_stage5c0_prerequisites()
    lock = build_final_pbrs_lock(
        prereq,
        git_commit="test",
        source_hashes={},
        configuration_sha256="abc",
    )
    assert lock["lambda_mean"] == EXPECTED_LAMBDA_MEAN == 0.2
    assert lock["lambda_min"] == EXPECTED_LAMBDA_MIN == 0.2
    assert lock["conditions"]["baseline"]["lambda"] == 0.0
    assert lock["equal_scales_across_shaped_conditions"] is True
    assert lock["pilot_comparative_outcomes_used_for_selection"] is False
    assert lock["pilot_behavioral_observations_read"] is False
    assert lock["pbrs_parameters_final"] is True
    assert lock["formal_training_started"] is False
    path = tmp_path / "final_pbrs_parameters.yaml"
    digest = write_final_pbrs_lock(path, lock)
    assert (
        tmp_path / "final_pbrs_parameters.sha256"
    ).read_text(encoding="utf-8").strip() == digest
    assert sha256_file(path) == digest
