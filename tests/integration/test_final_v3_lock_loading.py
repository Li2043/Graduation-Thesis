"""Stage 5A-0 — final lock loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from thesis.calibration.final_environment_trace_loader import sha256_file
from thesis.training.final_lock_loader import (
    EXPECTED_COMFORT_LOCK_SHA256,
    EXPECTED_ENVIRONMENT_LOCK_SHA256,
    FinalLockBlockedError,
    load_final_locks,
)

ENV_LOCK = Path(
    "experiments/pre_impl/stage4a_r1_final_environment_reselection/artifacts/"
    "20260730T003122Z_aee2d425/final_environment_lock.yaml"
)
COMFORT_LOCK = Path(
    "experiments/pre_impl/stage3b_r1_final_environment_comfort_calibration/artifacts/"
    "20260730T005639Z_c6992dd4/final_comfort_parameters.yaml"
)


def test_environment_and_comfort_lock_hashes():
    assert sha256_file(ENV_LOCK) == EXPECTED_ENVIRONMENT_LOCK_SHA256
    assert sha256_file(COMFORT_LOCK) == EXPECTED_COMFORT_LOCK_SHA256


def test_g1_i1_and_comfort_tuple_from_locks():
    bundle = load_final_locks()
    assert bundle.candidate_id == "G1-I1"
    assert bundle.observation_dimension == 27
    assert float(bundle.environment.lock["physics_dt"]) == 0.05
    assert float(bundle.environment.lock["policy_interval"]) == 0.20
    assert int(bundle.environment.lock["physics_substeps_per_action"]) == 4
    assert bundle.comfort.a_comfort == 1.5
    assert bundle.comfort.a_hard == 3.5
    assert bundle.comfort.eta_H == 0.015
    assert bundle.environment.lock.get("environment_parameters_final") is True
    assert bundle.comfort.comfort_parameters_final is True
    assert bundle.comfort.policy_training_started is False


def test_blocked_on_wrong_comfort_hash(tmp_path):
    bad = tmp_path / "bad_comfort.yaml"
    bad.write_text("a_comfort: 1.5\na_hard: 3.5\neta_H: 0.015\neta_hard_brake: 0.015\n", encoding="utf-8")
    with pytest.raises(FinalLockBlockedError, match="comfort lock SHA-256"):
        load_final_locks(comfort_lock_path=bad)
