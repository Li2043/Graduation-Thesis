"""Stage 5B-0 — pilot config freeze tests."""

from __future__ import annotations

from pathlib import Path

from thesis.calibration.final_environment_trace_loader import sha256_file
from thesis.training.final_lock_loader import (
    EXPECTED_COMFORT_LOCK_SHA256,
    EXPECTED_ENVIRONMENT_LOCK_SHA256,
)
from thesis.training.pilot_config import (
    PILOT_CONDITIONS,
    PILOT_SEEDS,
    PilotConfig,
    derive_run_seeds,
    epsilon_at_step,
)

ENV_LOCK = Path(
    "experiments/pre_impl/stage4a_r1_final_environment_reselection/artifacts/"
    "20260730T003122Z_aee2d425/final_environment_lock.yaml"
)
COMFORT_LOCK = Path(
    "experiments/pre_impl/stage3b_r1_final_environment_comfort_calibration/artifacts/"
    "20260730T005639Z_c6992dd4/final_comfort_parameters.yaml"
)


def test_lock_hashes_and_frozen_config():
    assert sha256_file(ENV_LOCK) == EXPECTED_ENVIRONMENT_LOCK_SHA256
    assert sha256_file(COMFORT_LOCK) == EXPECTED_COMFORT_LOCK_SHA256
    cfg = PilotConfig()
    cfg.validate()
    assert cfg.conditions == PILOT_CONDITIONS
    assert cfg.pilot_seeds == PILOT_SEEDS
    assert len(cfg.conditions) * len(cfg.pilot_seeds) == 6
    h1 = cfg.sha256()
    h2 = PilotConfig().sha256()
    assert h1 == h2
    assert len(h1) == 64
    assert cfg.pbrs.pbrs_parameters_final is False
    assert cfg.training_protocol_final is False
    assert cfg.formal_training_started is False


def test_seed_derivation_independent_of_condition_name():
    s1 = derive_run_seeds(51001)
    s2 = derive_run_seeds(51001)
    assert s1 == s2
    assert s1["learner_A"] != s1["learner_B"]


def test_epsilon_schedule_boundaries():
    cfg = PilotConfig().exploration
    assert epsilon_at_step(0, cfg) == 1.0
    assert epsilon_at_step(cfg.epsilon_decay_environment_steps, cfg) == 0.10
    assert epsilon_at_step(cfg.epsilon_decay_environment_steps + 100, cfg) == 0.10
    mid = epsilon_at_step(2000, cfg)
    assert 0.10 < mid < 1.0
