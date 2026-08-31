"""Stage 5B-0 integration pipeline smoke (reduced; full pilot via runner)."""

from __future__ import annotations

from pathlib import Path

from thesis.calibration.final_environment_trace_loader import sha256_file
from thesis.training.final_lock_loader import (
    EXPECTED_COMFORT_LOCK_SHA256,
    EXPECTED_ENVIRONMENT_LOCK_SHA256,
    load_final_locks,
)
from thesis.training.pilot_config import (
    PILOT_CONDITIONS,
    PILOT_SEEDS,
    PilotConfig,
    PilotDQNConfig,
    PilotDurationConfig,
    PilotExplorationConfig,
)
from thesis.training.pilot_training_loop import PilotTrainer

ENV_LOCK = Path(
    "experiments/pre_impl/stage4a_r1_final_environment_reselection/artifacts/"
    "20260730T003122Z_aee2d425/final_environment_lock.yaml"
)
COMFORT_LOCK = Path(
    "experiments/pre_impl/stage3b_r1_final_environment_comfort_calibration/artifacts/"
    "20260730T005639Z_c6992dd4/final_comfort_parameters.yaml"
)


def test_stage5b0_reduced_six_run_matrix(tmp_path):
    assert sha256_file(ENV_LOCK) == EXPECTED_ENVIRONMENT_LOCK_SHA256
    assert sha256_file(COMFORT_LOCK) == EXPECTED_COMFORT_LOCK_SHA256
    bundle = load_final_locks()
    cfg = PilotConfig(
        dqn=PilotDQNConfig(
            replay_warmup_per_controller=8,
            batch_size=8,
            target_sync_interval_updates=10,
            replay_capacity_per_controller=2000,
        ),
        duration=PilotDurationConfig(
            environment_steps_per_run=25,
            checkpoint_steps=(25,),
            evaluation_steps=(0, 25),
        ),
        exploration=PilotExplorationConfig(epsilon_decay_environment_steps=15),
    )
    cfg.validate()
    assert len(PILOT_CONDITIONS) == 3
    assert len(PILOT_SEEDS) == 2
    completed = 0
    for cond in PILOT_CONDITIONS:
        for seed in PILOT_SEEDS:
            t = PilotTrainer(
                bundle,
                condition=cond,
                pilot_seed=seed,
                config=cfg,
                checkpoint_dir=tmp_path / f"{cond}_{seed}",
                write_traces=False,
            )
            t.run()
            assert t.env_steps == 25
            assert t.pilot_training_started is True
            assert t.policy_training_started is True
            assert t.sustained_training_invoked is True
            completed += 1
    assert completed == 6
    assert cfg.pbrs.pbrs_parameters_final is False
    assert cfg.training_protocol_final is False
    assert cfg.formal_training_started is False
    assert sha256_file(ENV_LOCK) == EXPECTED_ENVIRONMENT_LOCK_SHA256
    assert sha256_file(COMFORT_LOCK) == EXPECTED_COMFORT_LOCK_SHA256
