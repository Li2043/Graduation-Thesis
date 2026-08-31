"""Checkpoint schema / cross-condition resume tests."""

from __future__ import annotations

import pytest

from thesis.agents.dqn_bootstrap import DQNTargetMode
from thesis.formal.formal_config import FormalConfig, FormalDurationConfig, derive_formal_job_seeds
from thesis.formal.formal_trainer import FormalEngineeringError, FormalTrainer
from thesis.pilots.stage7b_a1_checkpoint import (
    assert_resumable_payload,
    validate_resume_compatibility,
)
from thesis.training.final_lock_loader import load_final_locks


def _tiny_trainer(mode: DQNTargetMode, seed: int = 63001) -> FormalTrainer:
    cfg = FormalConfig(
        duration=FormalDurationConfig(
            environment_steps_per_run=8,
            checkpoint_steps=(8,),
            evaluation_steps=(),
            early_stopping=False,
        ),
        allow_test_budget=True,
    )
    return FormalTrainer(
        load_final_locks(),
        condition="baseline",
        master_seed=seed,
        seeds=derive_formal_job_seeds(seed),
        config=cfg,
        protocol_hash="testproto",
        target_mode=mode,
        algorithm_condition=mode.value,
    )


def test_checkpoint_contains_algorithm_mode():
    t = _tiny_trainer(DQNTargetMode.DOUBLE)
    t.run(n_steps=4)
    payload = t.export_checkpoint(step=4)
    assert_resumable_payload(payload)
    assert payload["algorithm_mode"] == "double_dqn"
    assert payload["algorithm_condition"] == "double_dqn"
    assert payload["condition"] == "baseline"
    assert "optimiser" in payload["learners"]["A"]
    assert "replay" in payload["learners"]["A"]
    assert "learner_rng" in payload["learners"]["A"]
    assert "ic_schedule" in payload
    assert "global_rng" in payload


def test_cross_condition_resume_rejected():
    a = _tiny_trainer(DQNTargetMode.VANILLA)
    a.run(n_steps=4)
    payload = a.export_checkpoint(step=4)
    with pytest.raises(ValueError, match="cross-condition"):
        validate_resume_compatibility(
            payload,
            algorithm_condition="double_dqn",
            algorithm_mode="double_dqn",
            protocol_hash="testproto",
        )
    b = _tiny_trainer(DQNTargetMode.DOUBLE)
    with pytest.raises(FormalEngineeringError, match="algorithm mode mismatch"):
        b.import_checkpoint(payload)


def test_stage6_hyperparams_unchanged():
    from thesis.formal.formal_config import FormalDQNConfig, FormalExplorationConfig

    dqn = FormalDQNConfig()
    assert dqn.hidden_sizes == (64, 64)
    assert dqn.learning_rate == 0.0005
    assert dqn.batch_size == 64
    assert dqn.replay_capacity_per_controller == 20_000
    assert dqn.target_sync_interval_updates == 250
    exp = FormalExplorationConfig()
    assert exp.epsilon_decay_environment_steps == 50_000
    assert exp.epsilon_after_decay == 0.10
