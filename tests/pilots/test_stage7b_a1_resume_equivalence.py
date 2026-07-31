"""Tiny resume equivalence for Stage 7B-A1 Double DQN path."""

from __future__ import annotations

import numpy as np

from thesis.agents.dqn_bootstrap import DQNTargetMode
from thesis.formal.formal_config import FormalConfig, FormalDurationConfig, derive_formal_job_seeds
from thesis.formal.formal_trainer import FormalTrainer
from thesis.pilots.stage7b_a1_checkpoint import atomic_hashed_torch_save
from thesis.training.final_lock_loader import load_final_locks
from thesis.training.pilot_checkpoint import load_checkpoint


def test_double_dqn_resume_equivalence(tmp_path):
    bundle = load_final_locks()
    seeds = derive_formal_job_seeds(63001)
    cfg = FormalConfig(
        duration=FormalDurationConfig(
            environment_steps_per_run=40,
            checkpoint_steps=(20, 40),
            evaluation_steps=(),
            early_stopping=False,
        ),
        allow_test_budget=True,
    )

    def make():
        return FormalTrainer(
            bundle,
            condition="baseline",
            master_seed=63001,
            seeds=seeds,
            config=cfg,
            protocol_hash="p",
            target_mode=DQNTargetMode.DOUBLE,
            algorithm_condition="double_dqn",
        )

    a = make()
    a.run(n_steps=40)
    b = make()
    while b.env_steps < 20:
        b.step_once()
    ckpt = tmp_path / "ckpt_step_20_full.pt"
    atomic_hashed_torch_save(ckpt, b.export_checkpoint(step=20))
    b2 = make()
    b2.import_checkpoint(load_checkpoint(ckpt))
    while b2.env_steps < 40:
        b2.step_once()
    for aid in ("A", "B"):
        for net in ("online", "target"):
            va = a.learners[aid].parameter_vector(network=net)
            vb = b2.learners[aid].parameter_vector(network=net)
            assert np.max(np.abs(va - vb)) == 0.0
    assert a.env_steps == b2.env_steps == 40
    assert a.schedule.export_state() == b2.schedule.export_state()
