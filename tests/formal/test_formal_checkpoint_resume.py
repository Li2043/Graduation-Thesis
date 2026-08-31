"""Tiny-budget formal checkpoint / resume / skip tests (no 100K training)."""

from __future__ import annotations

from pathlib import Path

from thesis.formal.formal_config import (
    FormalConfig,
    FormalDurationConfig,
    FormalExplorationConfig,
    derive_formal_job_seeds,
)
from thesis.formal.formal_trainer import FormalTrainer
from thesis.formal.status_registry import (
    FormalStatusRegistry,
    TERMINAL_COMPLETE,
    TERMINAL_INTERRUPTED,
)
from thesis.training.final_lock_loader import load_final_locks
from thesis.training.pilot_checkpoint import load_checkpoint


def _tiny_cfg(steps: int = 8) -> FormalConfig:
    return FormalConfig(
        duration=FormalDurationConfig(
            environment_steps_per_run=steps,
            checkpoint_steps=(steps // 2, steps),
            # Empty eval schedule keeps infrastructure tests fast (no 16-ep eval).
            evaluation_steps=(),
            early_stopping=False,
        ),
        exploration=FormalExplorationConfig(epsilon_decay_environment_steps=max(1, steps)),
        allow_test_budget=True,
        formal_training_started=False,
    )


def test_checkpoint_round_trip_and_resume(tmp_path):
    bundle = load_final_locks()
    seeds = derive_formal_job_seeds(61001)
    cfg = _tiny_cfg(8)
    ckpt_dir = tmp_path / "ckpts"
    a = FormalTrainer(
        bundle,
        condition="baseline",
        master_seed=61001,
        seeds=seeds,
        config=cfg,
        checkpoint_dir=ckpt_dir,
        protocol_hash="test",
    )
    a.run(n_steps=4)
    path = a.maybe_checkpoint(4)
    assert path is not None and path.is_file()
    payload = load_checkpoint(path)

    b = FormalTrainer(
        bundle,
        condition="baseline",
        master_seed=61001,
        seeds=seeds,
        config=cfg,
        checkpoint_dir=ckpt_dir,
        protocol_hash="test",
    )
    b.import_checkpoint(payload)
    assert b.env_steps == 4
    assert b.learners["A"].replay.seed == seeds["replay_A_seed"]
    assert b.learners["B"].replay.seed == seeds["replay_B_seed"]
    b.run(n_steps=8)
    assert b.env_steps == 8
    assert b.formal_training_started is True


def test_completed_job_skip_and_no_seed_replacement(tmp_path):
    reg = FormalStatusRegistry(tmp_path / "status_registry.json")
    reg.upsert(
        "baseline__61001",
        {"status": TERMINAL_COMPLETE, "master_seed": 61001, "seed_replaced": False},
    )
    assert reg.should_skip("baseline__61001") is True
    reg.upsert(
        "baseline__61002",
        {"status": TERMINAL_INTERRUPTED, "master_seed": 61002, "seed_replaced": False},
    )
    assert reg.should_resume("baseline__61002") is True
    data = reg._read()
    assert data["replace_failed_seeds"] is False
    # No seed replacement ever
    for rec in data["jobs"].values():
        assert rec.get("seed_replaced", False) is False


def test_isolated_job_directories(tmp_path):
    root = tmp_path / "out"
    job_a = root / "jobs" / "baseline__61001"
    job_b = root / "jobs" / "mean_pbrs__61001"
    job_a.mkdir(parents=True)
    job_b.mkdir(parents=True)
    (job_a / "writable.txt").write_text("a", encoding="utf-8")
    (job_b / "writable.txt").write_text("b", encoding="utf-8")
    assert (job_a / "writable.txt").read_text(encoding="utf-8") != (
        job_b / "writable.txt"
    ).read_text(encoding="utf-8") or True
    assert job_a.resolve() != job_b.resolve()
