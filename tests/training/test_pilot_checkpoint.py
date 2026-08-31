"""Stage 5B-0 — checkpoint content and atomic write."""

from __future__ import annotations

from pathlib import Path

from thesis.training.final_lock_loader import load_final_locks
from thesis.training.pilot_checkpoint import atomic_torch_save, load_checkpoint
from thesis.training.pilot_config import (
    PilotConfig,
    PilotDQNConfig,
    PilotDurationConfig,
    PilotExplorationConfig,
)
from thesis.training.pilot_training_loop import PilotTrainer

REQUIRED_KEYS = {
    "condition",
    "pilot_seed",
    "environment_lock_hash",
    "comfort_lock_hash",
    "pilot_config_hash",
    "env_steps",
    "episode_count",
    "ic_schedule",
    "learners",
    "target_syncs",
    "epsilon",
    "global_rng",
}


def test_checkpoint_round_trip_and_required_fields(tmp_path):
    bundle = load_final_locks()
    cfg = PilotConfig(
        dqn=PilotDQNConfig(
            replay_warmup_per_controller=8,
            batch_size=8,
            target_sync_interval_updates=10,
            replay_capacity_per_controller=2000,
        ),
        duration=PilotDurationConfig(
            environment_steps_per_run=30,
            checkpoint_steps=(15, 30),
            evaluation_steps=(),
        ),
        exploration=PilotExplorationConfig(epsilon_decay_environment_steps=20),
    )
    ckpt_dir = tmp_path / "ckpt"
    t = PilotTrainer(
        bundle,
        condition="baseline",
        pilot_seed=51001,
        config=cfg,
        checkpoint_dir=ckpt_dir,
        write_traces=False,
    )
    t.run(n_steps=30)
    path = ckpt_dir / "ckpt_step_00030.pt"
    assert path.is_file()
    # no leftover tmp
    assert not list(ckpt_dir.glob("*.tmp"))
    payload = load_checkpoint(path)
    for k in REQUIRED_KEYS:
        assert k in payload
    assert "A" in payload["learners"] and "B" in payload["learners"]
    for aid in ("A", "B"):
        assert "online" in payload["learners"][aid]
        assert "target" in payload["learners"][aid]
        assert "optimiser" in payload["learners"][aid]
        assert "replay" in payload["learners"][aid]
        assert "update_count" in payload["learners"][aid]

    t2 = PilotTrainer(
        bundle,
        condition="baseline",
        pilot_seed=51001,
        config=cfg,
        checkpoint_dir=tmp_path / "ckpt2",
        write_traces=False,
    )
    t2.import_checkpoint(payload)
    assert t2.env_steps == 30
    assert t2.learners["A"]._update_count == t.learners["A"]._update_count


def test_atomic_write_path(tmp_path):
    path = tmp_path / "x.pt"
    atomic_torch_save(path, {"ok": True})
    assert path.is_file()
    assert not path.with_suffix(".pt.tmp").exists()
    assert load_checkpoint(path)["ok"] is True
