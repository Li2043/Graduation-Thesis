"""Checkpoint schema tests."""

from __future__ import annotations

from thesis.pilots.stage7a1_checkpoint import checkpoint_contains_flags
from thesis.pilots.stage7a1_runner import make_trainer


def test_full_checkpoint_schema_fields(tmp_path):
    t = make_trainer(master_seed=62001, protocol_hash="t", checkpoint_dir=tmp_path)
    t.run(n_steps=4)
    payload = t.export_checkpoint(step=4)
    flags = checkpoint_contains_flags(payload)
    assert flags["contains_optimizer"]
    assert flags["contains_replay"]
    assert flags["contains_rng"]
    assert flags["contains_schedule_state"]
    assert flags["resumable"]
    assert "global_rng" in payload
    assert "ic_schedule" in payload
    la = payload["learners"]["A"]
    assert "optimiser" in la
    assert "replay" in la
    assert "learner_rng" in la
    assert "online" in la and "target" in la
