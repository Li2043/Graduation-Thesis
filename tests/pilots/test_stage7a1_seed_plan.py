"""Seed plan tests for Stage 7A-1."""

from __future__ import annotations

from thesis.formal.formal_config import derive_formal_job_seeds
from thesis.pilots.stage7a1_config import FORBIDDEN_FORMAL_SEEDS, PILOT_SEEDS
from thesis.pilots.stage7a1_runner import make_trainer


def test_seeds_exactly_62001_62020():
    assert PILOT_SEEDS == tuple(range(62001, 62021))
    assert len(PILOT_SEEDS) == 20


def test_old_formal_seeds_excluded():
    assert set(PILOT_SEEDS).isdisjoint(set(FORBIDDEN_FORMAL_SEEDS))
    for s in FORBIDDEN_FORMAL_SEEDS:
        assert s not in PILOT_SEEDS


def test_seed_derivation_ignores_condition_name():
    d = derive_formal_job_seeds(62001)
    assert d["environment_seed"] == 62001
    assert d["learner_A_seed"] == 162001
    assert d["learner_B_seed"] == 262001
    assert d["replay_A_seed"] == 362001
    assert d["replay_B_seed"] == 462001
    assert d["evaluation_seed"] == 562001
    assert d["schedule_seed"] == 662001


def test_make_trainer_rejects_formal_seed():
    import pytest

    with pytest.raises(RuntimeError):
        make_trainer(master_seed=61001, protocol_hash="x", checkpoint_dir=None)
