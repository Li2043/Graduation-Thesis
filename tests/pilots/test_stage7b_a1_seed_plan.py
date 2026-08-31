"""Seed plan tests for Stage 7B-A1."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thesis.pilots.stage7b_a1_config import (
    FORBIDDEN_FORMAL_SEEDS,
    FORBIDDEN_STAGE7A1_SEEDS,
    PILOT_SEEDS,
    assert_stage7b_guards,
)

AUDIT = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "pilots"
    / "stage7b_a1_double_dqn"
    / "manifests"
    / "seed_collision_audit.json"
)


def test_seeds_exactly_63001_63020():
    assert PILOT_SEEDS == tuple(range(63001, 63021))
    assert set(PILOT_SEEDS).isdisjoint(FORBIDDEN_FORMAL_SEEDS)
    assert set(PILOT_SEEDS).isdisjoint(FORBIDDEN_STAGE7A1_SEEDS)


def test_guards_reject_old_seeds():
    with pytest.raises(RuntimeError):
        assert_stage7b_guards(
            condition="vanilla_dqn",
            reward_shaping_enabled=False,
            shaping_coefficient=0.0,
            master_seed=61001,
            max_steps=300_000,
        )
    with pytest.raises(RuntimeError):
        assert_stage7b_guards(
            condition="double_dqn",
            reward_shaping_enabled=False,
            shaping_coefficient=0.0,
            master_seed=62001,
            max_steps=300_000,
        )


def test_seed_collision_audit_pass():
    assert AUDIT.is_file(), "run seed_collision_audit.py before commit"
    data = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert data["status"] == "PASS"
    assert data["collision_seeds"] == []
