"""Baseline-only / no-PBRS guards."""

from __future__ import annotations

import pytest

from thesis.pilots.stage7b_a1_config import assert_stage7b_guards


def test_pbrs_conditions_not_allowed_as_algorithm():
    with pytest.raises(RuntimeError):
        assert_stage7b_guards(
            condition="mean_pbrs",
            reward_shaping_enabled=False,
            shaping_coefficient=0.0,
            master_seed=63001,
            max_steps=300_000,
        )


def test_shaping_forbidden():
    with pytest.raises(RuntimeError):
        assert_stage7b_guards(
            condition="vanilla_dqn",
            reward_shaping_enabled=True,
            shaping_coefficient=0.0,
            master_seed=63001,
            max_steps=300_000,
        )
    with pytest.raises(RuntimeError):
        assert_stage7b_guards(
            condition="double_dqn",
            reward_shaping_enabled=False,
            shaping_coefficient=0.2,
            master_seed=63001,
            max_steps=300_000,
        )
