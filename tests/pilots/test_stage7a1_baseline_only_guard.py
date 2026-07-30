"""Baseline-only guards for Stage 7A-1."""

from __future__ import annotations

import pytest

from thesis.pilots.stage7a1_config import assert_baseline_only


def test_baseline_ok():
    assert_baseline_only(
        condition="baseline", reward_shaping_enabled=False, shaping_coefficient=0.0
    )


@pytest.mark.parametrize("cond", ["mean_pbrs", "min_pbrs"])
def test_pbrs_conditions_fail(cond):
    with pytest.raises(RuntimeError):
        assert_baseline_only(
            condition=cond, reward_shaping_enabled=False, shaping_coefficient=0.0
        )


def test_shaping_enabled_fails():
    with pytest.raises(RuntimeError):
        assert_baseline_only(
            condition="baseline", reward_shaping_enabled=True, shaping_coefficient=0.0
        )


def test_nonzero_lambda_fails():
    with pytest.raises(RuntimeError):
        assert_baseline_only(
            condition="baseline", reward_shaping_enabled=False, shaping_coefficient=0.2
        )
