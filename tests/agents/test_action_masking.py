"""Tests for action masking (selection + legal indices)."""

from __future__ import annotations

import numpy as np
import pytest

from thesis.agents.action_masking import (
    masked_argmax,
    masked_max_q,
    masked_random_action,
    role_action_mask,
    validate_action_mask,
)


def test_03_legal_greedy_action():
    # Illegal action 1 has the largest overall Q
    q = np.array([1.0, 100.0, 5.0])
    mask = np.array([True, False, True])
    assert masked_argmax(q, mask) == 2


def test_04_legal_epsilon_exploration():
    mask = np.array([True, False, True])
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = masked_random_action(mask, 3, rng)
        assert a in (0, 2)


def test_05_all_false_mask():
    with pytest.raises(ValueError, match="all-False"):
        validate_action_mask([False, False, False], 3)


def test_06_mask_shape_mismatch():
    with pytest.raises(ValueError, match="length"):
        validate_action_mask([True, True], 3)


def test_27_role_based_masks():
    m_main = role_action_mask("mainline")
    m_ramp = role_action_mask("ramp")
    assert m_main.tolist() == m_ramp.tolist() == [True, True, True]
    # Same role semantics regardless of calling context (A vs B)
    assert role_action_mask("mainline").tolist() == role_action_mask("mainline").tolist()


def test_masked_max_excludes_illegal():
    assert masked_max_q([2.0, 100.0, 5.0], [True, False, True]) == pytest.approx(5.0)
