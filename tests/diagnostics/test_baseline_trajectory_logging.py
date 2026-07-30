"""Trajectory logging invariants."""

from thesis.diagnostics.stage7a0_trajectory_eval import _joint_category, _q_diagnostics
import numpy as np


def test_q_margin_definition():
    q = np.array([1.0, 3.0, 2.0])
    mask = np.array([True, True, True])
    d = _q_diagnostics(q, mask)
    assert d["greedy_action"] == 1
    assert abs(d["Q_margin"] - 1.0) < 1e-12


def test_joint_category_inactive():
    assert _joint_category(0, 1, False, True) == "inactive-active"
    assert _joint_category(1, 0, True, False) == "active-inactive"
