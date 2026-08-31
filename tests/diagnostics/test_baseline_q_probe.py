"""Q probe helpers."""

import numpy as np

from thesis.diagnostics.stage7a0_trajectory_eval import _q_diagnostics


def test_masked_q_ignores_illegal():
    q = np.array([5.0, 1.0, 4.0])
    mask = np.array([False, True, True])
    d = _q_diagnostics(q, mask)
    assert d["greedy_action"] == 2
    assert abs(d["Q_margin"] - 3.0) < 1e-12
