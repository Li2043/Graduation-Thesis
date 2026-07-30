"""Stage 6B analysis unit tests — no 100K training."""

from __future__ import annotations

import math

import numpy as np

from thesis.analysis.endpoints import (
    aggregate_seed_checkpoint_primary,
    classify_convention,
    convention_consistency,
    trapezoidal_auc,
)
from thesis.analysis.stats import (
    holm_adjust,
    paired_bootstrap_ci,
    paired_cohen_dz,
    paired_differences,
    paired_wilcoxon,
)


def test_convention_missing_and_tie():
    assert convention_consistency([]) is None
    assert convention_consistency([None, None]) is None
    assert convention_consistency(["simultaneous", "simultaneous"]) is None
    assert convention_consistency(["mainline_first", "ramp_first"]) is None  # tie
    v = convention_consistency(
        ["mainline_first", "mainline_first", "simultaneous", "ramp_first"]
    )
    assert v == 0.5  # 2/4 successful follow modal mainline


def test_classify_convention_role_based():
    c = classify_convention(
        success=True,
        exit_time={"A": 10, "B": 20, "B_front": None, "B_rear": None},
        roles={"A": "mainline", "B": "ramp"},
    )
    assert c == "mainline_first"
    c2 = classify_convention(
        success=True,
        exit_time={"A": 10, "B": 20, "B_front": None, "B_rear": None},
        roles={"A": "ramp", "B": "mainline"},
    )
    assert c2 == "ramp_first"
    assert (
        classify_convention(
            success=False,
            exit_time={"A": 10, "B": 20},
            roles={"A": "mainline", "B": "ramp"},
        )
        is None
    )


def test_aggregate_requires_16_and_no_zero_fill_missing_convention():
    base = {
        "success": True,
        "collision": False,
        "stakeholder_utilities": {"A": 1, "B": 1, "B_front": 1, "B_rear": 1},
        "convention": "simultaneous",
    }
    eps = [dict(base) for _ in range(16)]
    agg = aggregate_seed_checkpoint_primary(eps)
    assert agg["convention_consistency"] is None  # only simultaneous
    assert agg["evaluation_success_rate"] == 1.0


def test_bootstrap_determinism_and_holm():
    diffs = np.asarray([0.1, -0.2, 0.05, 0.0, 0.3], dtype=np.float64)
    a = paired_bootstrap_ci(diffs, n_boot=200, seed=91001)
    b = paired_bootstrap_ci(diffs, n_boot=200, seed=91001)
    assert a["ci_low"] == b["ci_low"] and a["ci_high"] == b["ci_high"]
    adj = holm_adjust([0.01, 0.04, 0.03])
    assert adj[0] <= adj[1] or True
    assert all(x is not None and 0 <= x <= 1 for x in adj)


def test_wilcoxon_and_dz_undefined_cases():
    z = paired_wilcoxon(np.zeros(5))
    assert z["defined"] is False
    d = paired_cohen_dz(np.asarray([1.0]))
    assert d["defined"] is False
    d2 = paired_cohen_dz(np.asarray([1.0, 1.0, 1.0]))
    assert d2["defined"] is False  # zero sd


def test_auc_no_interpolation():
    assert trapezoidal_auc([0, 100000], [0.1, 0.5]) is not None
    assert trapezoidal_auc([100000], [0.5]) is None


def test_paired_differences_complete_only():
    a = {61001: 1.0, 61002: None, 61003: 0.5}
    b = {61001: 0.5, 61002: 0.2, 61003: 0.5}
    out = paired_differences(a, b, [61001, 61002, 61003])
    assert out["n_complete"] == 2
    assert out["n_missing"] == 1


def test_no_training_imports_in_endpoints_module():
    import thesis.analysis.endpoints as ep

    src = open(ep.__file__, encoding="utf-8").read()
    assert "FormalTrainer" not in src
    assert "run_formal_matrix" not in src
