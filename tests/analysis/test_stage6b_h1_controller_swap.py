"""Additional Stage 6B-H1 tests: evaluator guards, swap, manifest helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thesis.analysis.episode_utility_accumulator import (
    collided_ids_from_pairs,
    derive_utility_fields,
)


def test_collided_ids_from_pairs_stable() -> None:
    ids = collided_ids_from_pairs([["B", "A"], ("B_front", "A")])
    assert ids == {"A", "B", "B_front"}


def test_derive_utility_fields_tie() -> None:
    u = {"A": 0.2, "B": 0.2, "B_front": 0.5, "B_rear": 0.8}
    d = derive_utility_fields(u)
    assert d["worst_off_tie"] is True
    assert d["worst_off_stakeholder_ids_json"] == ["A", "B"]
    assert d["minimum_stakeholder_utility"] == pytest.approx(0.2)


def test_controller_swap_missing_not_zero() -> None:
    # Import function from runner module path
    import importlib.util

    repo = Path(__file__).resolve().parents[2]
    path = repo / "experiments/formal/stage6b_h1/scripts/run_stage6b_h1.py"
    spec = importlib.util.spec_from_file_location("run_stage6b_h1", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    episodes = [
        {
            "condition": "mean_pbrs",
            "master_seed": 61001,
            "block_id": "validation_001",
            "assignment": 0,
            "roles": {"A": "mainline", "B": "ramp"},
            "convention": None,
        },
        {
            "condition": "mean_pbrs",
            "master_seed": 61001,
            "block_id": "validation_001",
            "assignment": 1,
            "roles": {"A": "ramp", "B": "mainline"},
            "convention": None,
        },
    ]
    rows = mod.controller_swap_diagnostics(episodes)
    assert rows[0]["D_swap_estimable"] is False
    assert rows[0]["D_swap"] is None


def test_no_learning_curve_in_h1_runner_source() -> None:
    src = (
        Path(__file__).resolve().parents[2]
        / "experiments/formal/stage6b_h1/scripts/run_stage6b_h1.py"
    ).read_text(encoding="utf-8")
    assert "Only one formal endpoint is available" in src
    assert 'fig.savefig' in src or "make_endpoint_figures" in src
    # Must not generate AUC as if multi-checkpoint curves existed.
    assert "trapezoidal_auc(" not in src
