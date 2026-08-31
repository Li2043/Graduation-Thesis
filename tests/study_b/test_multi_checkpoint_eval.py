"""Unit tests for multi_checkpoint_eval.py's classify_failure_type --
VDN_Conditional_Amendment_Protocol.md sec 5's Type A/B/C/D taxonomy,
exercised with synthetic checkpoint metrics (no training/eval needed)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "pilots" / "study_b_fairness_mappo" / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


multi_checkpoint_eval = _load_script("multi_checkpoint_eval")
classify_failure_type = multi_checkpoint_eval.classify_failure_type


def _metrics(*rows):
    """``rows``: (step, completion, collision, timeout) tuples."""
    return [
        {"step": s, "completion_rate": c, "collision_rate": col, "timeout_rate": t}
        for s, c, col, t in rows
    ]


def test_qualifying_final_checkpoint_returns_none():
    metrics = _metrics((200_000, 0.3, 0.1, 0.6), (800_000, 0.95, 0.02, 0.03))
    assert classify_failure_type(metrics) is None


def test_type_a_never_learns():
    metrics = _metrics(
        (200_000, 0.03, 0.5, 0.47), (400_000, 0.01, 0.5, 0.49),
        (600_000, 0.00, 0.5, 0.50), (800_000, 0.00, 0.5, 0.50),
    )
    assert classify_failure_type(metrics) == "A"


def test_type_b_learns_then_collapses():
    metrics = _metrics(
        (200_000, 0.25, 0.3, 0.45), (400_000, 0.72, 0.2, 0.08),
        (600_000, 0.84, 0.1, 0.06), (800_000, 0.05, 0.9, 0.05),
    )
    assert classify_failure_type(metrics) == "B"


def test_type_c_frozen_timeout_attractor():
    metrics = _metrics(
        (200_000, 0.0, 0.0, 1.0), (400_000, 0.0, 0.0, 1.0),
        (600_000, 0.0, 0.0, 1.0), (800_000, 0.0, 0.0, 1.0),
    )
    assert classify_failure_type(metrics) == "C"


def test_type_d_aggressive_collision_attractor():
    metrics = _metrics(
        (200_000, 0.05, 0.9, 0.05), (400_000, 0.05, 0.92, 0.03),
        (600_000, 0.05, 0.95, 0.0), (800_000, 0.05, 0.97, 0.0),
    )
    assert classify_failure_type(metrics) == "D"


def test_ambiguous_pattern_reports_mixed():
    metrics = _metrics(
        (200_000, 0.2, 0.3, 0.5), (400_000, 0.2, 0.35, 0.45),
        (600_000, 0.2, 0.3, 0.5), (800_000, 0.2, 0.3, 0.5),
    )
    assert classify_failure_type(metrics) == "mixed"


def test_rejects_empty_metrics():
    import pytest

    with pytest.raises(ValueError):
        classify_failure_type([])
