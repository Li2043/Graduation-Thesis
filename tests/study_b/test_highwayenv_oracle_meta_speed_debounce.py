"""Regression test for the 2026-08-16 CONTROL_AUTHORITY amendment's
oracle-side fix: without debouncing, driving a ``meta_speed`` vehicle
directly from the oracle's raw per-step decision causes unbounded
``target_speed`` windup (confirmed to run to -108 m/s during one
sustained yield in the pre-fix investigation). This locks in both the
windup failure mode (so a future refactor can't silently reintroduce it)
and the fix's effectiveness end-to-end against the real Q bank."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "pilots" / "study_b_fairness_mappo" / "scripts"))

from run_oracle_controller_highwayenv import run_oracle_controller_highwayenv  # noqa: E402

_Q_BANK = Path(__file__).resolve().parents[2] / "experiments" / "pilots" / "study_b_fairness_mappo" / "scenario_banks" / "Q.json"


def _skip_if_no_bank():
    if not _Q_BANK.exists():
        pytest.skip("requires the frozen Q scenario bank on disk")


def test_meta_speed_oracle_reaches_strong_pass_with_debounce_and_lateral_check():
    _skip_if_no_bank()
    rows = run_oracle_controller_highwayenv(scenario_bank=_Q_BANK, action_representation="meta_speed")
    n = len(rows)
    completion_rate = sum(r["completion"] for r in rows) / n
    collision_rate = sum(r["collision"] for r in rows) / n
    timeout_rate = sum(r["timeout"] for r in rows) / n
    assert completion_rate >= 0.98, f"completion_rate={completion_rate:.3f}"
    assert collision_rate <= 0.01, f"collision_rate={collision_rate:.3f}"
    assert timeout_rate <= 0.01, f"timeout_rate={timeout_rate:.3f}"


def test_direct_accel_oracle_unaffected_by_meta_speed_only_changes():
    _skip_if_no_bank()
    rows = run_oracle_controller_highwayenv(scenario_bank=_Q_BANK, action_representation="direct_accel")
    n = len(rows)
    completion_rate = sum(r["completion"] for r in rows) / n
    assert completion_rate == pytest.approx(1.0)


def test_debounce_prevents_target_speed_windup_directly():
    """Isolated unit check of the windup fix itself, independent of the
    full oracle/env rollout: repeatedly debouncing a BRAKE decision while
    already leaning that direction must yield HOLD, not a further nudge."""
    import sys as _sys

    sys_path_entry = str(Path(__file__).resolve().parents[2] / "experiments" / "pilots" / "study_b_fairness_mappo" / "scripts")
    if sys_path_entry not in _sys.path:
        _sys.path.insert(0, sys_path_entry)
    from run_oracle_controller_highwayenv import _debounce_meta_speed_action
    from thesis.study_b.oracle_controller import ACCELERATE, DECELERATE, MAINTAIN

    class _FakeVehicle:
        def __init__(self, speed, target_speed):
            self.speed = speed
            self.target_speed = target_speed

    # Already leaning down (target < speed) -> repeated BRAKE decisions debounce to HOLD.
    v = _FakeVehicle(speed=20.0, target_speed=15.0)
    assert _debounce_meta_speed_action(DECELERATE, v) == MAINTAIN

    # Not yet leaning down -> BRAKE is forwarded once.
    v2 = _FakeVehicle(speed=20.0, target_speed=20.0)
    assert _debounce_meta_speed_action(DECELERATE, v2) == DECELERATE

    # Symmetric check for ACCELERATE.
    v3 = _FakeVehicle(speed=20.0, target_speed=25.0)
    assert _debounce_meta_speed_action(ACCELERATE, v3) == MAINTAIN
    v4 = _FakeVehicle(speed=20.0, target_speed=20.0)
    assert _debounce_meta_speed_action(ACCELERATE, v4) == ACCELERATE
