"""Stage 7A-0 diagnostic unit tests (fast; no full 160-episode run)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thesis.diagnostics.stage7a0_failure_taxonomy import classify_truncated_episode
from thesis.diagnostics.stage7a0_manifest import is_absolute_path_string, verify_manifest_hashes
from thesis.diagnostics.stage7a0_inventory import FORMAL_BASELINE_SEEDS


def _steps(n=50, speed=0.5, progress0=10.0, progress1=10.2, action=0, active=True, exited=False):
    rows = []
    for t in range(n):
        prog = progress0 + (progress1 - progress0) * t / max(1, n - 1)
        for ctrl in ("A", "B"):
            rows.append(
                {
                    "master_seed": 61001,
                    "validation_block_id": "validation_001",
                    "assignment": 0,
                    "policy_step": t,
                    "controller": ctrl,
                    "active": active,
                    "exited": exited,
                    "route_progress": prog,
                    "speed": speed,
                    "commanded_action": action,
                    "joint_action_category": "maintain-maintain",
                    "Q_margin": 0.001,
                    "Q_maintain": 0.0,
                    "Q_accelerate": -1.0,
                    "Q_decelerate": -1.0,
                }
            )
    return pd.DataFrame(rows)


def test_baseline_seeds_only():
    assert FORMAL_BASELINE_SEEDS == list(range(61001, 61011))


def test_taxonomy_mutual_yield():
    ep = {"truncated": True, "first_exit_controller": None, "second_exit_step": None}
    steps = _steps(speed=0.5, progress0=10.0, progress1=10.1, action=0)
    out = classify_truncated_episode(ep, steps)
    assert out["primary_failure_label"] in {
        "mutual_yielding",
        "merge_zone_deadlock",
        "stable_but_maladaptive",
        "low_confidence_policy",
        "other_unresolved",
    }
    assert out["flag_mutual_yielding"] or out["flag_merge_zone_deadlock"]


def test_taxonomy_post_exit_stall():
    rows = []
    for t in range(50):
        rows.append(
            {
                "master_seed": 61001,
                "validation_block_id": "validation_001",
                "assignment": 0,
                "policy_step": t,
                "controller": "A",
                "active": False,
                "exited": True,
                "route_progress": 100.0,
                "speed": 0.0,
                "commanded_action": 0,
                "joint_action_category": "inactive-active",
                "Q_margin": 1.0,
                "Q_maintain": 1.0,
                "Q_accelerate": 0.0,
                "Q_decelerate": 0.0,
            }
        )
        rows.append(
            {
                "master_seed": 61001,
                "validation_block_id": "validation_001",
                "assignment": 0,
                "policy_step": t,
                "controller": "B",
                "active": True,
                "exited": False,
                "route_progress": 20.0,
                "speed": 0.2,
                "commanded_action": 0,
                "joint_action_category": "inactive-active",
                "Q_margin": 1.0,
                "Q_maintain": 1.0,
                "Q_accelerate": 0.0,
                "Q_decelerate": 0.0,
            }
        )
    ep = {"truncated": True, "first_exit_controller": "A", "second_exit_step": None}
    out = classify_truncated_episode(ep, pd.DataFrame(rows))
    assert out["primary_failure_label"] == "post_exit_survivor_stall"


def test_taxonomy_environment_anomaly_precedence():
    steps = _steps()
    # inject discontinuous progress drop on A
    steps.loc[(steps["controller"] == "A") & (steps["policy_step"] == 40), "route_progress"] = 0.0
    ep = {"truncated": True, "first_exit_controller": None, "second_exit_step": None}
    out = classify_truncated_episode(ep, steps)
    assert out["primary_failure_label"] == "environment_or_exit_anomaly"


def test_taxonomy_oscillation():
    rows = []
    for t in range(50):
        act = 1 if t % 2 == 0 else 2
        for ctrl in ("A", "B"):
            rows.append(
                {
                    "master_seed": 61001,
                    "validation_block_id": "validation_001",
                    "assignment": 0,
                    "policy_step": t,
                    "controller": ctrl,
                    "active": True,
                    "exited": False,
                    "route_progress": 10.0 + 0.001 * t,
                    "speed": 2.0,
                    "commanded_action": act,
                    "joint_action_category": "accelerate-decelerate",
                    "Q_margin": 0.2,
                    "Q_maintain": 0.0,
                    "Q_accelerate": 1.0,
                    "Q_decelerate": 0.9,
                }
            )
    ep = {"truncated": True, "first_exit_controller": None, "second_exit_step": None}
    out = classify_truncated_episode(ep, pd.DataFrame(rows))
    assert out["flag_oscillatory_control"]


def test_paths_relative_helper():
    assert is_absolute_path_string(r"C:\Users\x")
    assert not is_absolute_path_string("output/figures/a.png")


def test_continuation_blocked_evidence_shape():
    # structural expectation used by runner
    payload = {
        "status": "BLOCKED",
        "available_resumable_100k_checkpoints": 0,
        "executed": False,
    }
    assert payload["status"] == "BLOCKED"
    assert payload["available_resumable_100k_checkpoints"] == 0
