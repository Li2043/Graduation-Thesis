"""Diagnostic_6_DQN_Pipeline_Verification_Protocol.md sec 11 (6G, Test
T9): deterministic PHYSICAL test of action-index semantics -- reset the
real environment to the same fixed state three times, apply each action
index (0/1/2) once, and read the actual resulting delta_v straight from
the physics, per the document's explicit instruction not to assume the
mapping from any prior document. This is the SAME encoding already used
by ``analyze_greedy_action_distribution.py``'s ``ACTION_NAMES`` and
``oracle_controller.py``'s ``MAINTAIN``/``ACCELERATE``/``DECELERATE``
constants -- this test is what actually justifies those constants."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(REPO_SRC))

from thesis.envs.stage10_symmetric_merge_env import HighLevelAction  # noqa: E402
from thesis.study_b.heterogeneous_env import StudyBEnvConfig, StudyBHeterogeneousEnv  # noqa: E402
from thesis.study_b.scenario_generator import generate_scenario  # noqa: E402

MAINTAIN, ACCELERATE, DECELERATE = 0, 1, 2


def _delta_v_for_action(action: int, *, vehicle_id: str = "V0") -> float:
    """Resets to the SAME fixed matched-TTC scenario, applies ``action``
    to every vehicle for exactly one step, returns vehicle_id's own
    delta_v (v_after - v_before)."""
    scenario = generate_scenario(
        scenario_id="action-mapping-test", episode_seed=42,
        role_members={"ramp": ["V0", "V1"], "mainline": ["V2", "V3"]}, traffic_type="heterogeneous",
    )
    env = StudyBHeterogeneousEnv(StudyBEnvConfig(episode_max_steps=200))
    env.reset(seed=0, scenario=scenario)
    v_before = env._env._vehicles[vehicle_id].speed  # noqa: SLF001 -- white-box physics test
    actions = {vid: action for vid in env.active_vehicle_ids}
    env.step(actions)
    v_after = env._env._vehicles[vehicle_id].speed  # noqa: SLF001
    return v_after - v_before


def test_action_encoding_matches_highlevelaction_enum():
    """Confirms this test's own constants match the actual environment
    code's enum -- not assumed from any external document."""
    assert MAINTAIN == int(HighLevelAction.MAINTAIN) == 0
    assert ACCELERATE == int(HighLevelAction.ACCELERATE) == 1
    assert DECELERATE == int(HighLevelAction.DECELERATE) == 2


def test_maintain_holds_speed_roughly_constant():
    delta_v = _delta_v_for_action(MAINTAIN)
    assert delta_v == pytest.approx(0.0, abs=1e-9)


def test_accelerate_increases_speed():
    delta_v = _delta_v_for_action(ACCELERATE)
    assert delta_v > 0.0


def test_decelerate_decreases_speed():
    delta_v = _delta_v_for_action(DECELERATE)
    assert delta_v < 0.0


def test_accelerate_and_decelerate_are_not_symmetric_by_coincidence():
    """Sanity check that the two directions are genuinely distinct
    magnitudes (accel_rate=2.0 vs decel_rate=3.0 m/s^2 per
    Stage10MergeEnvConfig), not e.g. both silently mapped to the same
    physical effect."""
    accel_delta = _delta_v_for_action(ACCELERATE)
    decel_delta = _delta_v_for_action(DECELERATE)
    assert abs(accel_delta) != pytest.approx(abs(decel_delta), rel=1e-3)


def test_action_direction_consistent_across_all_four_vehicles():
    """The same action index must mean the same physical direction for
    EVERY vehicle_id/role/speed-class -- not just V0 -- ruling out a
    role-dependent action-mapping bug."""
    for vid in ("V0", "V1", "V2", "V3"):
        assert _delta_v_for_action(ACCELERATE, vehicle_id=vid) > 0.0
        assert _delta_v_for_action(DECELERATE, vehicle_id=vid) < 0.0
