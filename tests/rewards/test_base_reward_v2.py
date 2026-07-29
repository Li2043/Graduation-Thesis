"""Deterministic unit tests for Stage-1 base reward v2 (Tests 1–18)."""

from __future__ import annotations

import math

import pytest

from thesis.rewards.base_reward_v2 import (
    AgentTransitionState,
    BaseRewardConfig,
    BaseRewardInputs,
    compute_base_reward_for_agents,
    compute_hard_braking_cost,
    compute_normalised_route_progress,
    compute_route_progress_delta,
)


def _no_collision() -> dict[str, bool]:
    return {"A": False, "B": False, "B_front": False, "B_rear": False}


def _agent(
    *,
    pos_t: float = 0.0,
    pos_t1: float = 0.0,
    start: float = 0.0,
    exit_p: float = 100.0,
    accel: float = 0.0,
    already_exited: bool = False,
) -> AgentTransitionState:
    return AgentTransitionState(
        route_position_t=pos_t,
        route_position_t1=pos_t1,
        route_start=start,
        route_exit=exit_p,
        acceleration=accel,
        already_exited=already_exited,
    )


def _cfg() -> BaseRewardConfig:
    # Explicit TEST-ONLY braking parameters (not final experimental values).
    return BaseRewardConfig(
        progress_weight=0.4,
        exit_weight=0.6,
        collision_penalty=1.0,
        eta_hard_brake=0.1,
        a_comfort=2.0,
        a_hard=6.0,
    )


# ---------------------------------------------------------------------------
# Test 1 — Stationary ordinary transition
# ---------------------------------------------------------------------------
def test_01_stationary_ordinary_transition():
    inputs = BaseRewardInputs(
        agents={"A": _agent(), "B": _agent()},
        stakeholder_collided=_no_collision(),
    )
    out = compute_base_reward_for_agents(inputs, _cfg())
    assert out["A"].total_reward == 0.0
    assert out["B"].total_reward == 0.0


# ---------------------------------------------------------------------------
# Test 2 — Positive route progress
# ---------------------------------------------------------------------------
def test_02_positive_route_progress():
    # start=0, exit=100 => Δpos=10 => Δρ=0.1
    inputs = BaseRewardInputs(
        agents={
            "A": _agent(pos_t=0.0, pos_t1=10.0),
            "B": _agent(pos_t=0.0, pos_t1=10.0),
        },
        stakeholder_collided=_no_collision(),
    )
    out = compute_base_reward_for_agents(inputs, _cfg())
    assert out["A"].progress_component == pytest.approx(0.04)
    assert out["A"].total_reward == pytest.approx(0.04)
    assert out["B"].progress_component == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# Test 3 — Different individual progress
# ---------------------------------------------------------------------------
def test_03_different_individual_progress():
    inputs = BaseRewardInputs(
        agents={
            "A": _agent(pos_t=0.0, pos_t1=10.0),  # Δρ=0.1
            "B": _agent(pos_t=0.0, pos_t1=20.0),  # Δρ=0.2
        },
        stakeholder_collided=_no_collision(),
    )
    out = compute_base_reward_for_agents(inputs, _cfg())
    assert out["A"].progress_component == pytest.approx(0.04)
    assert out["B"].progress_component == pytest.approx(0.08)
    assert out["A"].progress_component != out["B"].progress_component


# ---------------------------------------------------------------------------
# Test 4 — Safe first exit
# ---------------------------------------------------------------------------
def test_04_safe_first_exit():
    inputs = BaseRewardInputs(
        agents={
            "A": _agent(pos_t=99.0, pos_t1=101.0, exit_p=100.0),
            "B": _agent(pos_t=50.0, pos_t1=51.0, exit_p=100.0),
        },
        stakeholder_collided=_no_collision(),
    )
    out = compute_base_reward_for_agents(inputs, _cfg())
    assert out["A"].safe_exit_event == 1.0
    assert out["A"].exit_component == pytest.approx(0.6)
    assert out["B"].safe_exit_event == 0.0
    assert out["B"].exit_component == 0.0


# ---------------------------------------------------------------------------
# Test 5 — Exit bonus cannot repeat
# ---------------------------------------------------------------------------
def test_05_exit_bonus_cannot_repeat():
    inputs = BaseRewardInputs(
        agents={
            "A": _agent(
                pos_t=100.5, pos_t1=102.0, exit_p=100.0, already_exited=True
            ),
            "B": _agent(pos_t=50.0, pos_t1=51.0),
        },
        stakeholder_collided=_no_collision(),
    )
    out = compute_base_reward_for_agents(inputs, _cfg())
    assert out["A"].exit_component == 0.0
    assert out["A"].safe_exit_event == 0.0


# ---------------------------------------------------------------------------
# Test 6 — Simultaneous safe exits
# ---------------------------------------------------------------------------
def test_06_simultaneous_safe_exits():
    inputs = BaseRewardInputs(
        agents={
            "A": _agent(pos_t=99.0, pos_t1=101.0, exit_p=100.0),
            "B": _agent(pos_t=199.0, pos_t1=201.0, start=100.0, exit_p=200.0),
        },
        stakeholder_collided=_no_collision(),
    )
    out = compute_base_reward_for_agents(inputs, _cfg())
    assert out["A"].exit_component == pytest.approx(0.6)
    assert out["B"].exit_component == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Test 7 — Stakeholder collision (parameterised)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("who", ["A", "B", "B_front", "B_rear"])
def test_07_stakeholder_collision(who: str):
    collided = _no_collision()
    collided[who] = True
    inputs = BaseRewardInputs(
        agents={"A": _agent(), "B": _agent()},
        stakeholder_collided=collided,
    )
    out = compute_base_reward_for_agents(inputs, _cfg())
    assert out["A"].collision_component == pytest.approx(-1.0)
    assert out["B"].collision_component == pytest.approx(-1.0)
    assert out["A"].stakeholder_collision_event == 1.0
    assert out["B"].stakeholder_collision_event == 1.0


# ---------------------------------------------------------------------------
# Test 8 — Collision and exit on the same transition
# ---------------------------------------------------------------------------
def test_08_collision_blocks_exit():
    collided = _no_collision()
    collided["B_front"] = True
    inputs = BaseRewardInputs(
        agents={
            "A": _agent(pos_t=99.0, pos_t1=101.0, exit_p=100.0),
            "B": _agent(),
        },
        stakeholder_collided=collided,
    )
    out = compute_base_reward_for_agents(inputs, _cfg())
    assert out["A"].safe_exit_event == 0.0
    assert out["A"].exit_component == 0.0
    assert out["A"].collision_component == pytest.approx(-1.0)
    assert out["B"].collision_component == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Test 9 — Comfortable deceleration
# ---------------------------------------------------------------------------
def test_09_comfortable_deceleration():
    # a = -a_comfort and a > -a_comfort (less braking) => H=0
    for accel in (-2.0, -1.0, 0.0, 1.0):
        h = compute_hard_braking_cost(accel, a_comfort=2.0, a_hard=6.0)
        assert h == 0.0
        inputs = BaseRewardInputs(
            agents={
                "A": _agent(accel=accel),
                "B": _agent(accel=accel),
            },
            stakeholder_collided=_no_collision(),
        )
        out = compute_base_reward_for_agents(inputs, _cfg())
        assert out["A"].hard_braking_component == 0.0
        assert out["A"].hard_braking_cost == 0.0


# ---------------------------------------------------------------------------
# Test 10 — Intermediate hard braking
# ---------------------------------------------------------------------------
def test_10_intermediate_hard_braking():
    h = compute_hard_braking_cost(-4.0, a_comfort=2.0, a_hard=6.0)
    assert h == pytest.approx(0.25)
    inputs = BaseRewardInputs(
        agents={
            "A": _agent(accel=-4.0),
            "B": _agent(accel=0.0),
        },
        stakeholder_collided=_no_collision(),
    )
    out = compute_base_reward_for_agents(inputs, _cfg())
    assert out["A"].hard_braking_cost == pytest.approx(0.25)
    assert out["A"].hard_braking_component == pytest.approx(-0.025)
    assert out["B"].hard_braking_component == 0.0


# ---------------------------------------------------------------------------
# Test 11 — Hard-braking saturation
# ---------------------------------------------------------------------------
def test_11_hard_braking_saturation():
    for accel in (-6.0, -8.0, -100.0):
        h = compute_hard_braking_cost(accel, a_comfort=2.0, a_hard=6.0)
        assert h == pytest.approx(1.0)
        inputs = BaseRewardInputs(
            agents={
                "A": _agent(accel=accel),
                "B": _agent(),
            },
            stakeholder_collided=_no_collision(),
        )
        out = compute_base_reward_for_agents(inputs, _cfg())
        assert out["A"].hard_braking_component == pytest.approx(-0.1)


# ---------------------------------------------------------------------------
# Test 12 — Invalid braking configuration
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "a_comfort,a_hard",
    [
        (2.0, 2.0),
        (3.0, 2.0),
        (0.0, 6.0),
        (-1.0, 6.0),
    ],
)
def test_12_invalid_braking_configuration(a_comfort: float, a_hard: float):
    with pytest.raises(ValueError):
        compute_hard_braking_cost(-4.0, a_comfort=a_comfort, a_hard=a_hard)
    with pytest.raises(ValueError):
        BaseRewardConfig(a_comfort=a_comfort, a_hard=a_hard).validate()


# ---------------------------------------------------------------------------
# Test 13 — Route normalisation boundaries
# ---------------------------------------------------------------------------
def test_13_route_normalisation_boundaries():
    assert compute_normalised_route_progress(0.0, 0.0, 100.0) == 0.0
    assert compute_normalised_route_progress(100.0, 0.0, 100.0) == 1.0
    assert compute_normalised_route_progress(-10.0, 0.0, 100.0) == 0.0
    assert compute_normalised_route_progress(150.0, 0.0, 100.0) == 1.0
    with pytest.raises(ValueError):
        compute_normalised_route_progress(50.0, 100.0, 100.0)
    with pytest.raises(ValueError):
        compute_normalised_route_progress(50.0, 100.0, 50.0)


# ---------------------------------------------------------------------------
# Test 14 — Negative progress
# ---------------------------------------------------------------------------
def test_14_negative_progress():
    # ρ_t=0.5, ρ_t1=0.45 via positions on [0,100]
    inputs = BaseRewardInputs(
        agents={
            "A": _agent(pos_t=50.0, pos_t1=45.0),
            "B": _agent(),
        },
        stakeholder_collided=_no_collision(),
    )
    out = compute_base_reward_for_agents(inputs, _cfg())
    assert out["A"].delta_route_progress == pytest.approx(-0.05)
    assert out["A"].progress_component == pytest.approx(-0.02)
    assert out["A"].progress_component < 0.0


# ---------------------------------------------------------------------------
# Test 15 — External truncation
# ---------------------------------------------------------------------------
def test_15_external_truncation():
    inputs = BaseRewardInputs(
        agents={
            "A": _agent(pos_t=10.0, pos_t1=20.0, accel=-4.0),
            "B": _agent(pos_t=10.0, pos_t1=15.0),
        },
        stakeholder_collided=_no_collision(),
        terminated=False,
        truncated=True,
    )
    out = compute_base_reward_for_agents(inputs, _cfg())
    assert out["A"].exit_component == 0.0
    assert out["A"].collision_component == 0.0
    assert out["B"].exit_component == 0.0
    assert out["B"].collision_component == 0.0
    assert out["A"].progress_component == pytest.approx(0.04)
    assert out["A"].hard_braking_component == pytest.approx(-0.025)


# ---------------------------------------------------------------------------
# Test 16 — Reward decomposition identity
# ---------------------------------------------------------------------------
def test_16_reward_decomposition_identity():
    collided = _no_collision()
    collided["A"] = True
    inputs = BaseRewardInputs(
        agents={
            "A": _agent(pos_t=99.0, pos_t1=101.0, accel=-4.0),
            "B": _agent(pos_t=0.0, pos_t1=10.0, accel=-8.0),
        },
        stakeholder_collided=collided,
    )
    out = compute_base_reward_for_agents(inputs, _cfg())
    for aid in ("A", "B"):
        b = out[aid]
        expected = (
            b.progress_component
            + b.exit_component
            + b.collision_component
            + b.hard_braking_component
        )
        assert b.total_reward == pytest.approx(expected, abs=1e-12)
        # Collision blocks exit for A
        assert b.exit_component == 0.0


# ---------------------------------------------------------------------------
# Test 17 — Finite-value validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"pos_t": float("nan")},
        {"pos_t1": float("inf")},
        {"accel": float("nan")},
        {"start": float("-inf")},
    ],
)
def test_17_finite_value_validation(kwargs):
    base = dict(pos_t=0.0, pos_t1=1.0, start=0.0, exit_p=100.0, accel=0.0)
    base.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        inputs = BaseRewardInputs(
            agents={
                "A": _agent(**base),
                "B": _agent(),
            },
            stakeholder_collided=_no_collision(),
        )
        compute_base_reward_for_agents(inputs, _cfg())


def test_17_nan_rho_delta():
    with pytest.raises(ValueError):
        compute_route_progress_delta(float("nan"), 0.5)
    with pytest.raises(ValueError):
        compute_route_progress_delta(0.5, float("inf"))


# ---------------------------------------------------------------------------
# Test 18 — Missing stakeholder validation
# ---------------------------------------------------------------------------
def test_18_missing_stakeholder_validation():
    incomplete = {"A": False, "B": False, "B_front": False}  # missing B_rear
    inputs = BaseRewardInputs(
        agents={"A": _agent(), "B": _agent()},
        stakeholder_collided=incomplete,
    )
    with pytest.raises(ValueError, match="missing required stakeholders"):
        compute_base_reward_for_agents(inputs, _cfg())


def test_helper_delta_progress_math():
    assert compute_route_progress_delta(0.5, 0.45) == pytest.approx(-0.05)
    assert math.isfinite(compute_hard_braking_cost(-4.0, 2.0, 6.0))
