"""Environment unit/integration tests for MergeEnvV2 (Stage 2B-1)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from thesis.envs.merge_env_v2 import HighLevelAction, MergeEnvConfig, MergeEnvV2
from thesis.envs.route_coordinates import default_ramp_route, normalised_route_progress
from thesis.envs.scripted_scenarios import build_scenarios, run_scenario
from thesis.rewards.base_reward_v2 import STAKEHOLDER_SET


def _maintain():
    return {"A": int(HighLevelAction.MAINTAIN), "B": int(HighLevelAction.MAINTAIN)}


def _accel():
    return {"A": int(HighLevelAction.ACCELERATE), "B": int(HighLevelAction.ACCELERATE)}


# ---------------------------------------------------------------------------
# Tests 1–4
# ---------------------------------------------------------------------------
def test_01_environment_reset_identities_and_roles():
    env = MergeEnvV2(MergeEnvConfig(seed=0, role_A="mainline", role_B="ramp"))
    obs, info = env.reset(seed=0)
    assert set(obs.keys()) == {"A", "B"}
    assert info["registry"] == list(STAKEHOLDER_SET)
    assert info["roles"]["A"] == "mainline"
    assert info["roles"]["B"] == "ramp"
    for sid in STAKEHOLDER_SET:
        assert sid in env._vehicles
        assert env._vehicles[sid].identity == sid


def test_02_fixed_stakeholder_registry():
    env = MergeEnvV2(MergeEnvConfig(seed=0))
    env.reset(seed=0)
    assert tuple(env._vehicles.keys()) == STAKEHOLDER_SET
    env.step(_maintain())
    assert set(env.last_info["events"]["stakeholder_collided"].keys()) == set(
        STAKEHOLDER_SET
    )


def test_03_deterministic_reset():
    cfg = MergeEnvConfig(seed=42)
    env1 = MergeEnvV2(cfg)
    env2 = MergeEnvV2(MergeEnvConfig(seed=42))
    o1, i1 = env1.reset(seed=42)
    o2, i2 = env2.reset(seed=42)
    assert i1["seed"] == i2["seed"] == 42
    for aid in ("A", "B"):
        np.testing.assert_allclose(o1[aid], o2[aid])
    for sid in STAKEHOLDER_SET:
        assert env1._vehicles[sid].route_position == pytest.approx(
            env2._vehicles[sid].route_position
        )


def test_04_different_seed_traceability():
    env = MergeEnvV2(MergeEnvConfig(seed=0))
    _, i0 = env.reset(seed=0)
    state0 = {sid: env._vehicles[sid].route_position for sid in ("A", "B")}
    _, i1 = env.reset(seed=99)
    state1 = {sid: env._vehicles[sid].route_position for sid in ("A", "B")}
    assert i0["seed"] == 0
    assert i1["seed"] == 99
    assert i1["seed"] != i0["seed"]
    # Jitter differs with seed — positions should not silently reuse seed-0 state
    assert state0 != state1


# ---------------------------------------------------------------------------
# Tests 5–7 route progress
# ---------------------------------------------------------------------------
def test_05_route_progress_bounds():
    env = MergeEnvV2(MergeEnvConfig(seed=0, max_steps=30))
    env.reset(seed=0)
    for _ in range(20):
        _, _, term, trunc, info = env.step(_accel())
        for sid, v in info["vehicles_t1"].items():
            assert 0.0 <= v["rho"] <= 1.0
        if term or trunc:
            break


def test_06_ramp_route_continuity():
    cfg = MergeEnvConfig(
        seed=7,
        role_A="mainline",
        role_B="ramp",
        spawn_route_B=40.0,
        spawn_speed_B=18.0,
        spawn_route_A=5.0,
        spawn_speed_A=10.0,
        spawn_route_B_front=300.0,
        spawn_route_B_rear=-50.0,
        max_steps=80,
        discontinuity_report_threshold=0.25,
    )
    env = MergeEnvV2(cfg)
    env.reset(seed=7)
    prev_s = env._vehicles["B"].route_position
    prev_rho = normalised_route_progress(prev_s, default_ramp_route())
    large_neg = 0
    for _ in range(50):
        _, _, term, trunc, info = env.step(
            {"A": int(HighLevelAction.MAINTAIN), "B": int(HighLevelAction.ACCELERATE)}
        )
        s = info["vehicles_t1"]["B"]["route_position"]
        rho = info["vehicles_t1"]["B"]["rho"]
        ds = s - prev_s
        dr = rho - prev_rho
        # Unexplained large negative jump
        if ds < -1.0 or dr < -0.2:
            large_neg += 1
        prev_s, prev_rho = s, rho
        # Cross the join
        if s > default_ramp_route().ramp_approach_length + 5:
            break
        if term or trunc:
            break
    assert large_neg == 0


def test_07_mainline_route_continuity_forward():
    cfg = MergeEnvConfig(
        seed=8,
        role_A="mainline",
        role_B="ramp",
        spawn_route_A=10.0,
        spawn_speed_A=15.0,
        spawn_route_B_front=300.0,
        spawn_route_B_rear=-40.0,
        max_steps=40,
    )
    env = MergeEnvV2(cfg)
    env.reset(seed=8)
    prev = env._vehicles["A"].route_position
    for _ in range(25):
        _, _, term, trunc, info = env.step(
            {"A": int(HighLevelAction.ACCELERATE), "B": int(HighLevelAction.MAINTAIN)}
        )
        cur = info["vehicles_t1"]["A"]["route_position"]
        assert cur >= prev - 1e-9  # forward scripted motion
        prev = cur
        if term or trunc:
            break


# ---------------------------------------------------------------------------
# Tests 8–10 exits / success
# ---------------------------------------------------------------------------
def test_08_first_safe_exit():
    spec = build_scenarios()["A_exits_first"]
    env, records = run_scenario(spec)
    exits = [
        r
        for r in records
        if r["info"]["events"]["exit_event"]["A"] >= 1.0
    ]
    assert len(exits) >= 1
    first = exits[0]
    assert first["info"]["diagnostics"]["per_agent"]["A"]["exit_component"] == pytest.approx(
        0.6
    )


def test_09_repeated_exit_suppression():
    spec = build_scenarios()["A_exits_first"]
    env, records = run_scenario(spec)
    exit_steps = [
        r["info"]["step"]
        for r in records
        if r["info"]["events"]["exit_event"]["A"] >= 1.0
    ]
    assert len(exit_steps) == 1
    # After exit, further steps (if any) must not re-award
    after = [r for r in records if r["info"]["step"] > exit_steps[0]]
    for r in after:
        assert r["info"]["events"]["exit_event"]["A"] == 0.0
        assert r["info"]["diagnostics"]["per_agent"]["A"]["exit_component"] == 0.0


def test_10_both_vehicles_exit_success():
    # Both start just before their exits, separated in world-x (> collision_distance).
    # A mainline world_x ≈ route; B ramp world_x ≈ route - 10.
    cfg = MergeEnvConfig(
        seed=5,
        role_A="mainline",
        role_B="ramp",
        spawn_route_A=190.0,
        spawn_speed_A=20.0,
        spawn_route_B=206.0,
        spawn_speed_B=20.0,
        max_steps=30,
        spawn_route_B_front=2000.0,
        spawn_route_B_rear=-200.0,
        spawn_speed_B_front=0.0,
        spawn_speed_B_rear=0.0,
        collision_distance=4.0,
    )
    env = MergeEnvV2(cfg)
    env.reset(seed=5)
    env._vehicles["A"].route_position = 190.0
    env._vehicles["B"].route_position = 206.0
    env._sync_world(env._vehicles["A"])
    env._sync_world(env._vehicles["B"])
    saw_success = False
    for _ in range(25):
        _, _, term, trunc, info = env.step(_accel())
        if term and info["term_reason"] == "success":
            assert trunc is False
            assert info["completion"]["A"] is True
            assert info["completion"]["B"] is True
            saw_success = True
            break
        if term or trunc:
            break
    assert saw_success


# ---------------------------------------------------------------------------
# Tests 11–12 collisions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "scenario_id,must_include",
    [
        ("controlled_collision_A", "A"),
        ("controlled_collision_B", "B"),
        ("background_front_collision", "B_front"),
        ("background_rear_collision", "B_rear"),
    ],
)
def test_11_stakeholder_collision_termination(scenario_id: str, must_include: str):
    spec = build_scenarios()[scenario_id]
    env, records = run_scenario(spec)
    last = records[-1]
    assert last["terminated"] is True
    assert last["truncated"] is False
    assert last["info"]["events"]["stakeholder_collided"][must_include] is True
    assert last["info"]["events"]["stakeholder_collision_event"] == 1.0
    for aid in ("A", "B"):
        assert last["info"]["diagnostics"]["per_agent"][aid][
            "collision_component"
        ] == pytest.approx(-1.0)
    diag = last["info"]["diagnostics"]
    assert diag["actual_mean_potential_t1"] == 0.0
    assert diag["actual_min_potential_t1"] == 0.0


def test_12_collision_blocks_exit():
    cfg = MergeEnvConfig(
        seed=30,
        role_A="mainline",
        role_B="ramp",
        max_steps=5,
        fixture_mode="controlled_collision",
        fixture_payload={
            "collide_at_step": 1,
            "target_ids": ["A", "B_front"],
            # Place collision at exit boundary so A would otherwise cross
            "collision_world_x": 200.0,
            "collision_speed": 15.0,
        },
        spawn_route_A=198.0,
        spawn_speed_A=20.0,
        spawn_route_B=10.0,
        spawn_route_B_front=190.0,
        spawn_route_B_rear=-40.0,
    )
    env = MergeEnvV2(cfg)
    env.reset(seed=30)
    # Manually set A just before exit so dynamics+fixture collide at exit
    env._vehicles["A"].route_position = 198.0
    _, _, term, trunc, info = env.step(_accel())
    assert term is True
    assert trunc is False
    assert info["events"]["exit_event"]["A"] == 0.0
    assert info["diagnostics"]["per_agent"]["A"]["exit_component"] == 0.0
    assert info["diagnostics"]["per_agent"]["A"]["collision_component"] == pytest.approx(
        -1.0
    )


# ---------------------------------------------------------------------------
# Tests 13–14 truncation / flags
# ---------------------------------------------------------------------------
def test_13_external_truncation():
    spec = build_scenarios()["external_truncation"]
    env, records = run_scenario(spec)
    last = records[-1]
    assert last["terminated"] is False
    assert last["truncated"] is True
    assert last["info"]["events"]["stakeholder_collision_event"] == 0.0
    assert last["info"]["events"]["exit_event"]["A"] == 0.0
    assert last["info"]["events"]["exit_event"]["B"] == 0.0
    diag = last["info"]["diagnostics"]
    assert diag["actual_mean_potential_t1"] == pytest.approx(diag["raw_mean_potential_t1"])
    assert diag["actual_min_potential_t1"] == pytest.approx(diag["raw_min_potential_t1"])
    assert diag["actual_mean_potential_t1"] != 0.0 or diag["raw_mean_potential_t1"] == 0.0


def test_14_invalid_simultaneous_flags_never_emitted():
    for spec in build_scenarios().values():
        _, records = run_scenario(spec)
        for r in records:
            assert not (r["terminated"] and r["truncated"])
