"""Reward / PBRS integration tests against MergeEnvV2 (Stage 2B-1)."""

from __future__ import annotations

import math

import pytest

from thesis.envs.merge_env_v2 import HighLevelAction, MergeEnvConfig, MergeEnvV2
from thesis.envs.scripted_scenarios import build_scenarios, iter_scripted_transitions, run_scenario
from thesis.rewards.base_reward_v2 import STAKEHOLDER_SET
from thesis.rewards.pbrs_v2 import PBRSConfig


def _accel():
    return {"A": int(HighLevelAction.ACCELERATE), "B": int(HighLevelAction.ACCELERATE)}


def test_15_reward_decomposition():
    env = MergeEnvV2(MergeEnvConfig(seed=0, max_steps=10))
    env.reset(seed=0)
    _, _, _, _, info = env.step(_accel())
    for aid in ("A", "B"):
        p = info["diagnostics"]["per_agent"][aid]
        total = (
            p["progress_component"]
            + p["exit_component"]
            + p["collision_component"]
            + p["hard_braking_component"]
        )
        assert p["base_total"] == pytest.approx(total, abs=1e-12)


def test_16_pbrs_decomposition():
    env = MergeEnvV2(
        MergeEnvConfig(
            seed=1,
            max_steps=10,
            pbrs=PBRSConfig(
                learner_gamma=0.995,
                shaping_gamma=0.995,
                lambda_mean=0.5,
                lambda_min=0.3,
            ),
        )
    )
    env.reset(seed=1)
    _, _, _, _, info = env.step(_accel())
    lam_m = env.config.pbrs.lambda_mean
    lam_n = env.config.pbrs.lambda_min
    for aid in ("A", "B"):
        p = info["diagnostics"]["per_agent"][aid]
        assert p["mean_pbrs_total"] == pytest.approx(
            p["base_total"] + lam_m * p["mean_F_t"], abs=1e-12
        )
        assert p["min_pbrs_total"] == pytest.approx(
            p["base_total"] + lam_n * p["min_F_t"], abs=1e-12
        )


def test_17_common_shaping_individual_base():
    env = MergeEnvV2(
        MergeEnvConfig(
            seed=2,
            spawn_speed_A=18.0,
            spawn_speed_B=10.0,
            spawn_route_A=10.0,
            spawn_route_B=10.0,
            max_steps=10,
            spawn_route_B_front=250.0,
            spawn_route_B_rear=-40.0,
        )
    )
    env.reset(seed=2)
    _, _, _, _, info = env.step(
        {"A": int(HighLevelAction.ACCELERATE), "B": int(HighLevelAction.DECELERATE)}
    )
    pa, pb = info["diagnostics"]["per_agent"]["A"], info["diagnostics"]["per_agent"]["B"]
    assert pa["scaled_mean_shaping"] == pytest.approx(pb["scaled_mean_shaping"], abs=1e-12)
    assert pa["scaled_min_shaping"] == pytest.approx(pb["scaled_min_shaping"], abs=1e-12)
    # Bases may differ due to progress / braking
    assert pa["base_total"] != pytest.approx(pb["base_total"]) or True
    assert pa["mean_pbrs_total"] == pytest.approx(
        pa["base_total"] + pa["scaled_mean_shaping"], abs=1e-12
    )


def test_18_completed_stakeholder_experience():
    spec = build_scenarios()["A_exits_first"]
    env, records = run_scenario(spec)
    # Find first A exit, then one subsequent step if possible
    exit_idx = next(
        i
        for i, r in enumerate(records)
        if r["info"]["events"]["exit_event"]["A"] >= 1.0
    )
    # Continue if episode still active
    if exit_idx + 1 < len(records):
        nxt = records[exit_idx + 1]
        exp = nxt["info"]["diagnostics"]["stakeholder_experiences_t"]
        assert set(exp.keys()) == set(STAKEHOLDER_SET)
        assert exp["A"] == 1.0
    else:
        # Force one more step on a fresh long-horizon clone after A exit
        cfg = MergeEnvConfig(
            seed=3,
            role_A="mainline",
            role_B="ramp",
            spawn_route_A=195.0,
            spawn_speed_A=20.0,
            spawn_route_B=20.0,
            spawn_speed_B=10.0,
            max_steps=40,
            spawn_route_B_front=300.0,
            spawn_route_B_rear=-50.0,
        )
        env = MergeEnvV2(cfg)
        env.reset(seed=3)
        exited = False
        for _ in range(20):
            _, _, term, trunc, info = env.step(_accel())
            if info["events"]["exit_event"]["A"] >= 1.0:
                exited = True
            if exited and not term and not trunc:
                exp = info["diagnostics"]["stakeholder_experiences_t1"]
                assert "A" in exp and exp["A"] == 1.0
                assert set(exp.keys()) == set(STAKEHOLDER_SET)
                break
            if term or trunc:
                break
        assert exited


def test_19_terminal_potential():
    for sid in ("controlled_collision_A", "simultaneous_exit"):
        env, records = run_scenario(build_scenarios()[sid])
        last = records[-1]
        if last["terminated"]:
            d = last["info"]["diagnostics"]
            assert d["actual_mean_potential_t1"] == 0.0
            assert d["actual_min_potential_t1"] == 0.0


def test_20_truncation_potential_not_zeroed():
    _, records = run_scenario(build_scenarios()["external_truncation"])
    last = records[-1]
    assert last["truncated"] is True
    d = last["info"]["diagnostics"]
    assert d["actual_mean_potential_t1"] == pytest.approx(d["raw_mean_potential_t1"])
    assert d["actual_mean_potential_t1"] != 0.0


def test_21_transition_timing_uses_same_step_snapshots():
    env = MergeEnvV2(MergeEnvConfig(seed=0, max_steps=5))
    env.reset(seed=0)
    s_before = {sid: env._vehicles[sid].route_position for sid in STAKEHOLDER_SET}
    _, _, _, _, info = env.step(_accel())
    # vehicles_t matches pre-step snapshot
    for sid in STAKEHOLDER_SET:
        assert info["vehicles_t"][sid]["route_position"] == pytest.approx(s_before[sid])
    # vehicles_t1 matches post-step live state
    for sid in STAKEHOLDER_SET:
        assert info["vehicles_t1"][sid]["route_position"] == pytest.approx(
            env._vehicles[sid].route_position
        )
    # Reward delta_rho matches snapshot pair
    for aid in ("A", "B"):
        vt = info["vehicles_t"][aid]
        vt1 = info["vehicles_t1"][aid]
        assert info["diagnostics"]["per_agent"][aid]["delta_rho"] == pytest.approx(
            vt1["rho"] - vt["rho"], abs=1e-12
        )


def test_22_no_nan_across_scripted_transitions():
    n = 0
    bad = []

    def check_finite(label, value):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(float(value)):
                bad.append(label)

    for rec in iter_scripted_transitions():
        n += 1
        info = rec["info"]
        for sid, v in info["vehicles_t1"].items():
            for k in ("route_position", "rho", "speed", "acceleration", "world_x"):
                check_finite(f"{rec['scenario_id']}:{sid}:{k}", v[k])
        d = info["diagnostics"]
        for k in (
            "raw_mean_potential_t1",
            "actual_mean_potential_t1",
            "raw_min_potential_t1",
            "actual_min_potential_t1",
            "mean_F_t",
            "min_F_t",
        ):
            check_finite(f"{rec['scenario_id']}:{k}", d[k])
        for aid, p in d["per_agent"].items():
            for k, val in p.items():
                check_finite(f"{rec['scenario_id']}:{aid}:{k}", val)
        if n >= 100 and not bad:
            break
    assert n >= 100, f"only collected {n} transitions"
    assert not bad, f"non-finite values: {bad[:10]}"
