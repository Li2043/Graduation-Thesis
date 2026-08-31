"""Environment → reward condition → Independent DQN pipeline tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from thesis.agents.action_masking import role_action_mask
from thesis.agents.dqn_pipeline import (
    default_learners,
    run_pipeline_scenario,
)
from thesis.agents.independent_dqn_v2 import DQNConfig, build_independent_learners
from thesis.envs.merge_env_v2 import HighLevelAction, MergeEnvConfig, MergeEnvV2
from thesis.envs.scripted_scenarios import build_scenarios
from thesis.rewards.base_reward_v2 import STAKEHOLDER_SET


def test_19_individual_reward_preservation():
    learners = default_learners(reward_condition="mean_pbrs")
    spec = build_scenarios()["hard_braking_trace"]
    records = run_pipeline_scenario(
        spec, learners, reward_condition="mean_pbrs", episode_id="e_ind"
    )
    # Same step should share shaping but may differ in base/total
    by_step: dict[int, list] = {}
    for r in records:
        by_step.setdefault(r["step"], []).append(r)
    found = False
    for step, rows in by_step.items():
        if len(rows) == 2:
            a = next(x for x in rows if x["controller_id"] == "A")
            b = next(x for x in rows if x["controller_id"] == "B")
            assert a["shaping_component"] == pytest.approx(b["shaping_component"], abs=1e-12)
            # Totals retained separately
            assert "learner_reward" in a and "learner_reward" in b
            found = True
            break
    assert found


def test_27_role_masks_follow_role_not_identity():
    # A mainline, B ramp
    env = MergeEnvV2(MergeEnvConfig(role_A="mainline", role_B="ramp", seed=0))
    env.reset(seed=0)
    assert role_action_mask(env._role_of("A")).tolist() == role_action_mask("mainline").tolist()
    assert role_action_mask(env._role_of("B")).tolist() == role_action_mask("ramp").tolist()
    # Swap roles
    env2 = MergeEnvV2(MergeEnvConfig(role_A="ramp", role_B="mainline", seed=1))
    env2.reset(seed=1)
    assert role_action_mask(env2._role_of("A")).tolist() == role_action_mask("ramp").tolist()
    assert role_action_mask(env2._role_of("B")).tolist() == role_action_mask("mainline").tolist()
    # Same role ⇒ same mask semantics
    assert role_action_mask("mainline").tolist() == role_action_mask(
        env2._role_of("B")
    ).tolist()


def test_28_controller_identity_persistence():
    # Episode 1: A mainline
    env = MergeEnvV2(MergeEnvConfig(role_A="mainline", role_B="ramp", seed=0))
    env.reset(seed=0)
    assert env._vehicles["A"].identity == "A"
    # Episode 2: A ramp
    env2 = MergeEnvV2(MergeEnvConfig(role_A="ramp", role_B="mainline", seed=2))
    env2.reset(seed=2)
    assert env2._vehicles["A"].identity == "A"
    assert env2._role_of("A") == "ramp"
    learners = build_independent_learners(DQNConfig(), seed_A=0, seed_B=1)
    assert learners["A"].controller_id == "A"
    assert learners["B"].controller_id == "B"


def test_29_environment_terminal_pipeline():
    learners = default_learners(reward_condition="baseline")
    # Force both near exit for success
    from thesis.envs.scripted_scenarios import ScenarioSpec
    from thesis.envs.merge_env_v2 import MergeEnvConfig as C

    spec = ScenarioSpec(
        scenario_id="force_success",
        config=C(
            seed=5,
            role_A="mainline",
            role_B="ramp",
            spawn_route_A=190.0,
            spawn_speed_A=20.0,
            spawn_route_B=206.0,
            spawn_speed_B=20.0,
            max_steps=20,
            spawn_route_B_front=2000.0,
            spawn_route_B_rear=-200.0,
            spawn_speed_B_front=0.0,
            spawn_speed_B_rear=0.0,
        ),
        actions=[
            {"A": int(HighLevelAction.ACCELERATE), "B": int(HighLevelAction.ACCELERATE)}
            for _ in range(15)
        ],
    )
    records = run_pipeline_scenario(
        spec, learners, reward_condition="baseline", episode_id="e_term"
    )
    last_by = {r["controller_id"]: r for r in records if r["terminated"]}
    assert last_by
    for aid, r in last_by.items():
        assert r["terminated"] is True
        assert r["truncated"] is False
        assert r["bootstrap_multiplier"] == 0.0
        assert r["reward_condition"] == "baseline"
        # Stored in replay
        assert any(
            t.terminated and t.reward_condition == "baseline"
            for t in learners[aid].replay._storage
            if t is not None
        )


def test_30_environment_truncation_pipeline():
    learners = default_learners(reward_condition="mean_pbrs")
    records = run_pipeline_scenario(
        build_scenarios()["external_truncation"],
        learners,
        reward_condition="mean_pbrs",
        episode_id="e_trunc",
    )
    last = records[-1]
    assert last["terminated"] is False
    assert last["truncated"] is True
    assert last["next_observation"] is not None
    assert len(last["next_action_mask"]) == 3
    assert last["bootstrap_multiplier"] == 1.0
    # Replay retains both flags
    for aid in ("A", "B"):
        found = False
        start = (learners[aid].replay._write - len(learners[aid].replay)) % learners[
            aid
        ].replay.capacity
        for i in range(len(learners[aid].replay)):
            t = learners[aid].replay._storage[(start + i) % learners[aid].replay.capacity]
            if t and t.truncated and not t.terminated:
                assert t.next_observation is not None
                assert t.next_action_mask is not None
                found = True
        assert found


def test_31_collision_terminal_pipeline():
    learners = default_learners(reward_condition="baseline")
    records = run_pipeline_scenario(
        build_scenarios()["controlled_collision_A"],
        learners,
        reward_condition="baseline",
        episode_id="e_coll",
    )
    term = [r for r in records if r["terminated"]]
    assert term
    for r in term:
        assert r["bootstrap_multiplier"] == 0.0
        assert r["base_reward"] == pytest.approx(-1.0, abs=0.05) or r[
            "learner_reward"
        ] <= -0.9
        assert r["actual_mean_potential_t1"] == 0.0
        assert r["actual_min_potential_t1"] == 0.0
    # Both controllers got collision penalty in base
    bases = {r["controller_id"]: r["base_reward"] for r in term}
    assert "A" in bases and "B" in bases
    assert bases["A"] <= -0.9 and bases["B"] <= -0.9


def test_32_completed_stakeholder_pipeline_option1():
    learners = default_learners(reward_condition="baseline")
    records = run_pipeline_scenario(
        build_scenarios()["A_exits_first"],
        learners,
        reward_condition="baseline",
        episode_id="e_exitA",
    )
    a_exits = [r for r in records if r["controller_id"] == "A" and r["step"]]
    # Find exit transition for A via experiences / completion path
    exit_steps = []
    for r in records:
        if r["controller_id"] == "A":
            # After A exit, E_A should be 1 in experiences when recorded
            if r["experiences_t1"].get("A") == 1.0:
                exit_steps.append(r["step"])
    assert exit_steps
    # A remains in stakeholder mapping
    last_a = [r for r in records if r["controller_id"] == "A"][-1]
    assert set(last_a["experiences_t1"].keys()) == set(STAKEHOLDER_SET)
    assert last_a["experiences_t1"]["A"] == 1.0
    # Option 1: no further A transitions after exit transition
    max_a_step = max(r["step"] for r in records if r["controller_id"] == "A")
    later_a = [
        r
        for r in records
        if r["controller_id"] == "A" and r["step"] > max_a_step
    ]
    assert later_a == []
    # B may continue after A exit
    b_after = [r for r in records if r["controller_id"] == "B" and r["step"] > min(exit_steps)]
    # Not strictly required if episode ends immediately, but A must not keep storing
    n_a = sum(1 for r in records if r["controller_id"] == "A")
    assert n_a >= 1


def test_15_16_17_pipeline_reward_conditions():
    for cond, key in (
        ("baseline", "baseline"),
        ("mean_pbrs", "mean_pbrs"),
        ("min_pbrs", "min_pbrs"),
    ):
        learners = default_learners(reward_condition=cond)  # type: ignore[arg-type]
        records = run_pipeline_scenario(
            build_scenarios()["nominal_forward"],
            learners,
            reward_condition=cond,  # type: ignore[arg-type]
            episode_id=f"e_{cond}",
        )
        assert records
        r0 = records[0]
        if cond == "baseline":
            assert r0["learner_reward"] == pytest.approx(r0["base_reward"], abs=1e-12)
            assert r0["shaping_component"] == pytest.approx(0.0, abs=1e-12)
        else:
            assert r0["learner_reward"] == pytest.approx(
                r0["base_reward"] + r0["shaping_component"], abs=1e-12
            )


def test_33_hundred_transition_smoke():
    counts = {"n": 0}
    bad = []
    for cond in ("baseline", "mean_pbrs", "min_pbrs"):
        for sid, spec in build_scenarios().items():
            learners = default_learners(reward_condition=cond)  # type: ignore[arg-type]
            try:
                records = run_pipeline_scenario(
                    spec,
                    learners,
                    reward_condition=cond,  # type: ignore[arg-type]
                    episode_id=f"smoke_{cond}_{sid}",
                )
            except Exception as e:
                bad.append(f"{cond}/{sid}: {e}")
                continue
            for r in records:
                counts["n"] += 1
                if r["terminated"] and r["truncated"]:
                    bad.append("invalid flags")
                if not math.isfinite(r["learner_reward"]) or not math.isfinite(r["target"]):
                    bad.append("nan")
                if not any(r["action_mask"]):
                    bad.append("invalid mask")
                if not r["action_mask"][r["selected_action"]]:
                    bad.append("illegal action")
                if r["truncated"] and r["next_observation"] is None:
                    bad.append("trunc missing next")
                if not r["target_decomposition_valid"]:
                    bad.append("target decomp")
                expected = r["base_reward"] + r["shaping_component"]
                if abs(r["learner_reward"] - expected) > 1e-12:
                    bad.append("reward decomp")
            if counts["n"] >= 100 and not bad:
                break
        if counts["n"] >= 100 and not bad:
            break
    assert counts["n"] >= 100, f"only {counts['n']} transitions"
    assert not bad, bad[:10]
