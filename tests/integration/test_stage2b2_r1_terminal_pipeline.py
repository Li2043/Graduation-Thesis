"""Stage 2B-2R pipeline semantics: exit / collision / no post-exit storage."""

from __future__ import annotations

import numpy as np

from thesis.agents.dqn_pipeline import build_transition_for_controller, run_pipeline_scenario
from thesis.agents.independent_dqn_v2 import DQNConfig, build_independent_learners
from thesis.envs.scripted_scenarios import build_scenarios


def test_safe_exit_and_collision_are_controller_terminal():
    # Synthetic info: exit for A
    info_exit = {
        "diagnostics": {
            "per_agent": {
                "A": {
                    "base_total": 0.6,
                    "scaled_mean_shaping": 0.0,
                    "scaled_min_shaping": 0.0,
                }
            }
        },
        "vehicles_t": {"A": {"role": "mainline"}},
        "events": {"exit_event": {"A": 1.0, "B": 0.0}},
        "completion": {"A": True, "B": False},
        "step": 3,
    }
    tr = build_transition_for_controller(
        controller_id="A",
        obs=np.zeros(4),
        next_obs=np.ones(4),
        action=0,
        action_mask=np.array([True, True, True]),
        next_action_mask=np.array([True, True, True]),
        terminated=False,
        truncated=False,
        info=info_exit,
        reward_condition="baseline",
        episode_id="e",
    )
    assert tr.controller_terminal is True
    assert tr.learner_completed is True
    assert tr.next_observation is None
    assert tr.next_action_mask is None

    info_coll = {
        "diagnostics": {
            "per_agent": {
                "A": {
                    "base_total": -1.0,
                    "scaled_mean_shaping": 0.0,
                    "scaled_min_shaping": 0.0,
                }
            }
        },
        "vehicles_t": {"A": {"role": "mainline"}},
        "events": {"exit_event": {"A": 0.0, "B": 0.0}},
        "completion": {"A": False, "B": False},
        "step": 2,
    }
    tr2 = build_transition_for_controller(
        controller_id="A",
        obs=np.zeros(4),
        next_obs=np.ones(4),
        action=0,
        action_mask=np.array([True, True, True]),
        next_action_mask=np.array([True, True, True]),
        terminated=True,
        truncated=False,
        info=info_coll,
        reward_condition="baseline",
        episode_id="e",
    )
    assert tr2.controller_terminal is True


def test_first_learner_may_exit_while_other_continues_no_post_exit_rows():
    scenarios = build_scenarios()
    spec = scenarios["hard_braking_trace"]
    learners = build_independent_learners(DQNConfig(), seed_A=0, seed_B=1)
    records = run_pipeline_scenario(
        spec, learners, reward_condition="baseline", episode_id="pipe0", epsilon=0.0
    )
    for aid in ("A", "B"):
        rows = [r for r in records if r["controller_id"] == aid]
        seen_terminal = False
        for r in rows:
            if seen_terminal:
                raise AssertionError(f"post-exit replay row stored for {aid}")
            if r.get("controller_terminal"):
                seen_terminal = True
                assert r["bootstrap_multiplier"] == 0.0
    for r in records:
        if r.get("terminated"):
            assert r["controller_terminal"] is True
            assert r["bootstrap_multiplier"] == 0.0
