"""Stage 7C-Q1 Base Reward V2 active-time unit tests."""

from __future__ import annotations

import pytest

from thesis.agents.dqn_bootstrap import DQNTargetMode, compute_bootstrap_values
from thesis.pilots.stage7c_q1_config import (
    ACTIVE_TIME_COST_PER_STEP,
    CHECKPOINT_STEPS,
    MAX_STEPS,
    PILOT_SEEDS,
    assert_stage7c_q1_guards,
)
from thesis.rewards.base_reward_v2 import (
    AgentTransitionState,
    BaseRewardConfig,
    BaseRewardInputs,
    compute_base_reward_for_agents,
)
import torch
import torch.nn as nn


def _cfg(**kwargs) -> BaseRewardConfig:
    base = dict(
        a_comfort=1.5,
        a_hard=3.5,
        eta_hard_brake=0.015,
        active_time_cost_per_step=ACTIVE_TIME_COST_PER_STEP,
    )
    base.update(kwargs)
    return BaseRewardConfig(**base)


def _agent(
    *,
    pos_t: float,
    pos_t1: float,
    already_exited: bool = False,
    accel: float = 0.0,
    start: float = 0.0,
    exit_p: float = 100.0,
) -> AgentTransitionState:
    return AgentTransitionState(
        route_position_t=pos_t,
        route_position_t1=pos_t1,
        route_start=start,
        route_exit=exit_p,
        acceleration=accel,
        already_exited=already_exited,
    )


def _inputs(agents, *, collided=False, truncated=False, terminated=False):
    return BaseRewardInputs(
        agents=agents,
        stakeholder_collided={
            "A": collided,
            "B": collided,
            "B_front": False,
            "B_rear": False,
        },
        truncated=truncated,
        terminated=terminated,
    )


def test_active_ordinary_transition_charges_0005():
    out = compute_base_reward_for_agents(
        _inputs(
            {
                "A": _agent(pos_t=10.0, pos_t1=10.0),
                "B": _agent(pos_t=10.0, pos_t1=10.0),
            }
        ),
        _cfg(),
    )
    assert out["A"].active_time_component == pytest.approx(-0.0005)
    assert out["A"].active_indicator == 1.0
    assert out["A"].total_reward == pytest.approx(-0.0005)


def test_inactive_transition_no_time_cost():
    out = compute_base_reward_for_agents(
        _inputs(
            {
                "A": _agent(pos_t=100.0, pos_t1=100.0, already_exited=True),
                "B": _agent(pos_t=100.0, pos_t1=100.0, already_exited=True),
            }
        ),
        _cfg(),
    )
    assert out["A"].active_time_component == 0.0
    assert out["B"].active_time_component == 0.0


def test_exit_transition_charges_final_time_cost_and_exit_bonus():
    out = compute_base_reward_for_agents(
        _inputs(
            {
                "A": _agent(pos_t=99.0, pos_t1=100.0, already_exited=False),
                "B": _agent(pos_t=50.0, pos_t1=51.0, already_exited=False),
            }
        ),
        _cfg(),
    )
    assert out["A"].active_time_component == pytest.approx(-0.0005)
    assert out["A"].exit_component == pytest.approx(0.6)
    assert out["A"].safe_exit_event == 1.0


def test_post_exit_no_additional_time_cost():
    # After exit: already_exited True
    out = compute_base_reward_for_agents(
        _inputs(
            {
                "A": _agent(pos_t=100.0, pos_t1=100.0, already_exited=True),
                "B": _agent(pos_t=40.0, pos_t1=41.0, already_exited=False),
            }
        ),
        _cfg(),
    )
    assert out["A"].active_time_component == 0.0
    assert out["B"].active_time_component == pytest.approx(-0.0005)


def test_collision_transition_charges_time_cost():
    out = compute_base_reward_for_agents(
        _inputs(
            {
                "A": _agent(pos_t=30.0, pos_t1=31.0),
                "B": _agent(pos_t=32.0, pos_t1=33.0),
            },
            collided=True,
            terminated=True,
        ),
        _cfg(),
    )
    assert out["A"].active_time_component == pytest.approx(-0.0005)
    assert out["A"].collision_component == pytest.approx(-1.0)


def test_truncation_transition_charges_time_cost():
    out = compute_base_reward_for_agents(
        _inputs(
            {
                "A": _agent(pos_t=20.0, pos_t1=20.0),
                "B": _agent(pos_t=25.0, pos_t1=25.0),
            },
            truncated=True,
        ),
        _cfg(),
    )
    assert out["A"].active_time_component == pytest.approx(-0.0005)


def test_reward_decomposition_sums_to_total():
    out = compute_base_reward_for_agents(
        _inputs(
            {
                "A": _agent(pos_t=10.0, pos_t1=20.0, accel=-4.0),
                "B": _agent(pos_t=10.0, pos_t1=15.0),
            }
        ),
        _cfg(),
    )
    for aid in ("A", "B"):
        logs = out[aid].as_log_components()
        recon = (
            logs["reward_progress"]
            + logs["reward_exit"]
            + logs["reward_collision"]
            + logs["reward_hard_braking"]
            + logs["reward_active_time"]
        )
        assert recon == pytest.approx(logs["reward_total"])


class _Tiny(nn.Module):
    def __init__(self, q):
        super().__init__()
        self.register_buffer("q", q)

    def forward(self, x):
        return self.q.unsqueeze(0).expand(x.shape[0], -1)


def test_double_dqn_target_formula_unchanged():
    online = _Tiny(torch.tensor([0.0, 10.0, 1.0]))
    target = _Tiny(torch.tensor([1.0, 2.0, 9.0]))
    v = compute_bootstrap_values(
        online_network=online,
        target_network=target,
        next_observations=torch.zeros(1, 4),
        next_action_masks=torch.tensor([[True, True, True]]),
        mode=DQNTargetMode.DOUBLE,
    )
    # a*=1 from online; Q_target(s',1)=2
    assert float(v.item()) == 2.0


def test_truncation_bootstrap_semantics_match_stage7b():
    """Truncation is non-terminal bootstrap; controller_terminal exits are not."""
    from thesis.agents.independent_dqn_v2 import DQNConfig, IndependentDQNLearner
    from thesis.agents.replay_buffer_v2 import ReplayTransition
    import numpy as np

    cfg = DQNConfig(
        obs_dim=4,
        n_actions=3,
        hidden_sizes=(8,),
        batch_size=4,
        replay_capacity=32,
        target_mode=DQNTargetMode.DOUBLE,
        gamma=0.5,
    )
    learner = IndependentDQNLearner("A", cfg, seed=3, replay_seed=4)
    for i in range(8):
        learner.store_transition(
            ReplayTransition(
                observation=np.zeros(4),
                action=1,
                shaped_reward=1.0,
                next_observation=np.ones(4),
                terminated=False,
                truncated=True,
                controller_terminal=False,
                learner_completed=False,
                action_mask=np.array([True, True, True]),
                next_action_mask=np.array([True, True, True]),
                base_reward=1.0,
                shaping_component=0.0,
                reward_condition="baseline",
                episode_id="u",
                step=i,
                controller_id="A",
                traffic_role="ramp",
            )
        )
    stats = learner.update()
    assert stats["n_bootstrap_rows"] == 4
    assert stats["target_mode"] == "double_dqn"


def test_checkpoint_schedule_and_seeds_frozen():
    assert CHECKPOINT_STEPS == tuple(range(0, 400_001, 25_000))
    assert PILOT_SEEDS == tuple(range(64001, 64021))
    assert MAX_STEPS == 400_000


def test_config_forbids_pbrs_and_vanilla_and_over_budget():
    with pytest.raises(RuntimeError):
        assert_stage7c_q1_guards(
            algorithm="vanilla_dqn",
            condition="baseline",
            reward_shaping_enabled=False,
            shaping_coefficient=0.0,
            master_seed=64001,
            max_steps=400_000,
            active_time_cost_per_step=0.0005,
        )
    with pytest.raises(RuntimeError):
        assert_stage7c_q1_guards(
            algorithm="double_dqn",
            condition="baseline",
            reward_shaping_enabled=True,
            shaping_coefficient=0.1,
            master_seed=64001,
            max_steps=400_000,
            active_time_cost_per_step=0.0005,
        )
    with pytest.raises(RuntimeError):
        assert_stage7c_q1_guards(
            algorithm="double_dqn",
            condition="mean",
            reward_shaping_enabled=False,
            shaping_coefficient=0.0,
            master_seed=64001,
            max_steps=400_000,
            active_time_cost_per_step=0.0005,
            allow_mean_pbrs=True,
        )
    with pytest.raises(RuntimeError):
        assert_stage7c_q1_guards(
            algorithm="double_dqn",
            condition="baseline",
            reward_shaping_enabled=False,
            shaping_coefficient=0.0,
            master_seed=64001,
            max_steps=500_000,
            active_time_cost_per_step=0.0005,
        )
