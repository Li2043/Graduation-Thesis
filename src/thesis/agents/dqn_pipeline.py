"""Env → reward-condition → Independent DQN replay pipeline (Stage 2B-2).

Completed-controller policy (Option 1)
--------------------------------------
When a learning controller safely exits:

- its stakeholder experience remains E_i = 1 in potentials;
- it becomes inactive for physical control (env forces zero accel);
- the joint step API may still receive a placeholder MAINTAIN for that id;
- replay storage for that controller **stops after the exit transition**;
- no fictitious MERGE / NO_OP action is invented beyond the existing 3-action set.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from thesis.agents.action_masking import role_action_mask
from thesis.agents.independent_dqn_v2 import (
    DQNConfig,
    IndependentDQNLearner,
    RewardCondition,
    build_independent_learners,
    select_learner_reward,
)
from thesis.agents.replay_buffer_v2 import ReplayTransition
from thesis.envs.merge_env_v2 import HighLevelAction, MergeEnvConfig, MergeEnvV2
from thesis.envs.scripted_scenarios import ScenarioSpec, build_scenarios

RewardConditionName = Literal["baseline", "mean_pbrs", "min_pbrs"]


def build_transition_for_controller(
    *,
    controller_id: str,
    obs: np.ndarray,
    next_obs: np.ndarray,
    action: int,
    action_mask: np.ndarray,
    next_action_mask: np.ndarray,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
    reward_condition: RewardConditionName,
    episode_id: str,
) -> ReplayTransition:
    diag = info["diagnostics"]["per_agent"][controller_id]
    base = float(diag["base_total"])
    learner_r, shaping = select_learner_reward(
        condition=reward_condition,
        base_reward=base,
        scaled_mean_shaping=float(diag["scaled_mean_shaping"]),
        scaled_min_shaping=float(diag["scaled_min_shaping"]),
    )
    role = info["vehicles_t"][controller_id]["role"]
    return ReplayTransition(
        observation=np.asarray(obs, dtype=np.float64),
        action=int(action),
        shaped_reward=learner_r,
        next_observation=np.asarray(next_obs, dtype=np.float64),
        terminated=bool(terminated),
        truncated=bool(truncated),
        action_mask=np.asarray(action_mask, dtype=bool),
        next_action_mask=np.asarray(next_action_mask, dtype=bool),
        base_reward=base,
        shaping_component=shaping,
        reward_condition=reward_condition,
        episode_id=episode_id,
        step=int(info["step"]),
        controller_id=controller_id,
        traffic_role=str(role),
    )


def run_pipeline_scenario(
    spec: ScenarioSpec,
    learners: dict[str, IndependentDQNLearner],
    *,
    reward_condition: RewardConditionName,
    episode_id: str,
    epsilon: float = 0.0,
) -> list[dict[str, Any]]:
    """Run one scripted/learner scenario; store transitions under Option 1."""
    env = MergeEnvV2(spec.config)
    obs, reset_info = env.reset(seed=spec.config.seed)
    active = {"A": True, "B": True}
    records: list[dict[str, Any]] = []

    # Prefer scripted actions when provided; else learner selection
    scripted = list(spec.actions)
    step_i = 0
    while True:
        masks = {
            aid: role_action_mask(env._role_of(aid), n_actions=3)
            for aid in ("A", "B")
        }
        actions: dict[str, int] = {}
        for aid in ("A", "B"):
            if not active[aid]:
                # Placeholder for joint API only; env ignores accel when completed.
                actions[aid] = int(HighLevelAction.MAINTAIN)
            elif step_i < len(scripted):
                actions[aid] = int(scripted[step_i][aid])
            else:
                actions[aid] = learners[aid].select_action(
                    obs[aid], masks[aid], epsilon=epsilon, greedy=epsilon <= 0.0
                )

        next_obs, _reward, terminated, truncated, info = env.step(actions)
        next_masks = {
            aid: role_action_mask(info["vehicles_t1"][aid]["role"], n_actions=3)
            for aid in ("A", "B")
        }

        for aid in ("A", "B"):
            if not active[aid]:
                continue
            tr = build_transition_for_controller(
                controller_id=aid,
                obs=obs[aid],
                next_obs=next_obs[aid],
                action=actions[aid],
                action_mask=masks[aid],
                next_action_mask=next_masks[aid],
                terminated=terminated,
                truncated=truncated,
                info=info,
                reward_condition=reward_condition,
                episode_id=episode_id,
            )
            learners[aid].store_transition(tr)
            target = learners[aid].compute_target_for_transition(tr)
            records.append(
                {
                    "scenario_id": spec.scenario_id,
                    "episode_id": episode_id,
                    "step": info["step"],
                    "controller_id": aid,
                    "traffic_role": tr.traffic_role,
                    "selected_action": tr.action,
                    "action_mask": tr.action_mask.tolist(),
                    "reward_condition": reward_condition,
                    "base_reward": tr.base_reward,
                    "shaping_component": tr.shaping_component,
                    "learner_reward": tr.shaped_reward,
                    "terminated": tr.terminated,
                    "truncated": tr.truncated,
                    "observation": tr.observation.tolist(),
                    "next_observation": tr.next_observation.tolist(),
                    "next_action_mask": tr.next_action_mask.tolist(),
                    "target": target.target,
                    "bootstrap_multiplier": target.bootstrap_multiplier,
                    "masked_next_q_max": target.masked_next_q_max,
                    "target_decomposition_valid": abs(
                        target.target
                        - (
                            target.reward
                            + target.gamma
                            * target.bootstrap_multiplier
                            * target.masked_next_q_max
                        )
                    )
                    < 1e-12,
                    "experiences_t1": info["diagnostics"]["stakeholder_experiences_t1"],
                    "actual_mean_potential_t1": info["diagnostics"][
                        "actual_mean_potential_t1"
                    ],
                    "actual_min_potential_t1": info["diagnostics"][
                        "actual_min_potential_t1"
                    ],
                }
            )
            # Option 1: stop storing after exit transition for this controller
            if info["events"]["exit_event"].get(aid, 0.0) >= 1.0:
                active[aid] = False

        obs = next_obs
        step_i += 1
        if terminated or truncated:
            break
        if step_i >= max(len(scripted), 1) and not scripted:
            break
        # If using only scripted actions, stop when scripts exhausted unless episode ends
        if scripted and step_i >= len(scripted):
            break

    return records


def default_learners(
    *,
    seed_A: int = 0,
    seed_B: int = 1,
    reward_condition: RewardConditionName = "baseline",
) -> dict[str, IndependentDQNLearner]:
    cfg = DQNConfig(reward_condition=reward_condition)
    return build_independent_learners(cfg, seed_A=seed_A, seed_B=seed_B)
