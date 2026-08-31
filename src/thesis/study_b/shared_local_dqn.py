"""Study B parameter-shared LOCAL-observation DQN -- the fallback if
MAPPO's 48-hour integration gate is not met (new_research_plan.md's
"DQN fallback" section).

Deliberately reuses ``thesis.agents.stage10_shared_dqn.SharedDQNLearner``
(Stage 10 v5's ONE-network-for-every-vehicle architecture) UNCHANGED --
new_research_plan.md is explicit that the fallback must inherit this
project's already-validated replay/target-update/epsilon-schedule/batch-size
configuration rather than being redesigned in the final weeks. This module
is a thin adapter, not a new algorithm: it only wires that learner to
Study B's LOCAL observations (never the joint/concatenated observation the
old v12 architecture used -- see this file's module-level constant
docstring for why that distinction is load-bearing here specifically) and
builds correctly-flagged ``ReplayTransition`` rows from
``heterogeneous_env.py``'s per-step info.

``Q_theta(o_i, a)`` is called independently once per vehicle every step,
all four calls sharing the SAME ``theta`` (one ``SharedDQNLearner``
instance, one replay buffer, one optimiser) -- action for vehicle i is a
function of ``o_i`` alone. Concatenating ``[o_0, o_1, o_2, o_3]`` into one
joint input (the way the old Stage 11 v12 architecture worked) would
silently leak private target-speed information back into the decision
function through the other vehicles' own self-observation slots -- exactly
what this module must NOT do."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from thesis.agents.independent_dqn_v2 import DQNConfig
from thesis.agents.replay_buffer_v2 import ReplayTransition
from thesis.agents.stage10_shared_dqn import SharedDQNLearner
from thesis.pilots.stage11_dyad_merge_pilot_config import (
    BATCH_SIZE,
    GAMMA,
    HIDDEN_SIZES,
    LEARNING_RATE_START,
    REPLAY_CAPACITY_V5,
    REPLAY_WARMUP,
    epsilon_at_step_v12,
    lr_at_step_v12,
)
from thesis.pilots.stage11_dyad_merge_pilot_config import target_mode as v12_target_mode
from thesis.study_b.local_observation import LOCAL_OBS_DIM

__all__ = [
    "ALWAYS_LEGAL_ACTION_MASK",
    "build_study_b_dqn_config",
    "SharedLocalDQNAgent",
    "epsilon_at_step_v12",
    "lr_at_step_v12",
]

# action_mask() on the underlying Stage10SymmetricMergeEnv always returns
# all-True (no role/identity-dependent restriction in this env) -- see
# stage10_symmetric_merge_env.py's own action_mask() docstring. Reused as a
# module-level constant rather than rebuilt per call site.
ALWAYS_LEGAL_ACTION_MASK: np.ndarray = np.array([True, True, True], dtype=bool)


def build_study_b_dqn_config(
    *, reward_condition: str = "baseline", device: str = "cpu", obs_dim: int = LOCAL_OBS_DIM
) -> DQNConfig:
    """Every numeric value here is copied verbatim from
    ``stage11_dyad_merge_pilot_config.py`` (independently confirmed against
    that file during this package's design: HIDDEN_SIZES=(64,64),
    LEARNING_RATE_START=0.0005, GAMMA=0.995, REPLAY_CAPACITY_V5=100_000,
    BATCH_SIZE=64, target_mode=DOUBLE) -- only ``obs_dim`` (Study B's
    LOCAL_OBS_DIM=18, not Study A's OBS_DIM=9) and ``reward_condition``
    differ from a Study A v12 config.

    ``obs_dim`` (WSC addition): defaults to ``LOCAL_OBS_DIM`` (18), so
    every existing caller that does not pass it gets EXACTLY the same
    config as before. Formal WSC continuation passes
    ``obs_dim=LOCAL_OBS_DIM_WSC`` (22) explicitly. No other hyperparameter
    (hidden_sizes, gamma, learning_rate, batch_size, replay_capacity,
    target_mode) is affected by this parameter."""
    return DQNConfig(
        obs_dim=obs_dim,
        n_actions=3,
        hidden_sizes=HIDDEN_SIZES,
        learning_rate=LEARNING_RATE_START,
        gamma=GAMMA,
        epsilon=1.0,  # placeholder -- overwritten per-step via epsilon_at_step_v12, see mappo.py's analogous comment
        replay_capacity=REPLAY_CAPACITY_V5,
        batch_size=BATCH_SIZE,
        device=device,
        reward_condition=reward_condition,
        target_mode=v12_target_mode(),
    )


class SharedLocalDQNAgent:
    def __init__(self, config: DQNConfig, *, seed: int):
        self.learner = SharedDQNLearner(config, seed=seed)
        self.config = config

    def select_actions(
        self, obs: Mapping[str, np.ndarray], *, epsilon: float, greedy: bool = False
    ) -> dict[str, int]:
        return {
            vid: self.learner.select_action(o, ALWAYS_LEGAL_ACTION_MASK, epsilon=epsilon, greedy=greedy)
            for vid, o in obs.items()
        }

    def build_transition(
        self,
        *,
        vehicle_id: str,
        observation: np.ndarray,
        action: int,
        shaped_reward: float,
        next_observation: np.ndarray | None,
        terminated: bool,
        truncated: bool,
        controller_terminal: bool,
        learner_completed: bool,
        base_reward: float = 0.0,
        shaping_component: float = 0.0,
        episode_id: str = "",
        step: int = 0,
    ) -> ReplayTransition:
        """``controller_terminal``: True iff EITHER the whole episode
        terminated this step (collision or joint success -- every vehicle's
        controller-perspective ends together) OR this specific vehicle
        individually exited this step while others continue.
        ``learner_completed``: True iff this vehicle safely exited this
        step (implies ``controller_terminal``, per ``ReplayTransition``'s
        own validation rule)."""
        next_obs = None if controller_terminal else next_observation
        next_mask = None if controller_terminal else ALWAYS_LEGAL_ACTION_MASK
        transition = ReplayTransition(
            observation=observation,
            action=action,
            shaped_reward=shaped_reward,
            next_observation=next_obs,
            terminated=terminated,
            truncated=truncated,
            action_mask=ALWAYS_LEGAL_ACTION_MASK,
            next_action_mask=next_mask,
            controller_terminal=controller_terminal,
            learner_completed=learner_completed,
            base_reward=base_reward,
            shaping_component=shaping_component,
            reward_condition=self.config.reward_condition,
            episode_id=episode_id,
            step=step,
            controller_id=vehicle_id,
        )
        transition.validate(n_actions=self.config.n_actions, obs_dim=self.config.obs_dim)
        return transition

    def store_transition(self, transition: ReplayTransition) -> None:
        self.learner.store_transition(transition)

    def maybe_update(self, *, warmup: int = REPLAY_WARMUP) -> dict | None:
        """Returns ``None`` (no-op) until the replay buffer has at least
        ``warmup`` transitions -- matches this project's existing
        REPLAY_WARMUP convention rather than updating from a
        near-empty/unrepresentative buffer. The effective threshold is
        never allowed below ``config.batch_size`` regardless of what
        ``warmup`` the caller passes -- ``ReplayBuffer.sample()`` raises if
        asked to sample more than it currently holds, so a caller-supplied
        ``warmup`` smaller than the batch size (e.g. a shortened value for
        a fast test) would otherwise crash the first update instead of
        just waiting one more transition."""
        effective_warmup = max(int(warmup), int(self.config.batch_size))
        if len(self.learner.replay) < effective_warmup:
            return None
        return self.learner.update()

    def set_learning_rate(self, lr: float) -> None:
        self.learner.set_learning_rate(lr)
