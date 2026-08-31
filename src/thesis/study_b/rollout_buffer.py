"""On-policy rollout storage for Study B's MAPPO -- new_research_plan.md's
"No replay buffer... transitions retained only in the current rollout
buffer" section. Deliberately simple (Python lists + a final numpy/torch
conversion), sized for one PPO update's worth of data, then discarded --
not a circular/persistent buffer the way the DQN fallback's replay is.

Team-reward MAPPO shape: one scalar reward/value/advantage/return PER
TIMESTEP (not per agent -- see pbrs_reward.py's ``apply_team``), but
per-agent observations/actions/log-probs (one shared policy evaluated once
per agent per timestep)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["RolloutBuffer"]


@dataclass
class RolloutBuffer:
    agent_ids: tuple[str, ...]

    obs: list[dict[str, np.ndarray]] = field(default_factory=list)
    global_states: list[np.ndarray] = field(default_factory=list)
    actions: list[dict[str, int]] = field(default_factory=list)
    log_probs: list[dict[str, float]] = field(default_factory=list)
    team_rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)  # terminated OR truncated

    def add(
        self,
        *,
        obs: dict[str, np.ndarray],
        global_state: np.ndarray,
        actions: dict[str, int],
        log_probs: dict[str, float],
        team_reward: float,
        value: float,
        done: bool,
    ) -> None:
        self.obs.append(obs)
        self.global_states.append(global_state)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.team_rewards.append(float(team_reward))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def __len__(self) -> int:
        return len(self.team_rewards)

    def clear(self) -> None:
        self.obs.clear()
        self.global_states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.team_rewards.clear()
        self.values.clear()
        self.dones.clear()

    def compute_gae(
        self, *, last_value: float, gamma: float, gae_lambda: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Standard GAE over the team reward/value sequence. ``last_value``
        is the critic's estimate of the state AFTER the final stored
        transition (0.0 if that transition was a true terminal, i.e. the
        caller is responsible for passing 0.0 rather than a real bootstrap
        whenever ``dones[-1]`` came from ``terminated`` rather than
        ``truncated`` -- this buffer does not distinguish the two itself,
        matching new_research_plan.md's PBRS potential convention where
        that distinction is applied one level up, in
        ``pbrs_reward.compute_potential``/``actual_potential``)."""
        T = len(self)
        rewards = np.asarray(self.team_rewards, dtype=np.float64)
        values = np.asarray(self.values, dtype=np.float64)
        dones = np.asarray(self.dones, dtype=np.float64)

        advantages = np.zeros(T, dtype=np.float64)
        gae = 0.0
        next_value = last_value
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * mask - values[t]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[t] = gae
            next_value = values[t]
        returns = advantages + values
        return advantages, returns
