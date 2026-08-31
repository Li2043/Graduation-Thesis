"""Optional thin PettingZoo Parallel API wrapper around
``StudyBHeterogeneousEnv`` -- new_research_plan.md is explicit this is NOT
required to rewrite the environment (``mappo.py``/``shared_local_dqn.py``
train directly against ``StudyBHeterogeneousEnv``'s own Dict-keyed API, the
same shape this project's existing DQN runners already use). This wrapper
exists so ``pettingzoo.test.parallel_api_test`` can be run as an extra
compliance/reproducibility check (Phase 0's optional PettingZoo test item),
and as a standard interface if anyone later wants to plug in a
PettingZoo-native algorithm library instead.

Known deliberate deviation from strict per-agent-removal semantics: this
project's underlying env ends the WHOLE episode jointly (collision / all
completed / truncation), not per-agent -- a vehicle that finishes early
stays in ``self.agents`` (continuing to receive a no-op-equivalent action
every step, matching ``Stage10SymmetricMergeEnv.step()``'s own requirement
that every active id's action be supplied every call) until the shared
episode boundary, rather than being individually dropped from ``agents``
the moment it completes. This is a valid, common pattern for jointly-
terminating environments and does not affect ``parallel_api_test``'s core
checks (reset/step shape, space consistency, determinism)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from thesis.study_b.heterogeneous_env import StudyBEnvConfig, StudyBHeterogeneousEnv
from thesis.study_b.scenario_generator import ScenarioSpec

__all__ = ["StudyBParallelEnv"]


class StudyBParallelEnv(ParallelEnv):
    metadata = {"render_modes": [], "name": "study_b_heterogeneous_merge_v0"}

    def __init__(self, config: StudyBEnvConfig | None = None):
        self._inner = StudyBHeterogeneousEnv(config)
        self.possible_agents = ["V0", "V1", "V2", "V3"]
        self.agents: list[str] = []
        self._obs_dim = self._inner.observation_dim
        self._n_actions = self._inner.n_actions

    @lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float64)

    @lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        return spaces.Discrete(self._n_actions)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        scenario: ScenarioSpec | None = (options or {}).get("scenario")
        traffic_type: str | None = (options or {}).get("traffic_type")
        obs, info = self._inner.reset(
            seed=int(seed) if seed is not None else 0, scenario=scenario, traffic_type=traffic_type
        )
        self.agents = list(self._inner.active_vehicle_ids)
        infos = {vid: dict(info) for vid in self.agents}
        return obs, infos

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, np.ndarray], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]
    ]:
        obs, reward, terminated, truncated, info = self._inner.step(actions)
        terminations = {vid: bool(terminated) for vid in self.agents}
        truncations = {vid: bool(truncated) for vid in self.agents}
        infos = {vid: dict(info) for vid in self.agents}
        if terminated or truncated:
            self.agents = []
        return obs, reward, terminations, truncations, infos

    def render(self) -> None:  # pragma: no cover -- no renderer, matches base env
        return None

    def close(self) -> None:  # pragma: no cover -- nothing to release
        return None
