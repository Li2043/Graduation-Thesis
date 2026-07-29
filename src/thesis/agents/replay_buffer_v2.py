"""Replay buffer for Independent DQN (Stage 2B-2).

Circular FIFO buffer: when capacity is exceeded, the oldest transition is
overwritten (head advances). Behaviour is documented here and in the report.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from thesis.agents.action_masking import validate_action_mask


def _finite_array(name: str, arr: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values: {arr}")
    return arr


@dataclass
class ReplayTransition:
    observation: np.ndarray
    action: int
    shaped_reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool
    action_mask: np.ndarray
    next_action_mask: np.ndarray
    # Diagnostics
    base_reward: float = 0.0
    shaping_component: float = 0.0
    reward_condition: str = "baseline"
    episode_id: str = ""
    step: int = 0
    controller_id: str = ""
    traffic_role: str = ""

    def validate(self, *, n_actions: int, obs_dim: int) -> None:
        if bool(self.terminated) and bool(self.truncated):
            raise ValueError(
                "invalid transition: terminated=True and truncated=True simultaneously"
            )
        obs = np.asarray(self.observation, dtype=np.float64)
        nobs = np.asarray(self.next_observation, dtype=np.float64)
        if obs.shape != (obs_dim,):
            raise ValueError(
                f"observation dim {obs.shape} inconsistent with network input {(obs_dim,)}"
            )
        if nobs.shape != (obs_dim,):
            raise ValueError(
                f"next_observation dim {nobs.shape} inconsistent with network input {(obs_dim,)}"
            )
        _finite_array("observation", obs)
        _finite_array("next_observation", nobs)
        for name in ("shaped_reward", "base_reward", "shaping_component"):
            v = float(getattr(self, name))
            if not math.isfinite(v):
                raise ValueError(f"{name} must be finite, got {v}")
        mask = validate_action_mask(self.action_mask, n_actions)
        next_mask = validate_action_mask(self.next_action_mask, n_actions)
        a = int(self.action)
        if a < 0 or a >= n_actions:
            raise ValueError(f"action {a} out of range for n_actions={n_actions}")
        if not bool(mask[a]):
            raise ValueError(
                f"illegal stored action {a} under action_mask={mask.tolist()}"
            )
        self.observation = obs
        self.next_observation = nobs
        self.action_mask = mask
        self.next_action_mask = next_mask
        # Truncation must retain next obs/mask (already required by schema)
        if self.truncated and self.next_observation is None:
            raise ValueError("truncated transition missing next_observation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.tolist(),
            "action": int(self.action),
            "shaped_reward": float(self.shaped_reward),
            "next_observation": self.next_observation.tolist(),
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
            "action_mask": self.action_mask.astype(bool).tolist(),
            "next_action_mask": self.next_action_mask.astype(bool).tolist(),
            "base_reward": float(self.base_reward),
            "shaping_component": float(self.shaping_component),
            "reward_condition": self.reward_condition,
            "episode_id": self.episode_id,
            "step": int(self.step),
            "controller_id": self.controller_id,
            "traffic_role": self.traffic_role,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReplayTransition":
        return cls(
            observation=np.asarray(data["observation"], dtype=np.float64),
            action=int(data["action"]),
            shaped_reward=float(data["shaped_reward"]),
            next_observation=np.asarray(data["next_observation"], dtype=np.float64),
            terminated=bool(data["terminated"]),
            truncated=bool(data["truncated"]),
            action_mask=np.asarray(data["action_mask"], dtype=bool),
            next_action_mask=np.asarray(data["next_action_mask"], dtype=bool),
            base_reward=float(data.get("base_reward", 0.0)),
            shaping_component=float(data.get("shaping_component", 0.0)),
            reward_condition=str(data.get("reward_condition", "baseline")),
            episode_id=str(data.get("episode_id", "")),
            step=int(data.get("step", 0)),
            controller_id=str(data.get("controller_id", "")),
            traffic_role=str(data.get("traffic_role", "")),
        )


@dataclass
class ReplayBatch:
    observations: np.ndarray
    actions: np.ndarray
    shaped_rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    action_masks: np.ndarray
    next_action_masks: np.ndarray
    base_rewards: np.ndarray
    shaping_components: np.ndarray
    reward_conditions: list[str]
    indices: np.ndarray
    transitions: list[ReplayTransition] = field(default_factory=list)


class ReplayBuffer:
    """Circular FIFO replay buffer (overwrite oldest when full)."""

    def __init__(
        self,
        capacity: int,
        *,
        obs_dim: int,
        n_actions: int,
        seed: int = 0,
    ):
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self._storage: list[ReplayTransition | None] = [None] * self.capacity
        self._size = 0
        self._write = 0
        self._rng = np.random.default_rng(int(seed))
        self.seed = int(seed)

    def __len__(self) -> int:
        return self._size

    def append(self, transition: ReplayTransition) -> None:
        transition.validate(n_actions=self.n_actions, obs_dim=self.obs_dim)
        self._storage[self._write] = transition
        self._write = (self._write + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self._size < batch_size:
            raise ValueError(
                f"cannot sample {batch_size} from buffer of size {self._size}"
            )
        indices = self._rng.choice(self._size, size=batch_size, replace=False)
        # Physical indices in circular buffer: oldest at (write - size) mod capacity
        start = (self._write - self._size) % self.capacity
        physical = [(start + int(i)) % self.capacity for i in indices]
        transitions = [self._storage[p] for p in physical]
        assert all(t is not None for t in transitions)
        return self._stack(transitions, np.asarray(indices, dtype=np.int64))  # type: ignore[arg-type]

    def _stack(
        self, transitions: Sequence[ReplayTransition], indices: np.ndarray
    ) -> ReplayBatch:
        return ReplayBatch(
            observations=np.stack([t.observation for t in transitions]),
            actions=np.asarray([t.action for t in transitions], dtype=np.int64),
            shaped_rewards=np.asarray(
                [t.shaped_reward for t in transitions], dtype=np.float64
            ),
            next_observations=np.stack([t.next_observation for t in transitions]),
            terminated=np.asarray([t.terminated for t in transitions], dtype=bool),
            truncated=np.asarray([t.truncated for t in transitions], dtype=bool),
            action_masks=np.stack([t.action_mask for t in transitions]),
            next_action_masks=np.stack([t.next_action_mask for t in transitions]),
            base_rewards=np.asarray(
                [t.base_reward for t in transitions], dtype=np.float64
            ),
            shaping_components=np.asarray(
                [t.shaping_component for t in transitions], dtype=np.float64
            ),
            reward_conditions=[t.reward_condition for t in transitions],
            indices=indices,
            transitions=list(transitions),
        )

    def serialize(self) -> str:
        items = []
        start = (self._write - self._size) % self.capacity
        for i in range(self._size):
            t = self._storage[(start + i) % self.capacity]
            assert t is not None
            items.append(t.to_dict())
        return json.dumps(
            {
                "capacity": self.capacity,
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
                "seed": self.seed,
                "transitions": items,
            }
        )

    @classmethod
    def deserialize(cls, payload: str) -> "ReplayBuffer":
        data = json.loads(payload)
        buf = cls(
            int(data["capacity"]),
            obs_dim=int(data["obs_dim"]),
            n_actions=int(data["n_actions"]),
            seed=int(data.get("seed", 0)),
        )
        for item in data["transitions"]:
            buf.append(ReplayTransition.from_dict(item))
        return buf
