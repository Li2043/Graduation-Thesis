"""V1 policy interface.

Defines the abstract contract shared by the Egoistic and Rawlsian policies.
This module contains no reinforcement-learning logic; concrete behaviour lives
in the subclasses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BasePolicy(ABC):
    """Abstract base class defining the V1 policy interface."""

    @abstractmethod
    def select_action(self, state: Any) -> int:
        """Return the greedy (deterministic) action for ``state``."""
        raise NotImplementedError

    @abstractmethod
    def act(self, state: Any, epsilon: float) -> int:
        """Return an epsilon-greedy action for ``state``."""
        raise NotImplementedError

    @abstractmethod
    def remember(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        """Store a transition.

        The policy layer is decoupled from the environment: it accepts only a
        scalar reward and never sees the environment state. Reward computation
        (egoistic vs Rawlsian) belongs to the training loop via a RewardFunction.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, batch: Any) -> float:
        """Perform one learning update from ``batch`` and return the loss."""
        raise NotImplementedError
