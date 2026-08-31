"""Episode event registries for exits and stakeholder collisions (Stage 2B-1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from thesis.rewards.base_reward_v2 import STAKEHOLDER_SET


@dataclass
class TransitionEvents:
    """Events detected on a single s_t -> s_{t+1} transition."""

    exit_event: dict[str, float] = field(
        default_factory=lambda: {"A": 0.0, "B": 0.0}
    )
    stakeholder_collided: dict[str, bool] = field(
        default_factory=lambda: {sid: False for sid in STAKEHOLDER_SET}
    )
    collision_pairs: list[tuple[str, str]] = field(default_factory=list)
    stakeholder_collision_event: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def finalise(self) -> None:
        keys = set(self.stakeholder_collided.keys())
        expected = set(STAKEHOLDER_SET)
        if keys != expected:
            raise ValueError(
                f"stakeholder_collided must equal {list(STAKEHOLDER_SET)}; "
                f"got {sorted(keys)}"
            )
        self.stakeholder_collision_event = (
            1.0 if any(self.stakeholder_collided.values()) else 0.0
        )


@dataclass
class MergeVehicleRegistry:
    """Stable episode identities for the fixed stakeholder set V."""

    identities: tuple[str, ...] = STAKEHOLDER_SET
    roles: dict[str, str] = field(default_factory=dict)
    completed: dict[str, bool] = field(
        default_factory=lambda: {"A": False, "B": False}
    )

    def validate(self) -> None:
        if tuple(self.identities) != STAKEHOLDER_SET:
            raise ValueError(
                f"registry identities must be exactly {STAKEHOLDER_SET}, "
                f"got {self.identities}"
            )
        for sid in ("A", "B"):
            if sid not in self.roles:
                raise ValueError(f"missing traffic role for learning controller {sid}")
            if self.roles[sid] not in {"mainline", "ramp"}:
                raise ValueError(f"invalid role for {sid}: {self.roles[sid]!r}")
        for sid in ("B_front", "B_rear"):
            self.roles.setdefault(sid, "mainline")

    def mark_completed(self, controller: str) -> None:
        if controller not in ("A", "B"):
            raise ValueError(f"only learning controllers may complete, got {controller}")
        self.completed[controller] = True


def record_collision_pairs(
    collided_ids: Iterable[str],
) -> tuple[dict[str, bool], list[tuple[str, str]]]:
    """Build exact four-key registry and unordered collision pairs among V."""
    flags = {sid: False for sid in STAKEHOLDER_SET}
    ids = [sid for sid in collided_ids if sid in flags]
    for sid in ids:
        flags[sid] = True
    pairs: list[tuple[str, str]] = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            pairs.append(tuple(sorted((a, b))))
    return flags, pairs
