"""Formal IC schedule: 24-episode shuffled cycles (12 blocks × 2 assignments)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from thesis.envs.final_environment_config import InitialConditionBlock
from thesis.training.final_lock_loader import FinalLockBundle
from thesis.training.pilot_ic_schedule import build_env_for_block


@dataclass
class FormalScheduleCursor:
    cycle: int = 0
    index_in_cycle: int = 0
    order: list[tuple[str, int]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle": int(self.cycle),
            "index_in_cycle": int(self.index_in_cycle),
            "order": [list(x) for x in self.order] if self.order is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FormalScheduleCursor":
        order = data.get("order")
        return cls(
            cycle=int(data["cycle"]),
            index_in_cycle=int(data["index_in_cycle"]),
            order=[(str(a), int(b)) for a, b in order] if order is not None else None,
        )


class FormalICSchedule:
    """Every complete cycle contains each (block, assignment) pair exactly once."""

    CYCLE_LENGTH = 24

    def __init__(self, bundle: FinalLockBundle, *, schedule_seed: int):
        self._blocks = {b.block_id: b for b in bundle.environment.calibration_blocks}
        if len(self._blocks) != 12:
            raise ValueError(f"expected 12 calibration blocks, got {len(self._blocks)}")
        self._block_ids = sorted(self._blocks.keys())
        self._rng = np.random.default_rng(int(schedule_seed))
        self.cursor = FormalScheduleCursor()
        self._ensure_order()

    def _all_pairs(self) -> list[tuple[str, int]]:
        return [(bid, a) for bid in self._block_ids for a in (0, 1)]

    def _ensure_order(self) -> None:
        if self.cursor.order is None:
            pairs = self._all_pairs()
            assert len(pairs) == self.CYCLE_LENGTH
            perm = self._rng.permutation(len(pairs))
            self.cursor.order = [pairs[int(i)] for i in perm.tolist()]

    def peek(self) -> tuple[InitialConditionBlock, int, str]:
        self._ensure_order()
        assert self.cursor.order is not None
        bid, assignment = self.cursor.order[self.cursor.index_in_cycle]
        return self.materialize(bid, assignment), int(assignment), str(bid)

    def advance(self) -> None:
        self._ensure_order()
        assert self.cursor.order is not None
        self.cursor.index_in_cycle += 1
        if self.cursor.index_in_cycle >= len(self.cursor.order):
            self.cursor.index_in_cycle = 0
            self.cursor.cycle += 1
            pairs = self._all_pairs()
            perm = self._rng.permutation(len(pairs))
            self.cursor.order = [pairs[int(i)] for i in perm.tolist()]

    def materialize(self, block_id: str, assignment: int) -> InitialConditionBlock:
        base = self._blocks[block_id]
        if int(assignment) % 2 == 0:
            return base
        return replace(base, role_A=base.role_B, role_B=base.role_A)

    def export_state(self) -> dict[str, Any]:
        return {
            "cursor": self.cursor.to_dict(),
            "rng_state": self._rng.bit_generator.state,
            "block_ids": list(self._block_ids),
        }

    def import_state(self, payload: dict[str, Any]) -> None:
        self.cursor = FormalScheduleCursor.from_dict(payload["cursor"])
        self._rng.bit_generator.state = payload["rng_state"]


def evaluation_episode_seed(
    evaluation_seed: int,
    *,
    checkpoint_index: int,
    block_index: int,
    assignment_index: int,
) -> int:
    """Locked formal evaluation episode seed formula."""
    return (
        int(evaluation_seed)
        + 1000 * int(checkpoint_index)
        + 2 * int(block_index)
        + int(assignment_index)
    )


__all__ = [
    "FormalICSchedule",
    "FormalScheduleCursor",
    "build_env_for_block",
    "evaluation_episode_seed",
]
