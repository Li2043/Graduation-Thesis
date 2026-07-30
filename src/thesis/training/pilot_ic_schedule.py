"""Deterministic IC schedule for Stage 5B-0 pilot training."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from thesis.envs.final_environment_config import InitialConditionBlock
from thesis.envs.merge_env_candidate_v3 import MergeEnvCandidateV3, MergeEnvCandidateV3Config
from thesis.training.final_lock_loader import FinalLockBundle


@dataclass
class ICScheduleCursor:
    cycle: int = 0
    index_in_cycle: int = 0
    assignment_bit: int = 0  # 0 = locked roles, 1 = swapped
    order: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle": int(self.cycle),
            "index_in_cycle": int(self.index_in_cycle),
            "assignment_bit": int(self.assignment_bit),
            "order": list(self.order) if self.order is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ICScheduleCursor":
        return cls(
            cycle=int(data["cycle"]),
            index_in_cycle=int(data["index_in_cycle"]),
            assignment_bit=int(data["assignment_bit"]),
            order=list(data["order"]) if data.get("order") is not None else None,
        )


class PilotICSchedule:
    """Shuffled cycle over the 12 calibration blocks with alternating role assignments."""

    def __init__(self, bundle: FinalLockBundle, *, schedule_seed: int):
        self._blocks = {
            b.block_id: b for b in bundle.environment.calibration_blocks
        }
        if len(self._blocks) != 12:
            raise ValueError(f"expected 12 calibration blocks, got {len(self._blocks)}")
        self._block_ids = sorted(self._blocks.keys())
        self._rng = np.random.default_rng(int(schedule_seed))
        self.cursor = ICScheduleCursor()
        self._ensure_order()

    def _ensure_order(self) -> None:
        if self.cursor.order is None:
            perm = self._rng.permutation(self._block_ids)
            self.cursor.order = [str(x) for x in perm.tolist()]

    def peek(self) -> tuple[InitialConditionBlock, int, str]:
        self._ensure_order()
        assert self.cursor.order is not None
        bid = self.cursor.order[self.cursor.index_in_cycle]
        assignment = int(self.cursor.assignment_bit)
        return self.materialize(bid, assignment), assignment, bid

    def advance(self) -> None:
        """Advance after an episode completes."""
        self._ensure_order()
        assert self.cursor.order is not None
        # Alternate assignment each episode; advance block after both assignments
        if self.cursor.assignment_bit == 0:
            self.cursor.assignment_bit = 1
        else:
            self.cursor.assignment_bit = 0
            self.cursor.index_in_cycle += 1
            if self.cursor.index_in_cycle >= len(self.cursor.order):
                self.cursor.index_in_cycle = 0
                self.cursor.cycle += 1
                perm = self._rng.permutation(self._block_ids)
                self.cursor.order = [str(x) for x in perm.tolist()]

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
        self.cursor = ICScheduleCursor.from_dict(payload["cursor"])
        self._rng.bit_generator.state = payload["rng_state"]


def build_env_for_block(
    bundle: FinalLockBundle,
    block: InitialConditionBlock,
    *,
    max_policy_steps: int = 400,
) -> MergeEnvCandidateV3:
    cfg = MergeEnvCandidateV3Config(
        candidate=bundle.environment.candidate,
        block=block,
        timing=bundle.timing_from_lock(),
        vehicle=bundle.vehicle_from_lock(),
        dynamics=bundle.dynamics_from_lock(),
        comfort=bundle.comfort.to_base_reward_config(),
        max_policy_steps=int(max_policy_steps),
    )
    return MergeEnvCandidateV3(cfg)


def validation_blocks_with_assignments(
    bundle: FinalLockBundle,
) -> list[tuple[str, int, InitialConditionBlock]]:
    """8 validation blocks × 2 assignments."""
    out: list[tuple[str, int, InitialConditionBlock]] = []
    for b in bundle.environment.validation_blocks:
        out.append((b.block_id, 0, b))
        out.append((b.block_id, 1, replace(b, role_A=b.role_B, role_B=b.role_A)))
    if len(out) != 16:
        raise ValueError(f"expected 16 validation episodes, got {len(out)}")
    return out


__all__ = [
    "ICScheduleCursor",
    "PilotICSchedule",
    "build_env_for_block",
    "validation_blocks_with_assignments",
]
