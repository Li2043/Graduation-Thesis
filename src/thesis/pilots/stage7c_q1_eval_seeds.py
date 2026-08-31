"""Stable SHA-256 evaluation seed namespace for Stage 7C-Q1."""

from __future__ import annotations

import hashlib
from typing import Iterable

from thesis.pilots.stage7c_q1_config import (
    ASSIGNMENTS_PER_BLOCK,
    PROTOCOL_TAG,
    n_scenario_blocks,
)


def stable_eval_seed(
    *,
    master_seed: int,
    checkpoint_step: int,
    scenario_block: int,
    protocol_tag: str = PROTOCOL_TAG,
) -> int:
    """Deterministic positive 31-bit seed. Does NOT use Python's built-in hash()."""
    payload = (
        f"{protocol_tag}|master={int(master_seed)}|"
        f"ckpt={int(checkpoint_step)}|block={int(scenario_block)}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def swap_pair_id(
    *,
    master_seed: int,
    checkpoint_step: int,
    scenario_block: int,
    protocol_tag: str = PROTOCOL_TAG,
) -> str:
    return (
        f"{protocol_tag}:ms{int(master_seed)}:ckpt{int(checkpoint_step)}:b{int(scenario_block)}"
    )


def eval_plan_for_checkpoint(
    *,
    master_seed: int,
    checkpoint_step: int,
    protocol_tag: str = PROTOCOL_TAG,
) -> list[dict[str, int | str]]:
    """Return episode specs: scenario_block × assignment sharing the same base eval_seed."""
    n_blocks = n_scenario_blocks(int(checkpoint_step))
    rows: list[dict[str, int | str]] = []
    for block in range(n_blocks):
        seed = stable_eval_seed(
            master_seed=master_seed,
            checkpoint_step=checkpoint_step,
            scenario_block=block,
            protocol_tag=protocol_tag,
        )
        pair = swap_pair_id(
            master_seed=master_seed,
            checkpoint_step=checkpoint_step,
            scenario_block=block,
            protocol_tag=protocol_tag,
        )
        for assignment in range(ASSIGNMENTS_PER_BLOCK):
            rows.append(
                {
                    "master_seed": int(master_seed),
                    "checkpoint_step": int(checkpoint_step),
                    "scenario_block": int(block),
                    "assignment": int(assignment),
                    "eval_seed": int(seed),
                    "swap_pair_id": pair,
                    "template_validation_index": int(block % 8),
                }
            )
    return rows


def assert_no_eval_seed_overlap(
    master_seeds: Iterable[int],
    checkpoint_steps: Iterable[int],
    *,
    protocol_tag: str = PROTOCOL_TAG,
) -> None:
    seen: dict[int, tuple[int, int, int]] = {}
    for ms in master_seeds:
        for ckpt in checkpoint_steps:
            for row in eval_plan_for_checkpoint(
                master_seed=int(ms),
                checkpoint_step=int(ckpt),
                protocol_tag=protocol_tag,
            ):
                # Only one seed per scenario_block (assignments share it)
                if int(row["assignment"]) != 0:
                    continue
                es = int(row["eval_seed"])
                key = (int(ms), int(ckpt), int(row["scenario_block"]))
                if es in seen and seen[es] != key:
                    raise AssertionError(
                        f"eval_seed overlap {es}: {seen[es]} vs {key}"
                    )
                seen[es] = key


__all__ = [
    "assert_no_eval_seed_overlap",
    "eval_plan_for_checkpoint",
    "stable_eval_seed",
    "swap_pair_id",
]
