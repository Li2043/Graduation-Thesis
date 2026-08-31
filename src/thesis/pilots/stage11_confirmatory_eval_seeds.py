"""Stable SHA-256 evaluation seed namespace + held-out validation blocks for
the Stage 11 formal confirmatory protocol (DRAFT -- see STAGE11_PROTOCOL.md).

Design note (found while implementing this, not assumed in advance): unlike
chapter3's `InitialConditionBlock` (an explicit dataclass with `role_A`/
`role_B` fields that can be swapped independently of the scenario), Stage 11's
`Stage10SymmetricMergeEnv.reset(seed=...)` derives BOTH the role permutation
AND the spawn-position jitter from the SAME seeded RNG stream
(`np.random.default_rng(seed)`, role permutation drawn first, jitter drawn
second -- see `stage10_symmetric_merge_env.py` lines ~607-654). There is no
field to swap independently of the seed. For n_vehicles=2 there are only two
possible role permutations, so "assignment 1" (the swapped role case) is
implemented here as a deterministic, reproducible SEARCH over a small,
reserved offset window from the block's base seed, stopping at the first
candidate seed whose role permutation differs from assignment 0's. This
means the two assignments of a block do NOT share byte-identical spawn
jitter (unlike chapter3's exact field-swap) -- a minor loss of matching
precision relative to chapter3's design, not a correctness bug: both
episodes are still deterministic, reproducible, SHA-256-derived, and held
out from the training seed range by construction (see module docstring
below on the numeric-range argument).
"""

from __future__ import annotations

import hashlib
from typing import Iterable

from thesis.envs.stage10_symmetric_merge_env import Stage10MergeEnvConfig, Stage10SymmetricMergeEnv
from thesis.pilots.stage11_confirmatory_config import (
    ASSIGNMENTS_PER_BLOCK,
    N_VALIDATION_BLOCKS,
    PROTOCOL_TAG,
    STAGE11_RESERVED_SEED_BLOCK,
)

_ROLE_SEARCH_MAX_OFFSET = 1000
_TRAINING_SEED_MULTIPLIER = 1_000_003  # matches stage11_dyad_merge_runner.py's env.reset(seed=master_seed*1_000_003+step)


def _eval_env_config() -> Stage10MergeEnvConfig:
    """Matches stage11_dyad_merge_runner.py's env_config_kwargs exactly
    (n_vehicles=2, spawn_route_lead=0.0, include_role_zone_features=True) --
    imported by value, not duplicated as magic numbers, would be preferable,
    but the runner builds this dict inline rather than exporting it; this
    docstring is the cross-check trail if the runner's kwargs ever change."""
    return Stage10MergeEnvConfig(n_vehicles=2, spawn_route_lead=0.0, include_role_zone_features=True)


def stable_block_seed(*, block_index: int, protocol_tag: str = PROTOCOL_TAG) -> int:
    """Deterministic positive seed for a held-out validation block's
    assignment-0 episode. Never Python's built-in hash().

    Numeric-range argument for "held out from training": training reset
    seeds are always `master_seed * 1_000_003 + step` for
    `master_seed` in the confirmatory formal blocks (>=69121) and
    `step` in [0, 400000) -- i.e. always >= 69121 * 1_000_003 =
    69,121,000,363. This function (and stable_eval_seed below) reduce
    modulo 2**31-1 =~ 2.1 billion, an entirely different numeric range
    with zero possibility of collision -- not a probabilistic argument,
    a structural one.
    """
    payload = f"{protocol_tag}|validation-block|{int(block_index)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def stable_eval_seed(
    *,
    master_seed: int,
    checkpoint_step: int,
    scenario_block: int,
    protocol_tag: str = PROTOCOL_TAG,
) -> int:
    """Deterministic positive 31-bit seed, keyed by (master_seed, checkpoint,
    block) -- mirrors stage7c_q1_eval_seeds.stable_eval_seed's exact shape."""
    payload = (
        f"{protocol_tag}|master={int(master_seed)}|"
        f"ckpt={int(checkpoint_step)}|block={int(scenario_block)}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def role_of_ramp(seed: int) -> str:
    """Which physical vehicle id (V0/V1) plays 'ramp' under this reset seed."""
    env = Stage10SymmetricMergeEnv(_eval_env_config())
    _, info = env.reset(seed=int(seed))
    roles = dict(info["roles"])
    for vid, role in roles.items():
        if role == "ramp":
            return str(vid)
    raise RuntimeError(f"no vehicle assigned 'ramp' role for seed {seed}: {roles}")


def find_swapped_assignment_seed(base_seed: int, *, max_offset: int = _ROLE_SEARCH_MAX_OFFSET) -> int:
    """First seed in `base_seed+1, base_seed+2, ...` whose ramp-role
    assignment DIFFERS from `base_seed`'s -- deterministic, reproducible
    (see module docstring for why this search is needed instead of an
    explicit field swap)."""
    base_ramp = role_of_ramp(base_seed)
    for offset in range(1, max_offset + 1):
        candidate = base_seed + offset
        if role_of_ramp(candidate) != base_ramp:
            return candidate
    raise RuntimeError(
        f"no seed within +{max_offset} of {base_seed} produced a swapped role assignment "
        "-- role permutation may not be varying with seed as expected"
    )


def eval_plan_for_checkpoint(
    *,
    master_seed: int,
    checkpoint_step: int,
    protocol_tag: str = PROTOCOL_TAG,
    n_blocks: int = N_VALIDATION_BLOCKS,
) -> list[dict[str, int | str]]:
    """Return episode specs: block x assignment. Assignment 0 uses the
    block's own eval_seed directly; assignment 1 uses the seed-search
    swapped-role variant (see find_swapped_assignment_seed)."""
    rows: list[dict[str, int | str]] = []
    for block in range(n_blocks):
        seed0 = stable_eval_seed(
            master_seed=master_seed,
            checkpoint_step=checkpoint_step,
            scenario_block=block,
            protocol_tag=protocol_tag,
        )
        seed1 = find_swapped_assignment_seed(seed0)
        for assignment, seed in ((0, seed0), (1, seed1)):
            rows.append(
                {
                    "master_seed": int(master_seed),
                    "checkpoint_step": int(checkpoint_step),
                    "scenario_block": int(block),
                    "assignment": int(assignment),
                    "eval_seed": int(seed),
                }
            )
    if len(rows) != n_blocks * ASSIGNMENTS_PER_BLOCK:
        raise RuntimeError(f"expected {n_blocks * ASSIGNMENTS_PER_BLOCK} rows, got {len(rows)}")
    return rows


def assert_no_eval_seed_overlap(
    master_seeds: Iterable[int],
    checkpoint_steps: Iterable[int],
    *,
    protocol_tag: str = PROTOCOL_TAG,
) -> None:
    seen: dict[int, tuple[int, int, int, int]] = {}
    for ms in master_seeds:
        for ckpt in checkpoint_steps:
            for row in eval_plan_for_checkpoint(
                master_seed=int(ms), checkpoint_step=int(ckpt), protocol_tag=protocol_tag
            ):
                es = int(row["eval_seed"])
                key = (int(ms), int(ckpt), int(row["scenario_block"]), int(row["assignment"]))
                if es in seen and seen[es] != key:
                    raise AssertionError(f"eval_seed overlap {es}: {seen[es]} vs {key}")
                seen[es] = key


def assert_eval_seeds_disjoint_from_pilot_and_training(
    *,
    confirmatory_master_seeds: Iterable[int],
    pilot_seed_block: tuple[int, ...] = STAGE11_RESERVED_SEED_BLOCK,
) -> None:
    """Structural check (not just a spot check): every confirmatory master
    seed must fall outside the pilot's reserved block, and every derived
    eval_seed (bounded < 2**31-1) must fall outside the numeric range any
    training reset seed could ever take (>= smallest master_seed *
    1_000_003)."""
    pilot_set = set(pilot_seed_block)
    min_master = min(confirmatory_master_seeds)
    min_possible_training_seed = min_master * _TRAINING_SEED_MULTIPLIER
    for ms in confirmatory_master_seeds:
        if ms in pilot_set:
            raise AssertionError(f"confirmatory master seed {ms} overlaps the pilot block")
    if (2**31 - 1) >= min_possible_training_seed:
        raise AssertionError(
            "eval_seed numeric range is not structurally disjoint from the training "
            f"seed range (min possible training seed {min_possible_training_seed} <= 2**31-1)"
        )


__all__ = [
    "assert_eval_seeds_disjoint_from_pilot_and_training",
    "assert_no_eval_seed_overlap",
    "eval_plan_for_checkpoint",
    "find_swapped_assignment_seed",
    "role_of_ramp",
    "stable_block_seed",
    "stable_eval_seed",
]
