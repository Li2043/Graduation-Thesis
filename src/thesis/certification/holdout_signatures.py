"""Physical IC block signatures for calibration/validation separation (Stage 4A-0R).

Different IDs or seeds do **not** make identical deterministic physics a distinct
holdout case. This module only detects duplicates; it does not create replacement
blocks.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from thesis.envs.final_environment_config import InitialConditionBlock


def physical_block_signature(block: InitialConditionBlock, *, ndigits: int = 6) -> tuple:
    """Role-independent physical signature of an initial-condition block."""

    def r(x: float) -> float:
        return round(float(x), ndigits)

    # Sort learner physical slots by role-independent geometry, not controller ID.
    return (
        r(block.spawn_route_mainline),
        r(block.spawn_route_ramp),
        r(block.spawn_speed_mainline),
        r(block.spawn_speed_ramp),
        r(block.delta_arrival),
        r(block.background_time_headway),
        r(block.spawn_route_B_front),
        r(block.spawn_route_B_rear),
        r(block.spawn_speed_B_front),
        r(block.spawn_speed_B_rear),
    )


def find_duplicate_signatures(
    calibration: Sequence[InitialConditionBlock],
    validation: Sequence[InitialConditionBlock],
) -> list[dict]:
    """Return duplicate signature records across calibration vs validation."""
    cal_map: dict[tuple, list[str]] = {}
    for b in calibration:
        cal_map.setdefault(physical_block_signature(b), []).append(b.block_id)
    dupes: list[dict] = []
    for b in validation:
        sig = physical_block_signature(b)
        if sig in cal_map:
            dupes.append(
                {
                    "signature": sig,
                    "validation_block_id": b.block_id,
                    "calibration_block_ids": list(cal_map[sig]),
                }
            )
    return dupes


def assert_no_duplicate_holdout(
    calibration: Sequence[InitialConditionBlock],
    validation: Sequence[InitialConditionBlock],
) -> None:
    dupes = find_duplicate_signatures(calibration, validation)
    if dupes:
        raise ValueError(f"duplicate calibration/validation physical signatures: {dupes}")


def signatures_of(blocks: Iterable[InitialConditionBlock]) -> set[tuple]:
    return {physical_block_signature(b) for b in blocks}
