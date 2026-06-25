"""Shared merge-task signal (task-completion utility).

Derives a task-completion adjustment for the highway merge from raw, already
exposed state signals only. It does NOT import the environment, access env
internals, or use the experience function. The same adjustment is added to both
the Egoistic and the Rawlsian scalar reward so that both conditions solve the
identical merge task while differing only in their underlying objective
structure (individual utility vs. maximin experience).

The adjustment is terminal: a one-off merge-success bonus when the ego agent
completes the merge, or a one-off non-merge failure penalty when the episode
ends (max steps / end of rollout) with the ego agent unmerged and not crashed.
Collisions keep their existing handling and receive no extra non-merge penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

# Lane convention (matches the V1 environment's exposed ``lane`` field): the main
# carriageway is lane 0 and the on-ramp is lane 1. Documented here because the
# reward layer must not import the environment.
DEFAULT_MAIN_LANE_INDEX = 0

# Longitudinal position (same units as ``env_state["position"]``) at/after which
# a vehicle in the main lane is considered to have completed the merge. This is
# only a STATE-BASED FALLBACK: the training loop prefers the authoritative
# ``merged`` flag exposed in env ``info``. The value mirrors the environment's
# merge-completion line; it is configurable to avoid a fragile hard-coded magic.
DEFAULT_MERGE_COMPLETE_POSITION = 100.0


@dataclass(frozen=True)
class MergeTaskConfig:
    """Configurable constants for the shared merge-task / safety adjustments."""

    merge_success_bonus: float = 1.0
    non_merge_failure_penalty: float = 1.0
    # Shared terminal collision penalty applied identically to BOTH conditions.
    # Subtracted from the learning reward whenever the ego agent crashes. This is
    # a task-layer safety constraint and is independent of (and additional to)
    # any per-step collision penalty the egoistic objective already carries.
    terminal_collision_penalty: float = 10.0
    main_lane_index: int = DEFAULT_MAIN_LANE_INDEX
    merge_complete_position: float = DEFAULT_MERGE_COMPLETE_POSITION


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_collision(ego_state: Mapping[str, Any]) -> bool:
    """True if the ego agent's raw state reports a crash (env collision_flag)."""
    return bool(ego_state.get("crashed", False))


def is_merge_completed(
    ego_state: Mapping[str, Any], config: Optional[MergeTaskConfig] = None
) -> bool:
    """State-based merge-completion test: in the main lane past the merge line.

    Used as a fallback when the authoritative ``merged`` flag is not supplied.
    """
    config = config or MergeTaskConfig()
    lane = ego_state.get("lane")
    try:
        lane_ok = int(lane) == config.main_lane_index
    except (TypeError, ValueError):
        lane_ok = False
    position = _to_float(ego_state.get("position"))
    return bool(lane_ok and position >= config.merge_complete_position)


def is_merge_failed(
    ego_state: Mapping[str, Any],
    done: bool,
    truncated: bool,
    config: Optional[MergeTaskConfig] = None,
    merged: Optional[bool] = None,
) -> bool:
    """True if the episode is terminal, unmerged, and without collision."""
    config = config or MergeTaskConfig()
    completed = bool(merged) if merged is not None else is_merge_completed(ego_state, config)
    terminal = bool(done or truncated)
    return bool(terminal and not completed and not is_collision(ego_state))


def merge_bonus(config: Optional[MergeTaskConfig] = None) -> float:
    config = config or MergeTaskConfig()
    return float(config.merge_success_bonus)


def non_merge_penalty(config: Optional[MergeTaskConfig] = None) -> float:
    config = config or MergeTaskConfig()
    return float(config.non_merge_failure_penalty)


def classify_outcome(
    merge_completed: bool,
    collision: bool,
    terminal: bool,
) -> dict:
    """Map episode-level merge/collision flags to mutually-exclusive outcomes.

    ``merge_success_rate`` alone is misleading because an episode can both merge
    and collide on the merging step. The flags below split that case out:

    - safe_merge:             merged and not collided  (PRIMARY success metric)
    - unsafe_merge:           merged and collided
    - collision_without_merge: collided and not merged
    - non_merge_failure:      terminal, not merged, not collided (ran out of steps)
    """
    safe_merge = bool(merge_completed and not collision)
    unsafe_merge = bool(merge_completed and collision)
    collision_without_merge = bool(collision and not merge_completed)
    non_merge_failure = bool(terminal and not merge_completed and not collision)
    if safe_merge:
        reason = "safe_merge"
    elif unsafe_merge:
        reason = "unsafe_merge"
    elif collision_without_merge:
        reason = "collision_without_merge"
    elif non_merge_failure:
        reason = "max_steps_unmerged"
    else:
        reason = "unknown"
    return {
        "merge_completed": bool(merge_completed),
        "collision": bool(collision),
        "safe_merge": safe_merge,
        "unsafe_merge": unsafe_merge,
        "collision_without_merge": collision_without_merge,
        "non_merge_failure": non_merge_failure,
        "termination_reason": reason,
    }


def outcome_from_state(
    ego_state: Mapping[str, Any],
    done: bool,
    truncated: bool,
    config: Optional[MergeTaskConfig] = None,
    merged: Optional[bool] = None,
) -> dict:
    """Classify the episode outcome from raw terminal state signals."""
    config = config or MergeTaskConfig()
    completed = bool(merged) if merged is not None else is_merge_completed(ego_state, config)
    collision = is_collision(ego_state)
    terminal = bool(done or truncated)
    return classify_outcome(completed, collision, terminal)


def terminal_merge_adjustment(
    ego_state: Mapping[str, Any],
    done: bool,
    truncated: bool,
    config: Optional[MergeTaskConfig] = None,
    merged: Optional[bool] = None,
) -> float:
    """Return the one-off terminal task adjustment for this step.

    + merge_success_bonus  when the merge has just completed,
    - non_merge_failure_penalty  when the episode ends unmerged without a crash,
    0.0 otherwise (including mid-episode steps and collision terminals).
    """
    config = config or MergeTaskConfig()
    completed = bool(merged) if merged is not None else is_merge_completed(ego_state, config)
    if completed:
        return float(config.merge_success_bonus)
    if bool(done or truncated) and not is_collision(ego_state):
        return -float(config.non_merge_failure_penalty)
    return 0.0


def terminal_collision_adjustment(
    ego_state: Mapping[str, Any],
    config: Optional[MergeTaskConfig] = None,
) -> float:
    """Shared terminal collision penalty: ``-terminal_collision_penalty`` on crash.

    Identical function and config for both conditions. Applied in addition to the
    merge-task adjustment so that an unsafe merge (merge + collision) nets the
    merge bonus minus this penalty, making it strictly worse than a safe merge.
    """
    config = config or MergeTaskConfig()
    if is_collision(ego_state):
        return -float(config.terminal_collision_penalty)
    return 0.0
