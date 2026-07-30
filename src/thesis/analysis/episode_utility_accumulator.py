"""Trajectory-level stakeholder utility accumulation (Stage 6B-H1).

Episode utility is the mean of active-state speed-attainment samples along the
trajectory, not the final-state experience E_i(s_T).

Sampling semantics (decision-cycle timing used by Stage 6B-H1 evaluator):

1. After ``env.reset``, sample ``s_0`` for every stakeholder that is active and
   on-road.
2. After each ``env.step``, if the episode continues (not terminated and not
   truncated), sample the post-step state for active on-road stakeholders.
3. Absorbing / post-exit states (``active_on_road=False`` or ``completed=True``)
   are never sampled, so exit-absorbing experience ``1.0`` cannot enter the mean.
4. The terminal/truncated transition itself is not sampled after the step.
5. Stakeholders that appear in collision pairs receive utility ``0.0`` regardless
   of accumulated samples; non-colliding stakeholders keep their trajectory mean.
6. A non-colliding stakeholder with zero samples raises ``RuntimeError`` (no
   silent zero / NaN fallback).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, MutableMapping, Sequence
from typing import Any

import numpy as np

from thesis.analysis.endpoints import STAKEHOLDERS


def initialise_episode_utility_accumulator(
    stakeholder_ids: Sequence[str] | None = None,
) -> dict[str, list[float]]:
    """Create an empty per-stakeholder sample accumulator."""
    ids = tuple(stakeholder_ids) if stakeholder_ids is not None else STAKEHOLDERS
    return {str(sid): [] for sid in ids}


def _as_vehicle_view(vehicle: Any) -> dict[str, Any]:
    if isinstance(vehicle, Mapping):
        return dict(vehicle)
    return {
        "speed": float(vehicle.speed),
        "target_speed": float(getattr(vehicle, "target_speed")),
        "active_on_road": bool(vehicle.active_on_road),
        "completed": bool(getattr(vehicle, "completed", False)),
    }


def is_vehicle_active_and_on_road(vehicle: Any) -> bool:
    """True only while the stakeholder is still an active on-road participant."""
    view = _as_vehicle_view(vehicle)
    return bool(view.get("active_on_road", False)) and not bool(view.get("completed", False))


def clip_speed_attainment(speed: float, target_speed: float) -> float:
    """Return clip(speed / target_speed, 0, 1) with strict target-speed checks."""
    sp = float(speed)
    vt = float(target_speed)
    if not np.isfinite(vt) or vt <= 0.0:
        raise ValueError(f"Invalid target speed: {target_speed}")
    if not np.isfinite(sp):
        raise ValueError(f"Invalid speed: {speed}")
    ratio = sp / vt
    if ratio < 0.0:
        return 0.0
    if ratio > 1.0:
        return 1.0
    return float(ratio)


def collect_active_state_attainment(
    *,
    vehicles: Mapping[str, Any],
    stakeholder_ids: Sequence[str],
    target_speeds: Mapping[str, float] | None = None,
    accumulator: MutableMapping[str, list[float]],
) -> None:
    """Append one active-state attainment sample per eligible stakeholder.

    Parameters
    ----------
    vehicles:
        Mapping of stakeholder id -> vehicle state object or ``_veh_info`` dict.
    stakeholder_ids:
        Stakeholders to consider (order does not affect values).
    target_speeds:
        Optional overrides; otherwise ``vehicle.target_speed`` is used.
    accumulator:
        Mutable mapping created by :func:`initialise_episode_utility_accumulator`.
    """
    for sid in stakeholder_ids:
        key = str(sid)
        if key not in accumulator:
            raise KeyError(f"accumulator missing stakeholder {key}")
        if key not in vehicles:
            continue
        veh = vehicles[key]
        if not is_vehicle_active_and_on_road(veh):
            continue
        view = _as_vehicle_view(veh)
        if target_speeds is not None and key in target_speeds:
            vt = float(target_speeds[key])
        else:
            vt = float(view["target_speed"])
        attainment = clip_speed_attainment(float(view["speed"]), vt)
        accumulator[key].append(float(np.float64(attainment)))


def collided_ids_from_pairs(collision_pairs: Sequence[Any]) -> set[str]:
    """Extract unique stakeholder ids from collision pair records."""
    out: set[str] = set()
    for pair in collision_pairs:
        if pair is None:
            continue
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            out.add(str(pair[0]))
            out.add(str(pair[1]))
        else:
            raise ValueError(f"malformed collision pair: {pair!r}")
    return out


def finalise_episode_utilities(
    *,
    accumulator: Mapping[str, Sequence[float]],
    collided_stakeholder_ids: Collection[str],
) -> dict[str, float]:
    """Finalise trajectory utilities with collision override.

    Collided stakeholders receive exactly ``0.0``. Non-colliding stakeholders
    receive the float64 mean of their active-state samples. Empty samples for a
    non-colliding stakeholder raise ``RuntimeError``.
    """
    collided = {str(x) for x in collided_stakeholder_ids}
    utilities: dict[str, float] = {}
    for sid, samples in accumulator.items():
        key = str(sid)
        if key in collided:
            utilities[key] = 0.0
            continue
        if len(samples) == 0:
            raise RuntimeError(
                "No valid active-state attainment samples for "
                f"non-colliding stakeholder {key}"
            )
        arr = np.asarray(list(samples), dtype=np.float64)
        utilities[key] = float(arr.mean())
    return utilities


def utility_sample_counts(
    accumulator: Mapping[str, Sequence[float]],
) -> dict[str, int]:
    """Return the number of active-state samples per stakeholder."""
    return {str(sid): int(len(samples)) for sid, samples in accumulator.items()}


def derive_utility_fields(utilities: Mapping[str, float]) -> dict[str, Any]:
    """Recompute utility-derived episode fields from corrected utilities."""
    ordered = [str(s) for s in STAKEHOLDERS]
    vals = {s: float(utilities[s]) for s in ordered}
    mean_u = float(np.mean(np.asarray([vals[s] for s in ordered], dtype=np.float64)))
    min_u = float(min(vals[s] for s in ordered))
    worst = [s for s in ordered if vals[s] == min_u]
    return {
        "stakeholder_utilities": vals,
        "utility_A": vals["A"],
        "utility_B": vals["B"],
        "utility_background_front": vals["B_front"],
        "utility_background_rear": vals["B_rear"],
        "learner_A_utility": vals["A"],
        "learner_B_utility": vals["B"],
        "B_front_utility": vals["B_front"],
        "B_rear_utility": vals["B_rear"],
        "mean_stakeholder_utility": mean_u,
        "minimum_stakeholder_utility": min_u,
        "worst_off_stakeholder_id": worst[0],
        "worst_off_stakeholder_identity": worst[0],
        "worst_off_stakeholder_ids_json": list(worst),
        "worst_off_tie": bool(len(worst) > 1),
        "worst_off_utility": min_u,
        "utility_rank_order": sorted(ordered, key=lambda s: (vals[s], s)),
        "controlled_agent_mean_utility": float(
            np.mean(np.asarray([vals["A"], vals["B"]], dtype=np.float64))
        ),
        "controlled_agent_minimum_utility": float(min(vals["A"], vals["B"])),
        "background_mean_utility": float(
            np.mean(np.asarray([vals["B_front"], vals["B_rear"]], dtype=np.float64))
        ),
        "background_minimum_utility": float(min(vals["B_front"], vals["B_rear"])),
    }


__all__ = [
    "clip_speed_attainment",
    "collect_active_state_attainment",
    "collided_ids_from_pairs",
    "derive_utility_fields",
    "finalise_episode_utilities",
    "initialise_episode_utility_accumulator",
    "is_vehicle_active_and_on_road",
    "utility_sample_counts",
]
