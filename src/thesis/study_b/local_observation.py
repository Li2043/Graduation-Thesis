"""Study B local partial observation -- ``new_research_plan.md``'s
"Agent ignorance / partial observation" section.

Each vehicle sees its OWN role/speed/target-speed/acceleration/distance-to-
merge/previous-action exactly, and up to ``K=3`` neighbours' RELATIVE
motion only (presence, relative distance-to-merge, relative speed, lane
relation) -- never another vehicle's target speed, exact position, reward,
utility, value estimate, or intended action. This is the "hidden intrinsic
parameters" decentralization device the whole Study B fairness question
depends on: an agent can tell "that vehicle is currently slower than me"
from ``Δv`` but cannot tell "because it's a slow-class vehicle at its own
target" from "because it's yielding" -- see
``tests/study_b/test_local_observation_leakage.py`` for the unit test this
module's design is required to pass.

Deliberately environment-free (works on a plain ``VehicleSnapshot``
dataclass, not a live ``Stage10SymmetricMergeEnv``) for the same testability
reason as ``scenario_generator.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from thesis.envs.stage10_symmetric_merge_env import HighLevelAction, encode_peer_intent, encode_role

__all__ = [
    "R_OBS_DEFAULT",
    "V_SCALE_DEFAULT",
    "NEIGHBOUR_SLOTS",
    "SELF_OBS_DIM",
    "NEIGHBOUR_OBS_DIM",
    "LOCAL_OBS_DIM",
    "SELF_OBS_DIM_WSC",
    "NEIGHBOUR_OBS_DIM_WSC",
    "LOCAL_OBS_DIM_WSC",
    "WSC_NEUTRAL_M",
    "GLOBAL_STATE_N_VEHICLES",
    "GLOBAL_STATE_DIM",
    "VehicleSnapshot",
    "distance_to_merge",
    "build_self_observation",
    "build_neighbour_observations",
    "build_local_observation",
    "build_global_state",
]

# Phase-0 pilot defaults (new_research_plan.md) -- R_OBS in particular is
# explicitly flagged there as needing a Phase 0 check that it covers real
# conflict-relevant vehicles for this env's actual geometry before being
# frozen for Phase 3.
R_OBS_DEFAULT: float = 50.0
V_SCALE_DEFAULT: float = 10.0
NEIGHBOUR_SLOTS: int = 3  # fixed: N=4 means at most 3 other vehicles, ever.

SELF_OBS_DIM: int = 6  # [role, speed, target_speed, acceleration, dist_to_merge, prev_action]
NEIGHBOUR_OBS_DIM: int = 4  # [presence, delta_d_norm, delta_v_norm, lane_relation]
LOCAL_OBS_DIM: int = SELF_OBS_DIM + NEIGHBOUR_SLOTS * NEIGHBOUR_OBS_DIM  # 18 -- UNCHANGED, Original path

# WSC (Welfare-State Communication) additive observation mode -- new_research_plan.md's
# information-structure intervention (see thesis.study_b.utility.running_active_attainment
# for the M_i(t) definition this attaches). Adds exactly one scalar to the self block and
# one scalar per neighbour slot; every OTHER field/ordering/semantics is byte-identical to
# the Original 6/4-field layout above. LOCAL_OBS_DIM/SELF_OBS_DIM/NEIGHBOUR_OBS_DIM above are
# NEVER changed by the presence of this mode -- Original callers are completely unaffected.
SELF_OBS_DIM_WSC: int = SELF_OBS_DIM + 1  # 7: + M_i(t)
NEIGHBOUR_OBS_DIM_WSC: int = NEIGHBOUR_OBS_DIM + 1  # 5: + M_j(t)
LOCAL_OBS_DIM_WSC: int = SELF_OBS_DIM_WSC + NEIGHBOUR_SLOTS * NEIGHBOUR_OBS_DIM_WSC  # 7 + 3*5 = 22
WSC_NEUTRAL_M: float = 1.0  # padding value for an absent neighbour slot's M_j; matches
# thesis.study_b.utility.running_active_attainment's own empty-history neutral value, but
# is used here ONLY as the zero-padding convention for a masked-absent slot (presence=0
# already signals "ignore this row" -- see build_neighbour_observations below, which uses
# 0.0 for the padded M_j to match the existing all-zero row convention for absent slots,
# not this constant; WSC_NEUTRAL_M is exported for the "no history yet" case at the CALLER
# level, i.e. what a wrapper should pass in for welfare_state before any trace sample
# exists, mirroring running_active_attainment's own 1.0 return for that case).

GLOBAL_STATE_N_VEHICLES: int = 4
GLOBAL_STATE_DIM: int = GLOBAL_STATE_N_VEHICLES * SELF_OBS_DIM  # 24 -- see build_global_state


@dataclass(frozen=True)
class VehicleSnapshot:
    """Everything ``local_observation.py`` needs about one vehicle at one
    instant. ``target_speed`` is intentionally present here (the snapshot
    itself is allowed to know everything -- it represents privileged
    simulator state) but ``build_neighbour_observations`` never reads a
    neighbour's ``target_speed``, only its ``speed``/``route_position``/
    ``active`` (and, under WSC mode, its ``welfare_state``) -- that
    omission, not the snapshot's contents, is what enforces the
    information restriction.

    ``welfare_state`` (WSC addition, default 1.0 = neutral/no-history):
    the vehicle's OWN pre-decision M_i(t) (see
    ``thesis.study_b.utility.running_active_attainment``), populated by
    the caller from that vehicle's own episode trace. Additive field with
    a default -- every existing call site that constructs a
    ``VehicleSnapshot`` without this argument is completely unaffected;
    it is only READ when ``include_welfare_state=True`` is passed to
    ``build_self_observation``/``build_neighbour_observations``/
    ``build_local_observation`` below."""

    vehicle_id: str
    role: str
    speed: float
    route_position: float
    acceleration: float
    target_speed: float
    active: bool
    previous_action: int = HighLevelAction.MAINTAIN
    welfare_state: float = 1.0


def distance_to_merge(route_position: float, *, merge_start: float) -> float:
    """Positive while still approaching the merge-relevant coordinate,
    consistent with ``scenario_generator.py``'s spawn-distance convention
    (same reference coordinate, so a vehicle's own ``d_i,t^merge`` shrinks
    toward its nominal spawn distance's origin as it drives)."""
    return float(merge_start - route_position)


def build_self_observation(
    ego: VehicleSnapshot, *, merge_start: float, include_welfare_state: bool = False
) -> np.ndarray:
    """``include_welfare_state`` (WSC, default False = Original 6D behaviour
    byte-identical): appends ego's own ``welfare_state`` (M_i(t)) as a 7th
    feature. Every existing caller that omits this argument gets exactly
    the same 6-element array as before."""
    fields = [
        encode_role(ego.role),
        float(ego.speed),
        float(ego.target_speed),
        float(ego.acceleration),
        distance_to_merge(ego.route_position, merge_start=merge_start),
        encode_peer_intent(int(ego.previous_action)),
    ]
    if include_welfare_state:
        fields.append(float(ego.welfare_state))
    return np.array(fields, dtype=np.float64)


def build_neighbour_observations(
    ego: VehicleSnapshot,
    others: Sequence[VehicleSnapshot],
    *,
    merge_start: float,
    r_obs: float = R_OBS_DEFAULT,
    v_scale: float = V_SCALE_DEFAULT,
    k: int = NEIGHBOUR_SLOTS,
    local_sensing_range_m: float | None = None,
    include_welfare_state: bool = False,
) -> np.ndarray:
    """Returns shape ``(k, NEIGHBOUR_OBS_DIM)`` (or ``(k, NEIGHBOUR_OBS_DIM+1)``
    when ``include_welfare_state=True``, WSC mode), sorted by ascending
    |delta_d| (nearest conflict-relevant neighbour first), zero-padded with
    presence=0 rows if fewer than ``k`` OTHER ACTIVE vehicles exist (e.g.
    once some vehicles have already exited). Never reads ``other.target_speed``
    -- see this module's docstring.

    ``include_welfare_state`` (WSC, default False = Original 4-field
    behaviour byte-identical): appends the selected neighbour's OWN
    ``welfare_state`` (M_j(t)) as a 5th field, attached to the EXACT SAME
    slot/row as that neighbour's ``delta_d_norm``/``delta_v_norm``/
    ``lane_relation`` -- computed from the SAME ``scored[:k]`` selection
    and SAME sort order as the other 4 fields, never a separate ranking.
    A padded (absent) slot's M_j is ``0.0``, matching the existing
    all-zero-row convention for absent slots (the ``presence=0`` flag
    already marks the whole row, including this new field, as masked).

    ``local_sensing_range_m`` (LOCALITY AMENDMENT -- see
    ``LOCAL_SENSING_AMENDMENT.md``; frozen value R=50m per the
    2026-08-17 scientific decision): default ``None`` preserves the
    exact prior behaviour byte-for-byte (every active other vehicle is
    a selection candidate, regardless of distance -- at N=4 this means
    all active others are always visible, since there are never more
    than 3). When set to a finite value ``R``, a vehicle is only a
    selection candidate if ``abs(delta_d) <= R`` using the SAME
    road-relative ``distance_to_merge``-based coordinate already used
    for ``delta_d_norm`` elsewhere in this function (not Euclidean
    distance, and not a new coordinate system) -- vehicles strictly
    outside ``R`` are excluded entirely (not merely deprioritized), so
    with fewer than ``k`` vehicles inside range, the remaining slots
    stay ``presence=0`` rather than being filled by farther,
    out-of-range vehicles. Boundary is inclusive: a vehicle at exactly
    ``abs(delta_d) == R`` IS visible. This parameter changes WHO is
    selected; it does not change what is reported about a selected
    vehicle (see ``NEIGHBOUR_OBS_DIM``'s 4 fields, unchanged)."""
    ego_d = distance_to_merge(ego.route_position, merge_start=merge_start)
    active_others = [o for o in others if o.vehicle_id != ego.vehicle_id and o.active]

    scored: list[tuple[float, VehicleSnapshot]] = []
    for other in active_others:
        other_d = distance_to_merge(other.route_position, merge_start=merge_start)
        delta_d = other_d - ego_d
        if local_sensing_range_m is not None and abs(delta_d) > local_sensing_range_m:
            continue
        scored.append((abs(delta_d), other))
    scored.sort(key=lambda pair: pair[0])

    row_dim = NEIGHBOUR_OBS_DIM_WSC if include_welfare_state else NEIGHBOUR_OBS_DIM
    rows = np.zeros((k, row_dim), dtype=np.float64)
    for slot, (_abs_delta_d, other) in enumerate(scored[:k]):
        other_d = distance_to_merge(other.route_position, merge_start=merge_start)
        delta_d = other_d - ego_d
        delta_v = other.speed - ego.speed
        delta_d_norm = float(np.clip(delta_d / r_obs, -1.0, 1.0))
        delta_v_norm = float(np.clip(delta_v / v_scale, -1.0, 1.0))
        # Externally-observable lane geometry (same role == same lane
        # before the merge zone), NOT a private intrinsic parameter --
        # new_research_plan.md explicitly allows this.
        lane_relation = 1.0 if other.role == ego.role else -1.0
        row = [1.0, delta_d_norm, delta_v_norm, lane_relation]
        if include_welfare_state:
            row.append(float(other.welfare_state))
        rows[slot] = row
    return rows


def build_local_observation(
    ego: VehicleSnapshot,
    others: Sequence[VehicleSnapshot],
    *,
    merge_start: float,
    r_obs: float = R_OBS_DEFAULT,
    v_scale: float = V_SCALE_DEFAULT,
    k: int = NEIGHBOUR_SLOTS,
    local_sensing_range_m: float | None = None,
    include_welfare_state: bool = False,
) -> np.ndarray:
    """Flattened ``[self(6), neighbour_0(4), neighbour_1(4), neighbour_2(4)]``
    -- shape ``(LOCAL_OBS_DIM,)`` -- OR, when ``include_welfare_state=True``
    (WSC mode), ``[self(7), neighbour_0(5), neighbour_1(5), neighbour_2(5)]``
    -- shape ``(LOCAL_OBS_DIM_WSC,)``. This, not the environment's own
    ``_obs_from_vehicles()`` (which exposes exact neighbour gap/speed and
    is closer to a sliced-centralized observation), is what actor networks
    in Study B actually consume.

    ``local_sensing_range_m``: see ``build_neighbour_observations``'s
    docstring -- default ``None`` is byte-identical to the pre-amendment
    behaviour. ``include_welfare_state`` default ``False`` is likewise
    byte-identical to the pre-WSC 18D behaviour; every existing caller is
    unaffected unless it explicitly opts in."""
    self_obs = build_self_observation(ego, merge_start=merge_start, include_welfare_state=include_welfare_state)
    neighbour_obs = build_neighbour_observations(
        ego, others, merge_start=merge_start, r_obs=r_obs, v_scale=v_scale, k=k,
        local_sensing_range_m=local_sensing_range_m, include_welfare_state=include_welfare_state,
    )
    return np.concatenate([self_obs, neighbour_obs.reshape(-1)])


def build_global_state(snapshots: Sequence[VehicleSnapshot], *, merge_start: float) -> np.ndarray:
    """Centralized-critic-only input (CTDE -- MAPPO's ``V_phi(s_t)``, never
    exposed to any actor). Every vehicle's own SELF observation (the same 6
    features an agent sees about itself, per-vehicle -- i.e. this DOES
    reveal every vehicle's private target speed, deliberately, since the
    critic is training-time-only privileged state) concatenated in a fixed
    order (sorted by ``vehicle_id``, not by role -- role is itself part of
    the per-vehicle features, and a fixed id-based order keeps the critic's
    input layout stable across episodes regardless of that episode's
    role/speed-class permutation). Shape ``(GLOBAL_STATE_DIM,)``."""
    ordered = sorted(snapshots, key=lambda s: s.vehicle_id)
    if len(ordered) != GLOBAL_STATE_N_VEHICLES:
        raise ValueError(f"expected exactly {GLOBAL_STATE_N_VEHICLES} vehicles, got {len(ordered)}")
    return np.concatenate([build_self_observation(s, merge_start=merge_start) for s in ordered])
