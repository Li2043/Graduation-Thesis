"""Preregistered IC blocks, environment candidates, and GO/YIELD macros."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from thesis.envs.final_environment_config import (
    EnvironmentCandidate,
    GeometryCandidate,
    IDMProfile,
    InitialConditionBlock,
    TargetSpeeds,
)


MatrixCell = Literal["GO_GO", "GO_YIELD", "YIELD_GO", "YIELD_YIELD"]


GEOMETRY: list[GeometryCandidate] = [
    GeometryCandidate("G1", 0.0, 0.0, 80.0, 140.0, 260.0, 1),
    GeometryCandidate("G2", 0.0, 0.0, 80.0, 160.0, 280.0, 2),
    GeometryCandidate("G3", 0.0, 0.0, 100.0, 170.0, 300.0, 3),
]

IDM_PROFILES: list[IDMProfile] = [
    IDMProfile("I1", 20.0, 2.0, 1.5, 1.5, 2.0, 4.0, 1),
    IDMProfile("I2", 20.0, 2.5, 1.8, 1.2, 2.5, 4.0, 2),
    IDMProfile("I3", 20.0, 1.5, 1.2, 1.8, 2.0, 4.0, 3),
]


def build_environment_candidates() -> list[EnvironmentCandidate]:
    out: list[EnvironmentCandidate] = []
    rank = 1
    for g in GEOMETRY:
        for i in IDM_PROFILES:
            out.append(
                EnvironmentCandidate(
                    candidate_id=f"{g.geometry_id}-{i.profile_id}",
                    geometry=g,
                    idm=i,
                    priority_rank=rank,
                )
            )
            rank += 1
    return out


@dataclass(frozen=True)
class MacroProfile:
    profile_id: str
    kind: str  # GO | YIELD
    action: int  # ACC=1, DEC=2
    n_steps: int
    absolute_acceleration: float


GO_PROFILES = (
    MacroProfile("GO_1", "GO", 1, 4, 2.0),
    MacroProfile("GO_2", "GO", 1, 8, 2.0),
)
YIELD_PROFILES = (
    MacroProfile("YIELD_1", "YIELD", 2, 4, 3.0),
    MacroProfile("YIELD_2", "YIELD", 2, 8, 3.0),
)


def least_intervention_profile(profiles: tuple[MacroProfile, ...]) -> MacroProfile:
    return sorted(
        profiles,
        key=lambda p: (p.n_steps, p.absolute_acceleration, p.profile_id),
    )[0]


def _arrival_category(delta: float) -> str:
    if abs(delta) < 1e-9:
        return "near_simultaneous"
    if delta > 0:
        return "ramp_lead"
    return "mainline_lead"


def _spawn_from_delta(
    *,
    merge_start: float,
    v_m: float,
    v_r: float,
    delta_arrival: float,
    headway: float,
) -> tuple[float, float, float, float, float, float]:
    """Return mainline/ramp routes+speeds and B_front/B_rear routes at headway spacing.

    Positions are constructed from preregistered speeds and arrival-time deltas.
    Approach time is chosen so both remaining distances fit the geometry while
    preserving delta_arrival = t_mainline - t_ramp.
    """
    v_m = max(float(v_m), 1e-6)
    v_r = max(float(v_r), 1e-6)
    max_rem = max(25.0, float(merge_start) - 8.0)
    min_rem = 20.0
    # Feasible t_m such that rem_m, rem_r ∈ [min_rem, max_rem]
    # rem_m = t_m * v_m
    # rem_r = (t_m - delta) * v_r
    lo = max(min_rem / v_m, (min_rem / v_r) + delta_arrival)
    hi = min(max_rem / v_m, (max_rem / v_r) + delta_arrival)
    if lo > hi + 1e-9:
        # Fall back to mid feasible band ignoring min_rem floor slightly
        lo = max(0.5, (min_rem / v_r) + delta_arrival)
        hi = min(max_rem / v_m, (max_rem / v_r) + delta_arrival)
    # Prefer ~2.6 s approach when feasible (tight enough for GO/GO conflict)
    t_m = min(max(2.6, lo), hi) if hi >= lo else 2.6
    t_r = t_m - delta_arrival
    rem_m = min(max_rem, max(min_rem, t_m * v_m))
    rem_r = min(max_rem, max(min_rem, t_r * v_r))
    # Re-sync times from clamped rem to keep delta sign when possible
    t_m = rem_m / v_m
    t_r = rem_r / v_r
    # If clamping destroyed delta, nudge the lagging vehicle farther back
    realized = t_m - t_r
    if abs(realized - delta_arrival) > 0.05:
        if delta_arrival < 0:
            # mainline should be earlier: reduce rem_m or increase rem_r
            t_m = t_r + delta_arrival
            if t_m < 0.5:
                t_r = 0.5 - delta_arrival
                t_m = 0.5
            rem_m = min(max_rem, max(12.0, t_m * v_m))
            rem_r = min(max_rem, max(12.0, t_r * v_r))
        else:
            t_r = t_m - delta_arrival
            if t_r < 0.5:
                t_m = 0.5 + delta_arrival
                t_r = 0.5
            rem_m = min(max_rem, max(12.0, t_m * v_m))
            rem_r = min(max_rem, max(12.0, t_r * v_r))
    p_m = float(merge_start) - rem_m
    p_r = float(merge_start) - rem_r
    p_m = max(5.0, p_m)
    p_r = max(5.0, min(p_r, float(merge_start) - 5.0))
    # Place background inside the interaction region so IDM responds to learners
    front = float(merge_start) + max(22.0, headway * v_m)
    rear = max(0.0, min(p_m, p_r) - max(12.0, headway * v_m))
    return p_m, p_r, float(v_m), float(v_r), front, rear


def build_ic_blocks() -> tuple[list[InitialConditionBlock], list[InitialConditionBlock]]:
    """12 calibration + 8 validation blocks (immutable construction)."""
    # Preregistered (speed_m, speed_r, delta_arrival, headway).
    # Near-simultaneous blocks use unequal speeds so equal-action macros do not
    # place both learners on identical longitudinal world coordinates at merge join.
    cal_specs = [
        (16, 18, -0.4, 1.5),
        (16, 18, -0.2, 1.2),
        (18, 16, 0.0, 1.5),
        (18, 16, 0.2, 1.8),
        (20, 18, 0.4, 1.5),
        (18, 20, -0.4, 1.2),
        (16, 20, 0.0, 1.8),
        (18, 20, 0.2, 1.5),
        (20, 16, -0.2, 1.2),
        (18, 16, 0.4, 1.8),
        (16, 20, -0.4, 1.5),
        (20, 18, 0.0, 1.2),
    ]
    val_specs = [
        (16, 18, -0.4, 1.5),
        (18, 16, 0.4, 1.2),
        (20, 16, 0.0, 1.8),
        (18, 20, -0.2, 1.5),
        (16, 20, 0.2, 1.2),
        (18, 16, 0.0, 1.5),
        (20, 18, 0.4, 1.8),
        (16, 18, -0.2, 1.5),
    ]
    merge_ref = 80.0  # positions generated for G1; G2/G3 reuse relative geometry via merge_start

    def make(specs: list[tuple], prefix: str, block_set: str) -> list[InitialConditionBlock]:
        blocks: list[InitialConditionBlock] = []
        for i, (vm, vr, darr, th) in enumerate(specs, start=1):
            pm, pr, svm, svr, front, rear = _spawn_from_delta(
                merge_start=merge_ref, v_m=float(vm), v_r=float(vr), delta_arrival=float(darr), headway=float(th)
            )
            blocks.append(
                InitialConditionBlock(
                    block_id=f"{prefix}_{i:03d}",
                    block_set=block_set,
                    seed=1000 + i if block_set == "calibration" else 2000 + i,
                    role_A="mainline",
                    role_B="ramp",
                    spawn_route_mainline=pm,
                    spawn_route_ramp=pr,
                    spawn_speed_mainline=svm,
                    spawn_speed_ramp=svr,
                    spawn_route_B_front=front,
                    spawn_route_B_rear=rear,
                    spawn_speed_B_front=20.0,
                    spawn_speed_B_rear=20.0,
                    delta_arrival=float(darr),
                    arrival_category=_arrival_category(float(darr)),
                    background_time_headway=float(th),
                    target_speeds=TargetSpeeds(),
                )
            )
        return blocks

    return make(cal_specs, "calibration", "calibration"), make(val_specs, "validation", "validation")


def materialize_block_for_geometry(
    block: InitialConditionBlock, geometry: GeometryCandidate
) -> InitialConditionBlock:
    """Recompute absolute spawns from preregistered speed/delta/headway for a geometry."""
    pm, pr, svm, svr, front, rear = _spawn_from_delta(
        merge_start=float(geometry.merge_start),
        v_m=float(block.spawn_speed_mainline),
        v_r=float(block.spawn_speed_ramp),
        delta_arrival=float(block.delta_arrival),
        headway=float(block.background_time_headway),
    )
    return InitialConditionBlock(
        block_id=block.block_id,
        block_set=block.block_set,
        seed=block.seed,
        role_A=block.role_A,
        role_B=block.role_B,
        spawn_route_mainline=pm,
        spawn_route_ramp=pr,
        spawn_speed_mainline=svm,
        spawn_speed_ramp=svr,
        spawn_route_B_front=front,
        spawn_route_B_rear=rear,
        spawn_speed_B_front=block.spawn_speed_B_front,
        spawn_speed_B_rear=block.spawn_speed_B_rear,
        delta_arrival=block.delta_arrival,
        arrival_category=block.arrival_category,
        background_time_headway=block.background_time_headway,
        target_speeds=block.target_speeds,
    )


def expand_label_assignments(block: InitialConditionBlock) -> list[InitialConditionBlock]:
    """Both controller-label assignments with identical physical IC."""
    a1 = block
    a2 = InitialConditionBlock(
        block_id=block.block_id,
        block_set=block.block_set,
        seed=block.seed,
        role_A="ramp",
        role_B="mainline",
        spawn_route_mainline=block.spawn_route_mainline,
        spawn_route_ramp=block.spawn_route_ramp,
        spawn_speed_mainline=block.spawn_speed_mainline,
        spawn_speed_ramp=block.spawn_speed_ramp,
        spawn_route_B_front=block.spawn_route_B_front,
        spawn_route_B_rear=block.spawn_route_B_rear,
        spawn_speed_B_front=block.spawn_speed_B_front,
        spawn_speed_B_rear=block.spawn_speed_B_rear,
        delta_arrival=block.delta_arrival,
        arrival_category=block.arrival_category,
        background_time_headway=block.background_time_headway,
        target_speeds=block.target_speeds,
    )
    return [a1, a2]


def _completion_action(kind: str, t: int, macro_end: int, *, peer_kind: str) -> int:
    """Deterministic post-macro controller that finishes without reversing intent."""
    if t < macro_end:
        raise AssertionError("completion only after macro window")
    age = t - macro_end
    if kind == "GO":
        # Mutual assertion: keep accelerating — creates conflict when both GO
        if peer_kind == "GO":
            return 1 if age < 60 else 0
        return 1 if age < 28 else 0
    # Mutual yielding: remain slow much longer so YY is inefficient vs asymmetric cells
    if peer_kind == "YIELD":
        if age < 120:
            return 0
        return 1 if age % 6 == 0 else 0
    # Unilateral yield: hold, then gentle accelerate after the GO peer clears
    if age < 45:
        return 0
    return 1 if age % 3 == 0 else 0


def macro_action_sequence(
    mainline_kind: str,
    ramp_kind: str,
    *,
    go: MacroProfile,
    yield_p: MacroProfile,
    total_steps: int,
    role_A: str,
    go_ramp: MacroProfile | None = None,
    yield_ramp: MacroProfile | None = None,
) -> list[dict[str, int]]:
    """Build joint actions for physical mainline/ramp kinds under a label assignment."""
    if mainline_kind == "GO":
        ml_prof = go
    else:
        ml_prof = yield_p
    if ramp_kind == "GO":
        rp_prof = go_ramp if go_ramp is not None else go
    else:
        rp_prof = yield_ramp if yield_ramp is not None else yield_p
    acts: list[dict[str, int]] = []
    for t in range(total_steps):
        if t < ml_prof.n_steps:
            a_ml = ml_prof.action
        else:
            a_ml = _completion_action(mainline_kind, t, ml_prof.n_steps, peer_kind=ramp_kind)
        if t < rp_prof.n_steps:
            a_rp = rp_prof.action
        else:
            a_rp = _completion_action(ramp_kind, t, rp_prof.n_steps, peer_kind=mainline_kind)
        if role_A == "mainline":
            acts.append({"A": a_ml, "B": a_rp})
        else:
            acts.append({"A": a_rp, "B": a_ml})
    return acts


def cell_kinds(cell: MatrixCell) -> tuple[str, str]:
    return {
        "GO_GO": ("GO", "GO"),
        "GO_YIELD": ("GO", "YIELD"),
        "YIELD_GO": ("YIELD", "GO"),
        "YIELD_YIELD": ("YIELD", "YIELD"),
    }[cell]
