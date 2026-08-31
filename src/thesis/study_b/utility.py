"""Study B individual utility, coordination burden, and welfare/inequality
metrics -- ``new_research_plan.md`` sections "Individual utility",
"Coordination-burden proxy", "Gini coefficient", and the GGI evaluation
criterion.

Reuses ``thesis.pilots.stage11_welfare`` for everything it already provides
N-agnostically (target-speed attainment, absorbing experience, mean/min
welfare, PBRS potential/shaping) rather than reimplementing it -- that
module is Study A's own already-tested code and the formulas are identical
here, just applied to N=4 with per-vehicle (not shared) target speeds.
This module adds only what Study A never needed: Gini, GGI, and the
coordination-burden proxy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from thesis.pilots.stage11_welfare import (
    episode_mobility_outcome,
    mean_welfare,
    min_welfare,
    stakeholder_experience,
    target_speed_attainment,
)

__all__ = [
    "GGI_WEIGHTS_N4",
    "gini_coefficient",
    "generalized_gini_welfare",
    "coordination_burden",
    "utility_range",
    "burden_range",
    "EpisodeVehicleTrace",
    "episode_utilities",
    "episode_burdens",
    "running_active_attainment",
]

# new_research_plan.md: "本研究预注册 w = (0.4, 0.3, 0.2, 0.1)" -- frozen
# before any formal Study B result is seen. Index 0 applies to the WORST-off
# (smallest) utility after ascending sort, per the GGI convention used
# throughout that document.
GGI_WEIGHTS_N4: tuple[float, ...] = (0.4, 0.3, 0.2, 0.1)


def gini_coefficient(values: Sequence[float]) -> float | None:
    """G_U = sum_i sum_j |U_i - U_j| / (2 N sum_i U_i).

    Returns ``None`` (report as ``NA``, per new_research_plan.md's Gini
    section) when ``sum(values) == 0`` -- e.g. every vehicle failed the
    episode -- rather than 0.0, since "everyone tied at zero" is not
    meaningfully "perfect equality" and must not be silently coded as if it
    were. Callers must check for ``None`` before aggregating across
    episodes (e.g. exclude from a mean, or report the all-zero rate
    separately) rather than coercing it to 0.0."""
    n = len(values)
    if n == 0:
        raise ValueError("values must be non-empty")
    total = sum(values)
    if total == 0:
        return None
    numerator = sum(abs(a - b) for a in values for b in values)
    return float(numerator / (2.0 * n * total))


def generalized_gini_welfare(
    values: Sequence[float], weights: Sequence[float] = GGI_WEIGHTS_N4
) -> float:
    """W_GGI = sum_k w_k U_(k), utilities sorted ASCENDING (worst-off
    first) before weighting -- new_research_plan.md's GGI evaluation
    criterion. ``weights`` must be preregistered and frozen before formal
    results are seen (GGI_WEIGHTS_N4 is this project's frozen default for
    N=4); pass an explicit ``weights`` only for a different, also
    preregistered, vehicle count -- never re-fit weights after seeing
    results."""
    n = len(values)
    if n == 0:
        raise ValueError("values must be non-empty")
    if len(weights) != n:
        raise ValueError(f"weights length {len(weights)} must match values length {n}")
    ordered = sorted(values)
    return float(sum(w * u for w, u in zip(weights, ordered)))


def coordination_burden(
    *, dt: float, active_flags: Sequence[bool], attainments: Sequence[float]
) -> float:
    """C_i = dt * sum_t I_active(i,t) * (1 - u_i,t) -- cumulative
    mobility-loss coordination-burden PROXY (new_research_plan.md is
    explicit this is a proxy, not a causally-attributed cost: it cannot by
    itself prove another vehicle's action caused the deficit). ``attainments``
    is the RAW target-speed attainment ``u_i,t`` (not the absorbing
    ``stakeholder_experience`` -- once a vehicle exits it should stop
    accruing burden entirely, which ``active_flags`` already enforces, not
    get pinned at 0 burden-contribution via the absorbing 1.0 substitution
    that welfare potentials use)."""
    if len(active_flags) != len(attainments):
        raise ValueError("active_flags and attainments must be the same length")
    return float(dt * sum((1.0 - u) for active, u in zip(active_flags, attainments) if active))


def utility_range(values: Sequence[float]) -> float:
    """R_U = max_i U_i - min_i U_i."""
    if not values:
        raise ValueError("values must be non-empty")
    return float(max(values) - min(values))


def burden_range(values: Sequence[float]) -> float:
    """R_C = max_i C_i - min_i C_i (same shape as utility_range, kept as a
    separate name for readability at call sites)."""
    return utility_range(values)


@dataclass
class EpisodeVehicleTrace:
    """Per-vehicle raw per-step record for ONE episode, sufficient to
    compute U_i, C_i, and (via ``stage11_welfare.episode_collided_for_vehicle``
    at the caller) the collision-zeroing rule. ``target_speed`` is THIS
    vehicle's own, possibly-heterogeneous target (18/20/22 m/s depending on
    its assigned speed class) -- never a shared constant across vehicles in
    Study B, unlike Study A's single ``target_speed``."""

    vehicle_id: str
    target_speed: float
    speeds: list[float] = field(default_factory=list)
    active_flags: list[bool] = field(default_factory=list)
    collided: bool = False
    # Optional (only populated by heterogeneous_env.py, not by hand-built
    # traces in tests that don't need it) -- pre-step acceleration at each
    # sampled instant, same length as ``speeds`` when populated. Used by
    # behavioral analysis (hard-brake rate) in scripts/evaluate_policy.py.
    accelerations: list[float] = field(default_factory=list)

    def attainments(self) -> list[float]:
        return [target_speed_attainment(v, self.target_speed) for v in self.speeds]

    def hard_brake_count(self, *, threshold: float = -3.5) -> int:
        return sum(1 for a in self.accelerations if a <= threshold)


def episode_utilities(traces: Mapping[str, EpisodeVehicleTrace]) -> dict[str, float]:
    """U_i per vehicle for one episode -- ``stage11_welfare.episode_mobility_outcome``
    applied per-vehicle using each vehicle's OWN target speed."""
    out: dict[str, float] = {}
    for vid, trace in traces.items():
        active_attainments = [
            a for active, a in zip(trace.active_flags, trace.attainments()) if active
        ]
        out[vid] = episode_mobility_outcome(collided=trace.collided, attainment_samples=active_attainments)
    return out


def episode_burdens(traces: Mapping[str, EpisodeVehicleTrace], *, dt: float) -> dict[str, float]:
    """C_i per vehicle for one episode."""
    out: dict[str, float] = {}
    for vid, trace in traces.items():
        out[vid] = coordination_burden(dt=dt, active_flags=trace.active_flags, attainments=trace.attainments())
    return out


def running_active_attainment(trace: EpisodeVehicleTrace) -> float:
    """WSC (Welfare-State Communication) pre-decision mobility-history
    summary M_i(t) -- new_research_plan.md's information-structure
    intervention, NOT a second/independent welfare accumulator: this reuses
    the EXACT SAME ``active_flags``/``attainments()`` construct
    ``episode_utilities`` already uses (see that function immediately
    above), evaluated on whatever prefix of the trace has been recorded so
    far rather than waiting for episode end.

    M_i(t) = mean of all already-realised ACTIVE attainment samples in
    ``trace`` (identical selection rule to ``episode_utilities``'s
    ``active_attainments`` list). Returns 1.0 (neutral -- no mobility
    deficit yet accumulated) if the trace has zero active samples so far,
    deliberately DIFFERENT from ``episode_mobility_outcome``'s 0.0 empty
    convention, because that 0.0 encodes "failed episode with no data" at
    TERMINAL time, whereas this 1.0 encodes "nothing bad has happened yet"
    at DECISION time -- these are not the same situation and must not share
    a convention.

    Never zeroed by collision (unlike terminal ``U_i``): a collision is a
    global, episode-ending event handled entirely by the caller/network
    layer (a collided vehicle's trace simply stops receiving new samples
    the same way a completed vehicle's does -- see
    ``highwayenv_wrapper.py``'s ``_append_pre_step_trace_sample``), so this
    function only ever sees whatever active samples were already
    legitimately recorded before the episode ended, exactly like the
    terminal utility path would for a truncated (timeout) episode.

    Pure function, no side effects, does not mutate ``trace``."""
    active_samples = [a for active, a in zip(trace.active_flags, trace.attainments()) if active]
    if not active_samples:
        return 1.0
    return float(sum(active_samples) / len(active_samples))


# Re-exported for callers that want Study B's utility.py to be the single
# import surface for welfare math, without needing to know it's a thin
# layer over stage11_welfare underneath.
__all__ += ["mean_welfare", "min_welfare", "stakeholder_experience", "target_speed_attainment"]
