"""Dense (step-wise) welfare shaping for the Dense Reward Study.

Adds a discrete +/-c shaping signal on top of the existing per-step task
reward, based on the step-to-step change in a SHARED welfare-objective
snapshot ``Phi_t = condition.welfare_fn([M_1(t), ..., M_4(t)])``, where
``M_i(t)`` is the SAME ``running_active_attainment(trace)`` already used for
WSC observations and behavioural analysis -- no new welfare construct is
introduced (see ``thesis.study_b.utility.running_active_attainment`` and
``thesis.study_b.welfare_reward.WelfareCondition``).

Design record (``F:\\dense reward\\README.md`` Section 5 /
``configs\\FROZEN_EXPERIMENT_CONFIG.json``'s ``dense_reward_study_reserved``
block):

    DeltaPhi_t = Phi_(t+1) - Phi_t
    F_t = +c   if DeltaPhi_t >  epsilon
          -c   if DeltaPhi_t <  -epsilon
           0   otherwise
    r'_{i,t+1} = r_{i,t+1} + F_{t+1}

``F_t`` is a SINGLE shared value added identically to every active vehicle's
reward this step (not computed per-agent) -- this reuses
``exp01_ma_formal_v1``'s shared delta-min shaping DESIGN PATTERN only (see
``EXP01_DENSE_REWARD_FORMAL_V1_REFERENCE.md``). It does NOT reuse exp01's
``experience_score`` welfare definition: ``Phi_t`` here is built entirely
from the current Study B welfare construct (``running_active_attainment`` +
the condition's own ``welfare_fn``), never DER's four-weight aggregate.

``Phi_t`` is evaluated over the SAME FIXED vehicle-id set every step
(``env.active_vehicle_ids`` never shrinks when a vehicle exits -- see
``StudyBHeterogeneousHighwayEnv.active_vehicle_ids``), so a vehicle exiting
the active set does NOT change how many terms enter ``welfare_fn``: its
``running_active_attainment`` value simply stops updating (frozen at its
last value) rather than being dropped from the aggregate. This is
deliberate -- computing ``welfare_fn`` only over currently-active vehicles
would let a worst-off vehicle's exit produce a spurious ``DeltaPhi_t > 0``
the moment it leaves the minimum/mean, rewarding the OTHER vehicles for an
event they did not cause. This is the "active-set / exit artifact" this
design specifically avoids.

``c`` (``magnitude``) and ``epsilon`` are NOT decided here. They must be
explicitly supplied by the caller (CLI flags in the training scripts) once a
Dense Reward Study protocol freeze has chosen them -- this module
intentionally has no default value for either and raises if asked to shape
without them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from thesis.study_b.utility import EpisodeVehicleTrace, running_active_attainment
from thesis.study_b.welfare_reward import WelfareCondition

__all__ = [
    "DenseShapingConfig",
    "NEUTRAL_PHI",
    "welfare_objective_snapshot",
    "dense_shaping_term",
]

# running_active_attainment returns this same neutral value for a
# zero-sample trace; Phi at episode reset (all traces empty) is therefore
# always this value for mean/GGI/min alike (welfare_fn of four identical
# 1.0s is 1.0 for all three frozen conditions) -- stated explicitly here
# rather than recomputed via a welfare_fn call every reset, for clarity and
# to avoid a redundant call on the hot path.
NEUTRAL_PHI: float = 1.0


@dataclass(frozen=True)
class DenseShapingConfig:
    """Frozen, explicit dense-shaping parameters. ``enabled=False`` (the
    default and the only state exercised by any run before a Dense Reward
    Study protocol freeze) makes every function in this module a strict
    no-op, so training behaviour is byte-identical to a run built without
    this module at all."""

    enabled: bool = False
    mode: str = "discrete"
    magnitude: float | None = None
    epsilon: float | None = None

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if self.mode != "discrete":
            raise ValueError(
                f"dense shaping mode {self.mode!r} is not implemented -- only 'discrete' "
                "exists (README.md Section 5 / Priority 7's continuous ablation is not yet built)"
            )
        if self.magnitude is None or self.epsilon is None:
            raise ValueError(
                "dense_welfare_shaping=True requires an explicit, pre-frozen "
                "dense_shaping_magnitude (c) and dense_shaping_epsilon -- these are not "
                "guessed or defaulted by this module. Freeze the Dense Reward Study "
                "protocol and pass both values explicitly (e.g. --dense-shaping-magnitude, "
                "--dense-shaping-epsilon)."
            )
        if self.magnitude <= 0:
            raise ValueError(f"dense_shaping_magnitude must be positive, got {self.magnitude}")
        if self.epsilon < 0:
            raise ValueError(f"dense_shaping_epsilon must be non-negative, got {self.epsilon}")


def welfare_objective_snapshot(
    traces: Mapping[str, EpisodeVehicleTrace],
    vehicle_ids: Sequence[str],
    condition: WelfareCondition,
) -> float:
    """Phi_t = condition.welfare_fn([M_1(t), ..., M_N(t)]) over the FIXED
    ``vehicle_ids`` set -- never filtered by active status, see module
    docstring's "active-set / exit artifact" note. Pure function, no side
    effects, safe to call every step regardless of whether shaping is
    enabled (e.g. for --debug-reward-trace, which wants this value even
    when dense shaping itself is off)."""
    m_values = [running_active_attainment(traces[vid]) for vid in vehicle_ids]
    return condition.welfare_fn(m_values)


def dense_shaping_term(delta_phi: float, config: DenseShapingConfig) -> float:
    """F_t for one step transition, given DeltaPhi_t and a frozen config.
    Returns 0.0 unconditionally when ``config.enabled`` is False."""
    if not config.enabled:
        return 0.0
    assert config.magnitude is not None and config.epsilon is not None  # guaranteed by __post_init__
    if delta_phi > config.epsilon:
        return float(config.magnitude)
    if delta_phi < -config.epsilon:
        return float(-config.magnitude)
    return 0.0
