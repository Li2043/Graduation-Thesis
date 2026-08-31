"""Stage 11 pilot (E30) -- stakeholder welfare measurement.

Ported from `chap02.tex`'s Stage 9 theoretical framework (Sections 2.4-2.6:
distributive fairness / Rawlsian maximin, utilitarian vs. Rawlsian welfare
aggregation, operationalising stakeholder experience). Stage 9's stakeholder
set is V = {A, B, B_front, B_rear} (2 independently-learned persistent
controllers + 2 scripted background vehicles); Stage 11 has no background
vehicles at all (both vehicles are learners, per the E30 scope decision), so
V here is just the 2 active vehicles in the current episode. The formulas
below are otherwise a direct, literal port of chap02.tex's notation:
target-speed attainment e_i, the absorbing completed-status E_i, the
mean/min welfare potentials Phi_mean/Phi_min, and the episode-level
mobility outcome U_i.

For v1-v11, this module measured welfare ONLY -- nothing here was added to
the training reward (CONDITION="baseline", no PBRS). v12 (see
stage11_dyad_merge_runner.py's joint-network training path and
stage11_dyad_merge_pilot_config.py's PBRS_LAMBDA_V12/PILOT_V12_*_SEEDS
docstring) is the "future PBRS extension" this module's original docstring
anticipated: it adds ``actual_potential``/``pbrs_shaping_signal`` (below) and
actually wires mean_welfare/min_welfare into the reward for the mean_pbrs/
min_pbrs conditions, running Stage 9's own C / D-mean / D-min three-
condition design (STRUCTURE.md's E30 section records the full
Stage-9-to-Stage-11 formula mapping and the reasoning for this sequencing)
on this project's Stage 11 environment instead of Stage 9's.

Stage 9's D_swap (does merge order change when persistent controller
identity A/B is swapped) has NO direct analogue here: Stage 11 uses ONE
shared network (stage10_shared_dqn.SharedDQNLearner), not two independently
learned per-identity networks -- there is no "network A" or "network B" to
swap. The substitute this project uses is role-vs-physical-identity
cross-validation: does the passing-order statistic (p_RF, "ramp passes
first") hold regardless of which physical vehicle identity (V0..V5) is
currently playing the ramp/mainline role this episode? See
`stage11_dyad_merge_runner.py`'s per-episode crossing-order bookkeeping for
how this is actually computed across episodes; this module only supplies
the pure, unit-testable building blocks.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def target_speed_attainment(speed: float, target_speed: float) -> float:
    """e_i(s_t) = clip_01(v_i(s_t) / v_i^target) -- chap02.tex Eq. (target-speed).
    Values above target speed are capped at 1.0: overspeed is never counted
    as extra welfare -- safety stays in the base reward and the separate
    outcome measures, not here."""
    if target_speed <= 0:
        raise ValueError(f"target_speed must be > 0, got {target_speed}")
    return float(np.clip(speed / target_speed, 0.0, 1.0))


def stakeholder_experience(attainment: float, completed: bool) -> float:
    """E_i(s_t) -- chap02.tex Eq. (absorbing). Once a vehicle has exited
    without a collision (completed=True), its experience is pinned at 1.0
    for the remainder of the episode instead of silently dropping out of
    the stakeholder set and its weight in the aggregations below."""
    return 1.0 if completed else float(attainment)


def mean_welfare(experiences: Sequence[float]) -> float:
    """Phi_mean -- chap02.tex Eq. (unadjusted-potentials): utilitarian
    equal-weight mean aggregation over the current stakeholder set."""
    if not experiences:
        raise ValueError("experiences must be non-empty")
    return float(sum(experiences) / len(experiences))


def min_welfare(experiences: Sequence[float]) -> float:
    """Phi_min -- chap02.tex Eq. (maximin): Rawlsian worst-off-stakeholder
    aggregation over the current stakeholder set."""
    if not experiences:
        raise ValueError("experiences must be non-empty")
    return float(min(experiences))


def episode_mobility_outcome(*, collided: bool, attainment_samples: Sequence[float]) -> float:
    """U_i(tau) -- chap02.tex Eq. (episode-utility). A vehicle involved in a
    collision scores 0 regardless of how well it was doing beforehand;
    otherwise this is the plain time-average of e_i over the steps the
    vehicle was actually on the road (before it exited). Post-exit
    absorbing values feed the state potentials above but are never appended
    to this average -- chap02.tex is explicit that the training-time
    potential and the episode-level evaluation outcome are kept separate."""
    if collided:
        return 0.0
    if not attainment_samples:
        return 0.0
    return float(sum(attainment_samples) / len(attainment_samples))


def actual_potential(raw_potential: float, *, terminated: bool, truncated: bool) -> float:
    """Phi_c(s_t) -- chap02.tex Eq. (full-potential): zero ONLY for a true
    terminal state (collision or joint success); truncation (episode-length
    cap) preserves the raw potential unchanged, per Grzes (2017)'s point
    (chap02.tex Sec. 2.3) that a non-zero terminal potential at a genuine
    terminal state would break PBRS's policy-invariance guarantee, while a
    truncation is an external computational cap, not a task terminal state.

    Mirrors ``thesis.rewards.pbrs_v2.compute_actual_potential`` but is a
    fresh, stakeholder-count-agnostic implementation written here rather
    than importing that module: ``pbrs_v2.py`` hardcodes a 4-member
    stakeholder set (``STAKEHOLDER_ORDER = ("A", "B", "B_front",
    "B_rear")``, e.g. its mean-potential function divides by a literal
    ``4.0``) for Stage 9's own 2-controller-plus-2-background-vehicle scene,
    which Stage 9 still depends on unmodified. Stage 11 (v12) has exactly 2
    stakeholders (ramp, mainline, no background vehicles) -- this function
    (and ``pbrs_shaping_signal`` below) is the Stage-11-scoped, N-agnostic
    equivalent, built on this module's own already N-agnostic
    ``mean_welfare``/``min_welfare`` rather than duplicating their logic."""
    if terminated and truncated:
        raise ValueError("terminated=True and truncated=True simultaneously is invalid")
    if terminated:
        return 0.0
    return float(raw_potential)


def pbrs_shaping_signal(phi_t: float, phi_t1: float, *, gamma: float) -> float:
    """F(s_t, s_{t+1}) = gamma * Phi(s_{t+1}) - Phi(s_t) -- chap02.tex Eq.
    (pbrs-general)/(shaped-reward). Callers apply this SAME increment
    (scaled by the condition's lambda coefficient) to every stakeholder's
    own base reward -- the potential is a function of the shared game state,
    not of any one agent, so the shaping increment each step is identical
    across stakeholders (chap02.tex Eq. shaped-reward, Sec. 2.3)."""
    return float(gamma * phi_t1 - phi_t)


def clawback_tracking_contribution(
    *, step_reward: float, exit_event: bool, exit_reward_magnitude: float
) -> float:
    """Stage 11 v3 fix: the portion of this step's reward that should still
    count as "at risk" for a future clawback collision penalty.

    Bug found in v2 (2026-08-06): the collision-detection loop in
    ``stage10_symmetric_merge_env.py`` checks every pair of ACTIVE vehicle
    ids regardless of completed status -- a vehicle that has already exited
    keeps moving at constant speed and can still register a (physically
    real, e.g. a trailing vehicle rear-ending it) collision later in the
    same episode. Under the plain clawback (cancel whatever's been
    accumulated), that collision would retroactively erase an ALREADY-
    BANKED exit bonus this vehicle legitimately earned -- confirmed
    empirically in v2's own trajectory data (one episode where the clawback
    penalty hit ~1.0, the full progress+exit ceiling, on a vehicle that had
    already exited). A collision should only ever cancel reward this
    vehicle's CURRENT, still-ongoing attempt has put at risk -- not a
    completed success. The fix: once a vehicle's exit event fires, its exit
    bonus is excluded from the running total used for future clawback
    calculations (permanently banked); everything else (ordinary per-step
    progress reward) still accumulates and remains clawback-eligible exactly
    as before."""
    if exit_event:
        return float(step_reward - exit_reward_magnitude)
    return float(step_reward)


def episode_collided_for_vehicle(
    *, episode_collided: bool, was_already_completed_before_collision: bool
) -> bool:
    """Stage 11 v3 fix: whether THIS vehicle's episode-level U_i should be
    zeroed by the episode's collision outcome.

    Same root cause as ``clawback_tracking_contribution`` above: a vehicle
    that had already exited (and is therefore already absorbed at E_i=1.0
    for the state-potential purposes chap02.tex defines) should not have its
    episode-level mobility outcome zeroed just because the OTHER vehicle
    later collided with it while it continued cruising past the exit line --
    chap02.tex's own U_i definition is "0 if i is involved in a collision",
    and a vehicle that had already safely completed its own journey is not
    meaningfully "involved" in a later incident it can no longer avoid or
    be put at risk by. Before this fix, the runner used one blanket
    episode-level boolean for every vehicle, which is what let an already-
    completed vehicle's U_i get zeroed alongside the vehicle that was
    actually still attempting the merge when the collision occurred."""
    return bool(episode_collided and not was_already_completed_before_collision)


__all__ = [
    "target_speed_attainment",
    "stakeholder_experience",
    "mean_welfare",
    "min_welfare",
    "actual_potential",
    "pbrs_shaping_signal",
    "episode_mobility_outcome",
    "clawback_tracking_contribution",
    "episode_collided_for_vehicle",
]
