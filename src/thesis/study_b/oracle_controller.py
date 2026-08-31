"""VDN_Conditional_Amendment_Protocol.md sec 8 (Diagnostic 3): a
scripted/oracle N=4 controller for Study B's matched-TTC environment.

**Not a formal learned policy and never a thesis solver candidate** --
the document explicitly allows this diagnostic to read GLOBAL simulator
state (every vehicle's true position, speed, role, and matched-TTC
pairing) specifically because its only purpose is answering "is the
N=4 matched-TTC environment physically and procedurally solvable at
all?", independent of whatever local-observation/credit-assignment
problems the learned solver may have.

Three rules, checked in this priority order:

1. **Same-lane car-following safety** (checked FIRST, overrides rule 2):
   the two vehicles sharing a role ("ramp" or "mainline") share a lane the
   whole episode. If the nearest same-role vehicle ahead of ``vid`` is
   closer than ``min_following_gap`` (comfortably above the underlying
   env's own ``collision_distance=4.0`` m longitudinal collision
   threshold), ``vid`` decelerates regardless of the merge conflict below
   -- a same-lane rear-end is not a merge-timing problem and must not be
   overridden by cross-lane priority logic. (An earlier version of this
   controller omitted this rule and every observed collision under it was
   a same-lane rear-end, not a merge-point collision -- confirmed by
   inspecting ``collision_pairs`` on real Q-bank runs.)
2. **Cross-lane matched-TTC merge conflict**: each vehicle's other
   relevant conflict is with its matched-TTC "partner" -- the opposite-role
   vehicle sharing its ``ttc_slot`` ("front" or "rear" wave), exactly the
   pair ``scenario_generator.py`` constructed to have near-simultaneous
   arrival at ``merge_start``. Priority goes to whichever of the pair is
   currently closer to (or already past) ``merge_start``; the other
   vehicle decelerates to open a gap. Once either member of a pair has
   passed ``merge_end``, the lateral conflict is physically over and both
   are free to accelerate. Vehicles far from the interaction window
   (``lookahead`` metres from ``merge_start``) just hold their
   (already-at-target) spawn speed.
3. **Reaction-time margin is real, not a formality.** Discrete-time
   physics (dt=0.2s) with a bounded deceleration rate
   (``decel_rate=3.0`` m/s^2 -- see ``stage10_symmetric_merge_env.py``'s
   ``Stage10MergeEnvConfig``) means a vehicle cannot instantly cancel a
   closing-speed differential: it takes several steps of DECELERATE to
   actually reduce speed, during which the gap keeps shrinking. The
   thresholds below were empirically tuned (not guessed) against this
   physics -- see the calibration sweep referenced in this module's own
   development history -- to leave enough distance for deceleration to
   actually take effect before ``collision_distance`` (4.0m) is reached.
   The naive "start yielding right at the danger threshold" version of
   this controller (``min_following_gap=15``, ``lookahead=40``,
   ``min_gap_to_pass=20``) reached only ~17% completion with 83%
   same-lane rear-end collisions on the Q bank; the tuned values below
   reach 100% completion / 0% collision / 0% timeout across the Q (64),
   M (64), and H1 (256) scenario banks.

Deliberately environment-free (pure functions over positions/roles, no
``Stage10SymmetricMergeEnv`` dependency) -- same rationale as
``scenario_generator.py``: cheap, fast, large-scale unit testing without
paying for a single environment step."""

from __future__ import annotations

from typing import Mapping

from thesis.study_b.scenario_generator import ScenarioSpec

__all__ = ["MAINTAIN", "ACCELERATE", "DECELERATE", "partner_id", "oracle_action_for_vehicle", "oracle_actions"]

# Matches thesis.envs.stage10_symmetric_merge_env.HighLevelAction's integer
# encoding exactly (0=MAINTAIN, 1=ACCELERATE, 2=DECELERATE) -- not imported
# from there, to keep this module's core logic dependency-free/pure per its
# own docstring above.
MAINTAIN = 0
ACCELERATE = 1
DECELERATE = 2

DEFAULT_LOOKAHEAD_M = 80.0
DEFAULT_MIN_GAP_TO_PASS_M = 40.0
DEFAULT_MIN_FOLLOWING_GAP_M = 60.0  # empirically tuned -- see module docstring point 3


def partner_id(scenario: ScenarioSpec, vid: str) -> str:
    """The opposite-role vehicle sharing ``vid``'s ``ttc_slot`` -- the one
    vehicle ``vid`` was matched-TTC-constructed to have a temporal merge
    conflict with."""
    spec = scenario.vehicles[vid]
    for other_vid, other_spec in scenario.vehicles.items():
        if other_vid != vid and other_spec.ttc_slot == spec.ttc_slot and other_spec.role != spec.role:
            return other_vid
    raise ValueError(f"no matched-TTC partner found for {vid!r} in scenario {scenario.scenario_id!r}")


def oracle_action_for_vehicle(
    *, self_position: float, partner_position: float, merge_start: float, merge_end: float,
    same_lane_ahead_gap: float | None = None,
    lookahead: float = DEFAULT_LOOKAHEAD_M, min_gap_to_pass: float = DEFAULT_MIN_GAP_TO_PASS_M,
    min_following_gap: float = DEFAULT_MIN_FOLLOWING_GAP_M,
) -> int:
    """Pure function: given just this vehicle's and its partner's current
    ``route_position`` (plus, optionally, the gap to the nearest same-lane
    vehicle ahead), returns one action. See module docstring for the
    three rules and their priority order.

    ``same_lane_ahead_gap``: distance to the nearest same-role vehicle
    ahead of this one, or ``None``/``float("inf")`` if there isn't one
    (e.g. this vehicle is the lane's front-most). Checked FIRST -- a tight
    same-lane gap forces DECELERATE regardless of the merge-conflict rule
    below.

    Once EITHER vehicle in the merge-conflict pair has passed
    ``merge_end`` there is no more lateral conflict with this specific
    partner, so the caller is free to proceed on that rule -- this
    deliberately does not re-check ``self_position >= merge_end`` in
    isolation, since a vehicle that hasn't reached ``merge_end`` itself
    but whose partner already has is equally free (subject to rule 1
    still applying)."""
    if same_lane_ahead_gap is not None and same_lane_ahead_gap < min_following_gap:
        return DECELERATE

    if self_position >= merge_end or partner_position >= merge_end:
        return ACCELERATE

    self_dist = merge_start - self_position
    partner_dist = merge_start - partner_position

    if self_dist > lookahead and partner_dist > lookahead:
        return MAINTAIN  # not yet in the interaction window

    if self_dist <= partner_dist - min_gap_to_pass:
        return ACCELERATE  # clearly closer (or already past merge_start) -- take priority
    if partner_dist <= self_dist - min_gap_to_pass:
        return DECELERATE  # partner clearly closer -- yield

    return ACCELERATE if self_dist <= partner_dist else DECELERATE  # too close to call -- tie-break by distance


DEFAULT_LATERAL_SAME_LANE_THRESHOLD_M = 2.0


def _nearest_same_lane_ahead_gap(
    vid: str, scenario: ScenarioSpec, positions: Mapping[str, float], ids: list[str],
    *, lateral_positions: Mapping[str, float] | None = None,
    lateral_threshold: float = DEFAULT_LATERAL_SAME_LANE_THRESHOLD_M,
) -> float | None:
    """"Same lane" membership: when ``lateral_positions`` is given (real
    lane/y-coordinates, e.g. from a backend with actual lane geometry),
    membership is REAL-TIME physical proximity (|y_i - y_j| <
    lateral_threshold) -- this correctly treats a ramp vehicle that has
    already physically merged as same-lane with a mainline vehicle behind
    it, which the scenario's fixed ``role`` field alone cannot express
    (found 2026-08-16 via the HighwayEnv meta_speed oracle re-audit: a
    scenario-role-only same-lane check let an already-merged, slower ramp
    vehicle get rear-ended by a faster mainline vehicle, since they were
    never checked against each other -- different ``role`` values,
    despite being in the same physical lane at that point). When
    ``lateral_positions`` is omitted (default), falls back to the
    original ``scenario.vehicles[*].role`` grouping -- this is the ONLY
    lane concept the legacy 1D backend and the abstract ``ScenarioSpec``
    model have (no literal lane-change event exists there), and is
    UNCHANGED for any caller that doesn't pass the new parameter, so the
    already-validated legacy-backend and HighwayEnv-direct_accel oracle
    results (100% on all 384 scenarios) are unaffected."""
    self_position = positions[vid]
    if lateral_positions is not None:
        self_lateral = lateral_positions[vid]
        same_lane_ids = [
            other for other in ids
            if other != vid and abs(lateral_positions[other] - self_lateral) < lateral_threshold
        ]
    else:
        role = scenario.vehicles[vid].role
        same_lane_ids = [other for other in ids if other != vid and scenario.vehicles[other].role == role]
    gaps = [positions[other] - self_position for other in same_lane_ids if positions[other] > self_position]
    return min(gaps) if gaps else None


def oracle_actions(
    *, scenario: ScenarioSpec, positions: Mapping[str, float], merge_start: float, merge_end: float,
    active_vehicle_ids: Mapping[str, bool] | None = None,
    lookahead: float = DEFAULT_LOOKAHEAD_M, min_gap_to_pass: float = DEFAULT_MIN_GAP_TO_PASS_M,
    min_following_gap: float = DEFAULT_MIN_FOLLOWING_GAP_M,
    lateral_positions: Mapping[str, float] | None = None,
    lateral_threshold: float = DEFAULT_LATERAL_SAME_LANE_THRESHOLD_M,
) -> dict[str, int]:
    """``positions``: ``{vehicle_id: route_position}`` for every vehicle
    still needing an action this step. ``active_vehicle_ids``: optional
    filter (only vehicles present as keys, or with a truthy value, get an
    action) -- pass the env's own ``active_vehicle_ids``/``active`` info
    to skip already-exited vehicles. Same-lane following-gap checks only
    consider OTHER vehicles present in ``ids`` (i.e. still active) --
    an exited same-lane vehicle can no longer be rear-ended.
    ``lateral_positions``: optional ``{vehicle_id: world_y}`` -- when
    given, same-lane membership uses REAL physical lateral proximity
    instead of the scenario's fixed role field (see
    ``_nearest_same_lane_ahead_gap``'s docstring). Omit for byte-identical
    behavior to before this parameter existed."""
    ids = list(positions) if active_vehicle_ids is None else [vid for vid in positions if active_vehicle_ids.get(vid, True)]
    actions = {}
    for vid in ids:
        partner = partner_id(scenario, vid)
        partner_position = positions.get(partner, scenario.vehicles[partner].route_position)
        ahead_gap = _nearest_same_lane_ahead_gap(
            vid, scenario, positions, ids,
            lateral_positions=lateral_positions, lateral_threshold=lateral_threshold,
        )
        actions[vid] = oracle_action_for_vehicle(
            self_position=positions[vid], partner_position=partner_position,
            merge_start=merge_start, merge_end=merge_end, same_lane_ahead_gap=ahead_gap,
            lookahead=lookahead, min_gap_to_pass=min_gap_to_pass, min_following_gap=min_following_gap,
        )
    return actions
