"""Stage 10 pilot (E28) — symmetric merge environment, now 2/4/6-vehicle.

Replaces the Stage 8/9 asymmetric shape (2 independent learners A/B + 2
scripted IDM background vehicles B_front/B_rear) with a fully symmetric
shape: vehicles V0..V5, ALL learner-controlled, role (ramp/mainline)
randomised per episode across the active physical identities. No scripted
background traffic, no persistent controller identity (see
E28_STAGE10_PILOT_PROTOCOL_DRAFT.md S1 item 2 / S2.1).

Geometry history (protocol S2.2 / S0.1):
    v1 (round 1, Z-Merge-derived starting ratio): pre 150 / merging 100 / post 100
    v2 (round 2): lengthened post-merge 100->150 in response to round-1 step-
        density evidence -- later shown to be treating a symptom (policy
        competence), not the cause (geometry); reverted in v3.
    v3 (round 3): reverted post-merge back to 100m.
    v4 (round 4, THIS FILE): lengthens PRE-merge 150->200 instead -- a
        different lever, deliberately kept separate from the post-merge
        question so the two don't get entangled again. This is the "easier
        baseline geometry" lever (more approach/reaction room before the
        actual merge conflict), applied uniformly across all curriculum
        stages, not stage-specific.
    Current: pre-merge [0, 200) / merging [200, 300) / post-merge [300, 400].

Vehicle-count curriculum history:
    v2: hard step-40,000 cutover between 2 and 4 vehicles.
    v3: replaced the cutover with an episode-level probability blend
        (Portelas et al.'s ALP-GMM, arXiv:1910.07224, simplified) -- this
        made things WORSE (~2.7% vs v2's ~10.3% completion). Root cause: no
        observation feature ever told the network how many vehicles were in
        the current episode, so blending 2- and 4-vehicle episodes in the
        same time window fed contradictory optimal-action targets for
        near-identical inputs -- a genuine partial-observability gap, not
        just training noise.
    v4 (THIS FILE): (a) fixes the observability gap directly -- see
        ``scene_vehicle_count`` below; (b) replaces the blend with a
        threshold-triggered 3-stage curriculum (2 -> 4 -> 6 vehicles),
        modelled on Gupta, Egorov & Kochenderfer (2017)'s actual Algorithm 2
        (train until a performance threshold clears, then advance -- not a
        fixed split, not a blend); see
        ``stage10_role_phase_subpolicy_config.py``'s
        ``CURRICULUM_V4_*``/``stage_index_for_advance``. This is also a
        genuine SCOPE CHANGE (2026-08-05, user decision, recorded in the
        protocol doc S0.1): E28's final target is now a symmetric
        6-vehicle merge; 2 and 4 are curriculum stages en route to 6, not
        the terminus.

Scene-context observation feature (pilot v4, 9th obs dim): each vehicle's
observation now includes a scalar encoding how many vehicles are active in
the current episode -- (n_vehicles - 2) / 4.0, giving 0.0 / 0.5 / 1.0 for
2 / 4 / 6. This directly fixes the partial-observability gap identified
above: the network no longer has to guess the current regime from an
observation that would otherwise look identical across regimes.

Peer-intent observation feature (pilot v3, 8th obs dim, UNCHANGED here):
each vehicle's observation includes its nearest lane-comparable peer's most
recent action, encoded as -1.0/0.0/+1.0 (decelerate/maintain/accelerate). A
deliberately simplified proxy for Bouton, Nakhaei, Fujimura & Kochenderfer's
"Cooperation-Aware Reinforcement Learning for Merging in Dense Traffic"
(ITSC 2019, arXiv:1906.11021) belief-feature fix for the frozen_stall
failure mode round 2's trajectory logs confirmed.

N-per-side generalisation (pilot v4): rounds 1-3 only ever had 1 or 2
vehicles per role (ramp/mainline). This file generalises the spawn/queue
logic to 1, 2, or 3 vehicles per side (n_vehicles in {2, 4, 6}), and widens
the initial inter-vehicle spawn spacing by ~50% (protocol point 4: "as easy
as possible" baseline, tempered by not making the task uninformative).

Optional role/zone observation features (pilot v5, OFF by default --
``Stage10MergeEnvConfig.include_role_zone_features``): rounds 1-4's
independent (role, zone)-keyed sub-DQN architecture encoded role/zone
information ONLY implicitly, via which of 6 networks got called -- with
pilot v5's single shared network (see stage10_shared_dqn.py), that
information must become explicit or the one network genuinely cannot tell
vehicles apart. When this flag is True, the observation gains 4 more dims:
a role scalar (+1.0 mainline / -1.0 ramp, matching the convention already
used by this project's original formal 27-dim observation, see
final_observation.py) and a one-hot zone encoding (pre/merging/post) --
one-hot rather than an ordinal scalar because zone is categorical (merging
is not numerically "halfway between" pre and post in any meaningful sense).
Left False by default so pilot v1-v4's 9-dim observation and existing tests
are completely unaffected.

Both roles share identical route_start/merge_start/merge_end/route_exit
values so that, once a ramp vehicle enters the merging zone, its world_x is
directly comparable to a mainline vehicle's world_x on the same numeric
scale -- this is a deliberate pilot-scope simplification (not the old
asymmetric ramp route's separate approach-length/join-offset geometry).

Lateral position (world_y) gives "zone" concrete physical meaning: a ramp
vehicle is laterally offset (own lane) for the whole pre-merge zone, merges
linearly across the merging zone, and shares the through lane for the rest
of the route -- so "merging" is literally "actively crossing into the
shared lane", not an arbitrary label.

Reward revision (pilot v6, 2026-08-06): v5 (shared parameter + role/zone
features) raised 6-vehicle completion from v4's ~1% to ~13% average, but
left ~70-90% of episodes ending in collision (previously timeouts
dominated). This session's literature/GitHub research converged from three
angles on the same conclusion about the v1-v5 reward's collision handling:
(1) it is TERMINAL-ONLY -- zero gradient signal about approaching danger
before the exact moment of impact, whereas working systems commonly pair a
terminal penalty with a continuous risk-based (time-to-collision, TTC) term;
(2) it is applied UNIFORMLY to every active vehicle whenever ANY pair
collides, even vehicles not involved -- a MARL-driving survey (arXiv:
2408.09675) explicitly recommends penalising only the perpetrator(s); (3)
its magnitude (collision -1.0 vs completion +0.6, ~1.67:1) is low relative
to examples found in working systems (5:1 to 100:1+). All three changes
below are implemented; none is individually literature-proven as *the*
correct fix -- this is converging circumstantial evidence, not an
ablation-tested result, and is recorded as such (protocol doc S0.1).

    (a) Continuous TTC-based proximity-risk term, ``_ttc_risk_penalty``:
        penalises a vehicle continuously (not just at the collision instant)
        whenever it is closing on a lane-comparable neighbour (ahead or
        behind) with time-to-collision below ``TTC_PENALTY_THRESHOLD_SECONDS``,
        scaled by ``TTC_PENALTY_WEIGHT``.
    (b) Collision penalty now applies ONLY to vehicles in ``collided``
        (the actual colliding pair(s)), not to every active vehicle.
    (c) ``COLLISION_PENALTY_MAGNITUDE`` raised from 1.0 to 5.0 -- roughly an
        8.33:1 ratio against ``EXIT_REWARD_MAGNITUDE`` (0.6). The user opted
        to skip testing the intermediate 5:1 (3.0) value and jump straight
        to 5.0 (literature examples went as high as 100:1+, so this is
        still a moderate step, not the most aggressive value found).

This changes ONLY the reward computation in ``step()`` -- observation
dimensions, the shared-parameter architecture (stage10_shared_dqn.py), the
2->4->6 curriculum, LR/epsilon schedules, and geometry are all untouched
from pilot v5. See ``REWARD_VERSION_V6`` below for the self-describing
output tag.

Reward-magnitude curriculum (pilot v7, 2026-08-06): v6 cut collision rate
substantially (collision_free often 70-97% vs v5's 13-30%) but introduced a
NEW "too cautious" failure mode -- many seeds show high collision_free
alongside 0% completion (all-timeout), netting a WORSE average final
completion (~3.7%) than v5 (~13.3%) despite fixing the originally-diagnosed
problem.

User-directed research task: read, in full,
``Mohammad Alhazza -- Extending SYMPLEX for Real-Time Norm Induction in
Continuous Reinforcement Learning Environments`` (BSc dissertation, Univ. of
Bristol, provided by the user as a local PDF in this pilot's folder). This
dissertation runs an almost exactly analogous experiment (DQN-based highway
merge/driving agent, tuning a collision penalty against a speed-seeking
reward) and documents the SAME failure mode we just saw:

    - Its DQN002 (collision penalty jumped to -5.0, speed floor lowered from
      first step) "anchored at the new speed floor" and farmed survival time
      at minimum viable speed instead of actually completing tasks -- the
      dissertation's own words, "the floor became a ceiling". This is the
      same passive/over-cautious collapse our v6 shows.
    - Its fix, DQN004: a THREE-PHASE CONTINUOUS-RAMP curriculum on the
      collision-penalty magnitude itself -- Phase A (~24% of training) holds
      the penalty at a lenient fixed value so the network first builds a
      speed-seeking prior; Phase B (~32%) ramps it LINEARLY (not a discrete
      switch) up to the target value, updated in small steps as a "fine
      staircase approximation" of a continuous ramp; Phase C (~44%) locks it
      at the target for the remainder. Result: crash rate ~1.0->~0.6, and a
      genuine middle-ground driving speed rather than either extreme.
      The dissertation frames the CONTINUOUS-linear-interpolation mechanism
      (as opposed to a hard discrete phase switch) as its own design
      contribution, built on top of Freitag et al. (2025)'s two-stage reward
      curriculum concept (cited therein; not independently verified against
      the Freitag primary source by this project -- secondary citation only).
      Freitag et al.'s own ablation is reported (via Alhazza's citation) as
      showing the benefit comes from the PRETRAINED NETWORK carrying a prior
      forward through the transition, not from the replay buffer.
    - Its DQN005 (additional norm-penalty shaping terms applied at full
      strength from step zero) reproduced the SAME passive-collapse pattern.
      DQN006 fixed it with a SEPARATE curriculum on the norm-penalty scale
      alone (same phase boundaries, ramped independently of the collision
      penalty), showing the ramp-instead-of-static-strength fix generalises
      to ANY tightening reward-shaping term applied to a fresh agent, not
      just the primary collision penalty.

Applied here: BOTH ``COLLISION_PENALTY_MAGNITUDE`` and ``TTC_PENALTY_WEIGHT``
become curriculum-ramped rather than static, mirroring DQN004 (collision
penalty) and DQN006 (a second, independently-ramped shaping term) at once.
Implementation choice: ONE single continuous linear ramp across the whole
budget (steps 0 -> ``REWARD_CURRICULUM_RAMP_STEPS`` = 100,000, matching this
pilot's own Stage 1+2 combined safety-valve budget -- see
``stage10_role_phase_subpolicy_config.py``'s ``CURRICULUM_V4_STAGE_MAX_STEPS``
-- so the full-strength penalty is locked in for the entirety of Stage 3,
the hardest/final 6-vehicle stage, mirroring DQN004's Phase C "locked,
consolidating" period), NOT re-ramped at each curriculum vehicle-count
transition -- this matches DQN004's own single-ramp-across-training design
(it was not run alongside any other simultaneous curriculum to interleave
with). Start values: collision penalty resumes at 1.0 (this pilot's own
pre-v6 magnitude, already validated as not causing runaway reward-hacking in
v1-v5) rather than Alhazza's -0.5, since our absolute reward scale already
differs; TTC weight starts at a small 0.02 (proportionally similar
lenient-start ratio to its own 0.1 target as Alhazza's collision-penalty
ramp uses). The ramp schedule functions
(``collision_penalty_at_step``/``ttc_weight_at_step``) live in
``stage10_role_phase_subpolicy_config.py`` alongside ``epsilon_for_step``/
``lr_at_step`` (same "pure function of step, computed by the runner, passed
down" pattern); this env's ``step()`` accepts them as optional overrides
(``collision_penalty_magnitude``/``ttc_penalty_weight``), defaulting to the
module constants below when omitted so v1-v6 call sites/tests are
unaffected. See ``REWARD_VERSION_V7`` for the self-describing output tag.

Collision-penalty "clawback" (Stage 11 / E30 pilot v2, 2026-08-06): Stage 11
(a separate, later pilot reusing this same env unmodified -- see
``stage11_dyad_merge_pilot_config.py``/``stage11_dyad_merge_runner.py``) ran
its own v1 with this file's curriculum-ramped ``COLLISION_PENALTY_MAGNITUDE``
locked at 5.0 for the back 60% of training and found a NEW failure mode via
per-step trajectory analysis: both vehicles collapse to near-zero speed and
never leave the pre-merge zone, i.e. total avoidance rather than merely
over-caution. Diagnosis: TTC fired essentially never (fire rate ~0.001), so
the fixed terminal collision penalty was the sole driver -- and a FIXED
magnitude disconnected from how much reward a vehicle has actually earned
this episode makes "never attempt the merge" reward-dominant once the
network's perceived collision probability exceeds a modest threshold
(roughly ``EXIT_REWARD_MAGNITUDE / COLLISION_PENALTY_MAGNITUDE``), regardless
of curriculum timing. Stage 11 v2's fix -- the ``collision_penalty_override``
parameter above -- makes the collision penalty a per-vehicle "clawback"
instead: exactly cancel whatever positive reward that specific vehicle has
already earned this episode (computed by the runner, not this env, since it
requires per-episode reward bookkeeping the env itself does not keep), so a
collision can never leave a vehicle net ahead of where it started, but also
never costs more than what was actually put at risk -- removing the fixed,
position-independent catastrophic downside that made blanket avoidance
reward-dominant. See ``stage11_dyad_merge_runner.py`` for the exact
bookkeeping and STRUCTURE.md's E30 section for the full diagnosis.

Stage 9-based reward redesign (Stage 11 / E30 pilot v4, 2026-08-07): v1-v3
of Stage 11 (all built on THIS file's v6/v7 reward: progress + exit +
perpetrator-only terminal collision + TTC) topped out at ~13% average
completion, with per-step audits repeatedly showing the same "total
avoidance" collapse (near-zero speed, permanently stuck pre-merge) despite
three different collision-penalty mechanisms (static 5.0, curriculum-ramped,
clawback). User flagged that Stage 9's own base reward (formally locked,
``thesis.rewards.base_reward_v2``, 80% success on a HARDER 4-vehicle/2-
scripted-background setup with no zone/role observations) was never actually
ported into Stage 10/11 -- it has two terms this file's reward has always
lacked: a hard-braking cost and a per-step active-time cost. Diagnosis: this
file's collision penalty (5.0, ~8.33:1 vs exit 0.6) is also far above Stage
9's own proven 1.0 (~1.67:1), and with no time cost, "crawl slowly and let
the episode time out" was a near-zero-risk, ~0.3-0.35-payoff fallback (pure
progress reward, nothing charged against it) -- easily reward-competitive
with attempting completion once perceived collision risk is non-trivial.

Pilot v4 ports Stage 9's hard-braking/time-cost terms VERBATIM (same
``compute_hard_braking_cost`` function, same locked constants a_comfort=1.5/
a_hard=3.5/eta_hard_brake=0.015/active_time_cost_per_step=0.0005 -- see
``final_lock_loader.py``'s hard validation of these exact values), while
keeping this file's own already-existing perpetrator-only collision
attribution (an improvement over Stage 9's blanket 4-stakeholder attribution
-- see "Reward revision" above) and LOWERING ``COLLISION_PENALTY_MAGNITUDE``
back down to Stage 9's proven 1.0 for this version (the 5.0 v6/v7 value was
never validated against a reward that also charges a time cost, and is
itself a plausible contributor to the avoidance collapse). TTC and the
collision-magnitude curriculum are NOT carried into this version (TTC fire
rate was confirmed ~0.001, effectively inert; Stage 9 has no analog to
either mechanism) -- deliberately kept out to cleanly test whether Stage 9's
proven structure, plus the one perpetrator-only-attribution improvement, is
sufficient on its own before layering anything else back on. The 2->4->6
vehicle-count curriculum is also deliberately deferred to a later round, to
avoid conflating "is the reward right" with "does scaling to more vehicles
work" (the same conflation that made pilot v4's original failure hard to
diagnose).

Both new terms are added as optional, default-off ``step()`` kwargs
(``hard_braking_eta``/``time_cost_per_step``, both default ``None`` ->
disabled/0.0), so v1-v7 and Stage 11 v1-v3 call sites are byte-for-byte
unaffected. See ``REWARD_VERSION_STAGE9_BASED`` below and
``stage11_dyad_merge_runner.py`` for the Stage 11 v4 call site.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from thesis.rewards.base_reward_v2 import compute_hard_braking_cost

# Pilot v4: extended from 4 to 6 physical identities (3-per-side max).
VEHICLE_IDS: tuple[str, ...] = ("V0", "V1", "V2", "V3", "V4", "V5")
ROLE_VALUES: tuple[str, ...] = ("ramp", "mainline")
ZONE_VALUES: tuple[str, ...] = ("pre", "merging", "post")

LANE_OFFSET = 4.0
OBS_DIM = 9  # own_speed, own_progress, dist_to_zone_end, gap_ahead, relspeed_ahead, gap_behind, relspeed_behind, peer_intent, scene_vehicle_count
# Pilot v5 (shared-parameter architecture): +4 when include_role_zone_features
# is set -- role scalar (+1) + zone one-hot (+3). See module docstring.
OBS_DIM_WITH_ROLE_ZONE = OBS_DIM + 4  # 13
# Stage 11 v11 (priority-signal / symmetry-breaking): +2 when
# include_priority_signal (own_priority, peer_priority) -- see
# Stage10MergeEnvConfig.include_priority_signal's docstring.
OBS_DIM_WITH_ROLE_ZONE_AND_PRIORITY = OBS_DIM_WITH_ROLE_ZONE + 2  # 15
N_ACTIONS = 3
NO_NEIGHBOUR_GAP = 500.0  # large sentinel, not inf (keeps observation finite)

# Pilot v6 reward constants (2026-08-06, see module docstring "Reward
# revision" section for the full rationale/citations). Named explicitly
# (rather than left as inline literals in step()'s reward formula) so the
# exact reward this pilot used can be read back precisely without digging
# through the formula -- this will very likely need citing precisely in the
# thesis write-up.
PROGRESS_REWARD_WEIGHT = 0.4  # unchanged since pilot v1
EXIT_REWARD_MAGNITUDE = 0.6  # unchanged since pilot v1
COLLISION_PENALTY_MAGNITUDE = 5.0  # pilots v1-v5: 1.0 (~1.67:1 vs EXIT_REWARD_MAGNITUDE); v6: 5.0 (~8.33:1, user opted to skip the intermediate 5:1/3.0 test and jump straight here)
TTC_PENALTY_THRESHOLD_SECONDS = 1.2  # matches this session's literature finding
TTC_PENALTY_WEIGHT = 0.1  # deliberately modest starting coefficient -- NOT itself literature-validated,
# only the *existence* of a continuous (vs. terminal-only) term is; a tunable judgement call.

# Stage 9-based redesign (Stage 11 / E30 pilot v4, see module docstring
# "Stage 9-based reward redesign" section). Values copied VERBATIM from Stage
# 9's formally locked comfort calibration -- see
# ``final_lock_loader.py``'s ``FinalLockBlockedError`` guard, which hard-
# validates exactly these four numbers. Named separately from v1-v7's own
# constants above (rather than reusing/overloading them) since these two
# terms are opt-in (``step()``'s ``hard_braking_eta``/``time_cost_per_step``
# default to ``None`` -> disabled) and were never part of the v1-v7 reward.
HARD_BRAKING_ETA = 0.015  # Stage 9's eta_hard_brake
TIME_COST_PER_STEP = 0.0005  # Stage 9's active_time_cost_per_step
A_COMFORT = 1.5  # Stage 9's a_comfort
A_HARD = 3.5  # Stage 9's a_hard

REWARD_VERSION_V1_V5 = "terminal_uniform_collision_penalty"  # pilots v1-v5 (historical, not used by this file's current step())
REWARD_VERSION_V6 = "ttc_shaping_perpetrator_only_5to1"  # pilot v6 (historical -- static magnitudes, no curriculum)
REWARD_VERSION_V7 = "ttc_shaping_perpetrator_only_curriculum_ramped"  # this file's current reward, see module docstring
REWARD_VERSION_STAGE9_BASED = "stage9_based_hardbrake_timecost_perpetrator_collision_1to1"  # Stage 11 / E30 pilot v4


def _ttc_risk_penalty(gap: float, closing_speed: float, *, threshold: float) -> float:
    """Pilot v6: continuous proximity-risk shaping term. Returns 0.0 when
    there's no real neighbour (``gap`` is the ``NO_NEIGHBOUR_GAP`` sentinel or
    beyond), the neighbour isn't being approached (``closing_speed <= 0``), or
    time-to-collision is at/above ``threshold``; otherwise scales linearly
    from 0.0 at ``ttc == threshold`` to -1.0 at ``ttc == 0``.

    ``closing_speed`` must already be oriented so POSITIVE means closing/
    risky -- ahead-side and behind-side neighbours have OPPOSITE relspeed
    sign conventions (see the observation's ``relspeed_ahead``/
    ``relspeed_behind`` and this file's ``step()``, which orients each side
    correctly before calling this); this function itself is agnostic to
    which side it's being used for."""
    if gap is None or gap >= NO_NEIGHBOUR_GAP or closing_speed <= 0.0:
        return 0.0
    ttc = gap / closing_speed
    if ttc >= threshold:
        return 0.0
    return -(threshold - ttc) / threshold


class HighLevelAction:
    MAINTAIN = 0
    ACCELERATE = 1
    DECELERATE = 2


def encode_peer_intent(action: int) -> float:
    """Pilot v3 peer-intent feature encoding: -1.0/0.0/+1.0 for
    decelerate/maintain/accelerate -- a directly-observable "revealed intent"
    proxy (see module docstring), not a learned belief."""
    if action == HighLevelAction.ACCELERATE:
        return 1.0
    if action == HighLevelAction.DECELERATE:
        return -1.0
    return 0.0


def encode_scene_vehicle_count(n_vehicles: int) -> float:
    """Pilot v4 scene-context feature: (n_vehicles - 2) / 4.0, giving
    0.0 / 0.5 / 1.0 for 2 / 4 / 6 vehicles -- fixes the partial-observability
    gap identified in round 3 (see module docstring)."""
    return float(n_vehicles - 2) / 4.0


def encode_role(role: str) -> float:
    """Pilot v5: +1.0 mainline / -1.0 ramp -- matches this project's
    pre-existing convention from the original formal 27-dim observation
    (final_observation.py: "own traffic-role indicator (+1 mainline, -1 ramp)")."""
    if role not in ROLE_VALUES:
        raise ValueError(f"unknown role {role!r}, expected one of {ROLE_VALUES}")
    return 1.0 if role == "mainline" else -1.0


def encode_zone_onehot(zone: str) -> tuple[float, float, float]:
    """Pilot v5: one-hot (pre, merging, post) -- categorical, not ordinal
    (see module docstring for why this differs from scene_vehicle_count's
    scalar encoding)."""
    if zone not in ZONE_VALUES:
        raise ValueError(f"unknown zone {zone!r}, expected one of {ZONE_VALUES}")
    return (
        1.0 if zone == "pre" else 0.0,
        1.0 if zone == "merging" else 0.0,
        1.0 if zone == "post" else 0.0,
    )


def zone_for_position(route_position: float, merge_start: float, merge_end: float) -> str:
    """Deterministic, left-closed zone boundaries: exactly at merge_start -> 'merging',
    exactly at merge_end -> 'post'. Same rule used for routing (protocol S8 item 1)."""
    if route_position < merge_start:
        return "pre"
    if route_position < merge_end:
        return "merging"
    return "post"


def _zone_end(route_position: float, merge_start: float, merge_end: float, route_exit: float) -> float:
    zone = zone_for_position(route_position, merge_start, merge_end)
    if zone == "pre":
        return merge_start
    if zone == "merging":
        return merge_end
    return route_exit


def world_y_for(role: str, route_position: float, merge_start: float, merge_end: float) -> float:
    if role == "mainline":
        return 0.0
    if route_position <= merge_start:
        return -LANE_OFFSET
    if route_position >= merge_end:
        return 0.0
    frac = (route_position - merge_start) / (merge_end - merge_start)
    return -LANE_OFFSET * (1.0 - frac)


@dataclass(frozen=True)
class Stage10RouteGeometry:
    route_start: float = 0.0
    merge_start: float = 200.0  # pilot v4: 150 -> 200 (easier-baseline lever, see module docstring)
    merge_end: float = 300.0
    route_exit: float = 400.0  # pilot v3 reverted post-merge to 100m; v4 only changes pre-merge

    def validate(self) -> None:
        if not (self.route_start < self.merge_start < self.merge_end < self.route_exit):
            raise ValueError(
                "invalid Stage 10 geometry ordering: "
                f"start={self.route_start}, merge_start={self.merge_start}, "
                f"merge_end={self.merge_end}, exit={self.route_exit}"
            )


@dataclass
class Stage10MergeEnvConfig:
    dt: float = 0.2
    # Pilot v4: 400 -> 600 (route is only ~14% longer, but this is a
    # deliberately generous increase, not just a proportional one -- protocol
    # point 4 asked for "as easy as possible" on the time-budget axis
    # specifically, tempered by not making the task uninformative).
    max_steps: int = 600
    seed: int = 0
    n_vehicles: int = 4  # pilot v4 curriculum: 2, 4, or 6 (1/2/3 vehicles per side)
    geometry: Stage10RouteGeometry = field(default_factory=Stage10RouteGeometry)
    accel_rate: float = 2.0
    decel_rate: float = 3.0
    maintain_accel: float = 0.0
    v_min: float = 0.0
    v_max: float = 30.0
    collision_distance: float = 4.0
    collision_lateral_threshold: float = 1.5
    target_speed: float = 20.0
    # Stage 11 speed-asymmetry option (default None -- every existing
    # caller/version, Stage 10 and Stage 11 v1-v12, completely unaffected):
    # if set, ramp-role vehicles target THIS speed instead of the shared
    # ``target_speed`` above (which the ramp role then implicitly treats as
    # the mainline role's target -- there is deliberately no separate
    # ``target_speed_mainline`` field, since ``target_speed`` already plays
    # that role for every prior caller). Mirrors real on-ramp/mainline
    # dynamics: mainline vehicles cruise at highway speed, on-ramp vehicles
    # accelerate from a slower acceleration-lane speed (see
    # STAGE11_PROTOCOL.md Sec 12.4 for the literature grounding this was
    # added in response to -- fairness interventions need real baseline
    # asymmetry between roles to have something to correct, and a shared
    # target_speed for both roles under-supplies that). ``target_speed``
    # itself is not read anywhere inside this environment class (confirmed
    # by grep) -- it is only ever consumed externally, by
    # ``stage11_dyad_merge_runner.py``'s welfare/attainment calculations --
    # so this option requires no change to this class's own physics/reward
    # logic, only to the runner's per-vehicle target-speed lookup.
    target_speed_ramp: float | None = None
    # Spawn queue (protocol point 4: ~50% wider spacing than v1-v3's implicit
    # 40m gap). Generalised (pilot v4) from a fixed lead/trail pair to an
    # N-per-side queue: the i-th member of a role's queue (i=0 closest to the
    # merge) spawns at ``spawn_route_lead - i * spawn_queue_gap``.
    spawn_route_lead: float = 40.0
    spawn_queue_gap: float = 60.0  # v1-v3 implicit gap was 40.0 (spawn_route_lead - old spawn_route_trail=0); +50%
    spawn_speed: float = 18.0
    # Companion to target_speed_ramp above (same default-None,
    # every-prior-caller-unaffected contract): if set, ramp-role vehicles
    # SPAWN at this speed instead of the shared ``spawn_speed``. Unlike
    # target_speed, spawn_speed IS read inside this class's own reset()
    # (below), so this field does require an env-side lookup change.
    spawn_speed_ramp: float | None = None
    # Pilot v5 only (default False -- v1-v4 completely unaffected): append
    # explicit role/zone features (see module docstring). Only meaningful
    # when this env is driving a single shared network rather than routed
    # sub-policies.
    include_role_zone_features: bool = False
    # Stage 11 v11 (default False -- every existing caller/version, Stage 10
    # and Stage 11 v1-v10, completely unaffected): a fresh per-episode random
    # scalar in [0,1) is drawn for each active vehicle at reset() time (NOT
    # redrawn per-step -- deliberately adapted from Patil et al. 2026
    # ("Randomness is sometimes necessary for coordination" / "Diamond
    # Attention", arXiv:2605.06825)'s per-timestep redraw, since Stage 11 is
    # a single one-shot merge-conflict-resolution episode rather than their
    # continuous multi-step coordination setting -- redrawing every step
    # would make an approaching vehicle's "priority" flicker mid-approach,
    # which doesn't fit a single conflict resolution). When True, each
    # vehicle's observation gets two extra appended features: its OWN draw
    # and its PEER's draw (both raw values, not a pre-computed
    # comparison/boolean -- the policy must learn any comparison itself).
    #
    # Rationale: Patil et al. 2026 prove that under permutation-symmetric/
    # repeating observations, a deterministic policy structurally cannot
    # break a tie -- facing a literally-identical recurring situation, it can
    # only ever produce ONE fixed response. Their fix is a fresh random
    # scalar per decision point, visible to BOTH agents (their own ablation
    # shows private/unstructured noise gets 0% success -- "protocol-space
    # structure, not stochastic noise, is the operative ingredient"); on
    # their symmetric XOR game, deterministic baselines plateau at exactly
    # 0.5 while their mechanism reaches 1.0. This project's own pre-v6
    # per-step collision-signature audit found "matching action pair"
    # dominance at collision time -- both vehicles picking the SAME action
    # at a recurring symmetric decision point -- a plausible instance of
    # this exact pathology, since Stage 11's vehicles are deliberately
    # spawned at identical position/speed every episode (preserved, not a
    # bug). This is a coordination DEVICE the agents may learn to exploit,
    # not an imposed rule -- nothing forces any particular use of it; the
    # existing, unchanged reward signal is the only thing that can make the
    # network learn to condition on it.
    include_priority_signal: bool = False

    def validate(self) -> None:
        self.geometry.validate()
        if self.dt <= 0:
            raise ValueError("dt must be > 0")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        if self.n_vehicles not in (2, 4, 6):
            raise ValueError(f"n_vehicles must be 2, 4, or 6, got {self.n_vehicles}")
        if self.target_speed_ramp is not None and self.target_speed_ramp <= 0:
            raise ValueError(f"target_speed_ramp must be > 0, got {self.target_speed_ramp}")
        if self.spawn_speed_ramp is not None and self.spawn_speed_ramp < 0:
            raise ValueError(f"spawn_speed_ramp must be >= 0, got {self.spawn_speed_ramp}")


@dataclass
class Stage10VehicleState:
    identity: str
    role: str
    route_position: float
    speed: float
    acceleration: float = 0.0
    completed: bool = False


class Stage10SymmetricMergeEnv(gym.Env):
    """Fully symmetric 2/4/6-vehicle merge environment (Stage 10 pilot, E28)."""

    metadata = {"render_modes": []}

    def __init__(self, config: Stage10MergeEnvConfig | None = None):
        super().__init__()
        self.config = config or Stage10MergeEnvConfig()
        self.config.validate()
        # Pilot v5: effective per-vehicle obs dim depends on include_role_zone_features
        # (default False -- v1-v4 unaffected, obs_dim stays OBS_DIM=9). Pilot
        # v11 (Stage 11): +2 more when include_priority_signal is also set.
        self._obs_dim = OBS_DIM
        if self.config.include_role_zone_features:
            self._obs_dim += 4
        if self.config.include_priority_signal:
            self._obs_dim += 2
        self.action_space = spaces.Dict(
            {vid: spaces.Discrete(N_ACTIONS) for vid in VEHICLE_IDS}
        )
        self.observation_space = spaces.Dict(
            {
                vid: spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float64)
                for vid in VEHICLE_IDS
            }
        )
        self._np_random: np.random.Generator | None = None
        self._vehicles: dict[str, Stage10VehicleState] = {}
        self._roles: dict[str, str] = {}
        self._completed: dict[str, bool] = {}
        self._active_ids: tuple[str, ...] = VEHICLE_IDS
        self._last_action: dict[str, int] = {}  # pilot v3: peer-intent feature source
        self._step_count = 0
        self._seed_used: int | None = None
        self._episode_active = False
        self.last_info: dict[str, Any] = {}

    # ------------------------------------------------------------------ utils
    @property
    def active_vehicle_ids(self) -> tuple[str, ...]:
        """Currently-active vehicle identities (pilot v4 curriculum: 2, 4, or
        6, fixed for the duration of the current episode -- set at reset())."""
        return self._active_ids

    def role_of(self, identity: str) -> str:
        return self._roles[identity]

    def zone_of(self, identity: str) -> str:
        g = self.config.geometry
        return zone_for_position(self._vehicles[identity].route_position, g.merge_start, g.merge_end)

    def is_completed(self, identity: str) -> bool:
        return bool(self._completed[identity])

    def legal_actions(self) -> list[str]:
        return ["MAINTAIN", "ACCELERATE", "DECELERATE"]

    def action_mask(self, identity: str) -> np.ndarray:
        # No role/identity-dependent restriction in this pilot -- all 3 actions always legal.
        return np.array([True, True, True], dtype=bool)

    def _world_xy(self, veh: Stage10VehicleState) -> tuple[float, float]:
        g = self.config.geometry
        x = veh.route_position
        y = world_y_for(veh.role, veh.route_position, g.merge_start, g.merge_end)
        return x, y

    def _action_to_accel(self, act: int) -> float:
        if act == HighLevelAction.ACCELERATE:
            return float(self.config.accel_rate)
        if act == HighLevelAction.DECELERATE:
            return float(-self.config.decel_rate)
        return float(self.config.maintain_accel)

    # ----------------------------------------------------------------- reset
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.config.seed)
        self._seed_used = int(seed)
        self._np_random = np.random.default_rng(self._seed_used)
        self._step_count = 0
        self._episode_active = True

        n = int(self.config.n_vehicles)
        per_role = n // 2  # 1 (2 vehicles), 2 (4 vehicles), or 3 (6 vehicles)
        self._active_ids = VEHICLE_IDS[:n]
        self._completed = {vid: False for vid in self._active_ids}
        # Pilot v3: neutral default (MAINTAIN) -- no peer action has been
        # observed yet at the start of an episode.
        self._last_action = {vid: HighLevelAction.MAINTAIN for vid in self._active_ids}

        # Symmetric role randomisation: which physical identities are "ramp"
        # this episode vs "mainline" -- protocol S2.1, no persistent identity.
        # Generalised from round-1's fixed 2+2 pool to an arbitrary per_role
        # count so all three curriculum stages share code.
        roles_pool = ["ramp"] * per_role + ["mainline"] * per_role
        perm = self._np_random.permutation(n)
        self._roles = {self._active_ids[int(i)]: roles_pool[j] for j, i in enumerate(perm)}

        # Pilot v11 (priority-signal / symmetry-breaking, see
        # Stage10MergeEnvConfig.include_priority_signal's docstring): one
        # fresh draw per active vehicle for the whole episode, using the SAME
        # per-episode-reseeded self._np_random as role permutation/spawn
        # jitter above/below, so the episode stays fully deterministic given
        # the reset seed. Guarded behind the flag (not drawn unconditionally)
        # so every existing Stage 10/Stage 11 v1-v10 caller that doesn't set
        # include_priority_signal gets an IDENTICAL random-number-stream
        # sequence (role permutation -> spawn jitter -> ...) to before this
        # feature existed -- an unconditional draw here would silently shift
        # every subsequent np_random.uniform() call (spawn jitter, etc.) for
        # callers that never asked for this feature.
        self._priority: dict[str, float] = {}
        if self.config.include_priority_signal:
            self._priority = {
                vid: float(self._np_random.uniform(0.0, 1.0)) for vid in self._active_ids
            }

        # Pilot v4: generalised N-per-side spawn queue (was a hardcoded
        # lead/trail pair, or-else a single vehicle). The i-th member of a
        # role's queue (sorted by identity for determinism) spawns at
        # spawn_route_lead - i * spawn_queue_gap -- i=0 is closest to the
        # merge point. A single shared jitter is applied to the whole queue
        # (not per-member) so relative in-queue spacing stays exactly
        # spawn_queue_gap regardless of jitter -- a minor behavioural
        # simplification vs. v1-v3's lead-only jitter (harmless given jitter
        # is +/-0.5m).
        jitter = float(self._np_random.uniform(-0.5, 0.5))
        gap = float(self.config.spawn_queue_gap)
        lead = float(self.config.spawn_route_lead)
        self._vehicles = {}
        for role in ROLE_VALUES:
            members = sorted(vid for vid in self._active_ids if self._roles[vid] == role)
            if len(members) != per_role:
                raise RuntimeError(
                    f"expected exactly {per_role} vehicles with role={role}, got {members}"
                )
            spawn_speed_this_role = (
                float(self.config.spawn_speed_ramp)
                if role == "ramp" and self.config.spawn_speed_ramp is not None
                else float(self.config.spawn_speed)
            )
            for i, vid in enumerate(members):
                route_position = lead - i * gap + jitter
                self._vehicles[vid] = Stage10VehicleState(
                    identity=vid,
                    role=role,
                    route_position=float(route_position),
                    speed=spawn_speed_this_role,
                )

        info = {
            "seed": self._seed_used,
            "roles": dict(self._roles),
            "geometry": {
                "merge_start": self.config.geometry.merge_start,
                "merge_end": self.config.geometry.merge_end,
                "route_exit": self.config.geometry.route_exit,
            },
        }
        self.last_info = info
        return self._obs_from_vehicles(), info

    # ------------------------------------------------------------------ obs
    def _neighbours(self, identity: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Nearest vehicle ahead / behind by world_x among ALL active vehicles
        (protocol S2.3: own state + peer/context, no redundant role/zone
        feature). Each returned dict also carries the peer's ``id`` (pilot
        v3: needed to look up its last action for the peer-intent feature)."""
        me = self._vehicles[identity]
        mx, my = self._world_xy(me)
        ahead = None
        behind = None
        ahead_gap = None
        behind_gap = None
        for other_id, other in self._vehicles.items():
            if other_id == identity:
                continue
            ox, oy = self._world_xy(other)
            if abs(oy - my) >= 1.5:
                continue  # not in a laterally-comparable lane right now
            if ox >= mx:
                gap = ox - mx
                if ahead_gap is None or gap < ahead_gap:
                    ahead_gap = gap
                    ahead = {"id": other_id, "gap": gap, "speed": other.speed}
            else:
                gap = mx - ox
                if behind_gap is None or gap < behind_gap:
                    behind_gap = gap
                    behind = {"id": other_id, "gap": gap, "speed": other.speed}
        return ahead, behind

    def _peer_intent(self, ahead: dict[str, Any] | None, behind: dict[str, Any] | None) -> float:
        """Pilot v3: nearest lane-comparable peer's most recent action,
        encoded via ``encode_peer_intent``. Neutral (0.0) when isolated."""
        nearest = None
        if ahead is not None and behind is not None:
            nearest = ahead if ahead["gap"] <= behind["gap"] else behind
        elif ahead is not None:
            nearest = ahead
        elif behind is not None:
            nearest = behind
        if nearest is None:
            return 0.0
        return encode_peer_intent(self._last_action.get(nearest["id"], HighLevelAction.MAINTAIN))

    def _obs_from_vehicles(self) -> dict[str, np.ndarray]:
        g = self.config.geometry
        scene_vehicle_count = encode_scene_vehicle_count(len(self._active_ids))
        out: dict[str, np.ndarray] = {}
        for vid in self._active_ids:
            veh = self._vehicles[vid]
            progress = (veh.route_position - g.route_start) / (g.route_exit - g.route_start)
            dist_to_zone_end = _zone_end(veh.route_position, g.merge_start, g.merge_end, g.route_exit) - veh.route_position
            ahead, behind = self._neighbours(vid)
            gap_ahead = float(ahead["gap"]) if ahead else NO_NEIGHBOUR_GAP
            relspeed_ahead = float(veh.speed - ahead["speed"]) if ahead else 0.0
            gap_behind = float(behind["gap"]) if behind else NO_NEIGHBOUR_GAP
            relspeed_behind = float(veh.speed - behind["speed"]) if behind else 0.0
            peer_intent = self._peer_intent(ahead, behind)
            values = [
                veh.speed,
                float(np.clip(progress, 0.0, 1.0)),
                float(dist_to_zone_end),
                gap_ahead,
                relspeed_ahead,
                gap_behind,
                relspeed_behind,
                peer_intent,
                scene_vehicle_count,
            ]
            if self.config.include_role_zone_features:
                # Pilot v5: explicit role/zone -- see module docstring. zone_t
                # (this vehicle's zone at time of THIS observation, i.e. before
                # the action this observation will drive is taken) matches the
                # semantics of dist_to_zone_end/progress above, which are also
                # computed from the current (pre-action) route_position.
                values.append(encode_role(veh.role))
                values.extend(encode_zone_onehot(self.zone_of(vid)))
            if self.config.include_priority_signal:
                # Pilot v11: own draw + peer draw, both raw (not a
                # pre-computed comparison) -- see
                # Stage10MergeEnvConfig.include_priority_signal's docstring.
                # Stage 11 always has exactly 2 active vehicles, so
                # other_ids is a single-element list and peer_priority is
                # just that one other vehicle's draw; the np.mean is only
                # so this stays well-defined if this env is ever driven
                # with >2 vehicles elsewhere -- untested for this feature.
                own_priority = self._priority[vid]
                other_ids = [i for i in self._active_ids if i != vid]
                peer_priority = float(np.mean([self._priority[o] for o in other_ids])) if other_ids else 0.0
                values.append(own_priority)
                values.append(peer_priority)
            out[vid] = np.asarray(values, dtype=np.float64)
        return out

    # ------------------------------------------------------------------ step
    def step(
        self,
        action: Mapping[str, int],
        *,
        collision_penalty_magnitude: float | None = None,
        ttc_penalty_weight: float | None = None,
        collision_penalty_override: Mapping[str, float] | None = None,
        exit_reward_magnitude: float | None = None,
        hard_braking_eta: float | None = None,
        time_cost_per_step: float | None = None,
    ):
        """Pilot v7: ``collision_penalty_magnitude``/``ttc_penalty_weight``
        are optional per-call overrides of the module-level
        ``COLLISION_PENALTY_MAGNITUDE``/``TTC_PENALTY_WEIGHT`` constants,
        letting the runner drive a curriculum-ramped value (see module
        docstring "Reward-magnitude curriculum" section and
        ``stage10_role_phase_subpolicy_config.py``'s
        ``collision_penalty_at_step``/``ttc_weight_at_step``) without this
        env needing to know about global training step count itself. Both
        default to the static v6 constants, so pilot v1-v6 call sites and
        tests are unaffected.

        Stage 11 (E30) v2: ``collision_penalty_override`` is an optional
        per-vehicle dict of penalty magnitudes, keyed by vehicle id. When a
        colliding vehicle's id is present in this mapping, its value is used
        INSTEAD of ``collision_penalty_magnitude``/``COLLISION_PENALTY_MAGNITUDE``
        for that vehicle specifically -- this is how the Stage 11 "clawback"
        collision penalty (exactly cancel whatever positive reward this
        vehicle has already earned this episode, rather than an arbitrary
        fixed magnitude) is applied without this env needing to know about
        per-episode reward bookkeeping itself; the runner computes the
        per-vehicle clawback value and passes it in here each step. Vehicles
        not present in the mapping (or when the mapping is omitted entirely)
        fall back to ``collision_penalty_magnitude``, so v1-v7 and Stage 11
        v1 call sites are unaffected.

        Stage 11 v3: ``exit_reward_magnitude`` is an optional scalar override
        of the module-level ``EXIT_REWARD_MAGNITUDE`` constant, same pattern
        as ``collision_penalty_magnitude`` above -- lets a caller raise the
        completion bonus (e.g. to strengthen the incentive to actually finish
        the merge rather than crawl indefinitely without colliding) without
        touching the shared default every other pilot version relies on.
        Defaults to ``None`` (module constant), so v1-v7 and Stage 11 v1/v2
        call sites are unaffected.

        Stage 11 v4: ``hard_braking_eta``/``time_cost_per_step`` are optional
        scalar magnitudes for two NEW reward terms ported verbatim from Stage
        9's formally-locked base reward (see module docstring "Stage 9-based
        reward redesign" section) -- a per-step hard-braking cost
        (``compute_hard_braking_cost``, using this vehicle's own SI-signed
        acceleration this transition against ``A_COMFORT``/``A_HARD``) and a
        per-step active-time cost charged while the vehicle has not yet
        exited. Both default to ``None`` (term disabled / contributes 0.0),
        so v1-v7 and Stage 11 v1-v3 call sites are unaffected."""
        if not self._episode_active:
            raise RuntimeError("step() called before reset() or after episode end")
        if set(action.keys()) != set(self._active_ids):
            raise ValueError(f"action must contain exactly keys {self._active_ids}")

        # Pilot v3: record this step's actions as "last action" BEFORE building
        # the returned next-observation below, so the peer-intent feature in
        # that next observation reflects what each vehicle just did (revealed
        # intent), not a stale prior-step value.
        for vid, a in action.items():
            self._last_action[vid] = int(a)

        state_t = {vid: Stage10VehicleState(**vars(v)) for vid, v in self._vehicles.items()}
        zone_t = {vid: self.zone_of(vid) for vid in self._active_ids}

        dt = self.config.dt
        for vid, veh in self._vehicles.items():
            if self._completed[vid]:
                a = 0.0
            else:
                a = self._action_to_accel(int(action[vid]))
            v_new = float(np.clip(veh.speed + a * dt, self.config.v_min, self.config.v_max))
            v_avg = 0.5 * (veh.speed + v_new)
            veh.acceleration = a
            veh.speed = v_new
            veh.route_position = float(veh.route_position + v_avg * dt)
        self._step_count += 1

        state_t1 = {vid: Stage10VehicleState(**vars(v)) for vid, v in self._vehicles.items()}
        zone_t1 = {vid: self.zone_of(vid) for vid in self._active_ids}

        # Collisions: pairwise among all active vehicles (up to 15 pairs at
        # n=6, 6 at n=4, 1 at n=2), protocol S5.
        collided: set[str] = set()
        collision_pairs: list[tuple[str, str]] = []
        for i, a_id in enumerate(self._active_ids):
            for b_id in self._active_ids[i + 1 :]:
                xa, ya = self._world_xy(state_t1[a_id])
                xb, yb = self._world_xy(state_t1[b_id])
                if abs(xa - xb) <= self.config.collision_distance and abs(ya - yb) < self.config.collision_lateral_threshold:
                    collided.add(a_id)
                    collided.add(b_id)
                    collision_pairs.append(tuple(sorted((a_id, b_id))))
        collision_event = len(collided) > 0

        exit_event = {vid: False for vid in self._active_ids}
        if not collision_event:
            for vid in self._active_ids:
                if self._completed[vid]:
                    continue
                if state_t[vid].route_position < self.config.geometry.route_exit <= state_t1[vid].route_position:
                    exit_event[vid] = True

        # Stage 11 v4: snapshot completion status BEFORE this transition's
        # exits are applied below -- I_active(s_t) (the time-cost gate) must
        # reflect "still on-road at the START of the transition", per Stage
        # 9's own definition (module docstring "Stage 9-based reward
        # redesign" section), not the post-exit status ``self._completed``
        # is about to be mutated to.
        completed_before_step = dict(self._completed)

        for vid in self._active_ids:
            if exit_event[vid]:
                self._completed[vid] = True
                self._vehicles[vid].completed = True

        all_completed = all(self._completed.values())
        terminated = False
        truncated = False
        reason = "ongoing"
        if collision_event:
            terminated = True
            reason = "collision"
        elif all_completed:
            terminated = True
            reason = "success"
        elif self._step_count >= self.config.max_steps:
            truncated = True
            reason = "truncation"
        if terminated and truncated:
            raise ValueError("invalid simultaneous terminated and truncated")

        # Continuous collision-risk shaping (pilot v6, see module docstring
        # "Reward revision" section). Computed from the POST-step (state_t1,
        # i.e. self._vehicles' current live state at this point) neighbour
        # relationship on each side -- "did this step's action leave the
        # vehicle dangerously close", the same post-action state basis the
        # collision check above already uses.
        ttc_penalty: dict[str, float] = {}
        for vid in self._active_ids:
            ahead, behind = self._neighbours(vid)
            veh = self._vehicles[vid]
            gap_ahead_r = float(ahead["gap"]) if ahead else NO_NEIGHBOUR_GAP
            # Positive == THIS vehicle is closing on the one ahead (catching up) -- risky.
            # Same orientation as the observation's relspeed_ahead.
            closing_ahead = float(veh.speed - ahead["speed"]) if ahead else 0.0
            gap_behind_r = float(behind["gap"]) if behind else NO_NEIGHBOUR_GAP
            # Positive == the vehicle BEHIND is closing in on THIS one -- risky.
            # NOTE: this is -relspeed_behind, not relspeed_behind -- the sign
            # convention flips vs. the ahead side (see module docstring).
            closing_behind = float(behind["speed"] - veh.speed) if behind else 0.0
            ttc_penalty[vid] = _ttc_risk_penalty(
                gap_ahead_r, closing_ahead, threshold=TTC_PENALTY_THRESHOLD_SECONDS
            ) + _ttc_risk_penalty(
                gap_behind_r, closing_behind, threshold=TTC_PENALTY_THRESHOLD_SECONDS
            )

        # Base reward (protocol S1 item 3: base reward only, no PBRS this
        # pilot). Pilot v6: collision penalty is perpetrator-only (`vid in
        # collided`, not uniform to every active vehicle), plus the
        # continuous TTC risk term above. Pilot v7: the magnitude/weight used
        # are the per-call overrides when given (curriculum-ramped, see
        # module docstring "Reward-magnitude curriculum" section), else the
        # static module constants -- v1-v6 call sites that never pass the new
        # kwargs are byte-for-byte unaffected.
        cp_mag = COLLISION_PENALTY_MAGNITUDE if collision_penalty_magnitude is None else float(collision_penalty_magnitude)
        ttc_w = TTC_PENALTY_WEIGHT if ttc_penalty_weight is None else float(ttc_penalty_weight)
        exit_mag = EXIT_REWARD_MAGNITUDE if exit_reward_magnitude is None else float(exit_reward_magnitude)
        # Stage 11 v4: 0.0 (term disabled) when the caller omits the kwarg,
        # matching every other optional-override param above.
        hb_eta = 0.0 if hard_braking_eta is None else float(hard_braking_eta)
        time_cost = 0.0 if time_cost_per_step is None else float(time_cost_per_step)
        reward: dict[str, float] = {}
        collision_penalty_applied: dict[str, bool] = {}
        collision_penalty_used_per_vehicle: dict[str, float] = {}
        hard_braking_cost_used: dict[str, float] = {}
        time_cost_applied: dict[str, bool] = {}
        for vid in self._active_ids:
            g = self.config.geometry
            rho_t = float(np.clip((state_t[vid].route_position - g.route_start) / (g.route_exit - g.route_start), 0.0, 1.0))
            rho_t1 = float(np.clip((state_t1[vid].route_position - g.route_start) / (g.route_exit - g.route_start), 0.0, 1.0))
            delta = rho_t1 - rho_t
            vehicle_collided = vid in collided
            collision_penalty_applied[vid] = vehicle_collided
            if vehicle_collided and collision_penalty_override is not None and vid in collision_penalty_override:
                this_cp = float(collision_penalty_override[vid])
            else:
                this_cp = cp_mag
            collision_penalty_used_per_vehicle[vid] = this_cp
            # Stage 11 v4: hard-braking cost from THIS vehicle's own SI-signed
            # acceleration this transition (``state_t1[vid].acceleration`` was
            # set by the physics loop above from ``_action_to_accel``, already
            # SI-signed -- negative == braking). 0.0 whenever the term is
            # disabled (hb_eta == 0.0), skipping the compute_hard_braking_cost
            # call entirely rather than calling it with a zero weight.
            h_cost = (
                compute_hard_braking_cost(state_t1[vid].acceleration, A_COMFORT, A_HARD)
                if hb_eta != 0.0
                else 0.0
            )
            hard_braking_cost_used[vid] = h_cost
            # Stage 11 v4: time cost gated on completion status at the START
            # of this transition (``completed_before_step``, not the possibly
            # just-updated ``self._completed``) -- matches Stage 9's
            # I_active(s_t) definition exactly.
            is_active_at_start = not completed_before_step[vid]
            time_cost_applied[vid] = is_active_at_start
            r = (
                PROGRESS_REWARD_WEIGHT * delta
                + exit_mag * (1.0 if exit_event[vid] else 0.0)
                - this_cp * (1.0 if vehicle_collided else 0.0)
                + ttc_w * ttc_penalty[vid]
                - hb_eta * h_cost
                - time_cost * (1.0 if is_active_at_start else 0.0)
            )
            reward[vid] = float(r)

        if terminated or truncated:
            self._episode_active = False

        info = {
            "seed": self._seed_used,
            "step": self._step_count,
            "term_reason": reason,
            "roles": dict(self._roles),
            "zone_t": zone_t,
            "zone_t1": zone_t1,
            "collision_pairs": collision_pairs,
            "collision_event": collision_event,
            "exit_event": dict(exit_event),
            "completed": dict(self._completed),
            "action_masks": {vid: self.action_mask(vid).tolist() for vid in self._active_ids},
            # Pilot v6: per-vehicle reward-component breakdown, so the runner
            # (and trajectory logs) can record exactly which vehicles the
            # collision penalty applied to and what continuous risk penalty
            # each vehicle carried this step, without re-deriving either from
            # scratch -- this reward revision may itself need diagnosing the
            # same way frozen_stall was.
            "ttc_penalty": dict(ttc_penalty),
            "collision_penalty_applied": dict(collision_penalty_applied),
            # Pilot v7: the actual (possibly curriculum-ramped) magnitude/
            # weight used THIS step -- self-describing output, so trajectory
            # logs/manifests don't need the config's step->value schedule
            # re-derived to know what was in effect at any given step.
            "collision_penalty_magnitude_used": cp_mag,
            "ttc_penalty_weight_used": ttc_w,
            # Stage 11 v3: the actual exit-reward magnitude used this step --
            # same self-describing-output rationale as the two fields above.
            "exit_reward_magnitude_used": exit_mag,
            # Stage 11 v2: per-vehicle penalty actually applied THIS step --
            # differs from collision_penalty_magnitude_used when
            # collision_penalty_override supplied a per-vehicle "clawback"
            # value instead of the scalar (see step()'s docstring).
            "collision_penalty_used_per_vehicle": dict(collision_penalty_used_per_vehicle),
            # Stage 11 v4: per-vehicle hard-braking cost H_i and whether the
            # time-cost term was charged this step -- same self-describing-
            # output rationale as the fields above.
            "hard_braking_cost_used": dict(hard_braking_cost_used),
            "hard_braking_eta_used": hb_eta,
            "time_cost_applied": dict(time_cost_applied),
            "time_cost_per_step_used": time_cost,
        }
        self.last_info = info
        return self._obs_from_vehicles(), reward, bool(terminated), bool(truncated), info


__all__ = [
    "VEHICLE_IDS",
    "ROLE_VALUES",
    "ZONE_VALUES",
    "OBS_DIM",
    "OBS_DIM_WITH_ROLE_ZONE",
    "OBS_DIM_WITH_ROLE_ZONE_AND_PRIORITY",
    "N_ACTIONS",
    "NO_NEIGHBOUR_GAP",
    "PROGRESS_REWARD_WEIGHT",
    "EXIT_REWARD_MAGNITUDE",
    "COLLISION_PENALTY_MAGNITUDE",
    "TTC_PENALTY_THRESHOLD_SECONDS",
    "TTC_PENALTY_WEIGHT",
    "HARD_BRAKING_ETA",
    "TIME_COST_PER_STEP",
    "A_COMFORT",
    "A_HARD",
    "REWARD_VERSION_V1_V5",
    "REWARD_VERSION_V6",
    "REWARD_VERSION_V7",
    "REWARD_VERSION_STAGE9_BASED",
    "HighLevelAction",
    "encode_peer_intent",
    "encode_scene_vehicle_count",
    "encode_role",
    "encode_zone_onehot",
    "zone_for_position",
    "world_y_for",
    "_ttc_risk_penalty",
    "Stage10RouteGeometry",
    "Stage10MergeEnvConfig",
    "Stage10VehicleState",
    "Stage10SymmetricMergeEnv",
]
