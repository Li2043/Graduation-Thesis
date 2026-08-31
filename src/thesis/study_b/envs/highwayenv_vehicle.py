"""Project-local controlled-vehicle class for the HighwayEnv backend
migration.

``highway_env.vehicle.controller.ControlledVehicle.act()`` is a
target-SPEED cruise controller (``"FASTER"``/``"SLOWER"`` nudge
``target_speed`` by a fixed delta; a raw ``{"acceleration": a}`` dict is
silently ignored -- ``act()`` always recomputes acceleration itself via
``speed_control(self.target_speed)``). That is the wrong control paradigm
for this thesis: the DQN action must set acceleration DIRECTLY (three
fixed bins, see ``highwayenv_action.py``), not a target speed.

What ``ControlledVehicle`` gets right, and what this subclass keeps
unchanged, is ``follow_road()`` (automatically advances
``target_lane_index`` to the next lane in the network once the current
lane ends -- this is what makes a ramp vehicle automatically continue
from the ``j-k`` lane onto the curving ``k-b`` SineLane and then merge
into the rightmost mainline lane at node ``c``, entirely through
``RoadNetwork.next_lane()``'s built-in closest-lane heuristic, confirmed
by reading ``highway_env/road/road.py`` directly during M2) and
``steering_control()`` (a lane-centering controller that tracks whatever
``target_lane_index`` currently is). Together these give exactly the
thesis invariant from runbook sec 3.5/14: the action space stays
longitudinal-only, and lateral position is a deterministic function of
the vehicle's assigned route through the road network, never an agent
decision -- the direct HighwayEnv-backend analogue of the legacy
simulator's ``world_y_for()`` automatic lateral convergence.

Critical wiring detail (found by reading ``AbstractEnv._simulate()``
directly, not assumed): the acceleration command must be held in
PERSISTENT vehicle state (``commanded_acceleration``), not passed through
the transient ``action`` argument of ``act()``. ``_simulate()`` calls
``action_type.act(action)`` only once per policy step (at
``frame == 0``), but ``Road.act()`` then calls ``vehicle.act()`` with NO
arguments on every vehicle for EVERY physics substep of that same policy
step (3 substeps at this project's 15 Hz simulation / 5 Hz policy
frequency). If acceleration were read only from the transient ``action``
parameter, the commanded acceleration would apply for exactly 1 of every
3 substeps and silently reset to 0 for the other 2 -- this is exactly the
pattern the built-in ``ControlledVehicle.act()`` avoids by always
recomputing from persistent ``self.target_speed`` state regardless of
what (if anything) was passed in; this subclass does the same with a
persistent raw acceleration value instead of a target speed. See
``tests/study_b/test_highwayenv_action_persists_across_substeps.py``.
"""

from __future__ import annotations

import numpy as np
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

__all__ = ["ThesisControlledVehicle"]


class ThesisControlledVehicle(ControlledVehicle):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.commanded_acceleration: float = 0.0
        # Set True by ThesisHighwayMergeEnv._reward() the step a vehicle
        # safely completes. Found via audit (2026-08-16): without this,
        # a completed vehicle kept receiving and applying the DQN's
        # actions for the rest of the episode (including BRAKE),
        # diverging from the legacy simulator's explicit
        # "if self._completed[vid]: a = 0.0" freeze -- a completed
        # vehicle could be braked to a stop just past the exit line and
        # sit there as a same-lane obstacle for a still-active trailing
        # vehicle to "collide" with, which would wrongly terminate the
        # episode for everyone. See
        # tests/study_b/test_highwayenv_completed_vehicle_freeze.py.
        self.frozen: bool = False
        # HighwayEnv's own bounding-box collision physics (Vehicle.LENGTH=5.0/
        # WIDTH=2.0, applied inside Road.step() via handle_collisions() on
        # every substep) is a DIFFERENT collision model than the frozen Study
        # B definition (4.0m longitudinal / 1.5m lateral thresholds, a pure
        # flag with no dynamics effect -- runbook sec 16/25). Left enabled,
        # it silently pushes overlapping vehicles apart (an "impact" that
        # perturbs position/heading) BEFORE ThesisHighwayMergeEnv._reward()'s
        # own post-step distance check ever runs, which can mask a real
        # Study-B-definition collision entirely (found via the M4-H gate:
        # two vehicles spawned 1.0m apart in an obvious-overlap test case
        # were pushed to 2.0m apart by HighwayEnv's own collision response
        # before the frozen-definition check saw them). Disabling
        # ``collidable`` makes this vehicle invisible to HighwayEnv's native
        # collision machinery entirely, leaving the frozen-definition check
        # as the SOLE source of truth for collision, matching the legacy
        # simulator (which never had any position-perturbing collision
        # response of its own).
        self.collidable = False

    def act(self, action: dict | str | None = None) -> None:
        self.follow_road()
        steering = self.steering_control(self.target_lane_index)
        steering = float(
            np.clip(steering, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        )
        accel = 0.0 if self.frozen else self.commanded_acceleration
        Vehicle.act(self, {"steering": steering, "acceleration": accel})


class MetaSpeedControlledVehicle(ControlledVehicle):
    """M6-R3 (runbook sec 36) action-representation comparison B:
    desired-speed cruise control (HOLD/ACCELERATE/BRAKE dispatched as
    HighwayEnv's own ``"IDLE"``/``"FASTER"``/``"SLOWER"`` strings, nudging
    ``target_speed`` by ``ControlledVehicle.DELTA_SPEED``), with the
    REALIZED acceleration HighwayEnv's own unmodified
    ``speed_control()`` requests clipped to the frozen legacy
    longitudinal envelope ``[-3.0, +2.0]`` m/s^2 before it reaches
    vehicle physics (2026-08-16 amendment: CONTROL_AUTHORITY_MISMATCH --
    see ``output/highwayenv_migration/
    CONTROL_AUTHORITY_AMENDMENT_2026-08-16.md``. An acceleration audit
    found the UNBOUNDED controller could realize up to +-18 m/s^2, far
    outside the study's intended longitudinal authority).

    Why this can't just call ``super().act()``: ``ControlledVehicle.
    act()``'s own internal ``super().act(action)`` call resolves via
    Python's MRO relative to WHERE that code is defined
    (``ControlledVehicle``), so it always jumps straight to
    ``Vehicle.act()`` -- any override placed here in
    ``MetaSpeedControlledVehicle`` (earlier in the MRO) is skipped
    entirely, giving no way to intercept the computed acceleration in
    between. This method therefore reproduces
    ``ControlledVehicle.act()``'s dispatch logic explicitly (FASTER/
    SLOWER/target_speed update, ``follow_road()``, ``steering_control()``)
    so it can clip the acceleration ``speed_control()`` -- HighwayEnv's
    own, UNMODIFIED, still-called-as-is proportional controller -- returns,
    immediately before that value is stored into physics via
    ``Vehicle.act()``. This is clipping the controller's OUTPUT, not
    reimplementing the controller itself.

    Also adds the same ``collidable = False`` fix M4-H found necessary
    for representation A (HighwayEnv's own bounding-box collision physics
    otherwise masks the frozen Study-B collision definition), plus the
    same post-completion freeze as ``ThesisControlledVehicle``: HOLD
    semantics do NOT mean forcing acceleration to zero -- if current
    speed differs from (a possibly still-converging, or post-freeze
    permanently fixed) target_speed, ``speed_control()`` may still
    request nonzero acceleration; that realized value is allowed, but is
    always subject to the same ``[-3.0, +2.0]`` clip."""

    MIN_REALIZED_ACCEL_MPS2 = -3.0
    MAX_REALIZED_ACCEL_MPS2 = 2.0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.collidable = False
        self.frozen: bool = False
        self.last_requested_acceleration: float = 0.0  # pre-clip, for the M4-M audit/tests
        self.last_realized_acceleration: float = 0.0  # post-clip, what physics actually used

    def act(self, action: dict | str | None = None) -> None:
        if self.frozen:
            action = "IDLE"  # holds target_speed unchanged -- no further ACCELERATE/BRAKE nudges
        self.follow_road()
        if action == "FASTER":
            self.target_speed += self.DELTA_SPEED
        elif action == "SLOWER":
            self.target_speed -= self.DELTA_SPEED
        # LANE_LEFT/LANE_RIGHT intentionally not reproduced -- this action
        # space has no lateral action (runbook sec 3.5/11); "IDLE" and any
        # other value fall through unchanged, matching ControlledVehicle.act().

        steering = self.steering_control(self.target_lane_index)  # already clips to MAX_STEERING_ANGLE internally
        requested_accel = self.speed_control(self.target_speed)  # HighwayEnv's own, unmodified proportional controller
        realized_accel = float(
            np.clip(requested_accel, self.MIN_REALIZED_ACCEL_MPS2, self.MAX_REALIZED_ACCEL_MPS2)
        )
        self.last_requested_acceleration = float(requested_accel)
        self.last_realized_acceleration = realized_accel

        Vehicle.act(self, {"steering": float(steering), "acceleration": realized_accel})
