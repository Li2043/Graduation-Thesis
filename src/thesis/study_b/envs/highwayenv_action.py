"""Project-local longitudinal-only 3-bin discrete action type for the
HighwayEnv backend migration (Claude_Code_Autonomous_Experiment_Runbook_
HighwayEnv.md sec 11).

``highway_env.envs.common.action.DiscreteAction`` was tried first (runbook
sec 6.2 item 7) and rejected per the M2 fork-decision writeup
(``output/highwayenv_migration/M2_FORK_DECISION.md`` item 4): its 3-point
linspace over the normalized ``[-1, 1]`` interval, mapped through an
asymmetric ``acceleration_range`` (the legacy simulator's accel=+2.0/
decel=-3.0 m/s^2), gives a MIDDLE bin of -0.5 m/s^2, not ~0 -- it would
fail the M4-C action-semantics gate outright. This module implements the
sec 11.4 fallback (a minimal project-local action type) instead of forking
HighwayEnv, per the M2 decision.

Action index convention matches ``thesis.envs.stage10_symmetric_merge_env.
HighLevelAction`` NUMERICALLY (frozen, verified by the M4-C gate):
0=HOLD, 1=ACCELERATE, 2=BRAKE. **Terminology note (2026-08-16, user
instruction)**: this backend's formal action labels are ACCELERATE /
HOLD / BRAKE -- "DECELERATE" is not used as a label anywhere in this
backend's code, logs, or documentation, to avoid ambiguity with the
legacy module's own ``HighLevelAction.MAINTAIN``/``DECELERATE`` names
(which are NOT renamed -- they belong to shared, frozen, pre-existing
code outside this backend and are out of scope for this rename). The
underlying integer values (0/1/2) and physical meaning are unchanged;
only the labels used in THIS backend's code/docs/logs changed."""

from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np
from gymnasium import spaces
from highway_env.envs.common.action import ActionType
from highway_env.vehicle.kinematics import Vehicle

__all__ = [
    "HOLD",
    "ACCELERATE",
    "BRAKE",
    "ThesisLongitudinalAction",
    "ThesisMetaSpeedAction",
    "ThesisMultiAgentAction",
]

HOLD = 0
ACCELERATE = 1
BRAKE = 2


class ThesisLongitudinalAction(ActionType):
    """One controlled vehicle's action type: exactly 3 discrete bins, no
    lateral control at all (steering is left to the vehicle's own
    automatic lane-following -- see ``highwayenv_vehicle.
    ThesisControlledVehicle``, never to this action type)."""

    def __init__(
        self,
        env,
        *,
        accelerate_mps2: float = 2.0,
        brake_mps2: float = 3.0,
        hold_mps2: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(env)
        # {action_index: signed acceleration in m/s^2} -- fixed exact
        # values, no interpolation, no linspace. Matches
        # PRE_MIGRATION_CONFIG.json's legacy accel_rate/decel_rate exactly
        # by default.
        self._accel_by_action = {
            HOLD: float(hold_mps2),
            ACCELERATE: float(accelerate_mps2),
            BRAKE: -float(brake_mps2),
        }

    def space(self) -> spaces.Discrete:
        return spaces.Discrete(3)

    @property
    def vehicle_class(self):
        # Overridden by ThesisMultiAgentAction.vehicle_class in practice --
        # this env never lets HighwayEnv auto-construct vehicles from an
        # action type, but the property must still resolve to something
        # sane for API compatibility.
        return Vehicle

    def act(self, action: int) -> None:
        # Persistent state, not a transient argument -- see
        # highwayenv_vehicle.ThesisControlledVehicle's module docstring for
        # why this matters (Road.act() re-invokes vehicle.act() with no
        # arguments on every physics substep of this policy step).
        accel = self._accel_by_action[int(action)]
        self.controlled_vehicle.commanded_acceleration = accel
        self.controlled_vehicle.act()


class ThesisMetaSpeedAction(ActionType):
    """M6-R3 (runbook sec 36) comparison representation B: longitudinal-only
    desired-speed commands (BRAKE/HOLD/ACCELERATE, nudging ``target_speed``
    by a fixed delta) instead of representation A's direct fixed-
    acceleration bins. Dispatches to the CONTROLLED VEHICLE'S OWN
    unmodified ``highway_env.vehicle.controller.ControlledVehicle.act()``
    (via ``highwayenv_vehicle.MetaSpeedControlledVehicle``, which changes
    NOTHING about ``act()`` -- only disables HighwayEnv's native collision
    physics and adds the post-completion freeze, same as representation
    A's vehicle class) -- this is deliberately HighwayEnv's own built-in
    cruise-control mechanism (``speed_control``/``steering_control``), not
    a hand-rolled reimplementation, since the whole point of this
    comparison is testing whether a smoother, PID-like longitudinal
    controller is more robust to DQN exploration noise than representation
    A's raw bang-bang acceleration commands (see M6 recovery notes:
    ``output/highwayenv_migration/M6_R3_ACTION_REPRESENTATION_COMPARISON.md``).

    Discrete action label -> HighwayEnv desired-speed command:
    HOLD -> "IDLE" (target_speed unchanged), ACCELERATE -> "FASTER"
    (target_speed += DELTA_SPEED), BRAKE -> "SLOWER" (target_speed -=
    DELTA_SPEED). HighwayEnv's longitudinal controller then converts the
    resulting target_speed into a REQUESTED physical acceleration via
    ``speed_control()`` -- the discrete action label is never itself a
    physical acceleration value under this representation. As of the
    2026-08-16 CONTROL_AUTHORITY_MISMATCH amendment, that requested
    acceleration is clipped to the frozen legacy longitudinal envelope
    [-3.0, +2.0] m/s^2 (see ``highwayenv_vehicle.
    MetaSpeedControlledVehicle``) before it reaches vehicle physics --
    an unaudited unbounded controller was found to realize up to
    +-18 m/s^2, far outside the study's intended control authority."""

    def __init__(self, env, **kwargs) -> None:
        super().__init__(env)

    def space(self) -> spaces.Discrete:
        return spaces.Discrete(3)

    @property
    def vehicle_class(self):
        from thesis.study_b.envs.highwayenv_vehicle import MetaSpeedControlledVehicle

        return MetaSpeedControlledVehicle

    def act(self, action: int) -> None:
        label = {HOLD: "IDLE", ACCELERATE: "FASTER", BRAKE: "SLOWER"}[int(action)]
        self.controlled_vehicle.act(label)


class ThesisMultiAgentAction(ActionType):
    """Composes one sub-action-type instance (``sub_action_cls``, default
    ``ThesisLongitudinalAction``) per entry in ``env.controlled_vehicles``,
    in that list's order. Deliberately not routed through
    ``highway_env.envs.common.action.action_factory`` (that function's
    hardcoded string dispatch lives in third-party code and is not being
    modified/patched -- this class is instantiated directly by
    ``ThesisHighwayMergeEnv.define_spaces()``, a public override point,
    not a fork)."""

    def __init__(self, env, *, sub_action_cls: type[ActionType] = ThesisLongitudinalAction, **kwargs) -> None:
        super().__init__(env)
        self.agents_action_types: list[ActionType] = []
        for vehicle in self.env.controlled_vehicles:
            action_type = sub_action_cls(env, **kwargs)
            action_type.controlled_vehicle = vehicle
            self.agents_action_types.append(action_type)

    def space(self) -> spaces.Tuple:
        return spaces.Tuple([a.space() for a in self.agents_action_types])

    @property
    def vehicle_class(self):
        return Vehicle

    def act(self, action: Sequence[int]) -> None:
        assert len(action) == len(self.agents_action_types), (
            f"expected {len(self.agents_action_types)} actions (one per "
            f"controlled vehicle), got {len(action)}"
        )
        for agent_action, action_type in zip(action, self.agents_action_types):
            action_type.act(agent_action)

    def get_available_actions(self):
        return list(
            itertools.product(*[range(3) for _ in self.agents_action_types])
        )
