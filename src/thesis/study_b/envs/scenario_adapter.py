"""Translates a legacy-backend-agnostic ``ScenarioSpec``
(``thesis.study_b.scenario_generator``, unmodified by this migration --
see ``output/highwayenv_migration/CODE_PROVENANCE.md``) into concrete
HighwayEnv lane placements.

Coordinate mapping (documented here because it is the single most
important, non-obvious design decision of the whole migration -- see
``output/highwayenv_migration/validation/M4_D_SUMMARY.json`` for
the empirical validation this mapping must pass):

``ThesisHighwayMergeEnv``'s road (built by ``ConnectedLaneMergeGenericEnv.
_make_road``, unmodified) lays every lane out along a shared world x-axis:
the ramp's ``j-k`` straight lane starts at world x=0, exactly like the
mainline ``a-b`` lane's start. Both reach the reference point this module
calls ``merge_start`` (== ``before_merge_length`` in HighwayEnv's own
road-length config) after driving distance ``before_merge_length`` in a
straight line. This is structurally identical to the legacy simulator's
own ``route_position``/``world_y_for`` convention (world x == route
position, world y == lane offset) for the pre-merge zone specifically --
which is exactly the zone the matched-TTC formula (``scenario_generator.
generate_scenario``) reasons about (spawn distance FROM ``merge_start``).

So: a vehicle whose legacy spec says
``route_position = merge_start_legacy - target_speed * nominal_ttc``
(i.e., "arrives at merge_start after driving nominal_ttc seconds at
target_speed, undisturbed") is placed at HighwayEnv longitudinal
coordinate:

    s_i = before_merge_length - target_speed_i * nominal_ttc_i

along its assigned lane (ramp -> ("j","k",0); mainline -> ("a","b",
lanes_count-1) -- BOTH mainline vehicles share this one lane, matching
the legacy generator's "same role == same lane" spawn-validity check,
and it is deliberately the SAME lane index the ramp lane merges into at
node "c", so cross-role conflict only becomes physically possible near
the merge zone, matching the legacy model's intent).

Only ``nominal_ttc``/``target_speed`` (both unchanged, backend-agnostic
fields already on ``VehicleSpawnSpec``) are used -- role/speed_class/
ttc_slot assignments and their counterbalancing are entirely handled
upstream by the frozen generator and are NOT touched here.

Note: ``VehicleSpawnSpec.route_position`` (the LEGACY backend's own
placement coordinate) is intentionally NOT read here -- this adapter
recomputes the real spawn coordinate from ``target_speed``/
``nominal_ttc`` directly via ``spawn_longitudinal``. For every scenario
produced by ``generate_scenario`` these agree exactly (its
``route_position`` field is defined by the identical formula, just
reparametrized around ``merge_start`` instead of ``before_merge_length``
-- see that function's own docstring), so this only matters when a
caller hand-constructs a ``ScenarioSpec`` directly (as
``tests/study_b/test_highwayenv_m4f_leakage_through_env.py`` does, on
purpose, to control real spawn position independently of
``route_position``).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from highway_env.road.road import Road

from thesis.study_b.envs.highwayenv_vehicle import ThesisControlledVehicle
from thesis.study_b.scenario_generator import ScenarioSpec

__all__ = ["RAMP_LANE_INDEX", "mainline_lane_index", "spawn_longitudinal", "place_scenario"]

RAMP_LANE_INDEX = ("j", "k", 0)


def mainline_lane_index(lanes_count: int) -> tuple[str, str, int]:
    """Both mainline vehicles share this one lane -- the rightmost
    "a","b" lane, i.e. the one the ramp lane merges into at node "c"
    (see module docstring)."""
    return ("a", "b", lanes_count - 1)


def spawn_longitudinal(*, before_merge_length: float, target_speed: float, nominal_ttc: float) -> float:
    return float(before_merge_length - target_speed * nominal_ttc)


def place_scenario(
    road: Road,
    scenario: ScenarioSpec,
    *,
    before_merge_length: float,
    lanes_count: int,
    vehicle_cls: type = ThesisControlledVehicle,
) -> dict[str, ThesisControlledVehicle]:
    """Creates one ``vehicle_cls`` instance per vehicle in ``scenario``
    (default ``ThesisControlledVehicle`` -- pass
    ``highwayenv_vehicle.MetaSpeedControlledVehicle`` for the M6-R3
    action-representation-B comparison), appends each to ``road.vehicles``,
    and returns the id->vehicle map. Does NOT touch ``road.objects`` (the
    merge-point ``Obstacle`` from ``_make_road`` stays as-is; it's a
    static object, not a thesis vehicle, and none of the frozen
    collision/reward logic reads it)."""
    out: dict[str, ThesisControlledVehicle] = {}
    for vid, spec in scenario.vehicles.items():
        lane_index = RAMP_LANE_INDEX if spec.role == "ramp" else mainline_lane_index(lanes_count)
        lane = road.network.get_lane(lane_index)
        s = spawn_longitudinal(
            before_merge_length=before_merge_length,
            target_speed=spec.target_speed,
            nominal_ttc=spec.nominal_ttc,
        )
        if s < 0:
            raise ValueError(
                f"scenario {scenario.scenario_id!r} vehicle {vid!r}: spawn longitudinal "
                f"coordinate {s:.2f} < 0 -- before_merge_length={before_merge_length} is too "
                f"short for target_speed={spec.target_speed}*nominal_ttc={spec.nominal_ttc:.3f}; "
                "increase before_merge_length (M4-D FAIL branch D2)."
            )
        position = lane.position(s, 0.0)
        heading = lane.heading_at(s)
        vehicle = vehicle_cls(
            road,
            position,
            heading=heading,
            speed=spec.spawn_speed,
            target_lane_index=lane_index,
            target_speed=spec.target_speed,
        )
        road.vehicles.append(vehicle)
        out[vid] = vehicle
    return out
