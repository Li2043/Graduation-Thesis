"""M4-G -- connected-lane neighbour gate (runbook sec 24).

Architecture note (a deliberate amendment worth recording, not a gap):
this project's local observation (``local_observation.
build_neighbour_observations``) does NOT use HighwayEnv's own
``Road.neighbour_vehicles()`` lane-topology query at all -- it selects
the K=3 nearest OTHER ACTIVE vehicles by |delta distance-to-merge|
computed directly from real world x-coordinate (continuous across every
lane segment boundary, since every lane in this road is laid out along a
shared world x-axis -- see ``scenario_adapter.py``'s module docstring).
So cross-segment-boundary neighbour detectability is guaranteed by
construction for THIS observation design, not something that depends on
``neighbour_vehicles_connected_lanes`` or could silently fail at a
segment boundary the way a per-lane ``Road.neighbour_vehicles()`` query
could without that flag. This test verifies that guarantee empirically
at an actual boundary (node "b", where the ramp's own parallel lane and
the mainline lane meet) rather than assuming it, and separately confirms
the HighwayEnv-native flag is still set correctly on the constructed
road (in case any future code path starts relying on it).
"""

from __future__ import annotations

from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv
from thesis.study_b.scenario_generator import ScenarioSpec, VehicleSpawnSpec
from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig

_C = ThesisHighwayMergeEnvConfig()
_BOUNDARY_X = _C.before_merge_length + _C.converge_merge_length  # node "b"


def _ttc(target_speed: float, desired_x: float) -> float:
    return (_C.before_merge_length - desired_x) / target_speed


def test_neighbour_vehicles_connected_lanes_flag_is_set_on_the_constructed_road():
    env = StudyBHeterogeneousHighwayEnv()
    env.reset(seed=0)
    assert env._env.road.neighbour_vehicles_connected_lanes is True  # noqa: SLF001


def test_vehicles_straddling_a_lane_segment_boundary_still_see_each_other():
    """One vehicle just before node "b" (still in the converging ramp
    curve), one just after (already on the ramp's own parallel lane) --
    a real per-lane query at this exact boundary is the scenario
    ``neighbour_vehicles_connected_lanes`` exists to fix; this design
    doesn't need that fix, but must still get the right answer."""
    just_before = _BOUNDARY_X - 5.0
    just_after = _BOUNDARY_X + 5.0
    specs = {
        "V0": VehicleSpawnSpec(
            vehicle_id="V0", role="ramp", speed_class="slow", ttc_slot="front",
            target_speed=18.0, spawn_speed=18.0, route_position=just_before,
            nominal_ttc=_ttc(18.0, just_before),
        ),
        "V1": VehicleSpawnSpec(
            vehicle_id="V1", role="ramp", speed_class="slow", ttc_slot="rear",
            target_speed=18.0, spawn_speed=18.0, route_position=just_after,
            nominal_ttc=_ttc(18.0, just_after),
        ),
        "V2": VehicleSpawnSpec(
            vehicle_id="V2", role="mainline", speed_class="fast", ttc_slot="front",
            target_speed=22.0, spawn_speed=22.0, route_position=50.0,
            nominal_ttc=_ttc(22.0, 50.0),
        ),
        "V3": VehicleSpawnSpec(
            vehicle_id="V3", role="mainline", speed_class="fast", ttc_slot="rear",
            target_speed=22.0, spawn_speed=22.0, route_position=10.0,
            nominal_ttc=_ttc(22.0, 10.0),
        ),
    }
    scenario = ScenarioSpec(scenario_id="m4g_probe", episode_seed=0, traffic_type="heterogeneous", vehicles=specs)
    env = StudyBHeterogeneousHighwayEnv()
    obs, _info = env.reset(seed=0, scenario=scenario)

    # Confirm the two straddling vehicles really are on different lane
    # segments (otherwise this isn't testing a boundary at all).
    assert env._env._vehicle_by_id["V0"].lane_index[:2] != env._env._vehicle_by_id["V1"].lane_index[:2]  # noqa: SLF001

    # V0's nearest neighbour (smallest |delta_d|) must be V1 (10m apart in
    # x, vs. V2/V3 which are >=110m away) -- i.e. presence=1 and a small
    # delta_d in V0's first neighbour slot.
    v0_obs = obs["V0"]
    presence_slot0, delta_d_slot0 = v0_obs[6], v0_obs[7]
    assert presence_slot0 == 1.0
    assert abs(delta_d_slot0) < 0.5  # normalized by r_obs=50 -> |raw delta_d| ~10m -> ~0.2
