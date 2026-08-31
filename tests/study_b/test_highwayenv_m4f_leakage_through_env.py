"""M4-F -- local observation leakage gate (runbook sec 23), verified
THROUGH the real HighwayEnv pipeline (reset -> live vehicle state ->
wrapper._snapshot -> build_local_observation), not just at the abstract
``local_observation.py`` function level (already covered, unmodified, by
``tests/study_b/test_local_observation_leakage.py``).

Constructs two scenarios directly (bypassing ``generate_scenario``, whose
own convention ties ``spawn_speed`` to ``target_speed`` and therefore
can't hold "every externally observable neighbour state fixed" while
varying only the hidden target -- exactly what this gate needs to
isolate) that are IDENTICAL in every physically observable respect
(REAL spawn world-x, spawn_speed, role) for every vehicle, differing
ONLY in one neighbour's hidden ``target_speed`` field (18 vs 22). Ego's
observation, recomputed through the live environment, must be identical.

Important adapter detail this test had to account for (found while
writing it): ``scenario_adapter.place_scenario`` computes each vehicle's
REAL spawn coordinate from ``(target_speed, nominal_ttc)`` via
``spawn_longitudinal`` -- it does NOT read ``VehicleSpawnSpec.
route_position`` at all (that field is vestigial for this backend; it is
only meaningful for the legacy backend's own placement). So holding
"the same route_position" is not what holds the REAL spawn x-coordinate
fixed here; ``nominal_ttc`` must be chosen per target_speed so that
``before_merge_length - target_speed*nominal_ttc`` comes out identical
across the two scenarios.
"""

from __future__ import annotations

import numpy as np

from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig
from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv
from thesis.study_b.scenario_generator import ScenarioSpec, VehicleSpawnSpec

_BEFORE_MERGE_LENGTH = ThesisHighwayMergeEnvConfig().before_merge_length


def _ttc_for_fixed_spawn_x(*, target_speed: float, desired_spawn_x: float) -> float:
    return (_BEFORE_MERGE_LENGTH - desired_spawn_x) / target_speed


def _make_scenario(*, neighbour_target_speed: float) -> ScenarioSpec:
    # Real spawn world-x held fixed per vehicle across both calls (see
    # module docstring) -- only V1's hidden target_speed varies.
    v0_x, v1_x, v2_x, v3_x = 140.0, 100.0, 121.0, 77.0
    specs = {
        "V0": VehicleSpawnSpec(  # ego
            vehicle_id="V0", role="ramp", speed_class="slow", ttc_slot="front",
            target_speed=18.0, spawn_speed=18.0, route_position=v0_x,
            nominal_ttc=_ttc_for_fixed_spawn_x(target_speed=18.0, desired_spawn_x=v0_x),
        ),
        "V1": VehicleSpawnSpec(  # neighbour under test
            vehicle_id="V1", role="ramp", speed_class="slow", ttc_slot="rear",
            target_speed=neighbour_target_speed,  # <-- the only thing that varies
            spawn_speed=18.0,  # held fixed -- externally observable state unchanged
            route_position=v1_x,
            nominal_ttc=_ttc_for_fixed_spawn_x(target_speed=neighbour_target_speed, desired_spawn_x=v1_x),
        ),
        "V2": VehicleSpawnSpec(
            vehicle_id="V2", role="mainline", speed_class="fast", ttc_slot="front",
            target_speed=22.0, spawn_speed=22.0, route_position=v2_x,
            nominal_ttc=_ttc_for_fixed_spawn_x(target_speed=22.0, desired_spawn_x=v2_x),
        ),
        "V3": VehicleSpawnSpec(
            vehicle_id="V3", role="mainline", speed_class="fast", ttc_slot="rear",
            target_speed=22.0, spawn_speed=22.0, route_position=v3_x,
            nominal_ttc=_ttc_for_fixed_spawn_x(target_speed=22.0, desired_spawn_x=v3_x),
        ),
    }
    return ScenarioSpec(scenario_id="m4f_probe", episode_seed=0, traffic_type="heterogeneous", vehicles=specs)


def test_m4f_ego_observation_unaffected_by_neighbour_hidden_target_speed():
    env_a = StudyBHeterogeneousHighwayEnv()
    obs_a, _info_a = env_a.reset(seed=0, scenario=_make_scenario(neighbour_target_speed=18.0))

    env_b = StudyBHeterogeneousHighwayEnv()
    obs_b, _info_b = env_b.reset(seed=0, scenario=_make_scenario(neighbour_target_speed=22.0))

    # Sanity: the two scenarios really do differ (otherwise this test
    # would pass vacuously) -- V1's target_speed itself differs.
    assert env_a._scenario.vehicles["V1"].target_speed != env_b._scenario.vehicles["V1"].target_speed  # noqa: SLF001

    # Only OTHER agents' observations of V1 are in scope here -- V1's own
    # self-observation legitimately includes its own target_speed (an
    # agent knows its own private target; that is not leakage).
    for vid in ("V0", "V2", "V3"):
        np.testing.assert_allclose(
            obs_a[vid], obs_b[vid],
            err_msg=f"vehicle {vid}'s observation leaked V1's hidden target_speed",
        )

    # Also verify after one physics step (target_speed could theoretically
    # leak only once speeds start diverging under real dynamics).
    actions = {vid: 0 for vid in env_a.active_vehicle_ids}  # HOLD
    obs_a2, *_ = env_a.step(actions)
    obs_b2, *_ = env_b.step(actions)
    for vid in ("V0", "V2", "V3"):
        np.testing.assert_allclose(obs_a2[vid], obs_b2[vid])
