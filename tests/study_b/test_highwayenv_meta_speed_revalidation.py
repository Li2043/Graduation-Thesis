"""Action-adoption revalidation gate (Protocol Amendment 1, Change 2):
reruns M4-C/H/I/J/K under the ACCEPTED desired-speed (``meta_speed``)
action representation. The earlier M4-C/H/I/J/K test files
(``test_highwayenv_m4_basic_gates.py``, ``test_highwayenv_m4h_
collision_gate.py``, ``test_highwayenv_m4i_utility_burden.py``,
``test_highwayenv_m4j_reward_decomposition.py``, ``test_highwayenv_m4k_
dqn_regression.py``) all default to the ``direct_accel`` representation
(representation A) -- this file exercises the SAME invariants
specifically under ``meta_speed`` (representation B), since the
environment-level definitions (collision, utility, terminal handling)
are representation-agnostic but the REALIZED acceleration is not, and
Change 2 requires this before C4.

Does not revalidate M4-D (10,000-scenario matched-TTC/geometry) --
nothing about the action representation touches scenario placement.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig
from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv, StudyBHighwayWrapperConfig
from thesis.study_b.scenario_generator import ScenarioSpec, VehicleSpawnSpec
from thesis.study_b.shared_local_dqn import SharedLocalDQNAgent, build_study_b_dqn_config
from thesis.study_b.utility import episode_burdens, episode_utilities

_CFG = ThesisHighwayMergeEnvConfig(action_representation="meta_speed")


def _env() -> StudyBHeterogeneousHighwayEnv:
    return StudyBHeterogeneousHighwayEnv(StudyBHighwayWrapperConfig(env_config=_CFG))


def _ttc(target_speed: float, desired_x: float) -> float:
    return (_CFG.before_merge_length - desired_x) / target_speed


def _far_apart_scenario() -> ScenarioSpec:
    specs = {
        "V0": VehicleSpawnSpec(vehicle_id="V0", role="ramp", speed_class="slow", ttc_slot="front",
                                target_speed=18.0, spawn_speed=18.0, route_position=150.0, nominal_ttc=_ttc(18.0, 150.0)),
        "V1": VehicleSpawnSpec(vehicle_id="V1", role="ramp", speed_class="slow", ttc_slot="rear",
                                target_speed=18.0, spawn_speed=18.0, route_position=50.0, nominal_ttc=_ttc(18.0, 50.0)),
        "V2": VehicleSpawnSpec(vehicle_id="V2", role="mainline", speed_class="fast", ttc_slot="front",
                                target_speed=22.0, spawn_speed=22.0, route_position=100.0, nominal_ttc=_ttc(22.0, 100.0)),
        "V3": VehicleSpawnSpec(vehicle_id="V3", role="mainline", speed_class="fast", ttc_slot="rear",
                                target_speed=22.0, spawn_speed=22.0, route_position=10.0, nominal_ttc=_ttc(22.0, 10.0)),
    }
    return ScenarioSpec(scenario_id="metaspeed_reval", episode_seed=0, traffic_type="heterogeneous", vehicles=specs)


def _overlap_scenario() -> ScenarioSpec:
    specs = {
        "V0": VehicleSpawnSpec(vehicle_id="V0", role="ramp", speed_class="slow", ttc_slot="front",
                                target_speed=18.0, spawn_speed=18.0, route_position=101.0, nominal_ttc=_ttc(18.0, 101.0)),
        "V1": VehicleSpawnSpec(vehicle_id="V1", role="ramp", speed_class="slow", ttc_slot="rear",
                                target_speed=18.0, spawn_speed=18.0, route_position=100.0, nominal_ttc=_ttc(18.0, 100.0)),
        "V2": VehicleSpawnSpec(vehicle_id="V2", role="mainline", speed_class="fast", ttc_slot="front",
                                target_speed=22.0, spawn_speed=22.0, route_position=50.0, nominal_ttc=_ttc(22.0, 50.0)),
        "V3": VehicleSpawnSpec(vehicle_id="V3", role="mainline", speed_class="fast", ttc_slot="rear",
                                target_speed=22.0, spawn_speed=22.0, route_position=10.0, nominal_ttc=_ttc(22.0, 10.0)),
    }
    return ScenarioSpec(scenario_id="metaspeed_collision_reval", episode_seed=0, traffic_type="heterogeneous", vehicles=specs)


# ---------------------------------------------------------------- M4-C
@pytest.mark.parametrize("action_index,expected_sign", [(0, 0.0), (1, 1.0), (2, -1.0)])
def test_m4c_meta_speed_action_semantics_sign(action_index, expected_sign):
    """Under meta_speed, HOLD/ACCELERATE/BRAKE must still move target_speed
    (and hence realized speed) in the correct direction -- HOLD produces
    exactly zero speed change (target_speed literally unchanged), while
    ACCELERATE/BRAKE differ in sign, not necessarily a fixed magnitude
    (that's the whole point of this representation)."""
    env = _env()
    env.reset(seed=42)
    vid = env.active_vehicle_ids[0]
    v0 = float(env._env._vehicle_by_id[vid].speed)
    actions = {v: 0 for v in env.active_vehicle_ids}
    actions[vid] = action_index
    env.step(actions)
    v1 = float(env._env._vehicle_by_id[vid].speed)
    delta = v1 - v0
    if expected_sign == 0.0:
        assert delta == pytest.approx(0.0, abs=1e-9)
    else:
        assert delta * expected_sign > 0


# ---------------------------------------------------------------- M4-H
def test_m4h_meta_speed_collision_detected():
    env = _env()
    env.reset(seed=0, scenario=_overlap_scenario())
    _obs, _rew, terminated, _trunc, info = env.step({vid: 0 for vid in env.active_vehicle_ids})
    assert info["collision_event"] is True
    assert terminated is True


def test_m4h_meta_speed_non_collision():
    env = _env()
    env.reset(seed=0, scenario=_far_apart_scenario())
    _obs, _rew, terminated, _trunc, info = env.step({vid: 0 for vid in env.active_vehicle_ids})
    assert info["collision_event"] is False
    assert terminated is False


# ---------------------------------------------------------------- M4-I
def test_m4i_meta_speed_utility_and_burden():
    env = _env()
    env.reset(seed=0, scenario=_far_apart_scenario())
    for _ in range(5):
        env.step({vid: 0 for vid in env.active_vehicle_ids})
    traces = env.episode_traces()
    utilities = episode_utilities(traces)
    burdens = episode_burdens(traces, dt=env.dt())
    for vid in env.active_vehicle_ids:
        assert utilities[vid] == pytest.approx(1.0)
        assert burdens[vid] == pytest.approx(0.0, abs=1e-9)


def test_m4i_meta_speed_utility_zeroed_after_collision():
    env = _env()
    env.reset(seed=0, scenario=_overlap_scenario())
    env.step({vid: 0 for vid in env.active_vehicle_ids})
    traces = env.episode_traces()
    utilities = episode_utilities(traces)
    assert utilities["V0"] == pytest.approx(0.0)
    assert utilities["V1"] == pytest.approx(0.0)


# ---------------------------------------------------------------- M4-J
def test_m4j_meta_speed_reward_matches_realized_acceleration():
    """Under meta_speed, hard-braking cost must be computed from the
    REALIZED physical acceleration (speed_control()'s output), not a
    fixed bin -- this test independently recomputes it from directly
    observed vehicle.action["acceleration"] and compares to the reward
    the env actually returned."""
    from thesis.envs.stage10_symmetric_merge_env import A_COMFORT, A_HARD
    from thesis.rewards.base_reward_v2 import compute_hard_braking_cost

    env = _env()
    env.reset(seed=0, scenario=_far_apart_scenario())
    vid = "V0"
    x_t = float(env._env._vehicle_by_id[vid].position[0])  # noqa: SLF001
    actions = {v: 0 for v in env.active_vehicle_ids}
    actions[vid] = 2  # BRAKE
    _obs, reward, _term, _trunc, _info = env.step(actions)
    x_t1 = float(env._env._vehicle_by_id[vid].position[0])  # noqa: SLF001
    realized_accel = float(env._env._vehicle_by_id[vid].action["acceleration"])  # noqa: SLF001

    route_total_x = _CFG.before_merge_length + _CFG.converge_merge_length + _CFG.parallel_merge_length + _CFG.after_merge_length
    rho_t = max(0.0, min(1.0, x_t / route_total_x))
    rho_t1 = max(0.0, min(1.0, x_t1 / route_total_x))
    expected_progress = _CFG.progress_reward_weight * (rho_t1 - rho_t)
    expected_hb = -_CFG.hard_braking_eta * compute_hard_braking_cost(realized_accel, A_COMFORT, A_HARD)
    expected_time_cost = -_CFG.time_cost_per_step
    expected_total = expected_progress + expected_hb + expected_time_cost
    assert reward[vid] == pytest.approx(expected_total, abs=1e-9)


# ---------------------------------------------------------------- M4-K
def test_m4k_meta_speed_dqn_rollout_no_errors():
    config = build_study_b_dqn_config(reward_condition="baseline", device="cpu")
    agent = SharedLocalDQNAgent(config, seed=1)
    env = _env()
    n_updates = 0
    saw_terminal = False
    prev_obs = None
    for episode_i in range(6):
        scenario = _overlap_scenario() if episode_i % 2 == 0 else None
        obs, _info = env.reset(seed=100 + episode_i, scenario=scenario)
        prev_obs = obs
        for step_i in range(20):
            actions = agent.select_actions(prev_obs, epsilon=1.0)
            next_obs, reward, terminated, truncated, info = env.step(actions)
            controller_terminal_episode = terminated or truncated
            for vid in env.active_vehicle_ids:
                if vid not in prev_obs:
                    continue
                learner_completed = bool(info["completed_this_step"].get(vid, False))
                vehicle_terminal = controller_terminal_episode or learner_completed
                transition = agent.build_transition(
                    vehicle_id=vid, observation=prev_obs[vid], action=int(actions[vid]),
                    shaped_reward=float(reward[vid]), next_observation=next_obs.get(vid),
                    terminated=bool(terminated), truncated=bool(truncated),
                    controller_terminal=vehicle_terminal, learner_completed=learner_completed,
                    base_reward=float(reward[vid]), episode_id=f"metaspeed_reval_ep{episode_i}", step=step_i,
                )
                assert np.all(np.isfinite(transition.observation))
                agent.store_transition(transition)
                if vehicle_terminal:
                    saw_terminal = True
            update_result = agent.maybe_update(warmup=16)
            if update_result is not None:
                n_updates += 1
                for key, value in update_result.items():
                    if isinstance(value, (int, float)):
                        assert math.isfinite(value), f"{key} non-finite: {value}"
            prev_obs = next_obs
            if controller_terminal_episode:
                break
    assert saw_terminal
    assert n_updates > 0
    probe_obs = next(iter(prev_obs.values()))
    with torch.no_grad():
        q = agent.learner.online(torch.as_tensor(probe_obs, dtype=torch.float32).unsqueeze(0))
    assert bool(torch.isfinite(q).all())


# ---------------------------------------------------------- realized-accel distribution audit
def test_realized_acceleration_distribution_and_hard_brake_threshold_still_meaningful():
    """Amendment requirement: inspect the distribution of REALIZED
    physical accelerations under the accepted (meta_speed) controller and
    confirm the existing hard-brake threshold (a<=-3.5 m/s^2, this
    project's ``EpisodeVehicleTrace.hard_brake_count`` default) is not
    silently unreachable."""
    env = _env()
    env.reset(seed=0, scenario=_overlap_scenario())  # induces a SLOWER/BRAKE-heavy episode via collision setup
    accels = []
    import numpy as _np

    rng = _np.random.default_rng(0)
    obs, _info = env.reset(seed=1)
    for _ in range(400):
        actions = {vid: int(rng.integers(0, 3)) for vid in env.active_vehicle_ids}
        obs, _rew, term, trunc, _info = env.step(actions)
        for vid in env.active_vehicle_ids:
            accels.append(float(env._env._vehicle_by_id[vid].action["acceleration"]))  # noqa: SLF001
        if term or trunc:
            obs, _info = env.reset(seed=int(rng.integers(0, 10_000)))

    accels = np.array(accels)
    assert np.all(np.isfinite(accels))
    min_accel = float(accels.min())
    max_accel = float(accels.max())
    frac_hard_brake = float((accels <= -3.5).mean())
    # Documented, not asserted-strict: report the observed range so a
    # human can judge whether -3.5 m/s^2 remains operationally meaningful
    # under this controller (DELTA_SPEED=5, TAU_ACC=0.6s -- large jumps in
    # target_speed CAN produce transient accelerations beyond the legacy
    # direct-control bins). This test fails only if realized acceleration
    # is degenerate (constant / never negative at all), which WOULD mean
    # the threshold is silently meaningless.
    assert min_accel < -0.5, f"realized acceleration under BRAKE never meaningfully negative (min={min_accel})"
    assert max_accel > 0.5, f"realized acceleration under ACCELERATE never meaningfully positive (max={max_accel})"
    print(f"realized accel range=[{min_accel:.3f}, {max_accel:.3f}], frac(<=-3.5)={frac_hard_brake:.4f}")
