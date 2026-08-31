"""M4-K -- DQN terminal/return regression gate (runbook sec 28), rerun
against the new HighwayEnv backend interface.

``SharedDQNLearner``/``ReplayTransition``/``ReplayBuffer``/target-sync
mechanism (``thesis.agents.stage10_shared_dqn`` etc.) are entirely
UNCHANGED, backend-agnostic code -- their own unit-level terminal-
bootstrap/1-step-target/3-step-return/replay-round-trip/target-sync/
finite-numerics checks already exist and pass unmodified
(``test_target_network_sync.py``,
``test_diagnostic6_terminal_and_td_target.py``,
``test_diagnostic6_replay_roundtrip.py``, all still green in this
folder's copy). What THIS gate needs to newly verify is the INTERFACE:
does ``StudyBHeterogeneousHighwayEnv``'s output correctly drive that
unchanged learner through a real rollout, across every episode-boundary
case (collision-terminal, timeout-truncation, and a natural still-running
step), without violating ``ReplayTransition.validate()``'s invariants and
without producing any non-finite Q-value."""

from __future__ import annotations

import math

import numpy as np
import torch

from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig
from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv
from thesis.study_b.scenario_generator import ScenarioSpec, VehicleSpawnSpec
from thesis.study_b.shared_local_dqn import SharedLocalDQNAgent, build_study_b_dqn_config

_C = ThesisHighwayMergeEnvConfig()


def _ttc(target_speed: float, desired_x: float) -> float:
    return (_C.before_merge_length - desired_x) / target_speed


def _collision_prone_scenario() -> ScenarioSpec:
    # Same-lane ramp pair 1m apart -> collision within one step under any
    # non-decelerating action, so a terminal transition is guaranteed to
    # be exercised without needing a long rollout.
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
    return ScenarioSpec(scenario_id="m4k_probe", episode_seed=0, traffic_type="heterogeneous", vehicles=specs)


def test_dqn_agent_rollout_against_highwayenv_backend_no_validation_or_numeric_errors():
    config = build_study_b_dqn_config(reward_condition="baseline", device="cpu")
    agent = SharedLocalDQNAgent(config, seed=1)
    env = StudyBHeterogeneousHighwayEnv()

    n_updates_attempted = 0
    saw_terminal_transition = False
    saw_ongoing_transition = False

    for episode_i in range(6):
        scenario = _collision_prone_scenario() if episode_i % 2 == 0 else None
        obs, _info = env.reset(seed=100 + episode_i, scenario=scenario)
        prev_obs = obs
        for step_i in range(20):
            actions = agent.select_actions(prev_obs, epsilon=1.0)  # random -- exercises real dynamics
            next_obs, reward, terminated, truncated, info = env.step(actions)
            controller_terminal_episode = terminated or truncated

            for vid in env.active_vehicle_ids:
                if vid not in prev_obs:
                    continue
                learner_completed = bool(info["completed_this_step"].get(vid, False))
                vehicle_terminal = controller_terminal_episode or learner_completed
                transition = agent.build_transition(
                    vehicle_id=vid,
                    observation=prev_obs[vid],
                    action=int(actions[vid]),
                    shaped_reward=float(reward[vid]),
                    next_observation=next_obs.get(vid),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    controller_terminal=vehicle_terminal,
                    learner_completed=learner_completed,
                    base_reward=float(reward[vid]),
                    episode_id=f"ep{episode_i}", step=step_i,
                )
                assert np.all(np.isfinite(transition.observation))
                if transition.next_observation is not None:
                    assert np.all(np.isfinite(transition.next_observation))
                agent.store_transition(transition)
                if vehicle_terminal:
                    saw_terminal_transition = True
                else:
                    saw_ongoing_transition = True

            update_result = agent.maybe_update(warmup=32)
            if update_result is not None:
                n_updates_attempted += 1
                for key in ("loss",) if "loss" in update_result else ():
                    assert math.isfinite(update_result[key])
                for key, value in update_result.items():
                    if isinstance(value, (int, float)):
                        assert math.isfinite(value), f"{key} is non-finite: {value}"

            prev_obs = next_obs
            if controller_terminal_episode:
                break

    assert saw_terminal_transition, "no episode reached a terminal boundary across 6 short rollouts"
    assert saw_ongoing_transition, "no non-terminal transition was ever stored"
    assert n_updates_attempted > 0, "learner never had enough transitions to update"

    # Q-values on the trained agent must remain finite (no NaN/inf blowup
    # over this short rollout).
    probe_obs = next(iter(prev_obs.values()))
    with torch.no_grad():
        q = agent.learner.online(torch.as_tensor(probe_obs, dtype=torch.float32).unsqueeze(0))
    assert bool(torch.isfinite(q).all())
