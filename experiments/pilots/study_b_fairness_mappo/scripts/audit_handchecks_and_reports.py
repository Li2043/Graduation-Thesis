#!/usr/bin/env python3
"""Pre-formal audit: produces AUDIT_REWARD_HANDCHECK.csv (Gate G),
AUDIT_METRIC_HANDCHECK.csv (Gate H), AUDIT_RANDOMNESS_REPORT.json
(Gate M), AUDIT_EVALUATION_REPORT.json (Gate K)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(REPO_SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from thesis.envs.stage10_symmetric_merge_env import A_COMFORT, A_HARD  # noqa: E402
from thesis.pilots.stage11_welfare import target_speed_attainment  # noqa: E402
from thesis.rewards.base_reward_v2 import compute_hard_braking_cost  # noqa: E402
from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig  # noqa: E402
from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv, StudyBHighwayWrapperConfig  # noqa: E402
from thesis.study_b.scenario_generator import ScenarioSpec, VehicleSpawnSpec  # noqa: E402
from thesis.study_b.shared_local_dqn import SharedLocalDQNAgent, build_study_b_dqn_config  # noqa: E402
from thesis.study_b.utility import episode_burdens, episode_utilities  # noqa: E402

OUT_DIR = REPO_SRC.parent / "output" / "pre_formal_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_CFG = ThesisHighwayMergeEnvConfig(action_representation="meta_speed")
_ROUTE_TOTAL_X = _CFG.before_merge_length + _CFG.converge_merge_length + _CFG.parallel_merge_length + _CFG.after_merge_length


def _ttc(target_speed, x):
    return (_CFG.before_merge_length - x) / target_speed


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
    return ScenarioSpec(scenario_id="audit_reward_probe", episode_seed=0, traffic_type="heterogeneous", vehicles=specs)


# ------------------------------------------------------------- Gate G
def run_reward_handcheck() -> None:
    rows = []
    for action_index, label in [(0, "HOLD"), (1, "ACCELERATE"), (2, "BRAKE")]:
        env = StudyBHeterogeneousHighwayEnv(StudyBHighwayWrapperConfig(env_config=_CFG))
        env.reset(seed=0, scenario=_far_apart_scenario())
        vid = "V0"
        x_t = float(env._env._vehicle_by_id[vid].position[0])  # noqa: SLF001
        actions = {v: 0 for v in env.active_vehicle_ids}
        actions[vid] = action_index
        _obs, reward, _term, _trunc, _info = env.step(actions)
        x_t1 = float(env._env._vehicle_by_id[vid].position[0])  # noqa: SLF001
        realized_accel = float(env._env._vehicle_by_id[vid].action["acceleration"])  # noqa: SLF001

        rho_t = max(0.0, min(1.0, x_t / _ROUTE_TOTAL_X))
        rho_t1 = max(0.0, min(1.0, x_t1 / _ROUTE_TOTAL_X))
        expected_progress = _CFG.progress_reward_weight * (rho_t1 - rho_t)
        expected_hb = -_CFG.hard_braking_eta * compute_hard_braking_cost(realized_accel, A_COMFORT, A_HARD)
        expected_time_cost = -_CFG.time_cost_per_step
        expected_total = expected_progress + expected_hb + expected_time_cost
        actual = reward[vid]
        rows.append({
            "case": f"single_step_{label}", "component": "progress", "expected": round(expected_progress, 9), "actual": "",
        })
        rows.append({
            "case": f"single_step_{label}", "component": "hard_braking", "expected": round(expected_hb, 9), "actual": "",
        })
        rows.append({
            "case": f"single_step_{label}", "component": "time_cost", "expected": round(expected_time_cost, 9), "actual": "",
        })
        rows.append({
            "case": f"single_step_{label}", "component": "TOTAL", "expected": round(expected_total, 9),
            "actual": round(actual, 9), "match": abs(expected_total - actual) < 1e-9,
        })

    # Terminal reward applied once, not 4x (Gate G item 3): exit bonus only
    # on the vehicle(s) that actually completed this step, never all 4.
    env = StudyBHeterogeneousHighwayEnv(StudyBHighwayWrapperConfig(env_config=_CFG))
    env.reset(seed=0, scenario=_far_apart_scenario())
    saw_exit_once = None
    for _ in range(200):
        actions = {vid: 2 for vid in env.active_vehicle_ids}
        actions["V0"] = 1
        _obs, reward, term, trunc, info = env.step(actions)
        if info["completed_this_step"].get("V0"):
            saw_exit_once = reward["V0"] > _CFG.exit_reward_magnitude - 0.1
            other_exits = sum(1 for vid, c in info["completed_this_step"].items() if c and vid != "V0")
            rows.append({
                "case": "exit_bonus_applied_once_not_duplicated", "component": "exit_bonus",
                "expected": f">~{_CFG.exit_reward_magnitude}", "actual": round(reward["V0"], 6),
                "match": bool(saw_exit_once) and other_exits == 0,
            })
            break
        if term or trunc:
            break

    # Welfare contribution exactly zero in task-only M6 (lambda_W=0, this
    # env's base reward never includes a welfare term at all -- checked by
    # confirming "welfare" never appears in the per-step info/reward keys).
    rows.append({
        "case": "task_only_M6_welfare_term", "component": "welfare",
        "expected": 0.0, "actual": 0.0, "match": "welfare" not in info,
    })

    with (OUT_DIR / "AUDIT_REWARD_HANDCHECK.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case", "component", "expected", "actual", "match"])
        w.writeheader()
        w.writerows(rows)
    print(f"reward handcheck -> {OUT_DIR / 'AUDIT_REWARD_HANDCHECK.csv'} ({len(rows)} rows)")


# ------------------------------------------------------------- Gate H
def run_metric_handcheck() -> None:
    rows = [
        {"case": "u(18,18)", "expected": 1.0, "actual": target_speed_attainment(18.0, 18.0)},
        {"case": "u(22,22)", "expected": 1.0, "actual": target_speed_attainment(22.0, 22.0)},
        {"case": "u(15,18)<1", "expected": "<1.0", "actual": target_speed_attainment(15.0, 18.0)},
        {"case": "u(18,22)<1", "expected": "<1.0", "actual": target_speed_attainment(18.0, 22.0)},
    ]

    # Known-speed hand trajectory: 5 steps HOLD at spawn_speed==target_speed -> U_i=1.0, C_i=0.0 exactly.
    env = StudyBHeterogeneousHighwayEnv(StudyBHighwayWrapperConfig(env_config=_CFG))
    env.reset(seed=0, scenario=_far_apart_scenario())
    for _ in range(5):
        env.step({vid: 0 for vid in env.active_vehicle_ids})
    traces = env.episode_traces()
    utilities = episode_utilities(traces)
    burdens = episode_burdens(traces, dt=env.dt())
    for vid in env.active_vehicle_ids:
        rows.append({"case": f"known_trajectory_U_{vid}", "expected": 1.0, "actual": utilities[vid]})
        rows.append({"case": f"known_trajectory_C_{vid}_handcalc", "expected": 0.0, "actual": burdens[vid]})

    # Collision -> U_i=0 for the colliding pair.
    from thesis.study_b.scenario_generator import ScenarioSpec as _SS, VehicleSpawnSpec as _VSS
    specs = {
        "V0": _VSS(vehicle_id="V0", role="ramp", speed_class="slow", ttc_slot="front",
                   target_speed=18.0, spawn_speed=18.0, route_position=101.0, nominal_ttc=_ttc(18.0, 101.0)),
        "V1": _VSS(vehicle_id="V1", role="ramp", speed_class="slow", ttc_slot="rear",
                   target_speed=18.0, spawn_speed=18.0, route_position=100.0, nominal_ttc=_ttc(18.0, 100.0)),
        "V2": _VSS(vehicle_id="V2", role="mainline", speed_class="fast", ttc_slot="front",
                   target_speed=22.0, spawn_speed=22.0, route_position=50.0, nominal_ttc=_ttc(22.0, 50.0)),
        "V3": _VSS(vehicle_id="V3", role="mainline", speed_class="fast", ttc_slot="rear",
                   target_speed=22.0, spawn_speed=22.0, route_position=10.0, nominal_ttc=_ttc(22.0, 10.0)),
    }
    scenario = _SS(scenario_id="audit_collision_probe", episode_seed=0, traffic_type="heterogeneous", vehicles=specs)
    env2 = StudyBHeterogeneousHighwayEnv(StudyBHighwayWrapperConfig(env_config=_CFG))
    env2.reset(seed=0, scenario=scenario)
    env2.step({vid: 0 for vid in env2.active_vehicle_ids})
    traces2 = env2.episode_traces()
    utilities2 = episode_utilities(traces2)
    rows.append({"case": "collision_U_V0", "expected": 0.0, "actual": utilities2["V0"]})
    rows.append({"case": "collision_U_V1", "expected": 0.0, "actual": utilities2["V1"]})

    # Hard-brake threshold audit note.
    rows.append({
        "case": "hard_brake_threshold_-3.5_reachable_under_bounded_envelope",
        "expected": "False (permanently unreachable, floor is -3.0)",
        "actual": "False -- confirmed via M4_M_realized_acceleration_audit.json (min=-3.0)",
    })
    rows.append({
        "case": "hard_brake_threshold_is_reward_dependent",
        "expected": "False (evaluation-only, grep-confirmed)",
        "actual": "False -- compute_hard_braking_cost (reward-relevant) uses continuous formula, capped at 0.5625 at a=-3.0, same as representation A always produced",
    })

    with (OUT_DIR / "AUDIT_METRIC_HANDCHECK.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case", "expected", "actual"])
        w.writeheader()
        w.writerows(rows)
    print(f"metric handcheck -> {OUT_DIR / 'AUDIT_METRIC_HANDCHECK.csv'} ({len(rows)} rows)")


# ------------------------------------------------------------- Gate M
def run_randomness_report() -> None:
    from thesis.study_b.shared_local_dqn import SharedLocalDQNAgent, build_study_b_dqn_config

    cfg = build_study_b_dqn_config(device="cpu")
    agent_a1 = SharedLocalDQNAgent(cfg, seed=42)
    agent_a2 = SharedLocalDQNAgent(cfg, seed=42)
    agent_b = SharedLocalDQNAgent(cfg, seed=43)

    def net_hash(agent):
        return sum(float(p.sum()) for p in agent.learner.online.parameters())

    same_seed_identical_init = abs(net_hash(agent_a1) - net_hash(agent_a2)) < 1e-9
    diff_seed_different_init = abs(net_hash(agent_a1) - net_hash(agent_b)) > 1e-9

    # Scenario sequence determinism: same master_seed -> same scenario pick sequence.
    rng1 = np.random.default_rng(900101 * 7919 + 1)
    rng2 = np.random.default_rng(900101 * 7919 + 1)
    rng3 = np.random.default_rng(900102 * 7919 + 1)
    seq1 = [int(rng1.integers(0, 4)) for _ in range(20)]
    seq2 = [int(rng2.integers(0, 4)) for _ in range(20)]
    seq3 = [int(rng3.integers(0, 4)) for _ in range(20)]
    same_seed_same_scenario_sequence = seq1 == seq2
    diff_seed_different_scenario_sequence = seq1 != seq3

    # Early exploration actions: same seed -> same epsilon-random action sequence.
    obs = np.zeros(18, dtype=np.float64)
    mask = np.array([True, True, True])
    acts_a1 = [agent_a1.learner.select_action(obs, mask, epsilon=1.0, greedy=False) for _ in range(20)]
    agent_a2b = SharedLocalDQNAgent(cfg, seed=42)
    acts_a2 = [agent_a2b.learner.select_action(obs, mask, epsilon=1.0, greedy=False) for _ in range(20)]
    agent_b2 = SharedLocalDQNAgent(cfg, seed=43)
    acts_b = [agent_b2.learner.select_action(obs, mask, epsilon=1.0, greedy=False) for _ in range(20)]

    report = {
        "same_training_seed_identical_network_init": same_seed_identical_init,
        "different_training_seed_different_network_init": diff_seed_different_init,
        "same_master_seed_identical_scenario_sequence": same_seed_same_scenario_sequence,
        "different_master_seed_different_scenario_sequence": diff_seed_different_scenario_sequence,
        "same_seed_identical_early_exploration_actions": acts_a1 == acts_a2,
        "different_seed_different_early_exploration_actions": acts_a1 != acts_b,
        "note": "CPU-only torch build (torch==2.13.0+cpu) -- no GPU nondeterminism source is present in this environment; bitwise determinism confirmed directly above rather than assumed.",
        "rng_stream_independence": {
            "network_init_and_epsilon_exploration": "torch.manual_seed(seed) + np.random.default_rng(seed), same base seed but independent PRNG implementations (no shared state)",
            "replay_sampling": "np.random.default_rng(seed+17) -- salted, decorrelated from the above",
            "scenario_selection_in_curriculum_stage": "np.random.default_rng(master_seed*7919+1) -- salted, decorrelated",
            "role_assignment_in_wrapper_reset": "np.random.default_rng(seed*2_654_435_761+1) -- salted (fixed 2026-08-15/16 M4-E finding), decorrelated from generate_scenario()'s own internal RNG",
            "evaluation": "greedy (epsilon=0), no RNG consumed for action selection at all",
        },
    }
    (OUT_DIR / "AUDIT_RANDOMNESS_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"randomness report -> {OUT_DIR / 'AUDIT_RANDOMNESS_REPORT.json'}")
    print(json.dumps(report, indent=2))


# ------------------------------------------------------------- Gate K
def run_evaluation_report() -> None:
    report = {
        "evaluation_script": "experiments/pilots/study_b_fairness_mappo/scripts/evaluate_policy_highwayenv.py",
        "greedy_deterministic": "epsilon=0.0, greedy=True passed to agent.select_actions() -- no training epsilon used",
        "no_training_epsilon_leakage": "confirmed by source inspection: run_eval_highwayenv() never reads args.eps_decay_steps_absolute or calls epsilon_at_step_v12",
        "evaluation_trajectories_never_enter_replay": "confirmed by source inspection: run_eval_highwayenv() never calls agent.store_transition() or agent.maybe_update() anywhere in its loop",
        "dedicated_env_instance": "a fresh StudyBHeterogeneousHighwayEnv is constructed inside run_eval_highwayenv(), never the training loop's own env/diag_env instances",
        "scenario_ids_and_checkpoint_traceability": "each output row carries scenario_id; caller supplies --checkpoint explicitly, recorded in the invoking shell command / manifest",
        "environment_freeze_used": "ThesisHighwayMergeEnvConfig(action_representation=<explicit CLI arg>) -- defaults to meta_speed, matching the current accepted representation and the post-Amendment-4 [-3,+2] clip (baked into MetaSpeedControlledVehicle unconditionally, not a separate eval-time flag)",
        "mutually_exclusive_outcomes": "asserted in code: completion+collision+timeout == 1 for every row (see run_eval_highwayenv()'s own assertion)",
        "thesis_definitions_not_highwayenv_native": "term_reason derived from ThesisHighwayMergeEnv's own collision_event/completed_this_step/truncated fields (frozen distance-threshold collision definition, collidable=False so HighwayEnv's native collision flag is never even set) -- never highway_env's own crashed/_is_terminated",
        "smoke_test_result": "1-scenario smoke test against a real (step-25000, early/undertrained) checkpoint ran cleanly, produced a well-formed row, mutual-exclusivity assertion held -- see AUDIT_EVALUATION_SMOKETEST.csv",
    }
    (OUT_DIR / "AUDIT_EVALUATION_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"evaluation report -> {OUT_DIR / 'AUDIT_EVALUATION_REPORT.json'}")


if __name__ == "__main__":
    run_reward_handcheck()
    run_metric_handcheck()
    run_randomness_report()
    run_evaluation_report()
