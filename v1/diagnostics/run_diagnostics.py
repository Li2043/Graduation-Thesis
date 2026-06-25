"""Non-invasive diagnostic harness for the V1 training system.

This module ONLY observes the existing system. It imports the real environment,
policies, reward functions, and experience function and runs instrumented
roll-outs to produce diagnostic CSVs and short feasibility experiments. It does
NOT modify any reward, environment, policy, experience-function, or training-loop
behaviour. The learning code paths used here are exactly the production ones
(via ``v1.training.train`` builders); the only additions are read-only counters
and CSV logging.

Outputs are written under ``experiments/diagnostics/``:
    action_distribution.csv
    merge_diagnostics.csv
    reward_diagnostics.csv
    training_diagnostics.csv
    part3_summary.json

Run:
    python -m v1.diagnostics.run_diagnostics
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v1.env.highway_merge_env import (  # noqa: E402
    ACTION_LANE_CHANGE,
    MAIN_AGENT,
    RAMP_AGENT,
    HighwayMergeEnv,
)
from v1.rewards.egoistic_reward import EgoisticReward  # noqa: E402
from v1.training.train import (  # noqa: E402
    EVAL_SEEDS,
    RunConfig,
    build_experience_function,
    build_policy,
    build_reward_function,
    seed_everything,
)

DIAG_DIR = ROOT / "experiments" / "diagnostics"

LANE_CHANGE_MIN_X_OFFSET = 30.0  # env allows lane change when x >= conflict - 30
TRAIN_EPISODES = 25
EVAL_EPISODES = 10


# --------------------------------------------------------------------------- io
def _write_rows(filename: str, fieldnames: list[str], rows: list[dict]) -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    path = DIAG_DIR / filename
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _q_stats(policy: Any, observations: list[np.ndarray]) -> tuple:
    if not observations or not hasattr(policy, "q_network"):
        return "", ""
    arr = np.stack(observations).astype(np.float32)
    with torch.no_grad():
        q = policy.q_network(torch.as_tensor(arr))
    return float(q.mean().item()), float(q.std().item())


# ------------------------------------------------------------- instrumentation
def run_instrumented_episode(
    env: HighwayMergeEnv,
    policy: Any,
    reward_fn: Any,
    seed: int,
    epsilon: float,
    max_steps: int,
    learn: bool,
) -> dict:
    """Run one episode and collect read-only diagnostics (no logic changes)."""
    obs, env_state = env.reset(seed=seed)
    merge_zone_x = env.conflict_position - LANE_CHANGE_MIN_X_OFFSET

    action_counts = [0, 0, 0, 0]
    illegal_lane_change = 0
    successful_lane_change = 0
    step_rewards: list[float] = []
    ttc_values: list[float] = []
    waiting_values: list[float] = []
    observations: list[np.ndarray] = []
    losses: list[float] = []
    train_updates = 0

    reached_merge_zone = env_state[RAMP_AGENT]["position"] >= merge_zone_x
    entered_main_lane = env_state[RAMP_AGENT]["lane"] == 0
    merge_completed = False
    merge_step: Optional[int] = None
    collision_triggered = False
    terminal_bonus = 0
    terminal_penalty = 0
    collision_penalty_count = 0
    steps = 0

    for t in range(max_steps):
        observations.append(np.asarray(obs, dtype=np.float32))
        action = int(policy.act(obs, epsilon))
        if 0 <= action < 4:
            action_counts[action] += 1

        lane_before = env_state[RAMP_AGENT]["lane"]
        next_obs, terminated, truncated, info, next_env_state = env.step(action)
        done = bool(terminated or truncated)

        base_reward = reward_fn.compute(obs, next_obs, next_env_state, env_state)
        ego_state = next_env_state[RAMP_AGENT]
        # Mirror the production reward exactly: per-step + shared terminal merge
        # adjustment + shared terminal collision penalty.
        merge_adj = reward_fn.terminal_adjustment(
            ego_state, terminated, truncated, merged=info.get("merged")
        )
        collision_adj = reward_fn.terminal_collision_adjustment(ego_state)
        reward = float(base_reward) + float(merge_adj) + float(collision_adj)
        step_rewards.append(reward)
        if merge_adj > 0:
            terminal_bonus = 1
        elif merge_adj < 0:
            terminal_penalty = 1
        if collision_adj < 0:
            collision_penalty_count = 1

        lane_after = next_env_state[RAMP_AGENT]["lane"]
        if action == ACTION_LANE_CHANGE:
            if lane_before == 1 and lane_after == 0:
                successful_lane_change += 1
            else:
                illegal_lane_change += 1

        if learn:
            policy.remember(obs, action, reward, next_obs, done)
            loss = policy.train_step()
            if loss is not None:
                losses.append(float(loss))
                train_updates += 1

        ramp_state = next_env_state[RAMP_AGENT]
        if ramp_state["position"] >= merge_zone_x:
            reached_merge_zone = True
        if ramp_state["lane"] == 0:
            entered_main_lane = True
        if info.get("merged") and not merge_completed:
            merge_completed = True
            merge_step = t + 1
        if info.get("collision_flag"):
            collision_triggered = True

        ttc_r = ramp_state.get("ttc")
        if ttc_r is not None and not math.isinf(float(ttc_r)):
            ttc_values.append(float(ttc_r))
        waiting_values.append(float(ramp_state.get("waiting_time", 0.0)))

        steps += 1
        obs, env_state = next_obs, next_env_state
        if done:
            break

    safe_merge = bool(merge_completed and not collision_triggered)
    unsafe_merge = bool(merge_completed and collision_triggered)
    collision_without_merge = bool(collision_triggered and not merge_completed)
    if safe_merge:
        termination_reason = "safe_merge"
    elif unsafe_merge:
        termination_reason = "unsafe_merge"
    elif collision_without_merge:
        termination_reason = "collision_without_merge"
    else:
        termination_reason = "max_steps_unmerged"

    final = env_state[RAMP_AGENT]
    q_mean, q_std = _q_stats(policy, observations)

    return {
        "action_counts": action_counts,
        "illegal_lane_change": illegal_lane_change,
        "successful_lane_change": successful_lane_change,
        "reached_merge_zone": int(reached_merge_zone),
        "entered_main_lane": int(entered_main_lane),
        "merge_completed": int(merge_completed),
        "merge_failed": int(not merge_completed and not collision_triggered),
        "safe_merge": int(safe_merge),
        "unsafe_merge": int(unsafe_merge),
        "collision_without_merge": int(collision_without_merge),
        "time_to_merge": merge_step if merge_step is not None else "",
        "termination_reason": termination_reason,
        "collision_triggered": int(collision_triggered),
        "terminal_bonus": terminal_bonus,
        "terminal_penalty": terminal_penalty,
        "collision_penalty_count": collision_penalty_count,
        "final_position": float(final["position"]),
        "final_lane": int(final["lane"]),
        "final_velocity": float(final["velocity"]),
        "steps": steps,
        "episode_objective_reward": float(sum(step_rewards)),
        "mean_step_reward": float(np.mean(step_rewards)) if step_rewards else 0.0,
        "min_step_reward": float(np.min(step_rewards)) if step_rewards else 0.0,
        "max_step_reward": float(np.max(step_rewards)) if step_rewards else 0.0,
        "mean_ttc": float(np.mean(ttc_values)) if ttc_values else "",
        "min_ttc": float(np.min(ttc_values)) if ttc_values else "",
        "mean_waiting_time": float(np.mean(waiting_values)) if waiting_values else 0.0,
        "final_waiting_time": float(waiting_values[-1]) if waiting_values else 0.0,
        "replay_buffer_size": len(policy.buffer) if hasattr(policy, "buffer") else "",
        "train_updates": train_updates,
        "mean_loss": float(np.mean(losses)) if losses else "",
        "q_value_mean": q_mean,
        "q_value_std": q_std,
    }


# --------------------------------------------------------------------- part 2
def _log_episode(run_id, mode, seed, phase, episode, epsilon, diag) -> None:
    a = diag["action_counts"]
    _write_rows(
        "action_distribution.csv",
        [
            "run_id", "mode", "seed", "phase", "episode",
            "action_0_count", "action_1_count", "action_2_count", "action_3_count",
            "illegal_lane_change_attempts", "successful_lane_changes",
        ],
        [{
            "run_id": run_id, "mode": mode, "seed": seed, "phase": phase,
            "episode": episode,
            "action_0_count": a[0], "action_1_count": a[1],
            "action_2_count": a[2], "action_3_count": a[3],
            "illegal_lane_change_attempts": diag["illegal_lane_change"],
            "successful_lane_changes": diag["successful_lane_change"],
        }],
    )
    _write_rows(
        "merge_diagnostics.csv",
        [
            "run_id", "mode", "seed", "phase", "episode",
            "reached_merge_zone", "entered_main_lane", "merge_completed",
            "time_to_merge", "final_position", "final_lane", "final_velocity",
            "termination_reason",
        ],
        [{
            "run_id": run_id, "mode": mode, "seed": seed, "phase": phase,
            "episode": episode,
            "reached_merge_zone": diag["reached_merge_zone"],
            "entered_main_lane": diag["entered_main_lane"],
            "merge_completed": diag["merge_completed"],
            "time_to_merge": diag["time_to_merge"],
            "final_position": round(diag["final_position"], 3),
            "final_lane": diag["final_lane"],
            "final_velocity": round(diag["final_velocity"], 3),
            "termination_reason": diag["termination_reason"],
        }],
    )
    _write_rows(
        "reward_diagnostics.csv",
        [
            "run_id", "mode", "seed", "phase", "episode",
            "episode_objective_reward", "mean_step_reward", "min_step_reward",
            "max_step_reward", "collision_penalty_triggered", "mean_ttc",
            "min_ttc", "mean_waiting_time", "final_waiting_time",
        ],
        [{
            "run_id": run_id, "mode": mode, "seed": seed, "phase": phase,
            "episode": episode,
            "episode_objective_reward": round(diag["episode_objective_reward"], 4),
            "mean_step_reward": round(diag["mean_step_reward"], 4),
            "min_step_reward": round(diag["min_step_reward"], 4),
            "max_step_reward": round(diag["max_step_reward"], 4),
            "collision_penalty_triggered": diag["collision_triggered"],
            "mean_ttc": diag["mean_ttc"],
            "min_ttc": diag["min_ttc"],
            "mean_waiting_time": round(diag["mean_waiting_time"], 4),
            "final_waiting_time": round(diag["final_waiting_time"], 4),
        }],
    )
    if phase == "train":
        _write_rows(
            "training_diagnostics.csv",
            [
                "run_id", "mode", "seed", "episode", "replay_buffer_size",
                "train_updates", "mean_loss", "epsilon",
                "q_value_mean_if_available", "q_value_std_if_available",
            ],
            [{
                "run_id": run_id, "mode": mode, "seed": seed, "episode": episode,
                "replay_buffer_size": diag["replay_buffer_size"],
                "train_updates": diag["train_updates"],
                "mean_loss": diag["mean_loss"],
                "epsilon": round(epsilon, 4),
                "q_value_mean_if_available": diag["q_value_mean"],
                "q_value_std_if_available": diag["q_value_std"],
            }],
        )


def instrumented_run(mode: str, seed: int = 0) -> Any:
    """Run an instrumented train+eval for one mode; returns the trained policy."""
    config = RunConfig(mode=mode, seed=seed, episodes=TRAIN_EPISODES, max_steps=60)
    seed_everything(seed)
    env = HighwayMergeEnv(max_steps=config.max_steps)
    exp_fn = build_experience_function(config)
    policy = build_policy(config, action_dim=env.action_dim, obs_dim=env.obs_dim)
    reward_fn = build_reward_function(config, env.ego_agent, exp_fn)
    run_id = f"diag_{mode}_seed{seed}"

    for episode in range(config.episodes):
        frac = min(1.0, episode / config.epsilon_decay_episodes)
        epsilon = config.epsilon_start + frac * (config.epsilon_end - config.epsilon_start)
        diag = run_instrumented_episode(
            env, policy, reward_fn, seed=seed * 100000 + episode,
            epsilon=epsilon, max_steps=config.max_steps, learn=True,
        )
        _log_episode(run_id, mode, seed, "train", episode, epsilon, diag)

    for i, eval_seed in enumerate(EVAL_SEEDS[:EVAL_EPISODES]):
        diag = run_instrumented_episode(
            env, policy, reward_fn, seed=eval_seed,
            epsilon=0.0, max_steps=config.max_steps, learn=False,
        )
        _log_episode(run_id, mode, seed, "eval", i, 0.0, diag)

    return policy, env, reward_fn, config


# --------------------------------------------------------------------- part 3
def _aggregate(diags: list[dict]) -> dict:
    n = len(diags)
    a3 = sum(d["action_counts"][3] for d in diags)
    total_actions = sum(sum(d["action_counts"]) for d in diags)
    merges = [d for d in diags if d["merge_completed"]]
    safe_merges = [d for d in merges if not d["collision_triggered"]]
    return {
        "episodes": n,
        # Primary success metric is the SAFE merge rate (merge without collision).
        "safe_merge_success_rate": round(sum(d.get("safe_merge", 0) for d in diags) / n, 3),
        "unsafe_merge_rate": round(sum(d.get("unsafe_merge", 0) for d in diags) / n, 3),
        "collision_without_merge_rate": round(
            sum(d.get("collision_without_merge", 0) for d in diags) / n, 3
        ),
        "merge_success_rate": round(sum(d["merge_completed"] for d in diags) / n, 3),
        "non_merge_failure_rate": round(sum(d.get("merge_failed", 0) for d in diags) / n, 3),
        "collision_rate": round(sum(d["collision_triggered"] for d in diags) / n, 3),
        "reached_merge_zone_rate": round(sum(d["reached_merge_zone"] for d in diags) / n, 3),
        "entered_main_lane_rate": round(sum(d["entered_main_lane"] for d in diags) / n, 3),
        "action_3_frequency": round(a3 / total_actions, 4) if total_actions else 0.0,
        "successful_lane_changes_total": sum(d["successful_lane_change"] for d in diags),
        "terminal_merge_bonus_count": sum(d.get("terminal_bonus", 0) for d in diags),
        "terminal_non_merge_penalty_count": sum(d.get("terminal_penalty", 0) for d in diags),
        "terminal_collision_penalty_count": sum(
            d.get("collision_penalty_count", 0) for d in diags
        ),
        "avg_time_to_merge_when_safe_success": (
            round(float(np.mean([d["time_to_merge"] for d in safe_merges])), 2)
            if safe_merges
            else None
        ),
    }


def part_a_random(seed: int = 123, episodes: int = 50) -> dict:
    env = HighwayMergeEnv(max_steps=60)
    rng = random.Random(seed)

    class RandomPolicy:
        def act(self, state, epsilon):
            return rng.randrange(env.action_dim)

    exp_fn = build_experience_function(RunConfig(mode="egoistic"))
    reward_fn = build_reward_function(RunConfig(mode="egoistic"), env.ego_agent, exp_fn)
    diags = [
        run_instrumented_episode(env, RandomPolicy(), reward_fn, seed=1000 + i,
                                 epsilon=1.0, max_steps=60, learn=False)
        for i in range(episodes)
    ]
    return _aggregate(diags)


def part_b_forced_lane_change(episodes: int = 20) -> dict:
    env = HighwayMergeEnv(max_steps=60)

    class ScriptedPolicy:
        """Accelerate until inside the merge zone, then attempt lane change."""

        def act(self, state, epsilon):
            lane = state[2]
            dist_conflict_norm = state[3]  # (conflict - x)/goal
            in_zone = dist_conflict_norm <= (LANE_CHANGE_MIN_X_OFFSET / env.goal_position)
            if lane == 1 and in_zone:
                return ACTION_LANE_CHANGE
            return 1  # accelerate

    exp_fn = build_experience_function(RunConfig(mode="egoistic"))
    reward_fn = build_reward_function(RunConfig(mode="egoistic"), env.ego_agent, exp_fn)
    diags = [
        run_instrumented_episode(env, ScriptedPolicy(), reward_fn, seed=2000 + i,
                                 epsilon=0.0, max_steps=60, learn=False)
        for i in range(episodes)
    ]
    agg = _aggregate(diags)
    agg["blocked_by_main_examples"] = sum(
        1 for d in diags if d["collision_triggered"] and d["entered_main_lane"]
    )
    return agg


def part_c_trained_inspection(policy, env, reward_fn, episodes: int = 10) -> dict:
    diags = [
        run_instrumented_episode(env, policy, reward_fn, seed=EVAL_SEEDS[i % len(EVAL_SEEDS)],
                                 epsilon=0.0, max_steps=60, learn=False)
        for i in range(episodes)
    ]
    totals = [0, 0, 0, 0]
    for d in diags:
        for k in range(4):
            totals[k] += d["action_counts"][k]
    agg = _aggregate(diags)
    agg["greedy_action_counts"] = totals
    agg["action_3_ever_selected"] = any(d["action_counts"][3] > 0 for d in diags)
    return agg


def part_d_reward_components(policy, env, episodes: int = 10) -> dict:
    reward_fn = EgoisticReward(agent_id=env.ego_agent)
    progress_sum = risk_sum = waiting_sum = 0.0
    collision_count = 0
    step_count = 0
    for i in range(episodes):
        obs, env_state = env.reset(seed=EVAL_SEEDS[i % len(EVAL_SEEDS)])
        for _ in range(60):
            action = int(policy.act(obs, 0.0))
            next_obs, terminated, truncated, info, next_env_state = env.step(action)
            current = next_env_state[env.ego_agent]
            progress_sum += reward_fn._progress_reward(current, env_state)
            risk_sum += reward_fn._risk_penalty(current)
            waiting_sum += reward_fn._waiting_penalty(current)
            if reward_fn._collision_penalty(current) > 0:
                collision_count += 1
            step_count += 1
            obs, env_state = next_obs, next_env_state
            if terminated or truncated:
                break
    denom = step_count if step_count else 1
    return {
        "components_accessible": True,
        "avg_progress_reward": round(progress_sum / denom, 5),
        "avg_risk_penalty": round(risk_sum / denom, 5),
        "avg_waiting_penalty": round(waiting_sum / denom, 5),
        "collision_penalty_steps": collision_count,
        "steps_inspected": step_count,
    }


def main() -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    print("== Part 2: instrumented train+eval (egoistic) ==")
    ego_policy, ego_env, ego_reward, _ = instrumented_run("egoistic", seed=0)
    print("== Part 2: instrumented train+eval (rawlsian) ==")
    instrumented_run("rawlsian", seed=0)

    print("== Part 3A: random-action feasibility ==")
    part_a = part_a_random()
    print(json.dumps(part_a, indent=2))

    print("== Part 3B: forced lane-change feasibility ==")
    part_b = part_b_forced_lane_change()
    print(json.dumps(part_b, indent=2))

    print("== Part 3C: trained egoistic policy inspection ==")
    part_c = part_c_trained_inspection(ego_policy, ego_env, ego_reward)
    print(json.dumps(part_c, indent=2))

    print("== Part 3D: egoistic reward component inspection ==")
    part_d = part_d_reward_components(ego_policy, ego_env)
    print(json.dumps(part_d, indent=2))

    # Calibration parameters in effect for these diagnostics (RunConfig defaults).
    _diag_cfg = RunConfig(mode="rawlsian")
    calibration = {
        "rawlsian_objective_scale": _diag_cfg.rawlsian_objective_scale,
        "terminal_collision_penalty": _diag_cfg.terminal_collision_penalty,
        "merge_success_bonus": _diag_cfg.merge_success_bonus,
        "non_merge_failure_penalty": _diag_cfg.non_merge_failure_penalty,
    }
    print("== Calibration parameters used ==")
    print(json.dumps(calibration, indent=2))

    summary = {"calibration": calibration,
               "part_a_random": part_a, "part_b_scripted": part_b,
               "part_c_trained": part_c, "part_d_reward_components": part_d}
    with (DIAG_DIR / "part3_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nWrote diagnostics to {DIAG_DIR}")


if __name__ == "__main__":
    main()
