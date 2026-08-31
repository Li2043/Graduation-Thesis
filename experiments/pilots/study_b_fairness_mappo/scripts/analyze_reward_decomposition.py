#!/usr/bin/env python3
"""VDN_Conditional_Amendment_Protocol.md sec 7 (Diagnostic 2): full
per-step reward decomposition (progress / exit / collision / hard-brake /
time-cost / terminal welfare) for three representative episodes -- one
success, one timeout, one collision -- replayed from a frozen scenario
bank against a trained checkpoint. Verifies the basic reward-ordering
sanity condition: G_success > G_timeout and G_success > G_collision.

``Stage10SymmetricMergeEnv.step()`` (src/thesis/envs/stage10_symmetric_merge_env.py:1025-1032)
does not expose a bare "progress" scalar directly, but every other
signed component is in its ``info`` dict, so progress is recovered
algebraically: the components are constructed to sum back to the env's
own returned reward EXACTLY (this is checked, not assumed -- see
``decompose_step_reward``'s own internal consistency, exercised in this
diagnostic's unit tests). Study B always sets ``ttc_penalty_weight=0.0``,
so the ttc component is always 0 -- reported anyway for completeness/
future-proofing, not hidden.

There is no separate "timeout penalty" scalar in the underlying reward
formula -- under the direct-welfare qualification's default
(``include_time_cost=False``), the time-cost component is 0 on every
step, and a timed-out episode differs from a successful one only through
the missing exit reward and the terminal welfare bonus collapsing (a
timed-out vehicle's utility U_i=0). This is an expected property to
verify, not a gap.

Known simplification (documented, not silently wrong): the discounted
return uses one shared global step index for every agent's per-step
reward, and applies the terminal welfare bonus once per agent still
active at the true terminal step -- this matches how the actual training
scripts accumulate the team-level training signal (analogous to a VDN
``Q_tot = sum_i Q_i``), but is an approximation of each individual
agent's own discount horizon if agents exit at different times.

Read-only against an already-finished checkpoint -- runs no training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(REPO_SRC))
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_policy import load_policy  # noqa: E402

from thesis.study_b.heterogeneous_env import StudyBEnvConfig, StudyBHeterogeneousEnv  # noqa: E402
from thesis.study_b.training_common import load_scenario_bank  # noqa: E402
from thesis.study_b.utility import episode_utilities  # noqa: E402
from thesis.study_b.welfare_reward import condition_by_name, terminal_welfare_bonus  # noqa: E402

__all__ = ["decompose_step_reward", "replay_episode_with_decomposition", "find_representative_scenarios", "GAMMA"]

GAMMA = 0.995  # matches pbrs_reward.GAMMA / MAPPOConfig.gamma -- Study B's project-wide discount factor


def decompose_step_reward(vid: str, base_reward: dict[str, float], info: dict) -> dict[str, float]:
    """Signed per-component contributions to ``base_reward[vid]`` this
    step, recovered from ``Stage10SymmetricMergeEnv.step()``'s ``info``
    dict. By construction these sum EXACTLY back to ``base_reward[vid]``
    (``progress`` is defined as the residual, not independently measured)
    -- callers/tests should verify that invariant, not just trust it."""
    exit_signed = info["exit_reward_magnitude_used"] * (1.0 if info["exit_event"][vid] else 0.0)
    collision_signed = -info["collision_penalty_used_per_vehicle"][vid] * (1.0 if info["collision_penalty_applied"][vid] else 0.0)
    hard_brake_signed = -info["hard_braking_eta_used"] * info["hard_braking_cost_used"][vid]
    time_cost_signed = -info["time_cost_per_step_used"] * (1.0 if info["time_cost_applied"][vid] else 0.0)
    ttc_signed = info["ttc_penalty_weight_used"] * info["ttc_penalty"][vid]
    total = base_reward[vid]
    progress_signed = total - (exit_signed + collision_signed + hard_brake_signed + time_cost_signed + ttc_signed)
    return {
        "progress": progress_signed, "exit": exit_signed, "collision": collision_signed,
        "hard_brake": hard_brake_signed, "time_cost": time_cost_signed, "ttc": ttc_signed, "total": total,
    }


def find_representative_scenarios(rows: list[dict]) -> dict[str, str | None]:
    """``rows``: ``evaluate_policy.run_eval``'s output. Returns
    ``{"success": scenario_id or None, "timeout": ..., "collision": ...}``
    -- the first scenario found for each ``term_reason``."""
    picked: dict[str, str | None] = {"success": None, "timeout": None, "collision": None}
    reason_to_key = {"success": "success", "truncation": "timeout", "collision": "collision"}
    for row in rows:
        key = reason_to_key.get(row["term_reason"])
        if key is not None and picked[key] is None:
            picked[key] = row["scenario_id"]
    return picked


def replay_episode_with_decomposition(
    *, select, env: StudyBHeterogeneousEnv, scenario, condition_name: str,
    episode_max_steps: int = 200, gamma: float = GAMMA,
) -> dict:
    obs, _info = env.reset(seed=0, scenario=scenario)
    component_totals = {"progress": 0.0, "exit": 0.0, "collision": 0.0, "hard_brake": 0.0, "time_cost": 0.0, "ttc": 0.0}
    per_vehicle_return: dict[str, float] = dict.fromkeys(env.active_vehicle_ids, 0.0)
    discounted_total = 0.0
    step_index = 0
    term_reason = "truncation"
    for step_index in range(episode_max_steps):
        actions = select(obs)
        obs, base_reward, terminated, truncated, step_info = env.step(actions)
        step_sum = 0.0
        for vid, r in base_reward.items():
            parts = decompose_step_reward(vid, base_reward, step_info)
            for name in component_totals:
                component_totals[name] += parts[name]
            per_vehicle_return[vid] = per_vehicle_return.get(vid, 0.0) + r
            step_sum += r
        discounted_total += (gamma ** step_index) * step_sum
        if terminated or truncated:
            term_reason = step_info["term_reason"]
            break

    condition = condition_by_name(condition_name)
    traces = env.episode_traces()
    episode_u = episode_utilities(traces)
    welfare_bonus = terminal_welfare_bonus(condition, list(episode_u.values()))
    n_active_at_end = len(env.active_vehicle_ids)
    welfare_contribution = welfare_bonus * n_active_at_end
    for vid in env.active_vehicle_ids:
        per_vehicle_return[vid] = per_vehicle_return.get(vid, 0.0) + welfare_bonus

    undiscounted_G = sum(per_vehicle_return.values())
    discounted_G = discounted_total + (gamma ** step_index) * welfare_contribution

    return {
        "scenario_id": scenario.scenario_id,
        "term_reason": term_reason,
        "episode_length": step_index + 1,
        "component_totals": component_totals,
        "terminal_welfare_bonus": welfare_bonus,
        "n_active_at_terminal_step": n_active_at_end,
        "undiscounted_G": undiscounted_G,
        "discounted_G": discounted_G,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", type=str, default="dqn", choices=["mappo", "dqn"])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--scenario-bank", type=Path, required=True)
    p.add_argument("--condition", type=str, default="mean", choices=["mean", "ggi", "maximin"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episode-max-steps", type=int, default=200)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args(argv)

    from evaluate_policy import run_eval  # noqa: E402  (local import: avoids double env construction at module load)

    scenarios = load_scenario_bank(args.scenario_bank)
    scenario_by_id = {s.scenario_id: s for s in scenarios}

    rows = run_eval(
        algorithm=args.algorithm, checkpoint=args.checkpoint, scenario_bank=args.scenario_bank,
        episode_max_steps=args.episode_max_steps, device=args.device,
    )
    picked = find_representative_scenarios(rows)

    env = StudyBHeterogeneousEnv(StudyBEnvConfig(episode_max_steps=args.episode_max_steps))
    select = load_policy(algorithm=args.algorithm, checkpoint=args.checkpoint, env=env, device=args.device)

    results = {}
    for outcome, scenario_id in picked.items():
        if scenario_id is None:
            results[outcome] = None
            continue
        results[outcome] = replay_episode_with_decomposition(
            select=select, env=env, scenario=scenario_by_id[scenario_id], condition_name=args.condition,
            episode_max_steps=args.episode_max_steps,
        )

    ordering = {}
    if results["success"] is not None:
        for other in ("timeout", "collision"):
            if results[other] is not None:
                ordering[f"G_success > G_{other}"] = results["success"]["undiscounted_G"] > results[other]["undiscounted_G"]

    report = {"picked_scenarios": picked, "episodes": results, "reward_ordering_check": ordering}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for outcome, result in results.items():
        if result is None:
            print(f"{outcome:>10}: no example found in this scenario bank")
            continue
        print(f"{outcome:>10}: scenario={result['scenario_id']} len={result['episode_length']} "
              f"undiscounted_G={result['undiscounted_G']:.3f} discounted_G={result['discounted_G']:.3f}")
    print(f"ordering check: {ordering}")
    print(f"report written -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
