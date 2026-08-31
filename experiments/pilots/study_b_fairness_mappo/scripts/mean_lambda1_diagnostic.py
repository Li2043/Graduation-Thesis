#!/usr/bin/env python3
"""MEAN_LAMBDA1_TRAJECTORY_DIAGNOSTIC (read-only, evaluation-only).

Compares the authoritative C64 ensemble policy (K(1,200,000), stage
'C64') against the authoritative Mean-qualification-lambda_W=1.0
ensemble policy (K(2,000,000), stage 'MeanQual') for seeds 900101 and
900102, on the identical frozen Q.json (64-scenario) bank.

epsilon_eval=0 throughout (ensemble selection has no epsilon knob).
No training, no replay writes, no optimizer updates. Does not touch
the running lambda_W=0.5 (MeanQualMR2) job in any way -- reads only
its own frozen checkpoints for the two ALREADY-COMPLETED stages.

Produces, per seed, per scenario, under EACH policy:
  - term_reason, episode_length
  - undiscounted and discounted (gamma=0.995) task return
  - utilities/burdens (episode-level)
  - terminal welfare contribution (only nonzero for the MeanQual
    policy, since C64's lambda_W=0)
  - hard-brake count, time-below-target-speed
For the paired failure set (C64=success, MeanQual=collision, per
seed), additionally replays both policies in lockstep against the
SAME scenario to find the first action-divergence step and record
per-vehicle Q-values/margins/local-observation/physical state at and
around that point.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[4] / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent
for p in (REPO_SRC, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

from dr1_c4_failure_map import classify_collision_type, min_pairwise_ttc  # noqa: E402
from thesis.study_b.envs.highwayenv_action import ACCELERATE, BRAKE, HOLD  # noqa: E402
from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig  # noqa: E402
from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv, StudyBHighwayWrapperConfig  # noqa: E402
from thesis.study_b.q_ensemble import (  # noqa: E402
    load_ensemble_agents,
    q_ensemble_values,
    select_ensemble_actions,
)
from thesis.study_b.training_common import load_scenario_bank  # noqa: E402
from thesis.study_b.utility import episode_burdens, episode_utilities  # noqa: E402
from thesis.study_b.welfare_reward import MEAN, terminal_welfare_bonus  # noqa: E402

GAMMA = 0.995
_ACTION_NAME = {HOLD: "HOLD", ACCELERATE: "ACCELERATE", BRAKE: "BRAKE"}

_ENV_CONFIG = ThesisHighwayMergeEnvConfig()
_MERGE_START_X = _ENV_CONFIG.before_merge_length
_MERGE_END_X = _ENV_CONFIG.before_merge_length + _ENV_CONFIG.converge_merge_length
_PARALLEL_END_X = _MERGE_END_X + _ENV_CONFIG.parallel_merge_length


def _make_env(episode_max_steps: int = 200) -> StudyBHeterogeneousHighwayEnv:
    cfg = ThesisHighwayMergeEnvConfig(episode_max_steps=episode_max_steps, action_representation="meta_speed")
    return StudyBHeterogeneousHighwayEnv(StudyBHighwayWrapperConfig(env_config=cfg))


def zone_label(x: float) -> str:
    if x < _MERGE_START_X:
        return "before_merge"
    if x < _MERGE_END_X:
        return "converge"
    if x < _PARALLEL_END_X:
        return "parallel"
    return "after_merge"


def c64_checkpoint_paths(seed: int, root: Path) -> dict[int, Path]:
    return {step: root / f"ckpt_step_{step}.pt" for step in (1_050_000, 1_100_000, 1_150_000, 1_200_000)}


def meanqual_checkpoint_paths(seed: int, root: Path) -> dict[int, Path]:
    return {step: root / f"ckpt_step_{step}.pt" for step in (1_850_000, 1_900_000, 1_950_000, 2_000_000)}


def run_episode(
    *, agents, env: StudyBHeterogeneousHighwayEnv, scenario, episode_max_steps: int, lam: float,
) -> dict:
    """Deterministic single-episode replay under a given ensemble
    policy. Returns full step trace plus episode-level summary
    (returns, utilities, burdens, welfare contribution)."""
    obs, _info = env.reset(seed=0, scenario=scenario)
    steps: list[dict] = []
    term_reason = "ongoing"
    task_returns: dict[str, float] = dict.fromkeys(env.active_vehicle_ids, 0.0)
    discounted_task_returns: dict[str, float] = dict.fromkeys(env.active_vehicle_ids, 0.0)
    hard_brake_counts: dict[str, int] = dict.fromkeys(env.active_vehicle_ids, 0)
    below_target_steps: dict[str, int] = dict.fromkeys(env.active_vehicle_ids, 0)

    for t in range(episode_max_steps):
        actions = select_ensemble_actions(agents, obs)
        step_record = {"t": t, "actions": {vid: _ACTION_NAME[int(a)] for vid, a in actions.items()}}
        speeds_before = {vid: float(env._env._vehicle_by_id[vid].speed) for vid in env.active_vehicle_ids}  # noqa: SLF001
        accels_before = {vid: float(env._env._vehicle_by_id[vid].action.get("acceleration", 0.0)) for vid in env.active_vehicle_ids}  # noqa: SLF001

        obs, base_reward, terminated, truncated, step_info = env.step(actions)

        for vid, r in base_reward.items():
            task_returns[vid] = task_returns.get(vid, 0.0) + r
            discounted_task_returns[vid] = discounted_task_returns.get(vid, 0.0) + (GAMMA ** t) * r
        for vid in env.active_vehicle_ids:
            if accels_before.get(vid, 0.0) <= -3.5:
                hard_brake_counts[vid] += 1
        for vid, spec in scenario.vehicles.items():
            if speeds_before.get(vid, spec.target_speed) < spec.target_speed - 0.5:
                below_target_steps[vid] = below_target_steps.get(vid, 0) + 1

        steps.append(step_record)
        if step_info["collision_event"]:
            term_reason, episode_end_t = "collision", t + 1
            break
        if terminated:
            term_reason, episode_end_t = "success", t + 1
            break
        if truncated:
            term_reason, episode_end_t = "truncation", t + 1
            break
    else:
        episode_end_t = episode_max_steps

    traces = env.episode_traces()
    utilities = episode_utilities(traces)
    burdens = episode_burdens(traces, dt=env.dt())
    w_mean = MEAN.welfare_fn(list(utilities.values()))
    welfare_bonus = terminal_welfare_bonus(MEAN, list(utilities.values()), lam=lam) if lam != 0.0 else 0.0
    discounted_welfare = (GAMMA ** (episode_end_t - 1)) * welfare_bonus

    return {
        "term_reason": term_reason, "episode_length": episode_end_t,
        "undiscounted_task_return_mean": float(np.mean(list(task_returns.values()))),
        "discounted_task_return_mean": float(np.mean(list(discounted_task_returns.values()))),
        "discounted_welfare_contribution": float(discounted_welfare),
        "w_mean": float(w_mean), "utilities": utilities, "burdens": burdens,
        "mean_U": float(np.mean(list(utilities.values()))), "min_U": float(min(utilities.values())),
        "hard_brake_total": int(sum(hard_brake_counts.values())),
        "below_target_speed_steps_total": int(sum(below_target_steps.values())),
        "steps": steps,
    }


def run_distributional_pass(
    *, seed: int, c64_root: Path, meanqual_root: Path, scenario_bank_path: Path, scenario_ids: list[str],
    episode_max_steps: int, device: str,
) -> list[dict]:
    c64_agents = load_ensemble_agents(
        seed=seed, checkpoint_paths=c64_checkpoint_paths(seed, c64_root), device=device,
        expected_steps=(1_050_000, 1_100_000, 1_150_000, 1_200_000),
        expected_stage_by_step=dict.fromkeys((1_050_000, 1_100_000, 1_150_000, 1_200_000), "C64"),
    )
    mq_agents = load_ensemble_agents(
        seed=seed, checkpoint_paths=meanqual_checkpoint_paths(seed, meanqual_root), device=device,
        expected_steps=(1_850_000, 1_900_000, 1_950_000, 2_000_000),
        expected_stage_by_step=dict.fromkeys((1_850_000, 1_900_000, 1_950_000, 2_000_000), "MeanQual"),
    )

    scenarios = load_scenario_bank(scenario_bank_path)
    by_id = {s.scenario_id: s for s in scenarios}

    rows = []
    for policy_name, agents, lam in (("C64", c64_agents, 0.0), ("MeanQual_lambda1", mq_agents, 1.0)):
        env = _make_env(episode_max_steps)
        for sid in scenario_ids:
            scenario = by_id[sid]
            result = run_episode(agents=agents, env=env, scenario=scenario, episode_max_steps=episode_max_steps, lam=lam)
            rows.append({
                "seed": seed, "policy": policy_name, "scenario_id": sid,
                "term_reason": result["term_reason"], "episode_length": result["episode_length"],
                "undiscounted_task_return_mean": result["undiscounted_task_return_mean"],
                "discounted_task_return_mean": result["discounted_task_return_mean"],
                "discounted_welfare_contribution": result["discounted_welfare_contribution"],
                "w_mean": result["w_mean"], "mean_U": result["mean_U"], "min_U": result["min_U"],
                "hard_brake_total": result["hard_brake_total"],
                "below_target_speed_steps_total": result["below_target_speed_steps_total"],
            })
        print(f"[mean_lambda1_diag] seed={seed} policy={policy_name} done ({len(scenario_ids)} scenarios)")
    return rows


def replay_divergence(
    *, seed: int, scenario_id: str, c64_root: Path, meanqual_root: Path, scenario_bank_path: Path,
    episode_max_steps: int, device: str, context_window: int = 8,
) -> dict:
    c64_agents = load_ensemble_agents(
        seed=seed, checkpoint_paths=c64_checkpoint_paths(seed, c64_root), device=device,
        expected_steps=(1_050_000, 1_100_000, 1_150_000, 1_200_000),
        expected_stage_by_step=dict.fromkeys((1_050_000, 1_100_000, 1_150_000, 1_200_000), "C64"),
    )
    mq_agents = load_ensemble_agents(
        seed=seed, checkpoint_paths=meanqual_checkpoint_paths(seed, meanqual_root), device=device,
        expected_steps=(1_850_000, 1_900_000, 1_950_000, 2_000_000),
        expected_stage_by_step=dict.fromkeys((1_850_000, 1_900_000, 1_950_000, 2_000_000), "MeanQual"),
    )
    scenarios = load_scenario_bank(scenario_bank_path)
    scenario = next(s for s in scenarios if s.scenario_id == scenario_id)

    env_a, env_b = _make_env(episode_max_steps), _make_env(episode_max_steps)
    obs_a, _ = env_a.reset(seed=0, scenario=scenario)
    obs_b, _ = env_b.reset(seed=0, scenario=scenario)

    history: list[dict] = []
    divergence = None
    outcome_a, outcome_b = "ongoing", "ongoing"
    end_a, end_b = episode_max_steps, episode_max_steps

    # Lockstep phase: both envs have seen IDENTICAL actions so far (no
    # divergence yet), so their physical states remain identical; once
    # any vehicle's action differs, the two trajectories are no longer
    # comparable step-by-step and must be finished independently.
    t = 0
    while t < episode_max_steps:
        vids = env_a.active_vehicle_ids
        actions_a = select_ensemble_actions(c64_agents, obs_a)
        actions_b = select_ensemble_actions(mq_agents, obs_b)

        step_record = {"t": t, "vehicles": {}}
        any_diff = False
        for vid in vids:
            qa = q_ensemble_values(c64_agents, obs_a[vid])
            qb = q_ensemble_values(mq_agents, obs_b[vid])
            diff = int(actions_a[vid]) != int(actions_b[vid])
            any_diff = any_diff or diff
            qa_sorted = sorted(qa, reverse=True)
            qb_sorted = sorted(qb, reverse=True)
            vehicle = env_a._env._vehicle_by_id[vid]  # noqa: SLF001
            x, y = env_a._env.world_xy(vehicle)
            step_record["vehicles"][vid] = {
                "action_c64": _ACTION_NAME[int(actions_a[vid])], "action_meanqual": _ACTION_NAME[int(actions_b[vid])],
                "differs": diff,
                "q_c64": {_ACTION_NAME[k]: float(qa[i]) for i, k in enumerate((HOLD, ACCELERATE, BRAKE))},
                "q_meanqual": {_ACTION_NAME[k]: float(qb[i]) for i, k in enumerate((HOLD, ACCELERATE, BRAKE))},
                "margin_c64": float(qa_sorted[0] - qa_sorted[1]), "margin_meanqual": float(qb_sorted[0] - qb_sorted[1]),
                "x": x, "y": y, "speed": float(vehicle.speed), "zone": zone_label(x),
                "role": scenario.vehicles[vid].role, "speed_class": scenario.vehicles[vid].speed_class,
                "ttc_slot": scenario.vehicles[vid].ttc_slot, "target_speed": scenario.vehicles[vid].target_speed,
                "obs_c64": obs_a[vid].tolist(), "obs_meanqual": obs_b[vid].tolist(),
            }
        history.append(step_record)
        if any_diff:
            divergence = {"t": t, "detail": step_record}
            break

        obs_a, _r, term_a, trunc_a, info_a = env_a.step(actions_a)
        obs_b, _r, term_b, trunc_b, info_b = env_b.step(actions_b)
        t += 1

        if info_a["collision_event"] or term_a or trunc_a:
            outcome_a = "collision" if info_a["collision_event"] else ("success" if term_a else "truncation")
            end_a = t
        if info_b["collision_event"] or term_b or trunc_b:
            outcome_b = "collision" if info_b["collision_event"] else ("success" if term_b else "truncation")
            end_b = t
        if outcome_a != "ongoing" or outcome_b != "ongoing":
            break

    # Post-divergence phase: finish each env independently on its own
    # policy, no further cross-comparison (states have diverged).
    while outcome_a == "ongoing" and t < episode_max_steps:
        actions_a = select_ensemble_actions(c64_agents, obs_a)
        obs_a, _r, term_a, trunc_a, info_a = env_a.step(actions_a)
        t += 1
        if info_a["collision_event"] or term_a or trunc_a:
            outcome_a = "collision" if info_a["collision_event"] else ("success" if term_a else "truncation")
            end_a = t
    post_divergence_trace_b: list[dict] = []
    while outcome_b == "ongoing" and t < episode_max_steps:
        actions_b = select_ensemble_actions(mq_agents, obs_b)
        vids_b = env_b.active_vehicle_ids
        x_b = {vid: env_b._env.world_xy(env_b._env._vehicle_by_id[vid])[0] for vid in vids_b}  # noqa: SLF001
        y_b = {vid: env_b._env.world_xy(env_b._env._vehicle_by_id[vid])[1] for vid in vids_b}  # noqa: SLF001
        speed_b = {vid: float(env_b._env._vehicle_by_id[vid].speed) for vid in vids_b}  # noqa: SLF001
        step_ttc = min_pairwise_ttc(x_b, y_b, speed_b, vids_b)
        post_divergence_trace_b.append({
            "t": t, "actions": {vid: _ACTION_NAME[int(a)] for vid, a in actions_b.items()},
            "positions": {vid: round(x_b[vid], 2) for vid in vids_b}, "speeds": {vid: round(speed_b[vid], 2) for vid in vids_b},
            "min_ttc": (None if step_ttc == float("inf") else round(step_ttc, 4)),
            "zones": {vid: zone_label(x_b[vid]) for vid in vids_b},
        })
        obs_b, _r, term_b, trunc_b, info_b = env_b.step(actions_b)
        t += 1
        if info_b["collision_event"] or term_b or trunc_b:
            outcome_b = "collision" if info_b["collision_event"] else ("success" if term_b else "truncation")
            end_b = t
            if info_b["collision_event"]:
                post_divergence_trace_b.append({"t": t, "collision_pairs": info_b["collision_pairs"], "outcome": "collision"})

    div_t = divergence["t"] if divergence else None
    ctx_start = max(0, (div_t or 0) - context_window)
    ctx_end = min(len(history), (div_t or 0) + context_window + 1)

    return {
        "seed": seed, "scenario_id": scenario_id, "outcome_c64": outcome_a, "outcome_meanqual": outcome_b,
        "divergence_t": div_t, "divergence_detail": (divergence["detail"] if divergence else None),
        "context_window_steps": history[ctx_start:ctx_end],
        "final_steps": history[-min(20, len(history)):],
        "meanqual_post_divergence_trace": post_divergence_trace_b[-20:],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario-bank", type=Path, required=True)
    p.add_argument("--c64-checkpoint-root", type=str, required=True, help="seed:root_dir pairs")
    p.add_argument("--meanqual-checkpoint-root", type=str, required=True, help="seed:root_dir pairs")
    p.add_argument("--paired-failure-scenarios", type=str, default="", help="seed:sid1,sid2,... (only these get D4/D5/D6 divergence replay)")
    p.add_argument("--episode-max-steps", type=int, default=200)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    c64_roots = dict(pair.split(":", 1) for pair in args.c64_checkpoint_root.split(";"))
    mq_roots = dict(pair.split(":", 1) for pair in args.meanqual_checkpoint_root.split(";"))
    c64_roots = {int(k): Path(v) for k, v in c64_roots.items()}
    mq_roots = {int(k): Path(v) for k, v in mq_roots.items()}

    scenarios = load_scenario_bank(args.scenario_bank)
    all_sids = [s.scenario_id for s in scenarios]

    all_dist_rows = []
    for seed in c64_roots:
        rows = run_distributional_pass(
            seed=seed, c64_root=c64_roots[seed], meanqual_root=mq_roots[seed],
            scenario_bank_path=args.scenario_bank, scenario_ids=all_sids,
            episode_max_steps=args.episode_max_steps, device=args.device,
        )
        all_dist_rows.extend(rows)

    dist_csv = args.output_dir / "MEAN_LAMBDA1_DISTRIBUTIONAL.csv"
    with open(dist_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_dist_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_dist_rows)
    print(f"wrote {dist_csv}")

    divergence_results = []
    if args.paired_failure_scenarios:
        for entry in args.paired_failure_scenarios.split(";"):
            seed_str, sids_str = entry.split(":", 1)
            seed = int(seed_str)
            for sid in sids_str.split(","):
                if not sid:
                    continue
                result = replay_divergence(
                    seed=seed, scenario_id=sid, c64_root=c64_roots[seed], meanqual_root=mq_roots[seed],
                    scenario_bank_path=args.scenario_bank, episode_max_steps=args.episode_max_steps, device=args.device,
                )
                divergence_results.append(result)
                (args.output_dir / f"DIVERGENCE_seed{seed}_{sid}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(f"[divergence] seed={seed} scenario={sid} outcome_c64={result['outcome_c64']} "
                      f"outcome_meanqual={result['outcome_meanqual']} divergence_t={result['divergence_t']}")

    (args.output_dir / "MEAN_LAMBDA1_DIVERGENCE_SUMMARY.json").write_text(
        json.dumps(divergence_results, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(divergence_results)} divergence traces to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
