#!/usr/bin/env python3
"""Claude_Code_Autonomous_Experiment_Runbook.md Diagnostic 6N: trains the
corrected (target-sync-fixed) local shared DQN on a FIXED, deterministic
scenario SET (a curriculum stage -- C1/C4/C16/C64), task-only reward,
using the FROZEN ABSOLUTE-STEP epsilon/LR schedule (not a fraction of
this call's own --max-additional-steps -- see ABSOLUTE_SCHEDULE.json and
runbook sec 7's explicit "never restart or rescale the schedule" rule).

Generalizes ``train_single_scenario_overfit.py`` (Diagnostic 6L) from
one fixed scenario to a fixed SET of scenarios (sampled uniformly at
random each episode via a dedicated RNG, per ``curriculum_scenario_seed``
in FROZEN_CONFIG_SNAPSHOT.json), and adds true absolute-step checkpoint
continuation (``--resume-from`` + ``--start-step``) so a curriculum
stage transition (runbook sec 10: "continue from that stage checkpoint.
Do not reset network, optimizer, or schedule between stages") is a
genuine continuation, not a fresh run with a coincidentally-loaded
network. Replay buffer is NOT persisted across a resume (matches this
project's existing, already-documented warm-start convention -- see
``train_dqn_fallback.py``'s own docstring) -- refills past
``--replay-warmup`` again after any resume.

Per-scenario metrics are tracked (not just pooled) so a later
diversity-failure analysis (runbook sec 17) can be run without needing
to re-train."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(REPO_SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from thesis.study_b.heterogeneous_env import StudyBEnvConfig, StudyBHeterogeneousEnv  # noqa: E402
from thesis.study_b.oracle_controller import oracle_actions  # noqa: E402
from thesis.study_b.shared_local_dqn import SharedLocalDQNAgent, build_study_b_dqn_config, epsilon_at_step_v12, lr_at_step_v12  # noqa: E402
from thesis.study_b.training_common import StudyBEpisodeWindowStats, load_scenario_bank  # noqa: E402
from thesis.study_b.utility import episode_burdens, episode_utilities  # noqa: E402
from thesis.study_b.welfare_reward import WELFARE_LAMBDA, condition_by_name, terminal_welfare_bonus  # noqa: E402

__all__ = ["collect_fixed_oracle_batch_multi", "fixed_batch_q_stats", "TARGET_SYNC_INTERVAL_UPDATES"]

TARGET_SYNC_INTERVAL_UPDATES = 250  # frozen -- see FROZEN_CONFIG_SNAPSHOT.json


def collect_fixed_oracle_batch_multi(scenarios: list, *, episode_max_steps: int) -> np.ndarray:
    """ONE-TIME oracle-driven rollout of EVERY scenario in this stage,
    concatenated -- a stable, unchanging reference batch spanning the
    whole stage's scenario diversity, for cross-checkpoint Q comparisons
    (runbook sec 11: "Use fixed-reference Q statistics for cross-checkpoint
    numerical comparisons. Do not treat policy-visited mean Q alone as
    evidence of instability.")."""
    env = StudyBHeterogeneousEnv(StudyBEnvConfig(episode_max_steps=episode_max_steps, include_time_cost=False))
    rows = []
    for scenario in scenarios:
        obs, _info = env.reset(seed=0, scenario=scenario)
        rows.extend(obs.values())
        for _t in range(episode_max_steps):
            positions = {vid: env._env._vehicles[vid].route_position for vid in env.active_vehicle_ids}  # noqa: SLF001
            actions = oracle_actions(
                scenario=scenario, positions=positions, merge_start=200.0, merge_end=300.0,
                active_vehicle_ids={vid: True for vid in env.active_vehicle_ids},
            )
            obs, _r, terminated, truncated, _info = env.step(actions)
            rows.extend(obs.values())
            if terminated or truncated:
                break
    return np.stack(rows)


def fixed_batch_q_stats(agent: SharedLocalDQNAgent, fixed_batch: np.ndarray) -> dict:
    q_all = np.stack([agent.learner.q_values(obs, network="online") for obs in fixed_batch])
    spreads = q_all.max(axis=1) - q_all.min(axis=1)
    return {"mean_Q": float(q_all.mean()), "mean_Q_spread": float(spreads.mean())}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario-bank", type=Path, required=True)
    p.add_argument("--scenario-ids", type=str, nargs="+", required=True, help="the curriculum stage's scenario set")
    p.add_argument("--stage-name", type=str, required=True, help="e.g. C1/C4/C16/C64 -- used only for logging/naming")
    p.add_argument("--master-seed", type=int, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--start-step", type=int, default=0, help="absolute step to resume from (0 = fresh)")
    p.add_argument("--max-additional-steps", type=int, required=True, help="how many MORE steps to train from --start-step")
    p.add_argument("--episode-max-steps", type=int, default=200)
    p.add_argument("--checkpoint-every", type=int, default=25_000)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--replay-warmup", type=int, default=512)
    p.add_argument("--eps-decay-steps-absolute", type=int, default=640_000, help="frozen -- see ABSOLUTE_SCHEDULE.json")
    p.add_argument("--lr-decay-steps-absolute", type=int, default=800_000, help="frozen -- see ABSOLUTE_SCHEDULE.json")
    p.add_argument("--welfare-lambda", type=float, default=0.0, help="0.0 = task-only (6N default)")
    p.add_argument("--condition", type=str, default="mean", choices=["mean", "ggi", "maximin"], help="only matters if welfare_lambda != 0")
    p.add_argument("--resume-from", type=Path, default=None)
    p.add_argument("--dense-log-every", type=int, default=0)
    args = p.parse_args(argv)

    all_scenarios = load_scenario_bank(args.scenario_bank)
    by_id = {s.scenario_id: s for s in all_scenarios}
    stage_scenarios = [by_id[sid] for sid in args.scenario_ids]
    condition = condition_by_name(args.condition)

    env = StudyBHeterogeneousEnv(StudyBEnvConfig(episode_max_steps=args.episode_max_steps, include_time_cost=False))
    diag_env = StudyBHeterogeneousEnv(StudyBEnvConfig(episode_max_steps=args.episode_max_steps, include_time_cost=False))
    dqn_config = build_study_b_dqn_config(reward_condition="baseline", device=args.device)
    agent = SharedLocalDQNAgent(dqn_config, seed=args.master_seed)
    window_stats = StudyBEpisodeWindowStats()
    per_scenario_stats: dict[str, StudyBEpisodeWindowStats] = defaultdict(StudyBEpisodeWindowStats)
    scenario_rng = np.random.default_rng(args.master_seed * 7919 + 1)  # dedicated, deterministic, separate from training RNGs

    if args.resume_from is not None:
        ckpt = torch.load(args.resume_from, map_location="cpu")
        agent.learner.online.load_state_dict(ckpt["online"])
        agent.learner.target.load_state_dict(ckpt["target"])
        agent.learner.optimiser.load_state_dict(ckpt["optimiser"])
        agent.learner._update_count = int(ckpt["update_count"])
        print(f"resumed from {args.resume_from} (checkpoint step {ckpt['step']}), continuing at absolute step {args.start_step}")

    output_root = Path(args.output_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = checkpoint_root / f"seed_{args.master_seed}_{args.stage_name}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    max_total_step = args.start_step + args.max_additional_steps
    checkpoint_targets = sorted(
        {s for s in range(args.start_step, max_total_step + 1, args.checkpoint_every)} | {args.start_step, max_total_step}
    )
    checkpoint_records: list[dict] = []
    manifest_path = output_root / f"seed_{args.master_seed}_{args.stage_name}_manifest.json"

    dense_log_file = None
    fixed_oracle_batch = None
    if args.dense_log_every > 0:
        fixed_oracle_batch = collect_fixed_oracle_batch_multi(stage_scenarios, episode_max_steps=args.episode_max_steps)
        dense_log_file = open(output_root / f"seed_{args.master_seed}_{args.stage_name}_dense_log.jsonl", "a", encoding="utf-8")
    fixed_q_batch_for_checkpoints = fixed_oracle_batch if fixed_oracle_batch is not None else \
        collect_fixed_oracle_batch_multi(stage_scenarios, episode_max_steps=args.episode_max_steps)

    def collect_policy_visited_observations() -> np.ndarray:
        """Current-policy greedy rollout, over EVERY stage scenario --
        the OTHER Q measurement runbook sec 11 wants logged (policy-visited,
        as distinct from the fixed oracle-reference batch)."""
        rows = []
        for scenario in stage_scenarios:
            obs, _info = diag_env.reset(seed=0, scenario=scenario)
            rows.extend(obs.values())
            for _t in range(args.episode_max_steps):
                actions = agent.select_actions(obs, epsilon=0.0, greedy=True)
                obs, _r, terminated, truncated, _info = diag_env.step(actions)
                rows.extend(obs.values())
                if terminated or truncated:
                    break
        return np.stack(rows)

    def save_checkpoint(step: int) -> None:
        torch.save(
            {
                "step": step, "stage": args.stage_name, "scenario_ids": args.scenario_ids,
                "online": agent.learner.online.state_dict(), "target": agent.learner.target.state_dict(),
                "optimiser": agent.learner.optimiser.state_dict(), "update_count": agent.learner._update_count,
                "replay_size": len(agent.learner.replay),
            },
            checkpoint_dir / f"ckpt_step_{step}.pt",
        )
        policy_q = fixed_batch_q_stats(agent, collect_policy_visited_observations())
        oracle_ref_q = fixed_batch_q_stats(agent, fixed_q_batch_for_checkpoints)
        per_scenario = {sid: per_scenario_stats[sid].as_dict() for sid in args.scenario_ids}
        for sid in args.scenario_ids:
            per_scenario_stats[sid].reset()
        metrics = {
            "step": step, "window": window_stats.as_dict(),
            "q_diagnostics_policy_visited": policy_q, "q_diagnostics_fixed_oracle_ref": oracle_ref_q,
            "per_scenario": per_scenario,
        }
        window_stats.reset()
        checkpoint_records.append(metrics)
        print(f"[{args.stage_name}] step={step:>8}  completion={metrics['window']['completion_rate']:.3f}  "
              f"collision={metrics['window']['collision_rate']:.3f}  timeout={metrics['window']['truncation_rate']:.3f}  "
              f"mean_Q(policy)={policy_q['mean_Q']:.4f}  mean_Q(oracle_ref)={oracle_ref_q['mean_Q']:.4f}")

    def pick_scenario():
        return stage_scenarios[int(scenario_rng.integers(0, len(stage_scenarios)))]

    total_step = args.start_step
    current_scenario = pick_scenario()
    obs, _info = env.reset(seed=0, scenario=current_scenario)
    prev_active = {vid: True for vid in env.active_vehicle_ids}

    if args.start_step in checkpoint_targets and not checkpoint_records:
        save_checkpoint(args.start_step)

    start_time = time.time()
    while total_step < max_total_step:
        eps = epsilon_at_step_v12(total_step, decay_steps=args.eps_decay_steps_absolute)
        lr = lr_at_step_v12(total_step, decay_steps=args.lr_decay_steps_absolute)
        agent.set_learning_rate(lr)

        actions = agent.select_actions(obs, epsilon=eps)
        prev_obs = obs
        obs, base_reward, terminated, truncated, step_info = env.step(actions)
        episode_over = terminated or truncated

        welfare_bonus = 0.0
        if episode_over and args.welfare_lambda != 0.0:
            traces = env.episode_traces()
            episode_u = episode_utilities(traces)
            welfare_bonus = terminal_welfare_bonus(condition, list(episode_u.values()), lam=args.welfare_lambda)

        for vid in env.active_vehicle_ids:
            if not prev_active[vid]:
                continue
            exit_this_step = step_info["exit_event"][vid]
            controller_terminal = bool(terminated or exit_this_step)
            learner_completed = bool(exit_this_step and not step_info["collision_event"])
            shaped_reward = base_reward[vid] + (welfare_bonus if episode_over else 0.0)
            transition = agent.build_transition(
                vehicle_id=vid, observation=prev_obs[vid], action=actions[vid],
                shaped_reward=shaped_reward, next_observation=obs[vid],
                terminated=terminated, truncated=truncated,
                controller_terminal=controller_terminal, learner_completed=learner_completed,
                base_reward=base_reward[vid], shaping_component=(welfare_bonus if episode_over else 0.0),
                episode_id=f"seed_{args.master_seed}_{args.stage_name}_{current_scenario.scenario_id}", step=total_step,
            )
            agent.store_transition(transition)
        prev_active = dict(step_info["active"])

        update_info = agent.maybe_update(warmup=args.replay_warmup)
        if update_info is not None:
            if agent.learner._update_count % TARGET_SYNC_INTERVAL_UPDATES == 0:
                agent.learner.hard_sync_target()
            if dense_log_file is not None and agent.learner._update_count % args.dense_log_every == 0:
                q_stats = fixed_batch_q_stats(agent, fixed_oracle_batch)
                row = {
                    "step": total_step, "update_count": agent.learner._update_count, "loss": update_info["loss"],
                    "td_error_mean_abs": update_info["td_error_mean_abs"], "td_error_max_abs": update_info["td_error_max_abs"],
                    "grad_norm": update_info["grad_norm"], "epsilon": eps, "lr": lr, **q_stats,
                }
                dense_log_file.write(json.dumps(row) + "\n")
                dense_log_file.flush()
        total_step += 1

        if episode_over:
            traces = env.episode_traces()
            utilities = episode_utilities(traces)
            burdens = episode_burdens(traces, dt=env.dt())
            window_stats.record_episode(term_reason=step_info["term_reason"], utilities=utilities, burdens=burdens)
            per_scenario_stats[current_scenario.scenario_id].record_episode(
                term_reason=step_info["term_reason"], utilities=utilities, burdens=burdens
            )
            current_scenario = pick_scenario()
            obs, _info = env.reset(seed=0, scenario=current_scenario)
            prev_active = {vid: True for vid in env.active_vehicle_ids}

        if total_step in checkpoint_targets:
            save_checkpoint(total_step)

    if dense_log_file is not None:
        dense_log_file.close()

    elapsed = time.time() - start_time
    manifest = {
        "stage": args.stage_name, "scenario_ids": args.scenario_ids, "master_seed": args.master_seed,
        "start_step": args.start_step, "final_step": total_step, "elapsed_seconds": elapsed,
        "checkpoint_steps": checkpoint_targets, "checkpoints": checkpoint_records,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"final_step": total_step, "elapsed_seconds": elapsed, "manifest": str(manifest_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
