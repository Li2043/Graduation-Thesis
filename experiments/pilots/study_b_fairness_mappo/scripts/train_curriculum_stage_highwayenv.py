#!/usr/bin/env python3
"""M6 (runbook sec 32-36) and later C1->C64 curriculum training
(sec 37-42), HighwayEnv backend.

Directly adapted from `train_curriculum_stage.py` (the legacy-backend
script, unmodified, kept as-is for diagnostic/parity comparison) --
same algorithm, same absolute-step schedule discipline, same
target-sync-every-250-updates fix (re-applied explicitly here, since it
is a caller-side pattern that does not carry over automatically to a new
training script -- runbook sec 30's own M6-R1 checklist item). The ONLY
differences: uses `StudyBHeterogeneousHighwayEnv` instead of the legacy
env, and the oracle-driven reference-batch collector reads real
HighwayEnv world-x (via `env._env.world_xy`) instead of the legacy
scalar `route_position`, with merge_start/merge_end taken from
`ThesisHighwayMergeEnvConfig` instead of being hardcoded to the legacy
200.0/300.0."""

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

from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig  # noqa: E402
from thesis.study_b.envs.highwayenv_wrapper import (  # noqa: E402
    StudyBHeterogeneousHighwayEnv,
    StudyBHighwayWrapperConfig,
)
from thesis.study_b.oracle_controller import oracle_actions  # noqa: E402
from thesis.study_b.shared_local_dqn import SharedLocalDQNAgent, build_study_b_dqn_config, epsilon_at_step_v12, lr_at_step_v12  # noqa: E402
from thesis.study_b.training_common import StudyBEpisodeWindowStats, load_scenario_bank  # noqa: E402
from thesis.study_b.utility import episode_burdens, episode_utilities, running_active_attainment  # noqa: E402
from thesis.study_b.welfare_reward import condition_by_name, terminal_welfare_bonus  # noqa: E402
from thesis.study_b.dense_shaping import (  # noqa: E402
    NEUTRAL_PHI,
    DenseShapingConfig,
    dense_shaping_term,
    welfare_objective_snapshot,
)

# DENSE-REWARD COPY (2026-08-26), Section 12 scaffold: this file is otherwise
# an unmodified copy of the formal script. The --debug-reward-trace /
# --debug-reward-trace-episodes machinery is a default-off debug log
# distinct from the existing --dense-log-every Q-diagnostics log; when
# --debug-reward-trace is not passed (the default), that code path is a
# no-op.
#
# DENSE-REWARD IMPLEMENTATION (added after environment-setup review, still
# default-off): --dense-welfare-shaping wires in thesis.study_b.dense_shaping
# (see that module's docstring for the full design record and the
# active-set/exit-artifact guarantee). When --dense-welfare-shaping is not
# passed (the default, DenseShapingConfig(enabled=False)), dense_shaping_term
# always returns 0.0 and behaviour is byte-identical to the original script
# -- no new welfare construct, no change to base_reward, terminal_component,
# observation construction, or control flow. c (magnitude) and epsilon are
# never defaulted -- see dense_shaping.DenseShapingConfig.__post_init__.

__all__ = ["collect_fixed_oracle_batch_multi", "fixed_batch_q_stats", "TARGET_SYNC_INTERVAL_UPDATES"]

TARGET_SYNC_INTERVAL_UPDATES = 250  # frozen -- see ABSOLUTE_TRAINING_SCHEDULE.json

_ENV_CONFIG = ThesisHighwayMergeEnvConfig()
_MERGE_START = _ENV_CONFIG.before_merge_length
_MERGE_END = _ENV_CONFIG.before_merge_length + _ENV_CONFIG.converge_merge_length


def _make_env(
    episode_max_steps: int, *, action_representation: str = "meta_speed",
    local_sensing_range_m: float | None = None,
) -> StudyBHeterogeneousHighwayEnv:
    cfg = ThesisHighwayMergeEnvConfig(episode_max_steps=episode_max_steps, action_representation=action_representation)
    return StudyBHeterogeneousHighwayEnv(
        StudyBHighwayWrapperConfig(env_config=cfg, local_sensing_range_m=local_sensing_range_m)
    )


def collect_fixed_oracle_batch_multi(
    scenarios: list, *, episode_max_steps: int, action_representation: str = "meta_speed",
    local_sensing_range_m: float | None = None,
) -> np.ndarray:
    env = _make_env(episode_max_steps, action_representation=action_representation, local_sensing_range_m=local_sensing_range_m)
    rows = []
    for scenario in scenarios:
        obs, _info = env.reset(seed=0, scenario=scenario)
        rows.extend(obs.values())
        for _t in range(episode_max_steps):
            positions = {
                vid: env._env.world_xy(env._env._vehicle_by_id[vid])[0]  # noqa: SLF001
                for vid in env.active_vehicle_ids
            }
            actions = oracle_actions(
                scenario=scenario, positions=positions, merge_start=_MERGE_START, merge_end=_MERGE_END,
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
    p.add_argument("--scenario-ids", type=str, nargs="+", required=True)
    p.add_argument("--stage-name", type=str, required=True)
    p.add_argument("--master-seed", type=int, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--start-step", type=int, default=0)
    p.add_argument("--max-additional-steps", type=int, required=True)
    p.add_argument("--episode-max-steps", type=int, default=200)
    p.add_argument("--checkpoint-every", type=int, default=25_000)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--replay-warmup", type=int, default=512)
    p.add_argument("--eps-decay-steps-absolute", type=int, default=640_000)
    p.add_argument("--lr-decay-steps-absolute", type=int, default=800_000)
    p.add_argument("--welfare-lambda", type=float, default=0.0)
    p.add_argument("--condition", type=str, default="mean", choices=["mean", "ggi", "maximin"])
    p.add_argument("--resume-from", type=Path, default=None)
    p.add_argument("--dense-log-every", type=int, default=0)
    p.add_argument(
        "--action-representation", type=str, default="meta_speed",
        choices=["direct_accel", "meta_speed"],
        help="Accepted representation is meta_speed (Amendment 4, CONTROL_AUTHORITY_MISMATCH-fixed) -- "
             "direct_accel remains available for diagnostic comparison only, see runbook sec 36 / M6-R3.",
    )
    p.add_argument("--local-sensing-range-m", type=float, default=None,
                    help="LOCALITY AMENDMENT (2026-08-17): finite neighbour-visibility range in metres. "
                         "Default None preserves pre-amendment behaviour (unbounded nearest-3 selection) exactly.")
    p.add_argument("--debug-reward-trace", action="store_true", default=False,
                    help="Dense-reward-study scaffold (default OFF, no effect on training/reward/observation "
                         "when unset): for the first --debug-reward-trace-episodes episodes only, write one JSON "
                         "line per (step, agent) to seed_{seed}_{stage}_reward_trace.jsonl with step, agent_id, "
                         "base_reward, terminal_component, current M_i (running_active_attainment), the welfare "
                         "objective value evaluated over all vehicles' current M_i, done, collision, completion. "
                         "Does not compute any dense shaping term.")
    p.add_argument("--debug-reward-trace-episodes", type=int, default=5,
                    help="Number of episodes to log when --debug-reward-trace is set (default 5). Ignored otherwise.")
    p.add_argument("--dense-welfare-shaping", action="store_true", default=False,
                    help="Dense Reward Study (default OFF, byte-identical behaviour when unset): add a "
                         "discrete +/-c step-wise shaping term based on DeltaPhi_t, the step-to-step change "
                         "in condition.welfare_fn([M_i(t)]) over all vehicles. See thesis.study_b.dense_shaping "
                         "for the full design record. Requires --dense-shaping-magnitude and "
                         "--dense-shaping-epsilon to be set explicitly -- there is no default value for either.")
    p.add_argument("--dense-shaping-mode", type=str, default="discrete", choices=["discrete"],
                    help="Only 'discrete' is implemented (matches configs/FROZEN_EXPERIMENT_CONFIG.json's "
                         "dense_reward_study_reserved.dense_shaping_mode). A continuous variant is Priority 7 "
                         "in README.md and is explicitly lower priority than the discrete conditions.")
    p.add_argument("--dense-shaping-magnitude", type=float, default=None,
                    help="c: the fixed shaping magnitude added/subtracted when |DeltaPhi_t| exceeds epsilon. "
                         "Must be pre-frozen before formal training -- never chosen by observing results.")
    p.add_argument("--dense-shaping-epsilon", type=float, default=None,
                    help="epsilon: the dead-zone threshold below which DeltaPhi_t produces no shaping. "
                         "Must be pre-frozen before formal training -- never chosen by observing results.")
    args = p.parse_args(argv)

    # Fails fast, before any environment/agent is constructed, if
    # --dense-welfare-shaping is passed without both magnitude and epsilon
    # -- see DenseShapingConfig.__post_init__.
    dense_cfg = DenseShapingConfig(
        enabled=args.dense_welfare_shaping, mode=args.dense_shaping_mode,
        magnitude=args.dense_shaping_magnitude, epsilon=args.dense_shaping_epsilon,
    )

    all_scenarios = load_scenario_bank(args.scenario_bank)
    by_id = {s.scenario_id: s for s in all_scenarios}
    stage_scenarios = [by_id[sid] for sid in args.scenario_ids]
    condition = condition_by_name(args.condition)

    env = _make_env(args.episode_max_steps, action_representation=args.action_representation, local_sensing_range_m=args.local_sensing_range_m)
    diag_env = _make_env(args.episode_max_steps, action_representation=args.action_representation, local_sensing_range_m=args.local_sensing_range_m)
    dqn_config = build_study_b_dqn_config(reward_condition="baseline", device=args.device)
    agent = SharedLocalDQNAgent(dqn_config, seed=args.master_seed)
    window_stats = StudyBEpisodeWindowStats()
    per_scenario_stats: dict[str, StudyBEpisodeWindowStats] = defaultdict(StudyBEpisodeWindowStats)
    scenario_rng = np.random.default_rng(args.master_seed * 7919 + 1)

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
        fixed_oracle_batch = collect_fixed_oracle_batch_multi(stage_scenarios, episode_max_steps=args.episode_max_steps, action_representation=args.action_representation, local_sensing_range_m=args.local_sensing_range_m)
        dense_log_file = open(output_root / f"seed_{args.master_seed}_{args.stage_name}_dense_log.jsonl", "a", encoding="utf-8")
    fixed_q_batch_for_checkpoints = fixed_oracle_batch if fixed_oracle_batch is not None else \
        collect_fixed_oracle_batch_multi(stage_scenarios, episode_max_steps=args.episode_max_steps, action_representation=args.action_representation, local_sensing_range_m=args.local_sensing_range_m)

    # DENSE-REWARD COPY Section 12 scaffold -- see module-level note above.
    # Default-off; reward_trace_file stays None and reward_trace_episode_count
    # is never read when --debug-reward-trace is not passed.
    reward_trace_file = None
    reward_trace_episode_count = 0
    if args.debug_reward_trace:
        reward_trace_file = open(
            output_root / f"seed_{args.master_seed}_{args.stage_name}_reward_trace.jsonl", "a", encoding="utf-8"
        )

    def collect_policy_visited_observations() -> np.ndarray:
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
    phi_prev = NEUTRAL_PHI  # Phi_0: all traces empty -> running_active_attainment=1.0 for every vehicle

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

        # Compute the current-M_i welfare-objective snapshot ONCE per step,
        # whenever EITHER dense shaping OR tracing needs it (avoids a
        # redundant env.episode_traces() call when both are active; costs
        # nothing extra when neither is -- see dense_shaping.py's docstring
        # for why Phi is always evaluated over ALL vehicle_ids).
        trace_this_step = (
            reward_trace_file is not None and reward_trace_episode_count < args.debug_reward_trace_episodes
        )
        dense_term = 0.0
        delta_phi = None
        if dense_cfg.enabled or trace_this_step:
            live_traces = env.episode_traces()
            current_m = {vid: running_active_attainment(live_traces[vid]) for vid in env.active_vehicle_ids}
            welfare_objective_value = condition.welfare_fn(list(current_m.values()))
            if dense_cfg.enabled:
                phi_curr = welfare_objective_snapshot(live_traces, env.active_vehicle_ids, condition)
                delta_phi = phi_curr - phi_prev
                dense_term = dense_shaping_term(delta_phi, dense_cfg)
                phi_prev = phi_curr

        for vid in env.active_vehicle_ids:
            if not prev_active[vid]:
                continue
            exit_this_step = step_info["exit_event"][vid]
            controller_terminal = bool(terminated or exit_this_step)
            learner_completed = bool(exit_this_step and not step_info["collision_event"])
            shaping_this_step = (welfare_bonus if episode_over else 0.0) + dense_term
            shaped_reward = base_reward[vid] + shaping_this_step
            transition = agent.build_transition(
                vehicle_id=vid, observation=prev_obs[vid], action=actions[vid],
                shaped_reward=shaped_reward, next_observation=obs[vid],
                terminated=terminated, truncated=truncated,
                controller_terminal=controller_terminal, learner_completed=learner_completed,
                base_reward=base_reward[vid], shaping_component=shaping_this_step,
                episode_id=f"seed_{args.master_seed}_{args.stage_name}_{current_scenario.scenario_id}", step=total_step,
            )
            agent.store_transition(transition)

            if trace_this_step:
                reward_trace_file.write(json.dumps({
                    "step": total_step, "agent_id": vid,
                    "base_reward": base_reward[vid],
                    "terminal_component": (welfare_bonus if episode_over else 0.0),
                    "M_i": current_m[vid],
                    "welfare_objective_value": welfare_objective_value,
                    "delta_phi": delta_phi,
                    "dense_shaping_component": dense_term,
                    "done": bool(episode_over),
                    "collision": bool(step_info["collision_event"]),
                    "completion": learner_completed,
                }) + "\n")
        if trace_this_step:
            reward_trace_file.flush()
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
            if reward_trace_file is not None:
                reward_trace_episode_count += 1
            current_scenario = pick_scenario()
            obs, _info = env.reset(seed=0, scenario=current_scenario)
            prev_active = {vid: True for vid in env.active_vehicle_ids}
            phi_prev = NEUTRAL_PHI  # Phi_0: all traces empty -> running_active_attainment=1.0 for every vehicle

        if total_step in checkpoint_targets:
            save_checkpoint(total_step)

    if dense_log_file is not None:
        dense_log_file.close()
    if reward_trace_file is not None:
        reward_trace_file.close()

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
