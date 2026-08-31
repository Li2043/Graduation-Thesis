#!/usr/bin/env python3
"""Diagnostic_6_DQN_Pipeline_Verification_Protocol.md sec 16 (6L): the
decisive training diagnostic. A correct DQN implementation should be
able to overfit ONE fixed, oracle-solvable scenario much more easily
than the full scenario distribution. If the (now target-sync-FIXED,
see sec 12/6H) local shared DQN still cannot learn even this reduced
problem, the pipeline still has a fundamental issue (proceed to 6M);
if it CAN, the earlier multi-scenario failures point at
generalization/exploration/diversity, not basic DQN correctness.

Frozen for this diagnostic only: ONE scenario (same spawn/TTC/role-speed
assignment/physical IDs/jitter/dynamics every single episode -- scenario
randomization is disabled). Task-only reward (no direct welfare term,
matching sec 16.3's explicit "do not include direct welfare, do not add
a new time cost"). Same network/action-space/gamma/optimizer/replay
implementation as every other Study B local-DQN run (sec 16.4) -- this
script reuses ``SharedLocalDQNAgent``/``build_study_b_dqn_config``
unchanged, with the sec-12 target-sync fix already applied identically
to every other Study B training script.

Also folds in Diagnostic 6K's Q-value/TD-target numerical logging at
every checkpoint (mean/std/spread of online Q on a fixed diagnostic
batch drawn from this same scenario) -- cheap to add here since the
checkpoint files already exist; avoids a second training pass just for
numerical auditing.

Dense per-update logging (``--dense-log-every``, in UPDATES not
environment steps): added after the 400K extended run showed a
non-monotonic trajectory (completion collapsed 21.5%->6.5% around the
100K-200K window, coinciding with the CHECKPOINT-level mean_Q spiking to
5.657 -- 8-40x every other checkpoint's 0.02-0.7 range). That
per-checkpoint logging (every 10K-50K steps) is too coarse to see WHERE
inside that window the instability actually starts, or whether it
coincides with a loss/TD-error/gradient-norm spike. This logs loss,
mean/max |TD error|, gradient norm (all straight from
``SharedDQNLearner.update()``'s own per-update diagnostics -- see
``dqn_bootstrap``/``stage10_shared_dqn`` changes), plus mean_Q on a
FIXED (not re-rolled-out) diagnostic batch, every N updates -- cheap,
since it's just forward passes over an already-collected batch, not a
fresh environment rollout."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(REPO_SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from thesis.pilots.stage11_dyad_merge_pilot_config import TARGET_SYNC_INTERVAL_UPDATES  # noqa: E402
from thesis.study_b.heterogeneous_env import StudyBEnvConfig, StudyBHeterogeneousEnv  # noqa: E402
from thesis.study_b.oracle_controller import oracle_actions  # noqa: E402
from thesis.study_b.shared_local_dqn import SharedLocalDQNAgent, build_study_b_dqn_config, epsilon_at_step_v12, lr_at_step_v12  # noqa: E402
from thesis.study_b.training_common import StudyBEpisodeWindowStats, load_scenario_bank  # noqa: E402
from thesis.study_b.utility import episode_burdens, episode_utilities  # noqa: E402

__all__ = ["q_value_diagnostics", "collect_fixed_oracle_batch", "fixed_batch_q_stats"]


def q_value_diagnostics(agent: SharedLocalDQNAgent, observations: np.ndarray) -> dict:
    """6K: mean/std/spread of the online network's Q-values over a fixed
    batch of real observations (drawn from the frozen scenario)."""
    q_all = np.stack([agent.learner.q_values(obs, network="online") for obs in observations])
    spreads = q_all.max(axis=1) - q_all.min(axis=1)
    return {
        "mean_Q": float(q_all.mean()), "std_Q": float(q_all.std()), "mean_abs_Q": float(np.abs(q_all).mean()),
        "mean_Q_spread": float(spreads.mean()), "max_Q_spread": float(spreads.max()),
    }


def collect_fixed_oracle_batch(scenario, *, episode_max_steps: int) -> np.ndarray:
    """ONE-TIME, ORACLE-driven rollout of ``scenario`` (not the current
    policy's own rollout, which would change meaning as training
    progresses) -- a stable, physically-sensible, unchanging reference
    batch of states to track Q-value drift against across the whole run."""
    env = StudyBHeterogeneousEnv(StudyBEnvConfig(episode_max_steps=episode_max_steps, include_time_cost=False))
    obs, _info = env.reset(seed=0, scenario=scenario)
    rows = [arr for arr in obs.values()]
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
    p.add_argument("--scenario-id", type=str, default=None, help="defaults to the bank's first scenario")
    p.add_argument("--master-seed", type=int, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=200_000)
    p.add_argument("--episode-max-steps", type=int, default=200)
    p.add_argument("--checkpoint-steps", type=int, nargs="+", default=[10_000, 25_000, 50_000, 100_000, 200_000])
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--replay-warmup", type=int, default=512)
    p.add_argument(
        "--dense-log-every", type=int, default=0,
        help="If >0, log loss/TD-error/grad-norm/fixed-batch-Q every N UPDATES (not steps) "
             "to <output-root>/seed_<seed>_dense_log.jsonl. 0 = disabled (default).",
    )
    args = p.parse_args(argv)

    scenarios = load_scenario_bank(args.scenario_bank)
    scenario = scenarios[0] if args.scenario_id is None else next(s for s in scenarios if s.scenario_id == args.scenario_id)

    env = StudyBHeterogeneousEnv(StudyBEnvConfig(episode_max_steps=args.episode_max_steps, include_time_cost=False))
    # SEPARATE env instance for 6K's Q-diagnostic rollouts (save_checkpoint
    # below) -- must never share state with the training loop's own `env`,
    # since collecting a full diagnostic rollout mid-episode would otherwise
    # leave the training env terminated out from under the main loop.
    diag_env = StudyBHeterogeneousEnv(StudyBEnvConfig(episode_max_steps=args.episode_max_steps, include_time_cost=False))
    dqn_config = build_study_b_dqn_config(reward_condition="baseline", device=args.device)
    agent = SharedLocalDQNAgent(dqn_config, seed=args.master_seed)
    window_stats = StudyBEpisodeWindowStats()

    eps_decay_steps = max(1, round(0.8 * args.max_steps))
    lr_decay_steps = max(1, args.max_steps)

    output_root = Path(args.output_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = checkpoint_root / f"seed_{args.master_seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_targets = sorted(set(args.checkpoint_steps) | {args.max_steps, 0})
    checkpoint_records: list[dict] = []
    manifest_path = output_root / f"seed_{args.master_seed}_single_scenario_manifest.json"

    dense_log_file = None
    fixed_oracle_batch = None
    if args.dense_log_every > 0:
        fixed_oracle_batch = collect_fixed_oracle_batch(scenario, episode_max_steps=args.episode_max_steps)
        dense_log_path = output_root / f"seed_{args.master_seed}_dense_log.jsonl"
        dense_log_file = open(dense_log_path, "w", encoding="utf-8")

    # Fixed diagnostic batch for 6K's Q-value logging: every local
    # observation seen during a single fresh deterministic rollout of
    # THIS scenario, greedy at eps=0 using the CURRENT checkpoint --
    # collected once per checkpoint save (below).
    def collect_diagnostic_observations() -> np.ndarray:
        obs, _info = diag_env.reset(seed=0, scenario=scenario)
        rows = [arr for arr in obs.values()]
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
                "step": step, "diagnostic": "single_scenario_overfit", "scenario_id": scenario.scenario_id,
                "online": agent.learner.online.state_dict(), "target": agent.learner.target.state_dict(),
                "optimiser": agent.learner.optimiser.state_dict(), "update_count": agent.learner._update_count,
                "replay_size": len(agent.learner.replay),
            },
            checkpoint_dir / f"ckpt_step_{step}.pt",
        )
        q_diag = q_value_diagnostics(agent, collect_diagnostic_observations())
        metrics = {"step": step, "window": window_stats.as_dict(), "q_diagnostics": q_diag}
        window_stats.reset()
        checkpoint_records.append(metrics)
        print(f"step={step:>7}  completion={metrics['window']['completion_rate']:.3f}  "
              f"collision={metrics['window']['collision_rate']:.3f}  "
              f"mean_Q={q_diag['mean_Q']:.4f}  mean_Q_spread={q_diag['mean_Q_spread']:.4f}")

    seed = args.master_seed * 1_000_003
    obs, _info = env.reset(seed=0, scenario=scenario)
    prev_active = {vid: True for vid in env.active_vehicle_ids}

    total_step = 0
    if 0 in checkpoint_targets:
        save_checkpoint(0)

    start_time = time.time()
    while total_step < args.max_steps:
        eps = epsilon_at_step_v12(total_step, decay_steps=eps_decay_steps)
        lr = lr_at_step_v12(total_step, decay_steps=lr_decay_steps)
        agent.set_learning_rate(lr)

        actions = agent.select_actions(obs, epsilon=eps)
        prev_obs = obs
        obs, base_reward, terminated, truncated, step_info = env.step(actions)
        episode_over = terminated or truncated

        for vid in env.active_vehicle_ids:
            if not prev_active[vid]:
                continue
            exit_this_step = step_info["exit_event"][vid]
            controller_terminal = bool(terminated or exit_this_step)
            learner_completed = bool(exit_this_step and not step_info["collision_event"])
            transition = agent.build_transition(
                vehicle_id=vid, observation=prev_obs[vid], action=actions[vid],
                shaped_reward=base_reward[vid], next_observation=obs[vid],
                terminated=terminated, truncated=truncated,
                controller_terminal=controller_terminal, learner_completed=learner_completed,
                base_reward=base_reward[vid], shaping_component=0.0,
                episode_id=f"seed_{seed}_single_scenario", step=total_step,
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
                    "step": total_step, "update_count": agent.learner._update_count,
                    "loss": update_info["loss"], "td_error_mean_abs": update_info["td_error_mean_abs"],
                    "td_error_max_abs": update_info["td_error_max_abs"], "grad_norm": update_info["grad_norm"],
                    "epsilon": eps, "lr": lr, **q_stats,
                }
                dense_log_file.write(json.dumps(row) + "\n")
                dense_log_file.flush()
        total_step += 1

        if episode_over:
            traces = env.episode_traces()
            utilities = episode_utilities(traces)
            burdens = episode_burdens(traces, dt=env.dt())
            window_stats.record_episode(term_reason=step_info["term_reason"], utilities=utilities, burdens=burdens)
            seed = args.master_seed * 1_000_003 + total_step
            obs, _info = env.reset(seed=0, scenario=scenario)  # SAME scenario every episode
            prev_active = {vid: True for vid in env.active_vehicle_ids}

        if total_step in checkpoint_targets:
            save_checkpoint(total_step)

    if dense_log_file is not None:
        dense_log_file.close()

    elapsed = time.time() - start_time
    manifest = {
        "stage": "study_b_single_scenario_overfit", "scenario_id": scenario.scenario_id,
        "master_seed": args.master_seed, "final_step": total_step, "elapsed_seconds": elapsed,
        "checkpoint_steps": checkpoint_targets, "checkpoints": checkpoint_records,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"final_step": total_step, "elapsed_seconds": elapsed, "manifest": str(manifest_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
