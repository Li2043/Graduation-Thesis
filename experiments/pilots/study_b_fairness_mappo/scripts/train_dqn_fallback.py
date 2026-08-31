#!/usr/bin/env python3
"""Study B shared-local-DQN training entrypoint -- ONLY used if MAPPO's
48-hour integration gate is not met (new_research_plan.md). Same
CLI/manifest/checkpoint shape as ``train_mappo.py`` so downstream analysis
code does not need to special-case which algorithm produced a given run's
artifacts.

**Resuming across machines** (``--resume-from``): unlike ``train_mappo.py``,
this is a WARM START, not a full resume -- checkpoints save the online/
target networks and optimiser state (so the policy genuinely continues
from where it left off), but NOT the replay buffer contents (only its
size was ever recorded, matching this project's existing DQN checkpoint
convention elsewhere -- see E33's own stage11_dyad_merge_runner.py). After
resuming, the replay buffer starts EMPTY and must refill past
``--replay-warmup`` again before updates resume, exactly like the start of
a fresh run. The epsilon/learning-rate schedules do NOT reset, though --
both are pure functions of the (correctly continued) step counter, so
exploration/LR pick up from the correct point in the decay curve
immediately, even while replay is still refilling."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(REPO_SRC))

import torch  # noqa: E402

from thesis.pilots.stage11_dyad_merge_pilot_config import TARGET_SYNC_INTERVAL_UPDATES  # noqa: E402
from thesis.study_b.heterogeneous_env import StudyBEnvConfig, StudyBHeterogeneousEnv  # noqa: E402
from thesis.study_b.pbrs_reward import (  # noqa: E402
    PBRSRewardShaper,
    condition_by_name,
    experiences_from_step_info,
)
from thesis.study_b.shared_local_dqn import (  # noqa: E402
    SharedLocalDQNAgent,
    build_study_b_dqn_config,
    epsilon_at_step_v12,
    lr_at_step_v12,
)
from thesis.study_b.training_common import StudyBEpisodeWindowStats  # noqa: E402
from thesis.study_b.utility import episode_burdens, episode_utilities  # noqa: E402


def _initial_experiences(env: StudyBHeterogeneousEnv) -> list[float]:
    attain = {vid: 1.0 for vid in env.active_vehicle_ids}
    active = {vid: True for vid in env.active_vehicle_ids}
    return experiences_from_step_info(attain, active)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True, choices=["baseline", "mean_pbrs", "min_pbrs"])
    p.add_argument("--master-seed", type=int, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=1_000_000)
    p.add_argument("--episode-max-steps", type=int, default=200)
    p.add_argument("--checkpoint-every", type=int, default=50_000)
    p.add_argument("--heterogeneous-probability", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--r-obs", type=float, default=50.0)
    p.add_argument("--replay-warmup", type=int, default=512)
    p.add_argument(
        "--resume-from", type=Path, default=None,
        help="Path to a ckpt_step_<N>.pt to WARM-START from (see this script's module docstring -- replay buffer is NOT restored).",
    )
    args = p.parse_args(argv)

    condition = condition_by_name(args.condition)
    env_config = StudyBEnvConfig(
        episode_max_steps=args.episode_max_steps,
        heterogeneous_probability=args.heterogeneous_probability,
        r_obs=args.r_obs,
    )
    env = StudyBHeterogeneousEnv(env_config)
    dqn_config = build_study_b_dqn_config(reward_condition=condition.name, device=args.device)
    agent = SharedLocalDQNAgent(dqn_config, seed=args.master_seed)
    window_stats = StudyBEpisodeWindowStats()

    resumed_step = 0
    if args.resume_from is not None:
        ckpt = torch.load(args.resume_from, map_location="cpu")
        mismatches = []
        if ckpt.get("condition") != condition.name:
            mismatches.append(f"condition: checkpoint={ckpt.get('condition')!r} vs this run={condition.name!r}")
        if ckpt.get("obs_dim") != dqn_config.obs_dim:
            mismatches.append(f"obs_dim: checkpoint={ckpt.get('obs_dim')} vs this run={dqn_config.obs_dim}")
        if mismatches:
            raise ValueError(
                "--resume-from checkpoint is incompatible with this run's config:\n  "
                + "\n  ".join(mismatches)
                + "\nEvery flag that determines network shape or training condition must match the original run exactly."
            )
        agent.learner.online.load_state_dict(ckpt["online"])
        agent.learner.target.load_state_dict(ckpt["target"])
        agent.learner.optimiser.load_state_dict(ckpt["optimiser"])
        agent.learner._update_count = int(ckpt["update_count"])
        resumed_step = int(ckpt["step"])
        print(f"warm-started from {args.resume_from} at step {resumed_step} (replay buffer starts EMPTY)")

    # Same proportional-decay convention as this hub's own extended-
    # convergence DQN work (E33's stage11_dyad_merge_pilot_config.py):
    # epsilon decays over 80% of max_steps, LR over the full max_steps.
    eps_decay_steps = max(1, round(0.8 * args.max_steps))
    lr_decay_steps = max(1, args.max_steps)

    output_root = Path(args.output_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = checkpoint_root / f"seed_{args.master_seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_targets = sorted(set(range(0, args.max_steps + 1, args.checkpoint_every)) | {args.max_steps})
    checkpoint_records: list[dict] = []
    manifest_path = output_root / f"seed_{args.master_seed}_{condition.name}_manifest.json"
    if args.resume_from is not None and manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint_records = [rec for rec in prior.get("checkpoints", []) if rec["step"] <= resumed_step]

    def save_checkpoint(step: int) -> None:
        torch.save(
            {
                "step": step,
                "condition": condition.name,
                "obs_dim": dqn_config.obs_dim,
                "online": agent.learner.online.state_dict(),
                "target": agent.learner.target.state_dict(),
                "optimiser": agent.learner.optimiser.state_dict(),
                "update_count": agent.learner._update_count,
                "replay_size": len(agent.learner.replay),
            },
            checkpoint_dir / f"ckpt_step_{step}.pt",
        )
        metrics = {"step": step, "window": window_stats.as_dict()}
        window_stats.reset()
        checkpoint_records.append(metrics)

    seed = args.master_seed * 1_000_003 + resumed_step
    obs, _info = env.reset(seed=seed)
    shaper = PBRSRewardShaper(condition)
    shaper.reset(experiences=_initial_experiences(env))
    prev_active = {vid: True for vid in env.active_vehicle_ids}

    total_step = resumed_step
    if total_step == 0 and 0 in checkpoint_targets:
        save_checkpoint(0)

    start_time = time.time()
    while total_step < args.max_steps:
        eps = epsilon_at_step_v12(total_step, decay_steps=eps_decay_steps)
        lr = lr_at_step_v12(total_step, decay_steps=lr_decay_steps)
        agent.set_learning_rate(lr)

        actions = agent.select_actions(obs, epsilon=eps)
        prev_obs = obs
        obs, base_reward, terminated, truncated, step_info = env.step(actions)
        exps = experiences_from_step_info(step_info["attainments"], step_info["active"])
        shaping = shaper.step(experiences_next=exps, terminated=terminated, truncated=truncated)
        shaped = shaper.apply_per_vehicle(base_reward, shaping)
        episode_over = terminated or truncated

        for vid in env.active_vehicle_ids:
            if not prev_active[vid]:
                continue
            exit_this_step = step_info["exit_event"][vid]
            # Only a TRUE terminal (collision/joint success) or this
            # vehicle's own individual exit ends its controller-perspective
            # trajectory -- mere truncation (episode-length cutoff) must
            # NOT set controller_terminal for a still-active vehicle: it
            # should keep bootstrapping normally via next_observation, per
            # ReplayTransition's own documented semantics ("truncated
            # Bootstraps unless also controller_terminal").
            controller_terminal = bool(terminated or exit_this_step)
            learner_completed = bool(exit_this_step and not step_info["collision_event"])
            transition = agent.build_transition(
                vehicle_id=vid, observation=prev_obs[vid], action=actions[vid],
                shaped_reward=shaped[vid], next_observation=obs[vid],
                terminated=terminated, truncated=truncated,
                controller_terminal=controller_terminal, learner_completed=learner_completed,
                base_reward=base_reward[vid], shaping_component=shaping,
                episode_id=f"seed_{seed}", step=total_step,
            )
            agent.store_transition(transition)
        prev_active = dict(step_info["active"])

        if agent.maybe_update(warmup=args.replay_warmup) is not None:
            if agent.learner._update_count % TARGET_SYNC_INTERVAL_UPDATES == 0:
                agent.learner.hard_sync_target()
        total_step += 1

        if episode_over:
            traces = env.episode_traces()
            utilities = episode_utilities(traces)
            burdens = episode_burdens(traces, dt=env.dt())
            window_stats.record_episode(term_reason=step_info["term_reason"], utilities=utilities, burdens=burdens)
            seed = args.master_seed * 1_000_003 + total_step
            obs, _info = env.reset(seed=seed)
            shaper = PBRSRewardShaper(condition)
            shaper.reset(experiences=_initial_experiences(env))
            prev_active = {vid: True for vid in env.active_vehicle_ids}

        if total_step in checkpoint_targets:
            save_checkpoint(total_step)

    elapsed = time.time() - start_time
    manifest = {
        "stage": "study_b_dqn_fallback",
        "condition": condition.name,
        "master_seed": args.master_seed,
        "final_step": total_step,
        "elapsed_seconds": elapsed,
        "checkpoint_steps": checkpoint_targets,
        "checkpoints": checkpoint_records,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"final_step": total_step, "elapsed_seconds": elapsed, "manifest": str(manifest_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
