#!/usr/bin/env python3
"""Study B MAPPO training entrypoint (Phase 1 qualification and Phase 2/3
formal runs all use this same script -- only ``--condition``/``--master-seed``/
``--max-steps`` change between them, per new_research_plan.md's frozen-config
discipline).

"Parallel envs" here means ``--n-parallel-envs`` INDEPENDENT
``StudyBHeterogeneousEnv`` instances stepped in a synchronous Python loop
each rollout tick (not OS-level subprocess/multiprocess vectorization) --
correct and simple to reason about, at some wall-clock throughput cost
relative to true parallelism. If training speed becomes the bottleneck on
the actual experiment machine, replacing this loop with a multiprocess
VecEnv is the natural follow-up optimization; it does not change any of
the math in ``mappo.py``/``rollout_buffer.py``.

**Resuming across machines** (e.g. start on a CPU-only machine, copy the
checkpoint to a GPU machine and continue): pass ``--resume-from`` pointing
at a ``ckpt_step_<N>.pt`` file. This restores actor/critic/optimiser/
value-normalizer state and continues the step counter from ``N`` --
``--max-steps`` is still the TOTAL budget for the run (not an additional
amount), and every other flag (``--condition``, ``--hidden-size``, etc.)
must match the original run exactly; a mismatch in anything that changes
network shape is caught and rejected before touching torch, everything
else is trusted to the caller (same discipline as this project's frozen-
config convention elsewhere -- the burden is on not silently changing a
frozen config, not on this script re-deriving one). What does NOT survive
a resume: the on-policy rollout buffer (irrelevant -- it's discarded after
every PPO update even within a single uninterrupted run) and each
parallel env's mid-episode state (every env just starts a fresh episode
after resume -- harmless, this is exactly what happens after every
ordinary PPO update too, just with a discontinuity in which seeds get
used). This is a genuine, correct resume of the LEARNED POLICY, not a
byte-identical continuation of one specific run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(REPO_SRC))

import torch  # noqa: E402

from thesis.study_b.heterogeneous_env import StudyBEnvConfig, StudyBHeterogeneousEnv  # noqa: E402
from thesis.study_b.mappo import MAPPOConfig, MAPPOLearner  # noqa: E402
from thesis.study_b.pbrs_reward import (  # noqa: E402
    PBRSRewardShaper,
    condition_by_name,
    experiences_from_step_info,
)
from thesis.study_b.rollout_buffer import RolloutBuffer  # noqa: E402
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
    p.add_argument("--n-parallel-envs", type=int, default=16)
    p.add_argument("--rollout-length", type=int, default=200, help="per-env steps collected before each PPO update")
    p.add_argument("--episode-max-steps", type=int, default=200)
    p.add_argument("--checkpoint-every", type=int, default=50_000)
    p.add_argument("--heterogeneous-probability", type=float, default=0.5)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--actor-lr", type=float, default=5e-4)
    p.add_argument("--critic-lr", type=float, default=5e-4)
    p.add_argument("--clip-epsilon", type=float, default=0.10)
    p.add_argument("--ppo-epochs", type=int, default=5)
    p.add_argument("--minibatches", type=int, default=1)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--r-obs", type=float, default=50.0)
    p.add_argument(
        "--resume-from", type=Path, default=None,
        help="Path to a ckpt_step_<N>.pt to resume from (see this script's module docstring).",
    )
    args = p.parse_args(argv)

    condition = condition_by_name(args.condition)
    env_config = StudyBEnvConfig(
        episode_max_steps=args.episode_max_steps,
        heterogeneous_probability=args.heterogeneous_probability,
        r_obs=args.r_obs,
    )
    n_envs = args.n_parallel_envs
    envs = [StudyBHeterogeneousEnv(env_config) for _ in range(n_envs)]
    shapers = [PBRSRewardShaper(condition) for _ in range(n_envs)]
    window_stats = StudyBEpisodeWindowStats()

    mappo_config = MAPPOConfig(
        obs_dim=envs[0].observation_dim,
        global_state_dim=envs[0].global_state_dim,
        hidden_sizes=(args.hidden_size, args.hidden_size),
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        clip_epsilon=args.clip_epsilon,
        ppo_epochs=args.ppo_epochs,
        minibatches=args.minibatches,
        entropy_coef=args.entropy_coef,
        device=args.device,
    )
    learner = MAPPOLearner(mappo_config, seed=args.master_seed)

    resumed_step = 0
    if args.resume_from is not None:
        payload = torch.load(args.resume_from, map_location="cpu")
        mismatches = []
        if payload["condition"] != condition.name:
            mismatches.append(f"condition: checkpoint={payload['condition']!r} vs this run={condition.name!r}")
        if payload["obs_dim"] != mappo_config.obs_dim:
            mismatches.append(f"obs_dim: checkpoint={payload['obs_dim']} vs this run={mappo_config.obs_dim}")
        if payload["global_state_dim"] != mappo_config.global_state_dim:
            mismatches.append(
                f"global_state_dim: checkpoint={payload['global_state_dim']} vs this run={mappo_config.global_state_dim}"
            )
        if tuple(payload["hidden_sizes"]) != tuple(mappo_config.hidden_sizes):
            mismatches.append(f"hidden_sizes: checkpoint={payload['hidden_sizes']} vs this run={mappo_config.hidden_sizes}")
        if mismatches:
            raise ValueError(
                "--resume-from checkpoint is incompatible with this run's config:\n  "
                + "\n  ".join(mismatches)
                + "\nEvery flag that determines network shape or training condition must match the original run exactly."
            )
        learner.load_state_dict(payload["learner_state"])
        resumed_step = int(payload["step"])
        print(f"resumed from {args.resume_from} at step {resumed_step}")

    obs_list = []
    for i, env in enumerate(envs):
        seed_i = args.master_seed * 1_000_003 + i
        obs, _info = env.reset(seed=seed_i)
        shapers[i].reset(experiences=_initial_experiences(env))
        obs_list.append(obs)
    # Offset by resumed_step (not exact, but keeps post-resume episode
    # seeds from immediately repeating seeds already used pre-resume in
    # the common case) -- next fresh seed = master_seed*1_000_003 + next_seed_offset
    next_seed_offset = n_envs + resumed_step

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
        # Carry the pre-resume checkpoint history forward so the final
        # manifest is one continuous record, not just the post-resume tail.
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint_records = [rec for rec in prior.get("checkpoints", []) if rec["step"] <= resumed_step]

    def save_checkpoint(step: int) -> None:
        payload = {
            "step": step,
            "condition": condition.name,
            "obs_dim": mappo_config.obs_dim,
            "global_state_dim": mappo_config.global_state_dim,
            "hidden_sizes": list(mappo_config.hidden_sizes),
            "learner_state": learner.state_dict(),
        }
        torch.save(payload, checkpoint_dir / f"ckpt_step_{step}.pt")
        metrics = {"step": step, "window": window_stats.as_dict()}
        window_stats.reset()
        checkpoint_records.append(metrics)

    total_step = resumed_step
    if total_step == 0 and 0 in checkpoint_targets:
        save_checkpoint(0)

    start_time = time.time()
    while total_step < args.max_steps:
        buffers = [RolloutBuffer(agent_ids=tuple(envs[i].active_vehicle_ids)) for i in range(n_envs)]

        for _t in range(args.rollout_length):
            if total_step >= args.max_steps:
                break
            for i, env in enumerate(envs):
                actions, log_probs = learner.select_actions(obs_list[i])
                gstate = env.global_state()
                value = learner.compute_value(gstate)
                next_obs, base_reward, terminated, truncated, step_info = env.step(actions)
                exps = experiences_from_step_info(step_info["attainments"], step_info["active"])
                shaping = shapers[i].step(experiences_next=exps, terminated=terminated, truncated=truncated)
                team_reward = shapers[i].apply_team(base_reward, shaping)
                done = terminated or truncated

                buffers[i].add(
                    obs=obs_list[i], global_state=gstate, actions=actions, log_probs=log_probs,
                    team_reward=team_reward, value=value, done=done,
                )
                obs_list[i] = next_obs
                total_step += 1

                if done:
                    traces = env.episode_traces()
                    utilities = episode_utilities(traces)
                    burdens = episode_burdens(traces, dt=env.dt())
                    window_stats.record_episode(
                        term_reason=step_info["term_reason"], utilities=utilities, burdens=burdens
                    )
                    seed_i = args.master_seed * 1_000_003 + next_seed_offset
                    next_seed_offset += 1
                    obs_list[i], _info = env.reset(seed=seed_i)
                    shapers[i] = PBRSRewardShaper(condition)
                    shapers[i].reset(experiences=_initial_experiences(env))

                if total_step in checkpoint_targets:
                    save_checkpoint(total_step)
                if total_step >= args.max_steps:
                    break

        buffer_last_value_pairs = []
        for i, buffer in enumerate(buffers):
            if len(buffer) == 0:
                continue
            # Bootstrap from the CURRENT (post-rollout) state of that env --
            # correct whether or not this env's stream ended mid-rollout on
            # a done (compute_gae's own done-masking handles that case).
            last_value = learner.compute_value(envs[i].global_state())
            buffer_last_value_pairs.append((buffer, last_value))
        if buffer_last_value_pairs:
            learner.update(buffer_last_value_pairs, last_value=None)

    elapsed = time.time() - start_time
    manifest = {
        "stage": "study_b_mappo",
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
