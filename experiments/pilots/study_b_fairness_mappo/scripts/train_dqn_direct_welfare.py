#!/usr/bin/env python3
"""Study B shared-local-DQN training entrypoint, DIRECT WELFARE reward --
revised_new_research_plan.md / change.md's FINAL formal design (supersedes
``train_dqn_fallback.py``'s PBRS mechanism for formal Mean/GGI/Maximin
training; that script and ``pbrs_reward.py`` are left untouched and remain
valid for interpreting the earlier PBRS/baseline solver-diagnostic run).

Reward: non-terminal transitions use the base task reward unchanged; the
LAST transition of each episode (whichever step actually has
``terminated or truncated`` True) additionally receives
``welfare_reward.terminal_welfare_bonus(condition, episode_utilities)``,
computed ONCE per episode from ALL agents' full-episode utility and added
IDENTICALLY to every agent that is still receiving a transition at that
step (see ``welfare_reward.py``'s module docstring for why this is not
double/quadruple-counting).

Known simplification (documented, not a bug): if an agent individually
completes its own route well before the overall episode ends (the other
agents still active), that agent's own last stored transition happens at
ITS exit step -- BEFORE the team's full-episode utility is knowable -- so
it does NOT receive the terminal welfare bonus on that transition; only
agents still active at the step the WHOLE episode ends do. Retroactively
patching an already-stored early transition would require buffering an
entire episode's transitions before pushing any to replay, which this
script does not do. This does not affect the welfare CALCULATION itself
(``episode_utilities`` still correctly uses every agent's full trace
regardless of when they individually finished) -- only which specific
replay transitions carry the terminal bonus term.

Time cost: OFF by default (``--include-time-cost`` to restore it) --
revised_new_research_plan.md sec 12 drops the ``-0.0005`` active-step
cost; change.md's protocol-amendment fallback (sec 4 / "情况C") allows
reinstating it if Mean-direct-welfare qualification fails specifically on
timeout, which is exactly what this flag is for."""

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
from thesis.study_b.shared_local_dqn import (  # noqa: E402
    SharedLocalDQNAgent,
    build_study_b_dqn_config,
    epsilon_at_step_v12,
    lr_at_step_v12,
)
from thesis.study_b.training_common import StudyBEpisodeWindowStats  # noqa: E402
from thesis.study_b.utility import episode_burdens, episode_utilities  # noqa: E402
from thesis.study_b.welfare_reward import WELFARE_LAMBDA, condition_by_name, terminal_welfare_bonus  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True, choices=["mean", "ggi", "maximin"])
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
        "--include-time-cost", action="store_true",
        help="Protocol-amendment fallback (change.md sec 4/情况C): restore the -0.0005 active-step "
             "cost if direct-welfare Mean qualification fails specifically on timeout. Off by default.",
    )
    p.add_argument(
        "--resume-from", type=Path, default=None,
        help="Path to a ckpt_step_<N>.pt to WARM-START from (replay buffer is NOT restored).",
    )
    p.add_argument(
        "--welfare-lambda", type=float, default=WELFARE_LAMBDA,
        help="Overrides welfare_reward.WELFARE_LAMBDA for this run only. "
             "--welfare-lambda 0.0 reduces the reward to pure task reward (r_t = r_t^task for all t) -- "
             "VDN_Conditional_Amendment_Protocol.md sec 9's 'task-only local-DQN' diagnostic. "
             "Default (1.0) reproduces the formal direct-welfare mechanism exactly.",
    )
    args = p.parse_args(argv)

    condition = condition_by_name(args.condition)
    env_config = StudyBEnvConfig(
        episode_max_steps=args.episode_max_steps,
        heterogeneous_probability=args.heterogeneous_probability,
        r_obs=args.r_obs,
        include_time_cost=args.include_time_cost,
    )
    env = StudyBHeterogeneousEnv(env_config)
    # reward_condition is a bookkeeping-only string on DQNConfig, shared
    # with Study A's own frozen-protocol Literal type -- pass "baseline"
    # (always valid there) and record the REAL condition name ourselves,
    # in this script's own manifest/transition episode_id, instead of
    # widening a type shared with Study A's protocol-lock files.
    dqn_config = build_study_b_dqn_config(reward_condition="baseline", device=args.device)
    agent = SharedLocalDQNAgent(dqn_config, seed=args.master_seed)
    window_stats = StudyBEpisodeWindowStats()

    resumed_step = 0
    if args.resume_from is not None:
        ckpt = torch.load(args.resume_from, map_location="cpu")
        mismatches = []
        if ckpt.get("welfare_condition") != condition.name:
            mismatches.append(f"condition: checkpoint={ckpt.get('welfare_condition')!r} vs this run={condition.name!r}")
        if ckpt.get("obs_dim") != dqn_config.obs_dim:
            mismatches.append(f"obs_dim: checkpoint={ckpt.get('obs_dim')} vs this run={dqn_config.obs_dim}")
        if ckpt.get("welfare_lambda", WELFARE_LAMBDA) != args.welfare_lambda:
            mismatches.append(f"welfare_lambda: checkpoint={ckpt.get('welfare_lambda', WELFARE_LAMBDA)} vs this run={args.welfare_lambda}")
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
                "welfare_condition": condition.name,
                "obs_dim": dqn_config.obs_dim,
                "include_time_cost": args.include_time_cost,
                "welfare_lambda": args.welfare_lambda,
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
        episode_over = terminated or truncated

        # Direct welfare bonus: computed ONCE per episode, only at the
        # step the WHOLE episode ends (not per-vehicle, not per individual
        # exit) -- see this module's docstring for the early-exit
        # simplification this implies.
        welfare_bonus = 0.0
        if episode_over:
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
                episode_id=f"seed_{seed}_{condition.name}", step=total_step,
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
            prev_active = {vid: True for vid in env.active_vehicle_ids}

        if total_step in checkpoint_targets:
            save_checkpoint(total_step)

    elapsed = time.time() - start_time
    manifest = {
        "stage": "study_b_dqn_direct_welfare",
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
