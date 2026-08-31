#!/usr/bin/env python3
"""WSC (Welfare-State Communication) continuation -- a SIBLING of the real
formal script ``train_curriculum_stage_highwayenv.py`` (never modified by
this file). Mirrors that script's continuation protocol EXACTLY: same
absolute-step schedule discipline (start_step/max_additional_steps,
eps_decay_steps_absolute, lr_decay_steps_absolute), same DQN hyperparameters
(hidden sizes, gamma, batch size, replay warmup/capacity, target-sync
interval), same environment/scenario-sampling/reward construction, same
checkpoint interval, same seed handling.

The ONLY differences from the real formal script:
  1. WSC observation mode enabled (StudyBHighwayWrapperConfig(include_welfare_state=True)).
  2. obs_dim=22 (thesis.study_b.local_observation.LOCAL_OBS_DIM_WSC) instead of 18.
  3. --resume-from is loaded through wsc_checkpoint_expansion.expand_checkpoint
     (zero-column expansion of online/target/optimiser), not a plain torch.load.
  4. Additive WSC provenance fields in the written checkpoint/manifest.
  5. Checkpoint loading uses the CORRECTED semantic column mapping in
     thesis.study_b.wsc_checkpoint_expansion (see
     wsc_formal_campaign_incident_diagnosis.md for the full incident
     history). --wsc-weight-warmup-steps is accepted for DIAGNOSTIC use
     only and defaults to 0 (disabled) in the formal protocol -- see
     "WSC GRADIENT WARMUP -- DISABLED BY DEFAULT" below.

No other scientific difference is introduced. Reward construction
(terminal_welfare_bonus, base task reward, lambda, GGI weights) is called
via the EXACT SAME thesis.study_b.welfare_reward / thesis.study_b.utility
functions the real script uses, unchanged.

WSC GRADIENT WARMUP -- DISABLED BY DEFAULT (post-incident correction):
An earlier version of this script unconditionally installed a gradient-scale
ramp (thesis.study_b.wsc_gradient_warmup.NewColumnGradientRamp) on net.0
.weight's 4 new input columns, on the working hypothesis that the C64
checkpoint's razor-thin Q-value margins made it fragile to the first Adam
update touching the new zero-initialized columns. That hypothesis was
REFUTED: A/B tests at warmup_steps=20,000 and 50,000 (10% ramp strength for
most of the run) collapsed nearly identically to no-warmup. The TRUE root
cause was an unrelated bug -- wsc_checkpoint_expansion.py's old/new
observation-column mapping was semantically wrong (it assumed a
prefix/suffix split; the real WSC layout interleaves the new features). With
that mapping bug fixed, a full-strength (no warmup) short continuation
recovered healthy training dynamics matching the Original-18D control. The
formal protocol therefore does NOT use gradient warmup: --wsc-weight-warmup
-steps defaults to 0, and 0 means the ramp is never installed (net.0.weight
trains at full strength from step 0, identical to how the Original script
trains every column). The flag and NewColumnGradientRamp class are kept
only so a future diagnostic run can still opt in explicitly; no formal
launch command may pass a nonzero value without separate authorization.
"""
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
from thesis.study_b.local_observation import LOCAL_OBS_DIM_WSC  # noqa: E402
from thesis.study_b.oracle_controller import oracle_actions  # noqa: E402
from thesis.study_b.shared_local_dqn import SharedLocalDQNAgent, build_study_b_dqn_config, epsilon_at_step_v12, lr_at_step_v12  # noqa: E402
from thesis.study_b.training_common import StudyBEpisodeWindowStats, load_scenario_bank  # noqa: E402
from thesis.study_b.utility import episode_burdens, episode_utilities  # noqa: E402
from thesis.study_b.welfare_reward import condition_by_name, terminal_welfare_bonus  # noqa: E402
from thesis.study_b.wsc_checkpoint_expansion import ORIGINAL_OBS_DIM, WSC_OBS_DIM, expand_checkpoint  # noqa: E402
from thesis.study_b.wsc_gradient_warmup import NewColumnGradientRamp  # noqa: E402
from thesis.study_b.dense_shaping import (  # noqa: E402
    NEUTRAL_PHI,
    DenseShapingConfig,
    dense_shaping_term,
    welfare_objective_snapshot,
)

# DENSE-REWARD STUDY (added after environment-setup review, default-off):
# --dense-welfare-shaping wires in thesis.study_b.dense_shaping -- see that
# module's docstring for the full design record (shared discrete +/-c
# signal on DeltaPhi_t, active-set/exit-artifact guarantee). When
# --dense-welfare-shaping is not passed (the default), behaviour is
# byte-identical to this script's pre-existing WSC formal protocol. This is
# the "Priority 1: Maximin + WSC + Dense" wiring from
# F:\dense reward\README.md -- c and epsilon are never defaulted, see
# DenseShapingConfig.__post_init__.

assert WSC_OBS_DIM == LOCAL_OBS_DIM_WSC, "wsc_checkpoint_expansion and local_observation must agree on the WSC obs dim"

TARGET_SYNC_INTERVAL_UPDATES = 250  # frozen -- identical to the real formal script

_ENV_CONFIG = ThesisHighwayMergeEnvConfig()
_MERGE_START = _ENV_CONFIG.before_merge_length
_MERGE_END = _ENV_CONFIG.before_merge_length + _ENV_CONFIG.converge_merge_length


def _make_env(
    episode_max_steps: int, *, action_representation: str = "meta_speed",
    local_sensing_range_m: float | None = None, include_welfare_state: bool = True,
) -> StudyBHeterogeneousHighwayEnv:
    cfg = ThesisHighwayMergeEnvConfig(episode_max_steps=episode_max_steps, action_representation=action_representation)
    return StudyBHeterogeneousHighwayEnv(
        StudyBHighwayWrapperConfig(
            env_config=cfg, local_sensing_range_m=local_sensing_range_m,
            include_welfare_state=include_welfare_state,
        )
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
    p.add_argument("--resume-from", type=Path, required=True,
                    help="Path to a REAL 18D Original checkpoint (e.g. a C64 checkpoint). "
                         "Loaded through wsc_checkpoint_expansion.expand_checkpoint -- the source "
                         "file is never modified.")
    p.add_argument("--dense-log-every", type=int, default=0)
    p.add_argument(
        "--action-representation", type=str, default="meta_speed",
        choices=["direct_accel", "meta_speed"],
    )
    p.add_argument("--local-sensing-range-m", type=float, default=None)
    p.add_argument(
        "--wsc-weight-warmup-steps", type=int, default=0,
        help="DIAGNOSTIC USE ONLY, disabled by default (0). The gradient-warmup hypothesis "
             "for the v1 collapse was REFUTED (see wsc_formal_campaign_incident_diagnosis.md); "
             "the actual bug was a semantic column-mapping error in wsc_checkpoint_expansion.py, "
             "now fixed. The formal protocol trains net.0.weight at full strength from step 0 "
             "(0 = ramp never installed). Do not pass a nonzero value in a formal launch "
             "without separate, explicit authorization.",
    )
    p.add_argument("--dense-welfare-shaping", action="store_true", default=False,
                    help="Dense Reward Study (default OFF, byte-identical behaviour when unset): add a "
                         "discrete +/-c step-wise shaping term based on DeltaPhi_t. See "
                         "thesis.study_b.dense_shaping. Requires --dense-shaping-magnitude and "
                         "--dense-shaping-epsilon to be set explicitly.")
    p.add_argument("--dense-shaping-mode", type=str, default="discrete", choices=["discrete"],
                    help="Only 'discrete' is implemented.")
    p.add_argument("--dense-shaping-magnitude", type=float, default=None,
                    help="c: fixed shaping magnitude. Must be pre-frozen -- never chosen from results.")
    p.add_argument("--dense-shaping-epsilon", type=float, default=None,
                    help="epsilon: dead-zone threshold. Must be pre-frozen -- never chosen from results.")
    args = p.parse_args(argv)

    # Fails fast, before any environment/agent/checkpoint-expansion work, if
    # --dense-welfare-shaping is passed without both magnitude and epsilon.
    dense_cfg = DenseShapingConfig(
        enabled=args.dense_welfare_shaping, mode=args.dense_shaping_mode,
        magnitude=args.dense_shaping_magnitude, epsilon=args.dense_shaping_epsilon,
    )

    all_scenarios = load_scenario_bank(args.scenario_bank)
    by_id = {s.scenario_id: s for s in all_scenarios}
    stage_scenarios = [by_id[sid] for sid in args.scenario_ids]
    condition = condition_by_name(args.condition)

    env = _make_env(args.episode_max_steps, action_representation=args.action_representation,
                     local_sensing_range_m=args.local_sensing_range_m, include_welfare_state=True)
    diag_env = _make_env(args.episode_max_steps, action_representation=args.action_representation,
                          local_sensing_range_m=args.local_sensing_range_m, include_welfare_state=True)
    dqn_config = build_study_b_dqn_config(reward_condition="baseline", device=args.device, obs_dim=WSC_OBS_DIM)
    agent = SharedLocalDQNAgent(dqn_config, seed=args.master_seed)
    window_stats = StudyBEpisodeWindowStats()
    per_scenario_stats: dict[str, StudyBEpisodeWindowStats] = defaultdict(StudyBEpisodeWindowStats)
    scenario_rng = np.random.default_rng(args.master_seed * 7919 + 1)

    # --- WSC checkpoint loading ---
    # Fresh Priority-1 start: 18D C64/Original checkpoint, expanded in-memory to 22D
    # (source file never written). Pause/resume of an already-22D dense/WSC run:
    # load as-is -- expand_checkpoint refuses in_dim != 18.
    raw_ckpt = torch.load(args.resume_from, map_location="cpu")
    resume_in_dim = int(raw_ckpt["online"]["net.0.weight"].shape[1])
    if resume_in_dim == WSC_OBS_DIM:
        agent.learner.online.load_state_dict(raw_ckpt["online"])
        agent.learner.target.load_state_dict(raw_ckpt["target"])
        agent.learner.optimiser.load_state_dict(raw_ckpt["optimiser"])
        agent.learner._update_count = int(raw_ckpt["update_count"])
        expanded = {
            "step": int(raw_ckpt["step"]),
            "update_count": int(raw_ckpt["update_count"]),
            "source_obs_dim": WSC_OBS_DIM,
            "source_checkpoint_step": int(raw_ckpt.get("source_checkpoint_step", raw_ckpt["step"])),
        }
        print(f"[WSC] resumed from already-22D checkpoint {args.resume_from} "
              f"(checkpoint step {expanded['step']}), continuing at absolute step {args.start_step} "
              f"with obs_dim={WSC_OBS_DIM}")
    elif resume_in_dim == ORIGINAL_OBS_DIM:
        expanded = expand_checkpoint(args.resume_from, device="cpu")
        agent.learner.online.load_state_dict(expanded["online"])
        agent.learner.target.load_state_dict(expanded["target"])
        agent.learner.optimiser.load_state_dict(expanded["optimiser"])
        agent.learner._update_count = int(expanded["update_count"])
        print(f"[WSC] resumed from {args.resume_from} (checkpoint step {expanded['step']}, "
              f"source_obs_dim={expanded['source_obs_dim']}), continuing at absolute step {args.start_step} "
              f"with obs_dim={WSC_OBS_DIM}")
    else:
        raise SystemExit(
            f"--resume-from {args.resume_from} has net.0.weight input dim {resume_in_dim}, "
            f"expected {ORIGINAL_OBS_DIM} (C64/Original) or {WSC_OBS_DIM} (already-WSC/dense)"
        )

    # --- gradient warmup: DISABLED BY DEFAULT in the formal protocol (see
    # module docstring "WSC GRADIENT WARMUP -- DISABLED BY DEFAULT"). The
    # ramp hypothesis was refuted; the real bug was the (now-fixed) column
    # mapping in wsc_checkpoint_expansion.py. Only installed if a nonzero
    # --wsc-weight-warmup-steps is explicitly passed (diagnostic use only).
    gradient_ramp = None
    if args.wsc_weight_warmup_steps > 0:
        gradient_ramp = NewColumnGradientRamp(n_old_cols=ORIGINAL_OBS_DIM, warmup_steps=args.wsc_weight_warmup_steps)
        agent.learner.online.net[0].weight.register_hook(gradient_ramp.hook)
        print(f"[WSC] DIAGNOSTIC gradient ramp installed on net.0.weight[:, {ORIGINAL_OBS_DIM}:{WSC_OBS_DIM}], "
              f"warmup_steps={args.wsc_weight_warmup_steps} -- NOT the formal protocol default")
    else:
        print("[WSC] gradient warmup disabled (formal protocol default) -- net.0.weight trains at full strength")

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
                # Additive WSC provenance metadata -- does not remove any existing field above,
                # and is informational only (never read by the reward function).
                "obs_dim": WSC_OBS_DIM,
                "welfare_state_communication": True,
                "wsc_definition": "M_i(t)=running_active_attainment(trace); absolute M values; neutral=1.0",
                "source_checkpoint": str(args.resume_from),
                "source_checkpoint_step": int(expanded["source_checkpoint_step"]),
                "wsc_weight_warmup_steps": args.wsc_weight_warmup_steps,
                "wsc_weight_warmup_local_step_at_save": gradient_ramp.local_step if gradient_ramp is not None else 0,
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

        # --- reward construction: IDENTICAL to the real formal script, plus
        # optional dense shaping (default off, byte-identical when off) ---
        welfare_bonus = 0.0
        if episode_over and args.welfare_lambda != 0.0:
            traces = env.episode_traces()
            episode_u = episode_utilities(traces)
            welfare_bonus = terminal_welfare_bonus(condition, list(episode_u.values()), lam=args.welfare_lambda)

        dense_term = 0.0
        if dense_cfg.enabled:
            live_traces = env.episode_traces()
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
        if gradient_ramp is not None:
            gradient_ramp.advance()
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
            phi_prev = NEUTRAL_PHI  # Phi_0: all traces empty -> running_active_attainment=1.0 for every vehicle

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
        "obs_dim": WSC_OBS_DIM, "welfare_state_communication": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"final_step": total_step, "elapsed_seconds": elapsed, "manifest": str(manifest_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
