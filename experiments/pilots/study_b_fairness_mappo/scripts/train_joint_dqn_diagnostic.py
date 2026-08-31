#!/usr/bin/env python3
"""VDN_Conditional_Amendment_Protocol.md sec 11 (Diagnostic 5): a
joint-information DQN sanity test -- one centralized learner that sees
ALL 4 vehicles' true state (including hidden target speeds) every step,
via ``thesis.agents.joint_dqn`` (already N-generic, built for exactly
this N=4 case -- see that module's docstring) fed
``StudyBHeterogeneousEnv.global_state()`` (the same CTDE-only input
MAPPO's critic uses).

**Not a formal learned policy and never a thesis solver candidate** --
this diagnostic exists ONLY to separate "is the environment/task solvable
by SOME learner" from "is the specific local-observation/parameter-sharing/
independent-Q-learning formulation the bottleneck". If this succeeds where
the local-observation shared DQN fails, that is the document's strongest
trigger for adopting VDN (sec 11); if this also fails, VDN is NOT
justified and the remaining suspects are implementation bugs/reward
pathology/environment dynamics (sec 11's "If joint DQN also fails" list).

Slot-ordering note (the main correctness risk this diagnostic has to get
right -- see ``role_major_slot_order``'s docstring): ``joint_dqn.py``'s
network has one FIXED output head per ROLE-MAJOR slot (ramp slots, then
mainline slots), but WHICH vehicle_id occupies which role is re-randomised
every ``env.reset()``. ``role_major_slot_order`` + ``reorder_joint_observation``
below recompute the vehicle_id<->slot mapping fresh every episode from
``env.reset()``'s own ``info["roles"]`` and re-derive the joint observation
from ``env.global_state()`` (which is fixed in vehicle_id order) rather
than assuming any fixed identity<->slot correspondence.

``StudyBHeterogeneousEnv.active_vehicle_ids`` is fixed at all 4 vehicle
ids for an episode's ENTIRE duration (a vehicle that individually
completes just gets a physics no-op, per
``Stage10SymmetricMergeEnv.step()``'s ``if self._completed[vid]: a = 0.0``)
-- so, unlike a naive expectation, there is no need to pad/freeze slots
for early-exited vehicles: every step already has exactly 4 well-formed
observations/actions/rewards, and ``controller_terminal`` is simply
``terminated or truncated`` (the whole-episode-level flag), matching
``joint_dqn.py``'s original two-vehicle design assumption exactly."""

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

from thesis.agents.joint_dqn import JointDQNConfig, JointDQNLearner, JointReplayTransition  # noqa: E402
from thesis.pilots.stage11_dyad_merge_pilot_config import (  # noqa: E402
    BATCH_SIZE,
    GAMMA,
    HIDDEN_SIZES,
    LEARNING_RATE_START,
    REPLAY_CAPACITY_V5,
    TARGET_SYNC_INTERVAL_UPDATES,
    epsilon_at_step_v12,
    lr_at_step_v12,
)
from thesis.pilots.stage11_dyad_merge_pilot_config import target_mode as v12_target_mode  # noqa: E402
from thesis.study_b.heterogeneous_env import StudyBEnvConfig, StudyBHeterogeneousEnv  # noqa: E402
from thesis.study_b.local_observation import SELF_OBS_DIM  # noqa: E402
from thesis.study_b.training_common import StudyBEpisodeWindowStats, load_scenario_bank  # noqa: E402
from thesis.study_b.utility import episode_burdens, episode_utilities, generalized_gini_welfare, gini_coefficient  # noqa: E402
from thesis.study_b.welfare_reward import WELFARE_LAMBDA, condition_by_name, terminal_welfare_bonus  # noqa: E402

__all__ = [
    "role_major_slot_order", "reorder_joint_observation", "ALWAYS_LEGAL_ACTION_MASK",
    "build_joint_dqn_config", "run_eval_joint",
]

ALWAYS_LEGAL_ACTION_MASK = np.array([True, True, True], dtype=bool)


def role_major_slot_order(roles: dict[str, list[str]]) -> tuple[str, str, str, str]:
    """``roles``: ``{"ramp": [vid, vid], "mainline": [vid, vid]}`` (as
    returned by ``StudyBHeterogeneousEnv.reset()``'s ``info["roles"]``).
    Returns a 4-tuple of vehicle_ids in ``joint_dqn.py``'s expected slot
    order: role-major (all ramp slots first, then all mainline slots),
    each role's own two members sorted by vehicle_id for a stable,
    deterministic tie-break (the two members of one role are otherwise
    interchangeable -- either order is a valid "role-major" ordering, so
    fixing it by vehicle_id just keeps this function pure/deterministic,
    not because slot 0 vs slot 1 within a role carries any meaning)."""
    ramp = sorted(roles["ramp"])
    mainline = sorted(roles["mainline"])
    if len(ramp) != 2 or len(mainline) != 2:
        raise ValueError(f"expected exactly 2 ramp + 2 mainline vehicles, got roles={roles!r}")
    return (ramp[0], ramp[1], mainline[0], mainline[1])


def reorder_joint_observation(
    global_state: np.ndarray, *, vehicle_id_order: tuple[str, ...], slot_order: tuple[str, ...],
    per_vehicle_dim: int = SELF_OBS_DIM,
) -> np.ndarray:
    """``global_state``: ``build_global_state``'s output -- ``per_vehicle_dim``-wide
    chunks concatenated in ``vehicle_id_order`` (sorted-by-id, e.g.
    ``("V0","V1","V2","V3")``). Returns the same data reordered into
    ``slot_order`` (role-major), the layout ``joint_dqn.py`` expects."""
    n = len(vehicle_id_order)
    if global_state.shape != (n * per_vehicle_dim,):
        raise ValueError(f"global_state shape {global_state.shape} != expected {(n * per_vehicle_dim,)}")
    if set(slot_order) != set(vehicle_id_order):
        raise ValueError(f"slot_order {slot_order} is not a permutation of vehicle_id_order {vehicle_id_order}")
    chunks = {vid: global_state[i * per_vehicle_dim : (i + 1) * per_vehicle_dim] for i, vid in enumerate(vehicle_id_order)}
    return np.concatenate([chunks[vid] for vid in slot_order])


def build_joint_dqn_config(*, device: str = "cpu") -> JointDQNConfig:
    return JointDQNConfig(
        per_vehicle_obs_dim=SELF_OBS_DIM,
        n_actions=3,
        hidden_sizes=HIDDEN_SIZES,
        learning_rate=LEARNING_RATE_START,
        gamma=GAMMA,
        epsilon=1.0,  # placeholder, overwritten per-step via epsilon_at_step_v12
        replay_capacity=REPLAY_CAPACITY_V5,
        batch_size=BATCH_SIZE,
        device=device,
        target_mode=v12_target_mode(),
        n_vehicles=4,
    )


def run_eval_joint(
    *, checkpoint: Path, scenario_bank: Path, episode_max_steps: int = 200, device: str = "cpu",
) -> list[dict]:
    """Greedy (epsilon=0) evaluation of one joint-DQN checkpoint against a
    frozen scenario bank. Returns rows in the SAME shape as
    ``evaluate_policy.run_eval`` (same field names) so this diagnostic's
    checkpoints are drop-in compatible with ``analysis.welfare.seed_level_summary``,
    ``multi_checkpoint_eval.py``'s failure-type classifier, etc. -- NOT
    routed through ``evaluate_policy.load_policy`` since that function's
    ``--algorithm dqn`` path loads ``SharedLocalDQNAgent`` (local,
    18-dim-obs architecture), which does not match this checkpoint's
    ``JointQNetwork`` state_dict shape."""
    vid_order = ("V0", "V1", "V2", "V3")
    scenarios = load_scenario_bank(scenario_bank)
    env = StudyBHeterogeneousEnv(StudyBEnvConfig(episode_max_steps=episode_max_steps))

    dqn_config = build_joint_dqn_config(device=device)
    learner = JointDQNLearner(dqn_config, seed=0)
    ckpt = torch.load(checkpoint, map_location=device)
    learner.online.load_state_dict(ckpt["online"])

    rows = []
    for scenario in scenarios:
        _obs, info = env.reset(seed=0, scenario=scenario)
        slot_order = role_major_slot_order(info["roles"])
        joint_obs = reorder_joint_observation(env.global_state(), vehicle_id_order=vid_order, slot_order=slot_order)
        episode_return: dict[str, float] = dict.fromkeys(vid_order, 0.0)
        steps_taken = 0
        masks = [ALWAYS_LEGAL_ACTION_MASK] * 4
        for _t in range(episode_max_steps):
            actions_tuple = learner.select_action(joint_obs, masks, greedy=True)
            actions_dict = dict(zip(slot_order, actions_tuple))
            _next_obs, base_reward, terminated, truncated, step_info = env.step(actions_dict)
            for vid, r in base_reward.items():
                episode_return[vid] = episode_return.get(vid, 0.0) + r
            steps_taken += 1
            if terminated or truncated:
                break
            joint_obs = reorder_joint_observation(env.global_state(), vehicle_id_order=vid_order, slot_order=slot_order)
        else:
            step_info = {"term_reason": "truncation"}

        traces = env.episode_traces()
        utilities = episode_utilities(traces)
        burdens = episode_burdens(traces, dt=env.dt())
        u_values = list(utilities.values())
        min_vid = min(utilities, key=lambda v: utilities[v])

        row = {
            "scenario_id": scenario.scenario_id,
            "traffic_type": scenario.traffic_type,
            "term_reason": step_info["term_reason"],
            "completion": int(step_info["term_reason"] == "success"),
            "collision": int(step_info["term_reason"] == "collision"),
            "timeout": int(step_info["term_reason"] == "truncation"),
            "mean_U": sum(u_values) / len(u_values),
            "min_U": utilities[min_vid],
            "min_U_vehicle": min_vid,
            "min_U_role": scenario.vehicles[min_vid].role,
            "min_U_speed_class": scenario.vehicles[min_vid].speed_class,
            "ggi": generalized_gini_welfare(u_values),
            "gini": gini_coefficient(u_values),
            "C_max": max(burdens.values()),
            "C_mean": sum(burdens.values()) / len(burdens),
            "hard_brake_total": sum(traces[vid].hard_brake_count() for vid in traces),
            "episode_length": steps_taken,
            "mean_undiscounted_return": sum(episode_return.values()) / len(episode_return),
        }
        for vid in vid_order:
            row[f"role_{vid}"] = scenario.vehicles[vid].role
            row[f"speed_class_{vid}"] = scenario.vehicles[vid].speed_class
            row[f"U_{vid}"] = utilities[vid]
            row[f"C_{vid}"] = burdens[vid]
            row[f"hard_brake_{vid}"] = traces[vid].hard_brake_count()
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True, choices=["mean", "ggi", "maximin"])
    p.add_argument("--master-seed", type=int, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=800_000)
    p.add_argument("--episode-max-steps", type=int, default=200)
    p.add_argument("--checkpoint-every", type=int, default=50_000)
    p.add_argument("--heterogeneous-probability", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--replay-warmup", type=int, default=512)
    p.add_argument("--include-time-cost", action="store_true")
    p.add_argument("--welfare-lambda", type=float, default=WELFARE_LAMBDA)
    args = p.parse_args(argv)

    condition = condition_by_name(args.condition)
    env = StudyBHeterogeneousEnv(
        StudyBEnvConfig(
            episode_max_steps=args.episode_max_steps,
            heterogeneous_probability=args.heterogeneous_probability,
            include_time_cost=args.include_time_cost,
        )
    )
    dqn_config = build_joint_dqn_config(device=args.device)
    learner = JointDQNLearner(dqn_config, seed=args.master_seed)
    window_stats = StudyBEpisodeWindowStats()

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

    def save_checkpoint(step: int) -> None:
        torch.save(
            {
                "step": step, "diagnostic": "joint_dqn", "welfare_condition": condition.name,
                "welfare_lambda": args.welfare_lambda, "include_time_cost": args.include_time_cost,
                "online": learner.online.state_dict(), "target": learner.target.state_dict(),
                "optimiser": learner.optimiser.state_dict(), "update_count": learner._update_count,
                "replay_size": len(learner.replay),
            },
            checkpoint_dir / f"ckpt_step_{step}.pt",
        )
        metrics = {"step": step, "window": window_stats.as_dict()}
        window_stats.reset()
        checkpoint_records.append(metrics)

    seed = args.master_seed * 1_000_003
    _obs, info = env.reset(seed=seed)
    slot_order = role_major_slot_order(info["roles"])
    vid_order = tuple(sorted(env.active_vehicle_ids))
    joint_obs = reorder_joint_observation(env.global_state(), vehicle_id_order=vid_order, slot_order=slot_order)

    total_step = 0
    if 0 in checkpoint_targets:
        save_checkpoint(0)

    start_time = time.time()
    while total_step < args.max_steps:
        eps = epsilon_at_step_v12(total_step, decay_steps=eps_decay_steps)
        lr = lr_at_step_v12(total_step, decay_steps=lr_decay_steps)
        learner.set_learning_rate(lr)

        masks = [ALWAYS_LEGAL_ACTION_MASK] * 4
        actions_tuple = learner.select_action(joint_obs, masks, epsilon=eps)
        actions_dict = dict(zip(slot_order, actions_tuple))

        _next_obs, base_reward, terminated, truncated, _step_info = env.step(actions_dict)
        episode_over = bool(terminated or truncated)

        welfare_bonus = 0.0
        if episode_over:
            traces = env.episode_traces()
            episode_u = episode_utilities(traces)
            welfare_bonus = terminal_welfare_bonus(condition, list(episode_u.values()), lam=args.welfare_lambda)

        rewards_tuple = tuple(base_reward[vid] + welfare_bonus for vid in slot_order)

        if episode_over:
            next_joint_obs = None
            next_masks = None
        else:
            next_joint_obs = reorder_joint_observation(env.global_state(), vehicle_id_order=vid_order, slot_order=slot_order)
            next_masks = tuple(masks)

        transition = JointReplayTransition(
            joint_observation=joint_obs, actions=actions_tuple, rewards=rewards_tuple,
            next_joint_observation=next_joint_obs, terminated=terminated, truncated=truncated,
            action_masks=tuple(masks), next_action_masks=next_masks, controller_terminal=episode_over,
        )
        learner.store_transition(transition)

        effective_warmup = max(int(args.replay_warmup), int(dqn_config.batch_size))
        if len(learner.replay) >= effective_warmup:
            learner.update()
            if learner._update_count % TARGET_SYNC_INTERVAL_UPDATES == 0:
                learner.hard_sync_target()
        total_step += 1

        if episode_over:
            traces = env.episode_traces()
            utilities = episode_utilities(traces)
            burdens = episode_burdens(traces, dt=env.dt())
            window_stats.record_episode(term_reason=_step_info["term_reason"], utilities=utilities, burdens=burdens)
            seed = args.master_seed * 1_000_003 + total_step
            _obs, info = env.reset(seed=seed)
            slot_order = role_major_slot_order(info["roles"])
            joint_obs = reorder_joint_observation(env.global_state(), vehicle_id_order=vid_order, slot_order=slot_order)
        else:
            joint_obs = next_joint_obs

        if total_step in checkpoint_targets:
            save_checkpoint(total_step)

    elapsed = time.time() - start_time
    manifest = {
        "stage": "study_b_joint_dqn_diagnostic", "condition": condition.name, "master_seed": args.master_seed,
        "final_step": total_step, "elapsed_seconds": elapsed, "checkpoint_steps": checkpoint_targets,
        "checkpoints": checkpoint_records,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"final_step": total_step, "elapsed_seconds": elapsed, "manifest": str(manifest_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
