"""Stage 10 pilot (E28) single-seed training runner.

Simpler than ``FormalTrainer`` on purpose -- this is a diagnostic pilot
(protocol S0/S6), not a formal-gate run, so it does not replicate the full
resumable-checkpoint / git-hash-guard machinery Stage 6+ formal training
uses. It still checkpoints faithfully at the frozen checkpoint steps and
records everything T9's analysis needs (per-checkpoint completion rate,
collision-free rate, per-sub-policy replay/update counts for the
step-count-balance check, and -- pilot v4 -- which curriculum stage each
checkpoint/trajectory step belongs to).

Pilot v4: replaces v3's episode-level probability-blend curriculum with a
threshold-triggered 3-stage curriculum (2 -> 4 -> 6 vehicles), modelled on
Gupta, Egorov & Kochenderfer (2017)'s actual Algorithm 2 mechanism -- see
stage10_role_phase_subpolicy_config.py's ``stage_index_for_advance`` and
module-level docstring for the full rationale.
"""

from __future__ import annotations

import json
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from thesis.agents.independent_dqn_v2 import DQNConfig
from thesis.agents.subpolicy_dqn import (
    SubPolicyManager,
    make_bootstrap_transition,
    subpolicy_key,
)
from thesis.envs.stage10_symmetric_merge_env import (
    Stage10MergeEnvConfig,
    Stage10SymmetricMergeEnv,
)
from thesis.pilots.stage10_role_phase_subpolicy_config import (
    ALGORITHM,
    BATCH_SIZE,
    CHECKPOINT_STEPS_V4,
    CONDITION,
    CURRICULUM_V4_ADVANCE_THRESHOLD,
    CURRICULUM_V4_ROLLING_WINDOW_EPISODES,
    CURRICULUM_V4_STAGE_MAX_STEPS,
    CURRICULUM_V4_STAGE_VEHICLE_COUNTS,
    GAMMA,
    HIDDEN_SIZES,
    LEARNING_RATE,
    LEARNING_RATE_DECAY_STEPS_V4,
    MAX_STEPS_V4,
    N_ACTIONS,
    OBS_DIM,
    PROTOCOL_TAG,
    REPLAY_CAPACITY_PER_SUBPOLICY,
    REPLAY_WARMUP_PER_SUBPOLICY,
    TARGET_SYNC_INTERVAL_UPDATES,
    assert_stage10_pilot_guards,
    epsilon_for_step,
    lr_at_step,
    stage_index_for_advance,
    target_mode,
)


def _git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
        ).strip()
    except Exception:
        return "unknown"


@dataclass
class EpisodeWindowStats:
    """Rolling counters since the previous checkpoint (protocol S5/S9)."""

    episodes: int = 0
    completions: int = 0  # all vehicles exited without collision
    collisions: int = 0
    truncations: int = 0

    def as_dict(self) -> dict[str, Any]:
        n = max(self.episodes, 1)
        return {
            "episodes": self.episodes,
            "completion_rate": self.completions / n,
            "collision_free_rate": (self.episodes - self.collisions) / n,
            "truncation_rate": self.truncations / n,
        }

    def reset(self) -> None:
        self.episodes = 0
        self.completions = 0
        self.collisions = 0
        self.truncations = 0


def build_dqn_config() -> DQNConfig:
    cfg = DQNConfig(
        obs_dim=OBS_DIM,
        n_actions=N_ACTIONS,
        hidden_sizes=HIDDEN_SIZES,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon=1.0,  # overwritten per-step by epsilon_at_step; placeholder only
        replay_capacity=REPLAY_CAPACITY_PER_SUBPOLICY,
        batch_size=BATCH_SIZE,
        device="cpu",
        reward_condition="baseline",
        target_mode=target_mode(),
    )
    cfg.validate()
    return cfg


def run_pilot_training_job(
    *,
    master_seed: int,
    output_root: Path,
    checkpoint_root: Path,
    max_steps: int = MAX_STEPS_V4,
    strict: bool = True,
    stage_vehicle_counts: tuple[int, ...] = CURRICULUM_V4_STAGE_VEHICLE_COUNTS,
    stage_max_steps: tuple[int, ...] = CURRICULUM_V4_STAGE_MAX_STEPS,
    advance_threshold: float = CURRICULUM_V4_ADVANCE_THRESHOLD,
    rolling_window_episodes: int = CURRICULUM_V4_ROLLING_WINDOW_EPISODES,
    checkpoint_steps: tuple[int, ...] = CHECKPOINT_STEPS_V4,
    lr_decay_steps: int = LEARNING_RATE_DECAY_STEPS_V4,
    episode_max_steps: int | None = None,
    enable_trajectory_logging: bool = True,
) -> dict[str, Any]:
    """``stage_vehicle_counts``/``stage_max_steps``/``advance_threshold``/
    ``rolling_window_episodes``/``checkpoint_steps``/``lr_decay_steps``/
    ``episode_max_steps`` are exposed only so tests can use tiny windows for
    a fast smoke run of the pilot-v4 threshold-triggered curriculum -- the
    frozen values (``episode_max_steps=None`` meaning "use
    Stage10MergeEnvConfig's own default", currently 600) are always used in
    the real training command. ``episode_max_steps`` exists specifically so
    tests can force frequent episode boundaries (via a small per-episode
    truncation cap) within a tiny total training-step budget, rather than
    relying on random early-training action noise to happen to end an
    episode via collision/success quickly enough -- deterministic, not luck-
    dependent.

    Curriculum mechanism (protocol S0.1, Gupta et al. 2017 Algorithm 2
    style): starts at ``stage_vehicle_counts[0]`` (2 vehicles). At each
    episode boundary, a rolling completion rate over the trailing
    ``rolling_window_episodes`` episodes (of the CURRENT stage only -- the
    window is cleared on every stage transition, so an easier stage's good
    performance never carries over to justify skipping the next stage's own
    threshold) is checked via ``stage_index_for_advance``: once it reaches
    ``advance_threshold``, OR once the stage's own safety-valve step budget
    (``stage_max_steps[stage_idx]``) is exhausted, the NEXT episode starts
    the next stage. The terminal stage (index len(stage_vehicle_counts)-1)
    never advances further -- training simply continues until the overall
    ``max_steps`` ceiling. Learner weights/replay/optimiser state carry
    through every transition unchanged -- this is a continuation, not
    independent per-stage runs.
    """
    if strict:
        assert_stage10_pilot_guards(master_seed=master_seed, max_steps=max_steps)
    if len(stage_vehicle_counts) != len(stage_max_steps):
        raise ValueError("stage_vehicle_counts and stage_max_steps must be the same length")

    output_root = Path(output_root).resolve()
    checkpoint_root = Path(checkpoint_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve()
    while repo_root != repo_root.parent and not (repo_root / ".git").exists():
        repo_root = repo_root.parent
    code_commit = _git_head(repo_root)

    stage_idx = 0
    steps_in_stage = 0
    rolling: deque[bool] = deque(maxlen=rolling_window_episodes)

    env_config_kwargs: dict[str, Any] = {"seed": master_seed, "n_vehicles": stage_vehicle_counts[stage_idx]}
    if episode_max_steps is not None:
        env_config_kwargs["max_steps"] = episode_max_steps
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(**env_config_kwargs))
    manager = SubPolicyManager(build_dqn_config(), seed=master_seed)

    traj_file = None
    if enable_trajectory_logging:
        traj_dir = output_root / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        traj_file = open(traj_dir / f"seed_{master_seed}.jsonl", "a", encoding="utf-8")

    window = EpisodeWindowStats()
    checkpoint_records: list[dict[str, Any]] = []
    checkpoint_targets = sorted(s for s in checkpoint_steps if 0 <= s <= max_steps)

    def save_checkpoint(step: int) -> dict[str, Any]:
        payload = {
            "step": step,
            "master_seed": master_seed,
            "protocol_tag": PROTOCOL_TAG,
            "algorithm": ALGORITHM,
            "condition": CONDITION,
            "code_commit": code_commit,
            "learners": {
                key: {
                    "online": learner.online.state_dict(),
                    "target": learner.target.state_dict(),
                    "optimiser": learner.optimiser.state_dict(),
                    "update_count": learner._update_count,
                    "replay_size": len(learner.replay),
                }
                for key, learner in manager.learners.items()
            },
        }
        path = checkpoint_root / f"seed_{master_seed}" / f"ckpt_step_{step}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

        metrics = {
            "step": step,
            "curriculum_stage_idx": stage_idx,
            "curriculum_stage_vehicles": stage_vehicle_counts[stage_idx],
            "window": window.as_dict(),
            "per_subpolicy": {
                key: {
                    "replay_size": len(learner.replay),
                    "update_count": learner._update_count,
                }
                for key, learner in manager.learners.items()
            },
        }
        window.reset()
        checkpoint_records.append(metrics)
        return metrics

    obs_dict, info = env.reset(seed=master_seed)
    roles = dict(info["roles"])
    active_ids = tuple(roles.keys())
    step = 0
    if 0 in checkpoint_targets:
        save_checkpoint(0)

    try:
        while step < max_steps:
            eps = epsilon_for_step(
                step=step,
                stage_idx=stage_idx,
                steps_in_stage=steps_in_stage,
                stage_max_steps=stage_max_steps,
            )
            lr = lr_at_step(step, decay_steps=lr_decay_steps)
            manager.set_learning_rate_for_all(lr)

            zone_t = {vid: env.zone_of(vid) for vid in active_ids}
            masks = {vid: env.action_mask(vid) for vid in active_ids}
            actions = {
                vid: manager.select_action(
                    roles[vid], zone_t[vid], obs_dict[vid], masks[vid], epsilon=eps
                )
                for vid in active_ids
            }
            was_completed = {vid: env.is_completed(vid) for vid in active_ids}

            next_obs_dict, reward, terminated, truncated, step_info = env.step(actions)
            step += 1
            steps_in_stage += 1
            zone_t1 = step_info["zone_t1"]

            if traj_file is not None:
                traj_file.write(
                    json.dumps(
                        {
                            "step": step,
                            "seed": master_seed,
                            "curriculum_stage_idx": stage_idx,
                            "term_reason": step_info["term_reason"],
                            "collision_event": bool(step_info["collision_event"]),
                            "vehicles": [
                                {
                                    "id": vid,
                                    "role": roles[vid],
                                    "zone": zone_t[vid],
                                    "position": float(env._vehicles[vid].route_position),
                                    "speed": float(env._vehicles[vid].speed),
                                    "action": int(actions[vid]),
                                }
                                for vid in active_ids
                            ],
                        }
                    )
                    + "\n"
                )

            for vid in active_ids:
                if was_completed[vid]:
                    continue  # already exited before this step -- no transition to store
                exited_now = bool(step_info["exit_event"][vid])
                collided = bool(step_info["collision_event"])
                controller_terminal = exited_now or collided
                if controller_terminal:
                    next_obs = None
                    next_mask = None
                    bootstrap_id = ""
                else:
                    next_obs = next_obs_dict[vid]
                    next_mask = masks[vid]  # action mask is role/state-based only, unchanged shape
                    incoming_zone = zone_t1[vid]
                    bootstrap_id = (
                        "" if incoming_zone == zone_t[vid] else subpolicy_key(roles[vid], incoming_zone)
                    )
                transition = make_bootstrap_transition(
                    observation=obs_dict[vid],
                    action=int(actions[vid]),
                    shaped_reward=float(reward[vid]),
                    next_observation=next_obs,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    action_mask=masks[vid],
                    next_action_mask=next_mask,
                    controller_terminal=controller_terminal,
                    bootstrap_policy_id=bootstrap_id,
                    learner_completed=exited_now,
                )
                manager.store_transition(roles[vid], zone_t[vid], transition)

            update_stats = manager.update_all(min_buffer=REPLAY_WARMUP_PER_SUBPOLICY)
            for key, stats in update_stats.items():
                learner = manager.learners[key]
                if learner._update_count % TARGET_SYNC_INTERVAL_UPDATES == 0:
                    learner.hard_sync_target()

            obs_dict = next_obs_dict

            if terminated or truncated:
                window.episodes += 1
                success = step_info["term_reason"] == "success"
                if success:
                    window.completions += 1
                elif step_info["term_reason"] == "collision":
                    window.collisions += 1
                elif step_info["term_reason"] == "truncation":
                    window.truncations += 1
                rolling.append(success)

                # Pilot v4 threshold-triggered curriculum (protocol S0.1):
                # decide whether the NEXT episode should be in the next
                # stage. rolling_completion_rate is None (not yet eligible to
                # trigger) until the trailing window is completely full of
                # THIS stage's episodes.
                rolling_rate = (
                    (sum(rolling) / len(rolling))
                    if len(rolling) >= rolling_window_episodes
                    else None
                )
                next_stage_idx = stage_index_for_advance(
                    current_stage_idx=stage_idx,
                    rolling_completion_rate=rolling_rate,
                    steps_in_current_stage=steps_in_stage,
                    n_stages=len(stage_vehicle_counts),
                    advance_threshold=advance_threshold,
                    stage_max_steps=stage_max_steps,
                )
                if next_stage_idx != stage_idx:
                    stage_idx = next_stage_idx
                    steps_in_stage = 0
                    rolling.clear()
                    env.config.n_vehicles = stage_vehicle_counts[stage_idx]

                obs_dict, info = env.reset(seed=master_seed * 1_000_003 + step)
                roles = dict(info["roles"])
                active_ids = tuple(roles.keys())

            if step in checkpoint_targets:
                save_checkpoint(step)
    finally:
        if traj_file is not None:
            traj_file.close()

        if step in checkpoint_targets and (not checkpoint_records or checkpoint_records[-1]["step"] != step):
            save_checkpoint(step)

    manifest = {
        "stage": "stage10_role_phase_subpolicy_pilot",
        "protocol_tag": PROTOCOL_TAG,
        "seed": master_seed,
        "algorithm": ALGORITHM,
        "condition": CONDITION,
        "final_step": step,
        "final_curriculum_stage_idx": stage_idx,
        "final_curriculum_stage_vehicles": stage_vehicle_counts[stage_idx],
        "code_commit": code_commit,
        "checkpoint_steps": checkpoint_targets,
        "checkpoints": checkpoint_records,
    }
    manifest_path = output_root / f"seed_{master_seed}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


__all__ = ["EpisodeWindowStats", "build_dqn_config", "run_pilot_training_job"]
