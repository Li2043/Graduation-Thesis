"""Stage 10 pilot (E28) v5/v6 single-seed training runner -- shared-parameter architecture.

Structurally a close cousin of ``stage10_role_phase_subpolicy_runner.py``
(same threshold-triggered 2->4->6 curriculum, same LR/epsilon schedules,
same checkpoint cadence and trajectory logging) with the multi-network
(``SubPolicyManager``) machinery replaced by a single ``SharedDQNLearner``
(see ``stage10_shared_dqn.py`` for the architectural rationale). This is a
DELIBERATE controlled comparison against pilot v4: every constant reused
here (``CURRICULUM_V4_*``, ``EPSILON_STAGE_BUMP_*``, ``MAX_STEPS_V4``,
``CHECKPOINT_STEPS_V4``) is identical to what pilot v4 used -- the only
variable that changes between the two runners is the architecture itself.

Pilot v6 (2026-08-06): this SAME function (name kept as
``run_pilot_training_job_v5`` for historical continuity -- v5's already-
collected seeds/manifests used it too) now also serves pilot v6's new seeds.
Nothing in this file's control flow changed for v6 -- the reward-formula
revision (continuous TTC risk term, perpetrator-only collision penalty, 5:1
collision:completion ratio) lives entirely inside
``stage10_symmetric_merge_env.py``'s ``step()``; this runner just calls
``env.step(actions)`` and records whatever reward dict comes back, plus (new
for v6) the two extra per-vehicle diagnostic fields now present in
``step()``'s info dict (``ttc_penalty``, ``collision_penalty_applied``) in
the trajectory log, and a ``reward_version`` tag in checkpoints/manifest so
output data is self-describing about which reward formula produced it
without needing to cross-reference code history. v5's own manifests/
checkpoints (already on disk) do NOT have this tag -- only runs launched
after this change do.

Pilot v7 (2026-08-06): this SAME function now also serves pilot v7's new
seeds. Control-flow change (the only one): each step, this runner now
computes ``collision_penalty_at_step(step)``/``ttc_weight_at_step(step)``
(see stage10_role_phase_subpolicy_config.py -- a curriculum ramp modelled on
Alhazza (2026)'s DQN004/DQN006, see stage10_symmetric_merge_env.py's module
docstring) and passes them into ``env.step(actions, collision_penalty_magnitude=...,
ttc_penalty_weight=...)`` instead of relying on the env's static defaults.
The two ramped values are logged per-step in the trajectory log and the
``reward_version`` tag becomes ``REWARD_VERSION_V7``.
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
from thesis.agents.replay_buffer_v2 import ReplayTransition
from thesis.agents.stage10_shared_dqn import SharedDQNLearner
from thesis.envs.stage10_symmetric_merge_env import (
    REWARD_VERSION_V7,
    Stage10MergeEnvConfig,
    Stage10SymmetricMergeEnv,
)
from thesis.pilots.stage10_role_phase_subpolicy_config import (
    ALGORITHM,
    ARCHITECTURE_SHARED_PARAMETER,
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
    OBS_DIM_V5,
    PROTOCOL_TAG,
    REPLAY_CAPACITY_PER_SUBPOLICY,
    REPLAY_WARMUP_PER_SUBPOLICY,
    TARGET_SYNC_INTERVAL_UPDATES,
    assert_stage10_pilot_guards,
    collision_penalty_at_step,
    epsilon_for_step,
    lr_at_step,
    stage_index_for_advance,
    target_mode,
    ttc_weight_at_step,
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
    completions: int = 0
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


def build_shared_dqn_config() -> DQNConfig:
    cfg = DQNConfig(
        obs_dim=OBS_DIM_V5,
        n_actions=N_ACTIONS,
        hidden_sizes=HIDDEN_SIZES,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon=1.0,  # overwritten per-step by epsilon_for_step; placeholder only
        replay_capacity=REPLAY_CAPACITY_PER_SUBPOLICY,
        batch_size=BATCH_SIZE,
        device="cpu",
        reward_condition="baseline",
        target_mode=target_mode(),
    )
    cfg.validate()
    return cfg


def run_pilot_training_job_v5(
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
    """Same curriculum/LR/epsilon mechanism as pilot v4's
    ``run_pilot_training_job`` (see that function's docstring for the full
    mechanism description) -- the only structural difference is ONE
    ``SharedDQNLearner`` instead of a ``SubPolicyManager`` of six, and the
    environment is built with ``include_role_zone_features=True`` so role/
    zone are explicit inputs to that one network."""
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

    env_config_kwargs: dict[str, Any] = {
        "seed": master_seed,
        "n_vehicles": stage_vehicle_counts[stage_idx],
        "include_role_zone_features": True,
    }
    if episode_max_steps is not None:
        env_config_kwargs["max_steps"] = episode_max_steps
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(**env_config_kwargs))
    learner = SharedDQNLearner(build_shared_dqn_config(), seed=master_seed)

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
            "architecture": ARCHITECTURE_SHARED_PARAMETER,
            "reward_version": REWARD_VERSION_V7,
            "code_commit": code_commit,
            "learner": {
                "online": learner.online.state_dict(),
                "target": learner.target.state_dict(),
                "optimiser": learner.optimiser.state_dict(),
                "update_count": learner._update_count,
                "replay_size": len(learner.replay),
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
            "learner": {
                "replay_size": len(learner.replay),
                "update_count": learner._update_count,
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
            learner.set_learning_rate(lr)

            masks = {vid: env.action_mask(vid) for vid in active_ids}
            actions = {
                vid: learner.select_action(obs_dict[vid], masks[vid], epsilon=eps)
                for vid in active_ids
            }
            was_completed = {vid: env.is_completed(vid) for vid in active_ids}
            zone_t = {vid: env.zone_of(vid) for vid in active_ids}

            # Pilot v7: curriculum-ramped collision penalty / TTC weight,
            # keyed off the GLOBAL step count (not steps_in_stage) -- see
            # stage10_role_phase_subpolicy_config.py's docstring for why this
            # is one single ramp across the whole budget, not re-ramped per
            # curriculum stage transition.
            collision_penalty_mag = collision_penalty_at_step(step)
            ttc_weight = ttc_weight_at_step(step)
            next_obs_dict, reward, terminated, truncated, step_info = env.step(
                actions,
                collision_penalty_magnitude=collision_penalty_mag,
                ttc_penalty_weight=ttc_weight,
            )
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
                            "architecture": ARCHITECTURE_SHARED_PARAMETER,
                            # Pilot v7: the ramped values actually in effect this step (uniform
                            # across vehicles, but logged explicitly rather than re-derived).
                            "collision_penalty_magnitude": collision_penalty_mag,
                            "ttc_penalty_weight": ttc_weight,
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
                                    # Pilot v6 reward-revision diagnostics (see
                                    # stage10_symmetric_merge_env.py's module
                                    # docstring): per-vehicle continuous risk
                                    # penalty and whether the (now
                                    # perpetrator-only) collision penalty hit
                                    # this vehicle specifically this step.
                                    "ttc_penalty": float(step_info["ttc_penalty"][vid]),
                                    "collision_penalty_applied": bool(
                                        step_info["collision_penalty_applied"][vid]
                                    ),
                                }
                                for vid in active_ids
                            ],
                        }
                    )
                    + "\n"
                )

            # No cross-network bootstrap routing needed (pilot v5): every
            # transition -- regardless of zone/stage transition -- is an
            # entirely ordinary row for the ONE learner. Unlike pilot v4's
            # runner, there is no subpolicy_key/bootstrap_policy_id to compute.
            for vid in active_ids:
                if was_completed[vid]:
                    continue
                exited_now = bool(step_info["exit_event"][vid])
                collided = bool(step_info["collision_event"])
                controller_terminal = exited_now or collided
                if controller_terminal:
                    next_obs = None
                    next_mask = None
                else:
                    next_obs = next_obs_dict[vid]
                    next_mask = masks[vid]
                transition = ReplayTransition(
                    observation=obs_dict[vid],
                    action=int(actions[vid]),
                    shaped_reward=float(reward[vid]),
                    next_observation=next_obs,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    action_mask=masks[vid],
                    next_action_mask=next_mask,
                    controller_terminal=controller_terminal,
                    learner_completed=exited_now,
                )
                learner.store_transition(transition)

            if len(learner.replay) >= REPLAY_WARMUP_PER_SUBPOLICY:
                learner.update()
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
        "architecture": ARCHITECTURE_SHARED_PARAMETER,
        "reward_version": REWARD_VERSION_V7,
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


__all__ = ["EpisodeWindowStats", "build_shared_dqn_config", "run_pilot_training_job_v5"]
