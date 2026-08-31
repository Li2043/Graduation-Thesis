"""Stage 11 formal confirmatory protocol -- greedy (epsilon=0) held-out
evaluation (DRAFT, see STAGE11_PROTOCOL.md Sec 7).

Mirrors ``stage7c_q1_eval.py``'s shape (run_greedy_episode /
evaluate_checkpoint_*, learner-state-unmutated safety checks) adapted for
v12's single ``JointDQNLearner`` (one shared network, not two independent
per-controller learners) and reusing ``stage11_welfare.py``'s pure
functions for U_j rather than reimplementing them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from thesis.agents.joint_dqn import JointDQNConfig, JointDQNLearner
from thesis.envs.stage10_symmetric_merge_env import Stage10MergeEnvConfig, Stage10SymmetricMergeEnv
from thesis.pilots.stage11_confirmatory_config import PROTOCOL_TAG
from thesis.pilots.stage11_confirmatory_eval_seeds import eval_plan_for_checkpoint
from thesis.pilots.stage11_welfare import (
    episode_collided_for_vehicle,
    episode_mobility_outcome,
    stakeholder_experience,
    target_speed_attainment,
)

MAX_POLICY_STEPS = 400  # matches Stage10MergeEnvConfig's default max_steps


def build_eval_env() -> Stage10SymmetricMergeEnv:
    """Same env construction as stage11_dyad_merge_runner.py's
    env_config_kwargs (n_vehicles=2, spawn_route_lead=0.0,
    include_role_zone_features=True) -- must be kept in sync with the
    runner's own construction (see stage11_confirmatory_eval_seeds.py's
    _eval_env_config docstring for the same cross-check note)."""
    return Stage10SymmetricMergeEnv(
        Stage10MergeEnvConfig(n_vehicles=2, spawn_route_lead=0.0, include_role_zone_features=True)
    )


def load_learner_for_eval(checkpoint_path: str, *, joint_config: JointDQNConfig, seed: int = 0) -> JointDQNLearner:
    """Loads only the online network's weights (greedy eval never touches
    target/optimiser/replay) -- everything else in the learner is
    freshly constructed and irrelevant to greedy action selection."""
    learner = JointDQNLearner(joint_config, seed=seed)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    learner.online.load_state_dict(payload["learner"]["online"])
    return learner


def _role_vid_map(roles: dict[str, str]) -> dict[str, str]:
    return {role: vid for vid, role in roles.items()}


def _joint_obs_from(obs_d: dict[str, np.ndarray], role_vid: dict[str, str]) -> np.ndarray:
    return np.concatenate([obs_d[role_vid["ramp"]], obs_d[role_vid["mainline"]]])


def run_greedy_episode(
    learner: JointDQNLearner,
    *,
    episode_seed: int,
    max_policy_steps: int = MAX_POLICY_STEPS,
) -> dict[str, Any]:
    """One held-out greedy (epsilon=0) episode. Does not mutate ``learner``
    (no replay push, no optimiser step, no epsilon draw -- ``select_action``
    is called with ``greedy=True``, which this project's own
    ``JointDQNLearner.select_action`` implementation guarantees never calls
    ``self._rng``)."""
    env = build_eval_env()
    obs_dict, info = env.reset(seed=int(episode_seed))
    roles = dict(info["roles"])
    role_vid = _role_vid_map(roles)
    active_ids = tuple(roles.keys())
    target_speed = float(env.config.target_speed)

    on_road_attainment: dict[str, list[float]] = {vid: [] for vid in active_ids}
    already_completed_before_collision = {vid: False for vid in active_ids}
    first_crosser: tuple[str, str] | None = None
    zone_t = {vid: env.zone_of(vid) for vid in active_ids}

    step = 0
    terminated = False
    truncated = False
    term_reason = "ongoing"

    while step < max_policy_steps:
        was_completed = {vid: env.is_completed(vid) for vid in active_ids}
        masks = {vid: env.action_mask(vid) for vid in active_ids}
        joint_obs = _joint_obs_from(obs_dict, role_vid)
        action_ramp, action_mainline = learner.select_action(
            joint_obs, (masks[role_vid["ramp"]], masks[role_vid["mainline"]]), greedy=True
        )
        actions = {role_vid["ramp"]: action_ramp, role_vid["mainline"]: action_mainline}

        next_obs_dict, _reward, terminated, truncated, step_info = env.step(actions)
        step += 1
        zone_t1 = step_info["zone_t1"]

        if step_info["collision_event"]:
            for vid in active_ids:
                if was_completed[vid]:
                    already_completed_before_collision[vid] = True

        attainment = {vid: target_speed_attainment(env._vehicles[vid].speed, target_speed) for vid in active_ids}
        for vid in active_ids:
            if not was_completed[vid]:
                on_road_attainment[vid].append(attainment[vid])

        if first_crosser is None:
            for vid in active_ids:
                if zone_t[vid] == "merging" and zone_t1[vid] == "post":
                    first_crosser = (roles[vid], vid)
                    break
        zone_t = zone_t1

        obs_dict = next_obs_dict
        term_reason = str(step_info["term_reason"])
        if terminated or truncated:
            break

    episode_collided = term_reason == "collision"
    U_by_role = {
        roles[vid]: episode_mobility_outcome(
            collided=episode_collided_for_vehicle(
                episode_collided=episode_collided,
                was_already_completed_before_collision=already_completed_before_collision[vid],
            ),
            attainment_samples=on_road_attainment[vid],
        )
        for vid in active_ids
    }

    return {
        "success": term_reason == "success",
        "collision": term_reason == "collision",
        "truncated": bool(truncated),
        "term_reason": term_reason,
        "episode_length": int(step),
        "U_ramp": U_by_role["ramp"],
        "U_mainline": U_by_role["mainline"],
        "id_of_ramp": role_vid["ramp"],
        "first_crosser_role": first_crosser[0] if first_crosser is not None else None,
    }


def evaluate_checkpoint_stage11_confirmatory(
    checkpoint_path: str,
    *,
    joint_config: JointDQNConfig,
    master_seed: int,
    checkpoint_step: int,
    protocol_tag: str = PROTOCOL_TAG,
) -> dict[str, Any]:
    """Greedy eval over the full held-out plan (16 episodes: 8 blocks x 2
    role assignments). Loads a FRESH learner per call (no shared mutable
    state across calls to accidentally leak between checkpoints/seeds)."""
    learner = load_learner_for_eval(checkpoint_path, joint_config=joint_config, seed=int(master_seed))
    before = {k: v.clone() for k, v in learner.online.state_dict().items()}

    plan = eval_plan_for_checkpoint(master_seed=master_seed, checkpoint_step=checkpoint_step, protocol_tag=protocol_tag)
    episodes = []
    for row in plan:
        ep = run_greedy_episode(learner, episode_seed=int(row["eval_seed"]))
        ep.update(
            {
                "master_seed": int(master_seed),
                "checkpoint_step": int(checkpoint_step),
                "scenario_block": int(row["scenario_block"]),
                "assignment": int(row["assignment"]),
                "eval_seed": int(row["eval_seed"]),
            }
        )
        episodes.append(ep)

    after = learner.online.state_dict()
    for k, v in before.items():
        if not torch.equal(v, after[k]):
            raise RuntimeError(f"greedy evaluation mutated learner parameter {k!r}")

    return {"n_episodes": len(episodes), "episodes": episodes, "master_seed": int(master_seed), "checkpoint_step": int(checkpoint_step)}


__all__ = [
    "MAX_POLICY_STEPS",
    "build_eval_env",
    "evaluate_checkpoint_stage11_confirmatory",
    "load_learner_for_eval",
    "run_greedy_episode",
]
