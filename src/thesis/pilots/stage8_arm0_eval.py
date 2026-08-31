"""Stage 8 arm0 greedy evaluation: exact Stage 7C-Q1 reproduction + rich per-step logging.

This module deliberately does NOT reuse `thesis.pilots.stage7a1_eval.evaluate_checkpoint_rich`
/ `thesis.diagnostics.stage7a0_trajectory_eval.run_diagnostic_evaluation_episode` directly:
those functions call `build_env_for_block(bundle, block, max_policy_steps=...)` without an
`active_time_cost_per_step` argument, which silently defaults to Base Reward V1 (0.0), not
the V2 active-time-cost reward (0.0005) that Stage 7C-Q1 (and this reproduction) trains under.
Reusing them unmodified would evaluate under a different reward function than the one used in
training, invalidating the whole reproduction.

Instead, this module splices the env-construction / reward-accounting / failure-classification
path from `thesis.pilots.stage7c_q1_eval` (which does thread `active_time_cost_per_step`
correctly) with the rich per-step logging schema from
`thesis.diagnostics.stage7a0_trajectory_eval` (Q-values, front_gap, minimum_TTC, action_mask).
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from thesis.agents.action_masking import role_action_mask, validate_action_mask
from thesis.analysis.endpoints import classify_convention
from thesis.diagnostics.stage7a0_failure_taxonomy import classify_truncated_episode
from thesis.envs.merge_env_candidate_v3 import HighLevelAction
from thesis.pilots.stage7c_q1_eval import map_failure_category
from thesis.pilots.stage7c_q1_eval_seeds import eval_plan_for_checkpoint
from thesis.pilots.stage8_arm0_config import (
    ACTIVE_TIME_COST_PER_STEP,
    PROTOCOL_TAG,
    STAGE,
    episodes_per_seed_checkpoint,
)
from thesis.training.final_lock_loader import FinalLockBundle
from thesis.training.pilot_checkpoint import learner_fingerprint
from thesis.training.pilot_ic_schedule import build_env_for_block

ACTION_LABELS = ("maintain", "accelerate", "decelerate")


def _q_diagnostics(q: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Copied inline from stage7a0_trajectory_eval.py (private helper, not imported)."""
    legal = np.asarray(q, dtype=np.float64).copy()
    legal[~np.asarray(mask, dtype=bool)] = -np.inf
    order = np.argsort(-legal)
    best = float(legal[order[0]])
    second = float(legal[order[1]]) if np.isfinite(legal[order[1]]) else float("nan")
    margin = best - second if np.isfinite(second) else float("nan")
    return {
        "Q_maintain": float(q[0]),
        "Q_accelerate": float(q[1]),
        "Q_decelerate": float(q[2]),
        "greedy_action": int(order[0]),
        "best_Q": best,
        "second_best_Q": second,
        "Q_margin": margin,
        "Q_range": float(np.nanmax(q) - np.nanmin(q)),
    }


def _joint_category(a: int, b: int, active_a: bool, active_b: bool) -> str:
    """Copied inline from stage7a0_trajectory_eval.py (private helper, not imported)."""
    if not active_a and active_b:
        return "inactive-active"
    if active_a and not active_b:
        return "active-inactive"
    if not active_a and not active_b:
        return "inactive-inactive"
    return f"{ACTION_LABELS[a]}-{ACTION_LABELS[b]}"


def _progress(veh) -> float:
    return float(getattr(veh, "route_position", 0.0))


def _validation_blocks(bundle: FinalLockBundle) -> list[tuple[str, Any]]:
    """Copied inline from stage7c_q1_eval.py (private helper, not imported)."""
    blocks = list(bundle.environment.validation_blocks)
    if len(blocks) != 8:
        raise RuntimeError(f"expected 8 validation blocks, got {len(blocks)}")
    return [(b.block_id, b) for b in blocks]


def _block_for_template(
    bundle: FinalLockBundle, template_index: int, assignment: int
) -> tuple[str, Any]:
    """Copied inline from stage7c_q1_eval.py (private helper, not imported)."""
    blocks = _validation_blocks(bundle)
    block_id, block = blocks[int(template_index) % 8]
    if int(assignment) == 0:
        return block_id, block
    swapped = replace(block, role_A=block.role_B, role_B=block.role_A)
    return block_id, swapped


def run_greedy_episode_diagnostic(
    bundle: FinalLockBundle,
    learners: dict[str, Any],
    *,
    block,
    episode_seed: int,
    max_policy_steps: int = 400,
    active_time_cost_per_step: float = ACTIVE_TIME_COST_PER_STEP,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Greedy evaluation episode with rich per-step trajectory logging.

    Env construction, reward accounting, and failure classification are copied
    from `stage7c_q1_eval.run_greedy_episode` (V2-reward-correct). Per-step row
    content is the rich schema from `stage7a0_trajectory_eval.run_diagnostic_evaluation_episode`.
    """
    env = build_env_for_block(
        bundle,
        block,
        max_policy_steps=max_policy_steps,
        active_time_cost_per_step=float(active_time_cost_per_step),
    )
    obs, _ = env.reset(seed=int(episode_seed))
    roles0 = {aid: str(env._vehicles[aid].role) for aid in ("A", "B")}  # noqa: SLF001
    reward_totals = {
        aid: {
            "reward_progress": 0.0,
            "reward_exit": 0.0,
            "reward_collision": 0.0,
            "reward_hard_braking": 0.0,
            "reward_active_time": 0.0,
            "reward_total": 0.0,
        }
        for aid in ("A", "B")
    }
    exit_step = {"A": None, "B": None}
    step_rows: list[dict[str, Any]] = []
    steps = 0
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    term_reason = "ongoing"

    while True:
        actions: dict[str, int] = {}
        masks: dict[str, np.ndarray] = {}
        q_by: dict[str, dict[str, Any]] = {}
        for aid in ("A", "B"):
            veh = env._vehicles[aid]  # noqa: SLF001
            role = str(veh.role)
            mask = validate_action_mask(role_action_mask(role, 3), 3)
            masks[aid] = mask
            q = learners[aid].q_values(obs[aid], network="online")
            qd = _q_diagnostics(q, mask)
            q_by[aid] = qd
            if not veh.active_on_road:
                actions[aid] = int(HighLevelAction.MAINTAIN)
            else:
                actions[aid] = learners[aid].select_action(obs[aid], mask, greedy=True)
                if int(qd["greedy_action"]) != int(actions[aid]):
                    raise RuntimeError("Q logging greedy action mismatch")

        # Log pre-step state + commanded actions (rich schema).
        for aid in ("A", "B"):
            veh = env._vehicles[aid]  # noqa: SLF001
            peer = "B" if aid == "A" else "A"
            pveh = env._vehicles[peer]  # noqa: SLF001
            qd = q_by[aid]
            step_rows.append(
                {
                    "policy_step": steps,
                    "simulation_time": float(steps * 0.2),
                    "controller": aid,
                    "controller_role": str(veh.role),
                    "active": bool(veh.active_on_road),
                    "exited": bool(veh.completed),
                    "route_progress": _progress(veh),
                    "speed": float(veh.speed),
                    "commanded_action": int(actions[aid]),
                    "commanded_action_name": ACTION_LABELS[int(actions[aid])],
                    "commanded_acceleration": float(veh.commanded_acceleration),
                    "realised_acceleration": float(veh.realised_acceleration),
                    "peer_progress": _progress(pveh),
                    "peer_speed": float(pveh.speed),
                    "action_A": int(actions["A"]),
                    "action_B": int(actions["B"]),
                    "joint_action_category": _joint_category(
                        int(actions["A"]),
                        int(actions["B"]),
                        bool(env._vehicles["A"].active_on_road),  # noqa: SLF001
                        bool(env._vehicles["B"].active_on_road),  # noqa: SLF001
                    ),
                    "Q_maintain": qd["Q_maintain"],
                    "Q_accelerate": qd["Q_accelerate"],
                    "Q_decelerate": qd["Q_decelerate"],
                    "greedy_action": qd["greedy_action"],
                    "best_Q": qd["best_Q"],
                    "second_best_Q": qd["second_best_Q"],
                    "Q_margin": qd["Q_margin"],
                    "Q_range": qd["Q_range"],
                    "action_mask": json.dumps([bool(x) for x in masks[aid].tolist()]),
                    "collision_flag": False,
                    "termination_flag": False,
                    "truncation_flag": False,
                }
            )

        obs, _rewards, terminated, truncated, info = env.step(actions)
        steps += 1
        term_reason = str(info["term_reason"])

        for aid in ("A", "B"):
            comp = info["components"][aid]
            for k in reward_totals[aid]:
                reward_totals[aid][k] += float(comp.get(k, 0.0))
            if exit_step[aid] is None and float(info["events"]["exit_event"].get(aid, 0.0)) >= 1.0:
                exit_step[aid] = int(steps)

        # Annotate the two rows just logged with post-step outcomes.
        gap = info.get("min_bumper_gap")
        ttc = info.get("ttc")
        for row in step_rows[-2:]:
            aid = row["controller"]
            row["realised_acceleration"] = float(
                info["vehicles_t1"][aid]["realised_acceleration"]
            )
            row["collision_flag"] = bool(
                float(info["events"]["stakeholder_collision_event"]) >= 1.0
            )
            row["termination_flag"] = bool(terminated)
            row["truncation_flag"] = bool(truncated)
            if gap is not None and math.isfinite(float(gap)):
                row["front_gap"] = float(gap)
            if ttc is not None and math.isfinite(float(ttc)):
                row["minimum_TTC"] = float(ttc)

        if terminated or truncated:
            break

    roles = {aid: str(info["vehicles_t1"][aid]["role"]) for aid in ("A", "B")}
    success = term_reason == "success"
    collision = term_reason == "collision" or float(
        info["events"]["stakeholder_collision_event"]
    ) >= 1.0
    passing_order = classify_convention(
        success=success, exit_time=info["exit_time"], roles=roles
    )

    primary = None
    if truncated and not success and not collision:
        first_exit = None
        if exit_step["A"] is not None and exit_step["B"] is not None:
            first_exit = "A" if exit_step["A"] <= exit_step["B"] else "B"
        elif exit_step["A"] is not None:
            first_exit = "A"
        elif exit_step["B"] is not None:
            first_exit = "B"
        tax = classify_truncated_episode(
            {
                "truncated": True,
                "success": False,
                "collision": False,
                "first_exit_controller": first_exit,
                "second_exit_step": (
                    max(v for v in exit_step.values() if v is not None)
                    if sum(1 for v in exit_step.values() if v is not None) >= 2
                    else None
                ),
            },
            pd.DataFrame(step_rows),
        )
        primary = tax.get("primary_failure_label")

    failure_category = map_failure_category(
        success=success,
        collision=collision,
        truncated=bool(truncated),
        primary_failure_label=primary,
    )

    for aid in ("A", "B"):
        rt = reward_totals[aid]
        recon = (
            rt["reward_progress"]
            + rt["reward_exit"]
            + rt["reward_collision"]
            + rt["reward_hard_braking"]
            + rt["reward_active_time"]
        )
        if abs(recon - rt["reward_total"]) > 1e-9:
            raise RuntimeError(f"reward decomp mismatch for {aid}: {recon} vs {rt['reward_total']}")

    ep = {
        "success": bool(success),
        "collision": bool(collision),
        "truncation": bool(truncated),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": term_reason,
        "episode_length": int(steps),
        "exit_step_agent_0": exit_step["A"],
        "exit_step_agent_1": exit_step["B"],
        "passing_order": passing_order,
        "controller_role_mapping": {"A": roles0["A"], "B": roles0["B"]},
        "failure_category": failure_category,
        "primary_failure_label": primary,
        "reward_progress_A": reward_totals["A"]["reward_progress"],
        "reward_exit_A": reward_totals["A"]["reward_exit"],
        "reward_collision_A": reward_totals["A"]["reward_collision"],
        "reward_hard_braking_A": reward_totals["A"]["reward_hard_braking"],
        "reward_active_time_A": reward_totals["A"]["reward_active_time"],
        "reward_total_A": reward_totals["A"]["reward_total"],
        "reward_progress_B": reward_totals["B"]["reward_progress"],
        "reward_exit_B": reward_totals["B"]["reward_exit"],
        "reward_collision_B": reward_totals["B"]["reward_collision"],
        "reward_hard_braking_B": reward_totals["B"]["reward_hard_braking"],
        "reward_active_time_B": reward_totals["B"]["reward_active_time"],
        "reward_total_B": reward_totals["B"]["reward_total"],
    }
    return ep, step_rows


def evaluate_checkpoint_stage8_arm0(
    bundle: FinalLockBundle,
    learners: dict[str, Any],
    *,
    master_seed: int,
    checkpoint_step: int,
    code_commit: str,
    checkpoint_sha256: str = "",
    protocol_tag: str = PROTOCOL_TAG,
    max_policy_steps: int = 400,
    active_time_cost_per_step: float = ACTIVE_TIME_COST_PER_STEP,
    collect_trajectories: bool = True,
) -> dict[str, Any]:
    """Greedy eval across the 8-block/16-episode early plan; must not mutate learners."""
    before = {
        "A": learner_fingerprint(learners["A"]),
        "B": learner_fingerprint(learners["B"]),
    }
    rng_A = deepcopy(learners["A"]._rng.bit_generator.state)
    rng_B = deepcopy(learners["B"]._rng.bit_generator.state)
    replay_A = len(learners["A"].replay)
    replay_B = len(learners["B"].replay)
    updates_A = int(learners["A"]._update_count)
    updates_B = int(learners["B"]._update_count)

    plan = eval_plan_for_checkpoint(
        master_seed=master_seed,
        checkpoint_step=checkpoint_step,
        protocol_tag=protocol_tag,
    )
    expected_n = episodes_per_seed_checkpoint(checkpoint_step)
    if len(plan) != expected_n:
        raise RuntimeError(f"eval plan size {len(plan)} != expected {expected_n}")

    episodes: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    for i, row in enumerate(plan):
        block_id, block = _block_for_template(
            bundle, int(row["template_validation_index"]), int(row["assignment"])
        )
        ep, step_rows = run_greedy_episode_diagnostic(
            bundle,
            learners,
            block=block,
            episode_seed=int(row["eval_seed"]),
            max_policy_steps=max_policy_steps,
            active_time_cost_per_step=active_time_cost_per_step,
        )
        ep.update(
            {
                "stage": STAGE,
                "protocol_tag": protocol_tag,
                "code_commit": code_commit,
                "master_seed": int(master_seed),
                "checkpoint": int(checkpoint_step),
                "checkpoint_step": int(checkpoint_step),
                "checkpoint_sha256": checkpoint_sha256,
                "scenario_block": int(row["scenario_block"]),
                "assignment": int(row["assignment"]),
                "eval_seed": int(row["eval_seed"]),
                "episode_index": int(i),
                "swap_pair_id": str(row["swap_pair_id"]),
                "validation_block_id": block_id,
                "template_validation_index": int(row["template_validation_index"]),
            }
        )
        episodes.append(ep)
        if collect_trajectories:
            for srow in step_rows:
                srow["master_seed"] = int(master_seed)
                srow["checkpoint_step"] = int(checkpoint_step)
                srow["checkpoint_sha256"] = checkpoint_sha256
                srow["validation_block_id"] = block_id
                srow["assignment"] = int(row["assignment"])
                srow["scenario_block"] = int(row["scenario_block"])
                srow["eval_seed"] = int(row["eval_seed"])
                srow["episode_index"] = int(i)
            trajectories.extend(step_rows)

    learners["A"]._rng.bit_generator.state = rng_A
    learners["B"]._rng.bit_generator.state = rng_B
    after = {
        "A": learner_fingerprint(learners["A"]),
        "B": learner_fingerprint(learners["B"]),
    }
    if before != after:
        raise RuntimeError("evaluation mutated learner fingerprints")
    if len(learners["A"].replay) != replay_A or len(learners["B"].replay) != replay_B:
        raise RuntimeError("evaluation mutated replay")
    if int(learners["A"]._update_count) != updates_A or int(learners["B"]._update_count) != updates_B:
        raise RuntimeError("evaluation mutated update counts")

    return {
        "n_episodes": len(episodes),
        "episodes": episodes,
        "trajectories": trajectories,
        "master_seed": int(master_seed),
        "checkpoint_step": int(checkpoint_step),
    }


__all__ = [
    "evaluate_checkpoint_stage8_arm0",
    "run_greedy_episode_diagnostic",
]
