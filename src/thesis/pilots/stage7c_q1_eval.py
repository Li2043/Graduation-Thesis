"""Stage 7C-Q1 greedy evaluation with SHA-256 seeds and role-swap pairs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np

from thesis.agents.action_masking import role_action_mask, validate_action_mask
from thesis.analysis.endpoints import classify_convention
from thesis.diagnostics.stage7a0_failure_taxonomy import classify_truncated_episode
from thesis.envs.merge_env_candidate_v3 import HighLevelAction
from thesis.pilots.stage7c_q1_config import (
    ACTIVE_TIME_COST_PER_STEP,
    PROTOCOL_TAG,
    STAGE,
    episodes_per_seed_checkpoint,
)
from thesis.pilots.stage7c_q1_eval_seeds import eval_plan_for_checkpoint
from thesis.training.final_lock_loader import FinalLockBundle
from thesis.training.pilot_checkpoint import learner_fingerprint
from thesis.training.pilot_ic_schedule import build_env_for_block

FAILURE_CATEGORIES = (
    "success",
    "collision",
    "unilateral_stall",
    "mutual_yielding",
    "downstream_failure",
    "other_failure",
)


def map_failure_category(
    *,
    success: bool,
    collision: bool,
    truncated: bool,
    primary_failure_label: str | None = None,
) -> str:
    """Frozen 6-way taxonomy used by Stage 7C-Q1 (do not extend)."""
    if success:
        return "success"
    if collision:
        return "collision"
    label = str(primary_failure_label or "")
    if label == "unilateral_stall":
        return "unilateral_stall"
    if label == "mutual_yielding":
        return "mutual_yielding"
    if label in {"downstream_completion_failure", "post_exit_survivor_stall"}:
        return "downstream_failure"
    if truncated or label:
        return "other_failure"
    return "other_failure"


def _validation_blocks(bundle: FinalLockBundle) -> list[tuple[str, Any]]:
    blocks = list(bundle.environment.validation_blocks)
    if len(blocks) != 8:
        raise RuntimeError(f"expected 8 validation blocks, got {len(blocks)}")
    return [(b.block_id, b) for b in blocks]


def _block_for_template(
    bundle: FinalLockBundle, template_index: int, assignment: int
) -> tuple[str, Any]:
    blocks = _validation_blocks(bundle)
    block_id, block = blocks[int(template_index) % 8]
    if int(assignment) == 0:
        return block_id, block
    swapped = replace(block, role_A=block.role_B, role_B=block.role_A)
    return block_id, swapped


def run_greedy_episode(
    bundle: FinalLockBundle,
    learners: dict[str, Any],
    *,
    block,
    episode_seed: int,
    max_policy_steps: int = 400,
    active_time_cost_per_step: float = ACTIVE_TIME_COST_PER_STEP,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
        for aid in ("A", "B"):
            role = str(env._vehicles[aid].role)  # noqa: SLF001
            mask = validate_action_mask(role_action_mask(role, 3), 3)
            if not env._vehicles[aid].active_on_road:  # noqa: SLF001
                actions[aid] = int(HighLevelAction.MAINTAIN)
            else:
                actions[aid] = learners[aid].select_action(obs[aid], mask, greedy=True)
        obs, _rewards, terminated, truncated, info = env.step(actions)
        steps += 1
        term_reason = str(info["term_reason"])
        joint = (
            f"{('maintain','accelerate','decelerate')[int(actions['A'])]}"
            f"-{('maintain','accelerate','decelerate')[int(actions['B'])]}"
            if info["vehicles_t"]["A"]["active_on_road"]
            and info["vehicles_t"]["B"]["active_on_road"]
            else "mixed"
        )
        for aid in ("A", "B"):
            comp = info["components"][aid]
            for k in reward_totals[aid]:
                reward_totals[aid][k] += float(comp.get(k, 0.0))
            if exit_step[aid] is None and float(info["events"]["exit_event"].get(aid, 0.0)) >= 1.0:
                exit_step[aid] = int(steps)
            # Step log compatible with Stage 7A0 truncation taxonomy
            veh_t = info["vehicles_t"][aid]
            veh = info["vehicles_t1"][aid]
            step_rows.append(
                {
                    "controller": aid,
                    "policy_step": int(info["policy_step"]),
                    "action": int(actions[aid]),
                    "commanded_action": int(actions[aid]),
                    "speed": float(veh["speed"]),
                    "route_position": float(veh["route_position"]),
                    "route_progress": float(veh["route_position"]),
                    "active": bool(veh_t["active_on_road"]),
                    "active_on_road": bool(veh["active_on_road"]),
                    "exited": bool(veh.get("completed", False)),
                    "completed": bool(veh.get("completed", False)),
                    "joint_action_category": joint,
                    "joint_category": joint,
                    "Q_margin": float("nan"),
                }
            )
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
        import pandas as pd

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

    # Verify reward totals consistency per agent
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


def evaluate_checkpoint_stage7c(
    bundle: FinalLockBundle,
    learners: dict[str, Any],
    *,
    master_seed: int,
    checkpoint_step: int,
    code_commit: str,
    protocol_tag: str = PROTOCOL_TAG,
    max_policy_steps: int = 400,
    active_time_cost_per_step: float = ACTIVE_TIME_COST_PER_STEP,
) -> dict[str, Any]:
    """Greedy eval; must not mutate learners / replay / update counts."""
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
    for i, row in enumerate(plan):
        block_id, block = _block_for_template(
            bundle, int(row["template_validation_index"]), int(row["assignment"])
        )
        ep, _ = run_greedy_episode(
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
        "master_seed": int(master_seed),
        "checkpoint_step": int(checkpoint_step),
    }


def compute_swap_eligibility(episodes: list[dict[str, Any]]) -> float:
    """Strict swap eligibility: both assignments succeed with complementary orders."""
    by_block: dict[int, dict[int, dict[str, Any]]] = {}
    for ep in episodes:
        b = int(ep["scenario_block"])
        a = int(ep["assignment"])
        by_block.setdefault(b, {})[a] = ep
    if not by_block:
        return 0.0
    eligible = 0
    for block, assigns in by_block.items():
        if set(assigns.keys()) != {0, 1}:
            continue
        e0, e1 = assigns[0], assigns[1]
        if not (bool(e0.get("success")) and bool(e1.get("success"))):
            continue
        orders = {str(e0.get("passing_order")), str(e1.get("passing_order"))}
        if orders == {"mainline_first", "ramp_first"}:
            eligible += 1
    return float(eligible) / float(len(by_block))


def summarise_seed_checkpoint(episodes: list[dict[str, Any]]) -> dict[str, float]:
    n = max(len(episodes), 1)
    return {
        "success_rate": float(np.mean([float(e["success"]) for e in episodes])),
        "collision_rate": float(np.mean([float(e["collision"]) for e in episodes])),
        "truncation_rate": float(np.mean([float(e["truncation"]) for e in episodes])),
        "swap_eligibility": compute_swap_eligibility(episodes),
        "n_episodes": float(len(episodes)),
    }


__all__ = [
    "FAILURE_CATEGORIES",
    "compute_swap_eligibility",
    "evaluate_checkpoint_stage7c",
    "map_failure_category",
    "run_greedy_episode",
    "summarise_seed_checkpoint",
]
