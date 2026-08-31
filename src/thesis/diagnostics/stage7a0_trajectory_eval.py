"""Baseline-only diagnostic evaluation with per-step logging."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from thesis.agents.action_masking import role_action_mask, validate_action_mask
from thesis.analysis.endpoints import classify_convention
from thesis.analysis.episode_utility_accumulator import (
    collect_active_state_attainment,
    collided_ids_from_pairs,
    derive_utility_fields,
    finalise_episode_utilities,
    initialise_episode_utility_accumulator,
)
from thesis.analysis.reconstruct_eval import (
    CHECKPOINT_INDEX_100K,
    load_learners_from_final_weights,
)
from thesis.diagnostics.stage7a0_inventory import ACTION_NAMES, FORMAL_BASELINE_SEEDS, sha256_file
from thesis.envs.merge_env_candidate_v3 import HighLevelAction
from thesis.formal.formal_schedule import evaluation_episode_seed
from thesis.rewards.pbrs_v2 import STAKEHOLDER_ORDER
from thesis.training.final_lock_loader import FinalLockBundle, load_final_locks
from thesis.training.pilot_ic_schedule import build_env_for_block, validation_blocks_with_assignments

ACTION_LABELS = ("maintain", "accelerate", "decelerate")


def _q_diagnostics(q: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
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
    if not active_a and active_b:
        return "inactive-active"
    if active_a and not active_b:
        return "active-inactive"
    if not active_a and not active_b:
        return "inactive-inactive"
    return f"{ACTION_LABELS[a]}-{ACTION_LABELS[b]}"


def _progress(veh) -> float:
    # route_position is meters along route; use raw position as progress proxy
    return float(getattr(veh, "route_position", 0.0))


def run_diagnostic_evaluation_episode(
    bundle: FinalLockBundle,
    learners: dict[str, Any],
    *,
    block,
    assignment: int,
    block_id: str,
    block_index: int,
    episode_seed: int,
    master_seed: int,
    checkpoint_sha256: str,
    max_policy_steps: int = 400,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = build_env_for_block(bundle, block, max_policy_steps=max_policy_steps)
    obs, _ = env.reset(seed=int(episode_seed))
    target_speeds = env.config.block.target_speeds.as_map()
    util_acc = initialise_episode_utility_accumulator(STAKEHOLDER_ORDER)
    collect_active_state_attainment(
        vehicles=env._vehicles,  # noqa: SLF001
        stakeholder_ids=STAKEHOLDER_ORDER,
        target_speeds=target_speeds,
        accumulator=util_acc,
    )

    step_rows: list[dict[str, Any]] = []
    steps = 0
    term_reason = "ongoing"
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    min_gap = math.inf
    min_ttc = math.inf
    merge_entry = {"A": None, "B": None}
    first_exit_controller = None
    first_exit_step = None
    second_exit_step = None
    exited_seen: set[str] = set()
    evaluation_guard = {
        "optimizer_steps": 0,
        "replay_writes": 0,
        "target_syncs": 0,
        "epsilon_updates": 0,
        "network_updates": 0,
    }

    roles0 = {aid: str(env._vehicles[aid].role) for aid in ("A", "B")}  # noqa: SLF001

    while True:
        actions: dict[str, int] = {}
        q_by: dict[str, dict[str, Any]] = {}
        masks: dict[str, np.ndarray] = {}
        for aid in ("A", "B"):
            veh = env._vehicles[aid]  # noqa: SLF001
            role = str(veh.role)
            mask = validate_action_mask(role_action_mask(role, 3), 3)
            masks[aid] = mask
            if not veh.active_on_road:
                actions[aid] = int(HighLevelAction.MAINTAIN)
                q = learners[aid].q_values(obs[aid], network="online")
                q_by[aid] = _q_diagnostics(q, mask)
                q_by[aid]["greedy_action"] = int(HighLevelAction.MAINTAIN)
            else:
                q = learners[aid].q_values(obs[aid], network="online")
                qd = _q_diagnostics(q, mask)
                q_by[aid] = qd
                # Select using same Q path (greedy argmax) — do not call select_action
                # after a separate forward that could diverge; use computed greedy.
                actions[aid] = int(qd["greedy_action"])
                # Verify against learner.select_action for isolation tests
                alt = learners[aid].select_action(obs[aid], mask, greedy=True)
                if int(alt) != int(actions[aid]):
                    raise RuntimeError("Q logging greedy action mismatch")

        # Log pre-step state + commanded actions
        for aid in ("A", "B"):
            veh = env._vehicles[aid]  # noqa: SLF001
            peer = "B" if aid == "A" else "A"
            pveh = env._vehicles[peer]  # noqa: SLF001
            qd = q_by[aid]
            step_rows.append(
                {
                    "master_seed": master_seed,
                    "validation_block_id": block_id,
                    "assignment": int(assignment),
                    "policy_step": steps,
                    "simulation_time": float(steps * 0.2),
                    "controller": aid,
                    "controller_role": str(veh.role),
                    "active": bool(veh.active_on_road),
                    "exited": bool(veh.completed),
                    "route_progress": _progress(veh),
                    "speed": float(veh.speed),
                    "commanded_action": int(actions[aid]),
                    "commanded_action_name": ACTION_NAMES[int(actions[aid])],
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
                    **{k: qd[k] for k in (
                        "Q_maintain",
                        "Q_accelerate",
                        "Q_decelerate",
                        "greedy_action",
                        "best_Q",
                        "second_best_Q",
                        "Q_margin",
                        "Q_range",
                    )},
                    "action_mask": json.dumps([bool(x) for x in masks[aid].tolist()]),
                    "collision_flag": False,
                    "termination_flag": False,
                    "truncation_flag": False,
                }
            )

        obs, rewards, terminated, truncated, info = env.step(actions)
        steps += 1

        # Update merge entry / exits
        for aid in ("A", "B"):
            veh = info["vehicles_t1"][aid]
            if merge_entry[aid] is None and str(veh.get("physical_segment", "")) in {
                "merge",
                "shared",
                "downstream",
            }:
                merge_entry[aid] = steps
            if bool(veh.get("completed")) and aid not in exited_seen:
                exited_seen.add(aid)
                if first_exit_controller is None:
                    first_exit_controller = aid
                    first_exit_step = steps
                else:
                    second_exit_step = steps

        gap = info.get("min_bumper_gap")
        if gap is not None and math.isfinite(float(gap)):
            min_gap = min(min_gap, float(gap))
        ttc = info.get("ttc")
        if ttc is not None and math.isfinite(float(ttc)):
            min_ttc = min(min_ttc, float(ttc))

        # annotate last logged rows with post-step realised accel / flags
        for row in step_rows[-2:]:
            aid = row["controller"]
            row["realised_acceleration"] = float(
                info["vehicles_t1"][aid]["realised_acceleration"]
            )
            row["collision_flag"] = bool(
                info["events"].get("stakeholder_collision_event", 0) >= 1.0
            )
            row["termination_flag"] = bool(terminated)
            row["truncation_flag"] = bool(truncated)
            if "min_bumper_gap" in info and info["min_bumper_gap"] is not None:
                row["front_gap"] = float(info["min_bumper_gap"])
            if "ttc" in info and info["ttc"] is not None and math.isfinite(float(info["ttc"])):
                row["minimum_TTC"] = float(info["ttc"])

        term_reason = str(info["term_reason"])
        if terminated or truncated:
            break
        collect_active_state_attainment(
            vehicles=info["vehicles_t1"],
            stakeholder_ids=STAKEHOLDER_ORDER,
            target_speeds=target_speeds,
            accumulator=util_acc,
        )

    pairs = info.get("events", {}).get("collision_pairs") or []
    collided_ids = collided_ids_from_pairs(pairs)
    utilities = finalise_episode_utilities(
        accumulator=util_acc, collided_stakeholder_ids=collided_ids
    )
    derived = derive_utility_fields(utilities)
    roles = {aid: str(info["vehicles_t1"][aid]["role"]) for aid in ("A", "B")}
    success = term_reason == "success"
    collision = term_reason == "collision" or float(
        info["events"]["stakeholder_collision_event"]
    ) >= 1.0
    convention = classify_convention(
        success=success, exit_time=info["exit_time"], roles=roles
    )
    ep = {
        "master_seed": int(master_seed),
        "checkpoint_step": 100_000,
        "checkpoint_sha256": checkpoint_sha256,
        "validation_block_id": block_id,
        "assignment": int(assignment),
        "block_index": int(block_index),
        "evaluation_seed": int(episode_seed),
        "controller_A_role": roles["A"],
        "controller_B_role": roles["B"],
        "success": bool(success),
        "collision": bool(collision),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": term_reason,
        "episode_length": int(steps),
        "passing_order": convention,
        "first_exit_controller": first_exit_controller,
        "first_exit_step": first_exit_step,
        "second_exit_step": second_exit_step,
        "merge_entry_step_A": merge_entry["A"],
        "merge_entry_step_B": merge_entry["B"],
        "final_progress_A": float(info["vehicles_t1"]["A"]["route_position"]),
        "final_progress_B": float(info["vehicles_t1"]["B"]["route_position"]),
        "final_speed_A": float(info["vehicles_t1"]["A"]["speed"]),
        "final_speed_B": float(info["vehicles_t1"]["B"]["speed"]),
        "minimum_TTC": (None if min_ttc is math.inf else float(min_ttc)),
        "minimum_bumper_gap": (None if min_gap is math.inf else float(min_gap)),
        "mean_stakeholder_utility": derived["mean_stakeholder_utility"],
        "minimum_stakeholder_utility": derived["minimum_stakeholder_utility"],
        "utility_A": derived["utility_A"],
        "utility_B": derived["utility_B"],
        "utility_background_front": derived["utility_background_front"],
        "utility_background_rear": derived["utility_background_rear"],
        "worst_off_stakeholder_ids_json": json.dumps(
            derived["worst_off_stakeholder_ids_json"]
        ),
        "roles0_A": roles0["A"],
        "roles0_B": roles0["B"],
        "evaluation_guard": evaluation_guard,
        "condition": "baseline",
    }
    return ep, step_rows


def run_baseline_100k_diagnostics(
    *,
    stage6a_root: Path,
    out_root: Path,
    h1_episodes_csv: Path | None = None,
) -> dict[str, Any]:
    stage6a_root = Path(stage6a_root)
    out_root = Path(out_root)
    traj_dir = out_root / "trajectories"
    recon_dir = out_root / "reconstructed_evaluations"
    traj_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_final_locks()
    pairs = validation_blocks_with_assignments(bundle)
    val_ids = sorted({bid for bid, _, _ in pairs})
    block_index_map = {bid: i for i, bid in enumerate(val_ids)}

    episode_rows: list[dict[str, Any]] = []
    all_steps: list[dict[str, Any]] = []

    for seed in FORMAL_BASELINE_SEEDS:
        job = stage6a_root / "jobs" / f"baseline__{seed}"
        weights = job / "final_online_target_weights.pt"
        if not weights.is_file():
            raise FileNotFoundError(weights)
        # Guard: only baseline
        manifest = json.loads((job / "job_manifest.json").read_text(encoding="utf-8-sig"))
        if str(manifest["condition"]) != "baseline":
            raise RuntimeError(f"non-baseline job encountered: {job.name}")
        ck_sha = sha256_file(weights)
        learners = load_learners_from_final_weights(weights, condition="baseline")
        eval_seed = int(manifest["seeds"]["evaluation_seed"])
        for block_id, assignment, block in pairs:
            b_idx = block_index_map[block_id]
            ep_seed = evaluation_episode_seed(
                eval_seed,
                checkpoint_index=CHECKPOINT_INDEX_100K,
                block_index=b_idx,
                assignment_index=int(assignment),
            )
            ep, steps = run_diagnostic_evaluation_episode(
                bundle,
                learners,
                block=block,
                assignment=int(assignment),
                block_id=block_id,
                block_index=b_idx,
                episode_seed=ep_seed,
                master_seed=seed,
                checkpoint_sha256=ck_sha,
            )
            episode_rows.append(ep)
            all_steps.extend(steps)

    ep_df = pd.DataFrame(episode_rows)
    step_df = pd.DataFrame(all_steps)
    ep_path = recon_dir / "baseline_100k_diagnostic_episodes.csv"
    step_path = traj_dir / "baseline_100k_step_log.csv"
    ep_df.to_csv(ep_path, index=False)
    # step log can be large; write compressed-friendly csv
    step_df.to_csv(step_path, index=False)

    mismatches = []
    if h1_episodes_csv and Path(h1_episodes_csv).is_file():
        h1 = pd.read_csv(h1_episodes_csv)
        h1b = h1[h1["condition"] == "baseline"].copy()
        h1b = h1b.rename(columns={"block_id": "validation_block_id"})
        for _, ep in ep_df.iterrows():
            m = h1b[
                (h1b["master_seed"] == ep["master_seed"])
                & (h1b["validation_block_id"] == ep["validation_block_id"])
                & (h1b["assignment"] == ep["assignment"])
            ]
            if m.empty:
                mismatches.append(
                    {
                        "field": "row",
                        "reason": "missing_h1",
                        "master_seed": int(ep["master_seed"]),
                        "validation_block_id": ep["validation_block_id"],
                        "assignment": int(ep["assignment"]),
                    }
                )
                continue
            h = m.iloc[0]
            checks = [
                ("success", bool(ep["success"]), bool(h["success"])),
                ("collision", bool(ep["collision"]), bool(h["collision"])),
                ("truncated", bool(ep["truncated"]), bool(h["truncated"])),
                ("termination_reason", str(ep["termination_reason"]), str(h["termination_reason"])),
                ("episode_length", int(ep["episode_length"]), int(h["episode_length"])),
            ]
            for field, a, b in checks:
                if a != b:
                    mismatches.append(
                        {
                            "field": field,
                            "diag": a,
                            "h1": b,
                            "master_seed": int(ep["master_seed"]),
                            "validation_block_id": ep["validation_block_id"],
                            "assignment": int(ep["assignment"]),
                        }
                    )
            if abs(float(ep["mean_stakeholder_utility"]) - float(h["mean_stakeholder_utility"])) > 1e-6:
                mismatches.append(
                    {
                        "field": "mean_stakeholder_utility",
                        "diag": float(ep["mean_stakeholder_utility"]),
                        "h1": float(h["mean_stakeholder_utility"]),
                        "master_seed": int(ep["master_seed"]),
                        "validation_block_id": ep["validation_block_id"],
                        "assignment": int(ep["assignment"]),
                    }
                )

    mm_path = recon_dir / "baseline_h1_nonddiagnostic_mismatches.csv"
    if mismatches:
        pd.DataFrame(mismatches).to_csv(mm_path, index=False)
    else:
        mm_path.write_text(
            "field,diag,h1,master_seed,validation_block_id,assignment,reason\n",
            encoding="utf-8",
        )

    return {
        "episode_count": len(ep_df),
        "step_row_count": len(step_df),
        "nondiagnostic_mismatch_count": len(mismatches),
        "episode_path": str(ep_path.as_posix()),
        "step_path": str(step_path.as_posix()),
        "episodes": ep_df,
        "steps": step_df,
    }
