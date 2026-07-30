"""Scripted base-reward incentive audit (read-only; no reward changes)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from thesis.audits.audit_metrics import discounted_return, undiscounted_return
from thesis.envs.merge_env_candidate_v3 import HighLevelAction
from thesis.training.final_lock_loader import load_final_locks
from thesis.training.pilot_ic_schedule import build_env_for_block, validation_blocks_with_assignments

ACC = int(HighLevelAction.ACCELERATE)
MNT = int(HighLevelAction.MAINTAIN)
DEC = int(HighLevelAction.DECELERATE)
GAMMA = 0.995


SCRIPTS = {
    "maintain_only": lambda n: [{"A": MNT, "B": MNT} for _ in range(n)],
    "mutual_yield_decel": lambda n: [{"A": DEC, "B": DEC} for _ in range(min(40, n))]
    + [{"A": MNT, "B": MNT} for _ in range(n)],
    "accelerate_insist": lambda n: [{"A": ACC, "B": ACC} for _ in range(n)],
    "mainline_bias_success_attempt": lambda n: (
        [{"A": ACC, "B": MNT} for _ in range(min(30, n))]
        + [{"A": ACC, "B": DEC} for _ in range(min(40, n))]
        + [{"A": ACC, "B": ACC} for _ in range(n)]
    ),
}


def _run_script(env, actions: list[dict[str, int]]) -> dict[str, Any]:
    obs, _ = env.reset(seed=0)
    base_a: list[float] = []
    base_b: list[float] = []
    prog_a = prog_b = exit_a = exit_b = coll = brake = 0.0
    steps = 0
    term_reason = "ongoing"
    terminated = truncated = False
    info: dict[str, Any] = {}
    for act in actions:
        obs, rewards, terminated, truncated, info = env.step(act)
        steps += 1
        ca = info["components"]["A"]
        cb = info["components"]["B"]
        base_a.append(float(ca["total_base_reward"]))
        base_b.append(float(cb["total_base_reward"]))
        prog_a += float(ca.get("progress_component", 0.0))
        prog_b += float(cb.get("progress_component", 0.0))
        exit_a += float(ca.get("exit_component", 0.0))
        exit_b += float(cb.get("exit_component", 0.0))
        coll += float(ca.get("collision_component", 0.0)) + float(cb.get("collision_component", 0.0))
        brake += float(ca.get("hard_braking_component", 0.0)) + float(
            cb.get("hard_braking_component", 0.0)
        )
        term_reason = str(info["term_reason"])
        if terminated or truncated:
            break
    return {
        "episode_length": steps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "term_reason": term_reason,
        "G_A_undiscounted": undiscounted_return(base_a),
        "G_B_undiscounted": undiscounted_return(base_b),
        "G_A_discounted": discounted_return(base_a, GAMMA),
        "G_B_discounted": discounted_return(base_b, GAMMA),
        "progress_A": prog_a,
        "progress_B": prog_b,
        "exit_A": exit_a,
        "exit_B": exit_b,
        "collision_component_sum": coll,
        "hard_braking_component_sum": brake,
        "success": term_reason == "success",
    }


def run_reward_audit(*, out_csv: Path, max_steps: int = 400) -> dict[str, Any]:
    bundle = load_final_locks()
    pairs = validation_blocks_with_assignments(bundle)
    rows: list[dict[str, Any]] = []
    for block_id, assignment, block in pairs:
        env = build_env_for_block(bundle, block, max_policy_steps=max_steps)
        for script_id, builder in SCRIPTS.items():
            actions = builder(max_steps)
            out = _run_script(env, actions)
            rows.append(
                {
                    "validation_block_id": block_id,
                    "assignment": int(assignment),
                    "script_id": script_id,
                    **out,
                }
            )
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Separability: successful-ish scripts vs stall scripts per block/assignment
    sep_rows = []
    weak_any = False
    for (bid, asn), g in df.groupby(["validation_block_id", "assignment"]):
        succ = g[g["script_id"] == "mainline_bias_success_attempt"].iloc[0]
        stall = g[g["script_id"] == "mutual_yield_decel"].iloc[0]
        maintain = g[g["script_id"] == "maintain_only"].iloc[0]
        for agent, col in (("A", "G_A_discounted"), ("B", "G_B_discounted")):
            gs = float(succ[col])
            diff_stall = gs - float(stall[col])
            diff_m = gs - float(maintain[col])
            denom = abs(gs) + 1e-8
            r_stall = diff_stall / denom
            r_m = diff_m / denom
            weak = (diff_stall <= 0.05) or (r_stall < 0.05)
            weak_any = weak_any or weak
            sep_rows.append(
                {
                    "validation_block_id": bid,
                    "assignment": int(asn),
                    "controller": agent,
                    "success_minus_stall": diff_stall,
                    "success_minus_maintain": diff_m,
                    "R_separation_stall": r_stall,
                    "R_separation_maintain": r_m,
                    "weak_reward_separation": weak,
                }
            )
    sep = pd.DataFrame(sep_rows)
    sep_path = out_csv.with_name("baseline_reward_separability.csv")
    sep.to_csv(sep_path, index=False)
    return {
        "audit_csv": str(out_csv.as_posix()),
        "separability_csv": str(sep_path.as_posix()),
        "weak_reward_separation_any": bool(weak_any),
        "weak_fraction": float(sep["weak_reward_separation"].mean()) if len(sep) else 0.0,
        "mean_R_separation_stall": float(sep["R_separation_stall"].mean()) if len(sep) else 0.0,
    }
