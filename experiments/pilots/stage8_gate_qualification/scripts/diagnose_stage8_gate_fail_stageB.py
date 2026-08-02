#!/usr/bin/env python3
"""Stage B trajectory-level diagnosis of the Stage 8 formal gate FAIL,
targeted at what Stage A flagged: (1) is the downstream_failure collapse at
gate checkpoints the same frozen_stall mechanism arm0 found, (2) what does
the increasing collision rate at 400K look like at the Q-value level, (3)
why is swap_eligibility so low even when raw success is high.

Read-only, reuses arm0's post-peer-exit diagnostic logic
(analyze_stage8_arm0.py::build_downstream_failure_diagnostics) unmodified,
just pointed at the gate's results and restricted to the 3 gate checkpoints
(the only ones with rich trajectory logs, aside from checkpoint 0).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(r"C:\Users\HP\Desktop\毕业项目\thesis\final_new_stage8")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "pilots" / "stage8_arm0_diagnostic" / "scripts"))

from thesis.training.final_lock_loader import load_final_locks  # noqa: E402
from analyze_stage8_arm0 import build_downstream_failure_diagnostics  # noqa: E402

RESULTS_ROOT = REPO / "results" / "stage8_gate" / "v1"
GATE_CHECKPOINTS = (350_000, 375_000, 400_000)


def main() -> None:
    ep = pd.read_csv(RESULTS_ROOT / "raw" / "evaluation_episodes.csv")
    ep_gate = ep[ep["checkpoint_step"].isin(GATE_CHECKPOINTS)].copy()
    # episode_index is a plain int for the rich evaluator (used at all 3 gate
    # checkpoints) but a composite string ID for the lightweight evaluator
    # (used elsewhere) -- appending both to one CSV makes pandas infer the
    # whole column as object/string on read, so '7' vs int64 7 never matches
    # in the trajectory join below. Safe to cast here since ep_gate is
    # restricted to rich-only checkpoints.
    ep_gate["episode_index"] = ep_gate["episode_index"].astype(int)
    ep_gate["assignment"] = ep_gate["assignment"].astype(int)
    ep_gate["scenario_block"] = ep_gate["scenario_block"].astype(int)

    bundle = load_final_locks()
    a_comfort = float(bundle.comfort.a_comfort)
    a_hard = float(bundle.comfort.a_hard)

    traj_dir = RESULTS_ROOT / "raw" / "trajectories"
    diag, examples = build_downstream_failure_diagnostics(
        ep_gate, traj_dir, a_comfort=a_comfort, a_hard=a_hard
    )
    print(f"=== downstream_failure diagnostics at gate checkpoints: {len(diag)} episodes ===")
    if diag.empty:
        print("EMPTY -- nothing to classify")
        return

    diag["mode"] = (
        (diag["post_exit_action_switch_count"] <= 1) & (diag["post_exit_route_progress_gain"] < 1.0)
    ).map({True: "frozen_stall", False: "moving_with_switches"})

    print("\n=== 1. Sub-mode counts (all gate-checkpoint downstream_failure episodes) ===")
    print(diag["mode"].value_counts())

    print("\n=== 1b. Sub-mode counts, seed 65040 only ===")
    print(diag[diag["master_seed"] == 65040]["mode"].value_counts())
    print(diag[diag["master_seed"] == 65040].groupby("checkpoint_step")["mode"].value_counts())

    print("\n=== 2. Sub-mode summary stats (all seeds) ===")
    print(
        diag.groupby("mode")[
            [
                "post_exit_route_progress_gain",
                "post_exit_decelerate_fraction",
                "post_exit_action_switch_count",
                "post_exit_hard_braking_cost_sum",
                "post_exit_duration_steps",
            ]
        ].mean()
    )

    print("\n=== 3. Does frozen_stall concentrate in a few seeds, or spread across many? ===")
    stall = diag[diag["mode"] == "frozen_stall"]
    print(stall["master_seed"].value_counts())

    print("\n=== 4. frozen_stall by checkpoint (is it worse at 400K specifically?) ===")
    print(diag.groupby("checkpoint_step")["mode"].value_counts())

    # --- Collision episodes: quick Q-margin look near end-of-episode ---
    print("\n\n=== 5. Collision episodes at gate checkpoints: last-step Q_margin/action ===")
    coll_ep = ep_gate[ep_gate["failure_category"] == "collision"]
    print(f"{len(coll_ep)} collision episodes at gate checkpoints")
    print(coll_ep.groupby("checkpoint_step").size())

    traj_cache: dict[tuple[int, int], pd.DataFrame] = {}
    rows = []
    for _, row in coll_ep.iterrows():
        seed = int(row["master_seed"])
        ckpt = int(row["checkpoint_step"])
        key = (seed, ckpt)
        if key not in traj_cache:
            p = traj_dir / f"seed_{seed}_traj_step_{ckpt}.csv"
            traj_cache[key] = pd.read_csv(p) if p.is_file() else pd.DataFrame()
        traj = traj_cache[key]
        if traj.empty:
            continue
        mask = (
            (traj["validation_block_id"] == row["validation_block_id"])
            & (traj["assignment"] == row["assignment"])
            & (traj["scenario_block"] == row["scenario_block"])
            & (traj["episode_index"] == row["episode_index"])
        )
        ep_traj = traj.loc[mask]
        if ep_traj.empty:
            continue
        last_step = ep_traj["policy_step"].max()
        last_rows = ep_traj[ep_traj["policy_step"] >= last_step - 1]
        for _, r2 in last_rows.iterrows():
            rows.append(
                {
                    "seed": seed,
                    "checkpoint_step": ckpt,
                    "controller": r2.get("controller"),
                    "policy_step": r2.get("policy_step"),
                    "Q_margin": r2.get("Q_margin"),
                    "commanded_action_name": r2.get("commanded_action_name"),
                    "speed": r2.get("speed"),
                }
            )
    coll_df = pd.DataFrame(rows)
    if not coll_df.empty:
        print("\nlast-2-step Q_margin distribution (collision episodes):")
        print(coll_df["Q_margin"].describe())
        print("\nlast-2-step action distribution (collision episodes):")
        print(coll_df["commanded_action_name"].value_counts())
        print("\nby checkpoint:")
        print(coll_df.groupby("checkpoint_step")["Q_margin"].describe())
    else:
        print("no trajectory rows recovered for collision episodes (may be lightweight-only checkpoints)")

    # --- swap_eligibility root cause ---
    print("\n\n=== 6. swap_eligibility root cause: passing_order distribution ===")
    print(ep_gate["passing_order"].value_counts(dropna=False))
    print("\nby checkpoint:")
    print(ep_gate.groupby("checkpoint_step")["passing_order"].value_counts(dropna=False))
    succ = ep_gate[ep_gate["success"]]
    print("\namong SUCCESSFUL episodes only, passing_order distribution:")
    print(succ["passing_order"].value_counts(dropna=False))
    print("\nby checkpoint (successful only):")
    print(succ.groupby("checkpoint_step")["passing_order"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
