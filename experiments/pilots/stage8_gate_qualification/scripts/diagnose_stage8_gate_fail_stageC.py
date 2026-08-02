#!/usr/bin/env python3
"""Stage C: deeper mechanism dive on collision + swap_eligibility, requested
after Stage A/B already confirmed the frozen_stall/moving_with_switches
picture. Read-only, reuses the already-written evaluation_episodes.csv and
trajectory CSVs (gate checkpoints only: 350K/375K/400K).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS_ROOT = Path(r"C:\Users\HP\Desktop\毕业项目\thesis\final_new_stage8\results\stage8_gate\v1")
GATE_CHECKPOINTS = (350_000, 375_000, 400_000)


def load_gate_episodes() -> pd.DataFrame:
    ep = pd.read_csv(RESULTS_ROOT / "raw" / "evaluation_episodes.csv")
    ep_gate = ep[ep["checkpoint_step"].isin(GATE_CHECKPOINTS)].copy()
    ep_gate["episode_index"] = ep_gate["episode_index"].astype(int)
    ep_gate["assignment"] = ep_gate["assignment"].astype(int)
    ep_gate["scenario_block"] = ep_gate["scenario_block"].astype(int)
    return ep_gate


def collision_deep_dive(ep_gate: pd.DataFrame) -> None:
    coll_ep = ep_gate[ep_gate["failure_category"] == "collision"]
    print(f"=== COLLISION: {len(coll_ep)} episodes at gate checkpoints ===")

    print("\n-- concentration across seeds --")
    print(coll_ep["master_seed"].value_counts())

    print("\n-- concentration across seeds x checkpoint --")
    print(coll_ep.groupby(["master_seed", "checkpoint_step"]).size().unstack(fill_value=0))

    traj_dir = RESULTS_ROOT / "raw" / "trajectories"
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
        ep_traj = traj.loc[mask].sort_values("policy_step")
        if ep_traj.empty:
            continue
        last_step = ep_traj["policy_step"].max()
        final_rows = ep_traj[ep_traj["policy_step"] == last_step]
        role_map = {"A": row.get("controller_role_mapping"), }
        for _, r2 in final_rows.iterrows():
            rows.append(
                {
                    "seed": seed,
                    "checkpoint_step": ckpt,
                    "controller": r2.get("controller"),
                    "controller_role": r2.get("controller_role"),
                    "commanded_action_name": r2.get("commanded_action_name"),
                    "speed": r2.get("speed"),
                    "Q_margin": r2.get("Q_margin"),
                    "front_gap": r2.get("front_gap"),
                    "minimum_TTC": r2.get("minimum_TTC"),
                    "joint_action_category": r2.get("joint_action_category"),
                }
            )
    coll_df = pd.DataFrame(rows)
    if coll_df.empty:
        print("no trajectory rows recovered")
        return

    print(f"\n-- final-step rows recovered: {len(coll_df)} (expect ~2 per episode) --")
    print("\n-- joint_action_category at moment of collision --")
    print(coll_df["joint_action_category"].value_counts())

    print("\n-- action by role (mainline vs ramp) at moment of collision --")
    print(coll_df.groupby("controller_role")["commanded_action_name"].value_counts())

    print("\n-- front_gap at moment of collision --")
    print(coll_df["front_gap"].describe())

    print("\n-- minimum_TTC at moment of collision --")
    print(coll_df["minimum_TTC"].describe())

    print("\n-- speed at moment of collision, by role --")
    print(coll_df.groupby("controller_role")["speed"].describe())


def swap_eligibility_deep_dive(ep_gate: pd.DataFrame) -> None:
    print("\n\n=== SWAP_ELIGIBILITY deep dive ===")
    rows = []
    for (seed, ckpt, block), g in ep_gate.groupby(["master_seed", "checkpoint_step", "scenario_block"]):
        a = g[g["assignment"] == 0]
        b = g[g["assignment"] == 1]
        if len(a) != 1 or len(b) != 1:
            continue
        a = a.iloc[0]
        b = b.iloc[0]
        if not (a["success"] and b["success"]):
            cat = "one_or_both_failed"
            fixed_order = None
        else:
            oa, ob = str(a["passing_order"]), str(b["passing_order"])
            if {oa, ob} == {"mainline_first", "ramp_first"}:
                cat = "complementary"
                fixed_order = None
            else:
                cat = "same_order"
                fixed_order = oa
        rows.append(
            {
                "master_seed": seed,
                "checkpoint_step": ckpt,
                "scenario_block": block,
                "category": cat,
                "fixed_order": fixed_order,
                "role_map_a": a.get("controller_role_mapping"),
            }
        )
    df = pd.DataFrame(rows)
    print(f"total blocks analysed: {len(df)}")

    same = df[df["category"] == "same_order"]
    print("\n-- among 'same_order' blocks, which order dominates --")
    print(same["fixed_order"].value_counts())

    print("\n-- 'same_order' rate by seed (top/bottom 5) --")
    rate_by_seed = df.groupby("master_seed")["category"].apply(lambda s: (s == "same_order").mean())
    print(rate_by_seed.sort_values(ascending=False))

    print("\n-- 'same_order' rate by checkpoint --")
    print(df.groupby("checkpoint_step")["category"].apply(lambda s: (s == "same_order").mean()))

    # Check whether assignment=0 always maps to the same role (i.e. is
    # "mainline_first" simply always won by whichever controller is A?).
    print("\n-- role_map for assignment=0 rows, sample (does A/B<->mainline/ramp itself flip across the swap pair?) --")
    print(df["role_map_a"].value_counts().head(5))


def main() -> None:
    ep_gate = load_gate_episodes()
    collision_deep_dive(ep_gate)
    swap_eligibility_deep_dive(ep_gate)


if __name__ == "__main__":
    main()
