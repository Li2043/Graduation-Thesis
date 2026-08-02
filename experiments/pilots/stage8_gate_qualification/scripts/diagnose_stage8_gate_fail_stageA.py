#!/usr/bin/env python3
"""Stage A aggregate diagnosis of the Stage 8 formal gate FAIL.

Read-only analysis over the already-written evaluation_episodes.csv (20
seeds x 17 checkpoints, 14080 episodes). No new training, no trajectory
joins yet (that is stage B, targeted at whatever this stage flags).

Produces, printed to stdout and not written to disk (quick look first):
  1. Per-seed success at the 3 gate checkpoints (350K/375K/400K) + whether
     bimodal (reliable vs unreliable seeds) or wobbly (most seeds pass some
     checkpoints, fail others).
  2. failure_category breakdown among FAILing episodes at the 3 gate
     checkpoints, and separately for the two flagged seeds (65021, 65040).
  3. Full learning curve (200K-400K, 9 checkpoints) per seed, flagging the
     single largest adjacent drop per seed.
  4. episode_length distribution at 65040/400000 specifically (the
     background eval log showed abnormally large trajectory row counts
     there -- checking whether these are truncation/max-step episodes).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS_ROOT = Path(r"C:\Users\HP\Desktop\毕业项目\thesis\final_new_stage8\results\stage8_gate\v1")

GATE_CHECKPOINTS = (350_000, 375_000, 400_000)
LEARNING_CURVE_CHECKPOINTS = tuple(range(200_000, 400_001, 25_000))
GATE_SEED_SUCCESS_MIN = 61 / 64
FLAGGED_SEEDS = (65021, 65040)


def main() -> None:
    ep = pd.read_csv(RESULTS_ROOT / "raw" / "evaluation_episodes.csv")
    print(f"loaded {len(ep)} episodes")

    print("\n=== 1. Per-seed success at gate checkpoints ===")
    piv = ep[ep["checkpoint_step"].isin(GATE_CHECKPOINTS)].pivot_table(
        index="master_seed", columns="checkpoint_step", values="success", aggfunc="mean"
    )
    piv["qualified_all3"] = (piv[list(GATE_CHECKPOINTS)] >= GATE_SEED_SUCCESS_MIN).all(axis=1)
    piv["n_ckpts_qualified"] = (piv[list(GATE_CHECKPOINTS)] >= GATE_SEED_SUCCESS_MIN).sum(axis=1)
    print(piv.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nqualified at 0/1/2/3 gate checkpoints:", piv["n_ckpts_qualified"].value_counts().sort_index().to_dict())

    print("\n=== 2a. failure_category among FAILING episodes at gate checkpoints (all seeds) ===")
    gate_ep = ep[ep["checkpoint_step"].isin(GATE_CHECKPOINTS)]
    fail_ep = gate_ep[~gate_ep["success"]]
    print(fail_ep.groupby("checkpoint_step")["failure_category"].value_counts())

    print("\n=== 2b. failure_category for flagged seeds 65021/65040 at gate checkpoints ===")
    flagged = fail_ep[fail_ep["master_seed"].isin(FLAGGED_SEEDS)]
    print(flagged.groupby(["master_seed", "checkpoint_step"])["failure_category"].value_counts())

    print("\n=== 3. Full learning curve 200K-400K, per seed ===")
    lc = ep[ep["checkpoint_step"].isin(LEARNING_CURVE_CHECKPOINTS)].pivot_table(
        index="master_seed", columns="checkpoint_step", values="success", aggfunc="mean"
    )
    print(lc.to_string(float_format=lambda x: f"{x:.3f}"))
    drops = lc[list(LEARNING_CURVE_CHECKPOINTS)].diff(axis=1).min(axis=1) * -1
    print("\nlargest single adjacent-checkpoint drop per seed (>0.20 = material regression territory):")
    print(drops.sort_values(ascending=False).to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n=== 4. episode_length distribution, seed 65040 @ 400000 vs overall ===")
    e40 = ep[(ep["master_seed"] == 65040) & (ep["checkpoint_step"] == 400_000)]
    print(e40["episode_length"].describe())
    print("failure_category counts:", e40["failure_category"].value_counts().to_dict())
    print("terminated/truncated counts:", e40[["terminated", "truncated"]].sum().to_dict())
    print("\noverall episode_length distribution (all episodes, all checkpoints):")
    print(ep["episode_length"].describe())

    print("\n=== 4b. same for seed 65021 across gate checkpoints ===")
    e21 = ep[(ep["master_seed"] == 65021) & (ep["checkpoint_step"].isin(GATE_CHECKPOINTS))]
    print(e21.groupby("checkpoint_step")["episode_length"].describe())


if __name__ == "__main__":
    main()
