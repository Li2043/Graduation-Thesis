#!/usr/bin/env python3
"""One-off: compute a Stage-6B-H1-style corrected mean mobility (U_i) for the
two LEARNING controllers (A, B) from the Stage 8 gate's baseline rich
trajectories (350K/375K/400K), to anchor Stage 9's non-inferiority margin.

NOTE: this is A/B only, not the full 4-stakeholder V={A,B,B_front,B_rear}
Chapter 2 potential -- B_front/B_rear speed/target-speed telemetry is not
present in these per-step trajectory logs (only "controller" in {A,B} rows
exist). target_speed is a single fixed constant (20.0 m/s, see
merge_env_v2.py MergeEnvConfig.target_speed) shared by all stakeholders in
this environment, which is what makes an A/B-only proxy at least directly
comparable in scale to the full formula.

U_i(episode) = 0                              if i was in a collision
             = mean over active on-road steps of clip01(speed_i/20.0)   otherwise
"active on-road" = rows where the controller has not yet exited (exited==False)
and is not post-terminal.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

RESULTS = Path(r"C:\Users\HP\Desktop\毕业项目\thesis\final_new_stage8\results\stage8_gate\v1")
TARGET_SPEED = 20.0
GATE_CHECKPOINTS = (350_000, 375_000, 400_000)


def main() -> None:
    ep = pd.read_csv(RESULTS / "raw" / "evaluation_episodes.csv")
    ep_gate = ep[ep["checkpoint_step"].isin(GATE_CHECKPOINTS)].copy()
    ep_gate["episode_index"] = ep_gate["episode_index"].astype(int)
    ep_gate["assignment"] = ep_gate["assignment"].astype(int)
    ep_gate["scenario_block"] = ep_gate["scenario_block"].astype(int)

    traj_dir = RESULTS / "raw" / "trajectories"
    traj_cache: dict[tuple[int, int], pd.DataFrame] = {}

    rows = []
    for _, erow in ep_gate.iterrows():
        seed = int(erow["master_seed"])
        ckpt = int(erow["checkpoint_step"])
        key = (seed, ckpt)
        if key not in traj_cache:
            p = traj_dir / f"seed_{seed}_traj_step_{ckpt}.csv"
            traj_cache[key] = pd.read_csv(p) if p.is_file() else pd.DataFrame()
        traj = traj_cache[key]
        if traj.empty:
            continue
        mask = (
            (traj["validation_block_id"] == erow["validation_block_id"])
            & (traj["assignment"] == erow["assignment"])
            & (traj["scenario_block"] == erow["scenario_block"])
            & (traj["episode_index"] == erow["episode_index"])
        )
        et = traj.loc[mask]
        if et.empty:
            continue
        collided = bool(erow["collision"])
        for controller in ("A", "B"):
            sub = et[et["controller"] == controller].sort_values("policy_step")
            if sub.empty:
                continue
            active = sub[sub["active"] == True] if "active" in sub.columns else sub  # noqa: E712
            if collided:
                u = 0.0
            elif active.empty:
                u = float("nan")
            else:
                e = (active["speed"] / TARGET_SPEED).clip(0.0, 1.0)
                u = float(e.mean())
            rows.append(
                {
                    "master_seed": seed,
                    "checkpoint_step": ckpt,
                    "controller": controller,
                    "U_i": u,
                }
            )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["U_i"])
    print(f"episode-controller rows: {len(df)}")

    # mean over A,B per episode, matching Chapter 2's "mean stakeholder mobility" spirit (restricted to learners)
    ep_mean = df.groupby(["master_seed", "checkpoint_step"])["U_i"].mean().reset_index()
    print("\n=== per-checkpoint mean learner mobility (A,B averaged, then averaged over episodes/seeds implicitly via row-level mean) ===")
    print(df.groupby("checkpoint_step")["U_i"].agg(["mean", "std", "count"]))

    print("\n=== pooled across 350K/375K/400K ===")
    print("mean:", df["U_i"].mean(), "sd:", df["U_i"].std(ddof=1), "n_rows:", len(df))

    overall_mean = df["U_i"].mean()
    print(f"\nproposed margin anchor: baseline learner mean mobility ~= {overall_mean:.4f}")
    print(f"10% relative margin => acceptable floor for mean_pbrs/min_pbrs ~= {overall_mean*0.9:.4f}")


if __name__ == "__main__":
    main()
