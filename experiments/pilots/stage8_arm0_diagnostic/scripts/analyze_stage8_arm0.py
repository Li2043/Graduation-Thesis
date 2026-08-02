#!/usr/bin/env python3
"""Stage 8 arm0 analysis: descriptive checkpoint table + downstream_failure hypothesis test.

Lighter than analyze_stage7c_q1_v1.py -- no gate machinery, no integrity/hash
checks, no historical comparison (not applicable to a 2-seed diagnostic run).
Keeps two things:
  (a) a descriptive per-seed checkpoint table (no bootstrap CI: n=2 seeds is
      too small for meaningful CIs, reported per-seed instead), and
  (b) the actual hypothesis test: for every downstream_failure episode, join
      the post-peer-exit portion of the survivor's trajectory and compute
      progress/gap/TTC/oscillation/braking diagnostics.
"""

from __future__ import annotations

import json
import sys
from ast import literal_eval
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

from thesis.rewards.base_reward_v2 import compute_hard_braking_cost  # noqa: E402
from thesis.training.final_lock_loader import load_final_locks  # noqa: E402

RESULTS_ROOT = Path(__file__).resolve().parents[1] / ".." / ".." / ".." / "results" / "stage8_arm0" / "v1"
RESULTS_ROOT = RESULTS_ROOT.resolve()
OUT_ROOT = Path(__file__).resolve().parents[1] / ".." / ".." / ".." / "analysis" / "stage8_arm0" / "v1"
OUT_ROOT = OUT_ROOT.resolve()
FIG_DIR = OUT_ROOT / "figures"

# Two-color categorical pair (survivor vs peer), fixed assignment, colorblind-
# separated (blue / orange), used consistently across all figures.
COLOR_SURVIVOR = "#2E5FA3"
COLOR_PEER = "#D2691E"


def _parse_role_mapping(value) -> dict[str, str]:
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        try:
            return literal_eval(value)
        except (TypeError, ValueError, SyntaxError):
            return {}


def build_checkpoint_summary(ep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, ckpt), g in ep.groupby(["master_seed", "checkpoint_step"]):
        n = len(g)
        rows.append(
            {
                "master_seed": int(seed),
                "checkpoint_step": int(ckpt),
                "n_episodes": n,
                "success_rate": float(g["success"].mean()),
                "collision_rate": float(g["collision"].mean()),
                "truncation_rate": float(g["truncation"].mean()),
                "downstream_failure_rate": float((g["failure_category"] == "downstream_failure").mean()),
                "unilateral_stall_rate": float((g["failure_category"] == "unilateral_stall").mean()),
                "mutual_yielding_rate": float((g["failure_category"] == "mutual_yielding").mean()),
                "other_failure_rate": float((g["failure_category"] == "other_failure").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["master_seed", "checkpoint_step"]).reset_index(drop=True)


def _survivor_and_peer_exit(row: pd.Series) -> tuple[str, int] | None:
    role_map = _parse_role_mapping(row.get("controller_role_mapping"))
    exit0 = row.get("exit_step_agent_0")
    exit1 = row.get("exit_step_agent_1")
    a_exited = pd.notna(exit0)
    b_exited = pd.notna(exit1)
    if a_exited and not b_exited:
        return "B", int(exit0)
    if b_exited and not a_exited:
        return "A", int(exit1)
    return None


def build_downstream_failure_diagnostics(
    ep: pd.DataFrame, traj_dir: Path, *, a_comfort: float, a_hard: float
) -> tuple[pd.DataFrame, list[dict]]:
    df_eps = ep[ep["failure_category"] == "downstream_failure"].copy()
    diag_rows: list[dict] = []
    example_episodes: list[dict] = []
    traj_cache: dict[tuple[int, int], pd.DataFrame] = {}

    for _, row in df_eps.iterrows():
        res = _survivor_and_peer_exit(row)
        if res is None:
            continue
        survivor, peer_exit_step = res
        seed = int(row["master_seed"])
        ckpt = int(row["checkpoint_step"])
        key = (seed, ckpt)
        if key not in traj_cache:
            traj_path = traj_dir / f"seed_{seed}_traj_step_{ckpt}.csv"
            if not traj_path.is_file():
                traj_cache[key] = pd.DataFrame()
            else:
                traj_cache[key] = pd.read_csv(traj_path)
        traj = traj_cache[key]
        if traj.empty:
            continue

        ep_mask = (
            (traj["validation_block_id"] == row["validation_block_id"])
            & (traj["assignment"] == row["assignment"])
            & (traj["scenario_block"] == row["scenario_block"])
            & (traj["episode_index"] == row["episode_index"])
        )
        ep_traj = traj.loc[ep_mask]
        if ep_traj.empty:
            continue

        surv = ep_traj[ep_traj["controller"] == survivor].sort_values("policy_step")
        post = surv[surv["policy_step"] > peer_exit_step]
        if post.empty:
            continue

        actions = post["commanded_action_name"].tolist()
        switch_count = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i - 1])
        decel_fraction = float((post["commanded_action_name"] == "decelerate").mean())

        hb_sum = 0.0
        for accel in post["realised_acceleration"].dropna().tolist():
            hb_sum += compute_hard_braking_cost(float(accel), a_comfort, a_hard)

        progress_gain = (
            float(post["route_progress"].iloc[-1] - post["route_progress"].iloc[0])
            if len(post) >= 2
            else 0.0
        )

        diag_rows.append(
            {
                "master_seed": seed,
                "checkpoint_step": ckpt,
                "validation_block_id": row["validation_block_id"],
                "assignment": int(row["assignment"]),
                "episode_index": int(row["episode_index"]),
                "survivor": survivor,
                "peer_exit_step": peer_exit_step,
                "episode_length": int(row["episode_length"]),
                "post_exit_duration_steps": int(len(post)),
                "post_exit_route_progress_gain": progress_gain,
                "post_exit_mean_front_gap": float(post["front_gap"].dropna().mean())
                if "front_gap" in post and post["front_gap"].notna().any()
                else float("nan"),
                "post_exit_min_front_gap": float(post["front_gap"].dropna().min())
                if "front_gap" in post and post["front_gap"].notna().any()
                else float("nan"),
                "post_exit_min_ttc": float(post["minimum_TTC"].dropna().min())
                if "minimum_TTC" in post and post["minimum_TTC"].notna().any()
                else float("nan"),
                "post_exit_decelerate_fraction": decel_fraction,
                "post_exit_action_switch_count": switch_count,
                "post_exit_hard_braking_cost_sum": hb_sum,
            }
        )
        example_episodes.append(
            {
                "seed": seed,
                "checkpoint_step": ckpt,
                "survivor": survivor,
                "peer_exit_step": peer_exit_step,
                "trajectory": ep_traj,
            }
        )

    return pd.DataFrame(diag_rows), example_episodes


def plot_example_episode(example: dict, out_path: Path) -> None:
    traj = example["trajectory"]
    survivor = example["survivor"]
    peer = "B" if survivor == "A" else "A"
    peer_exit_step = example["peer_exit_step"]

    surv = traj[traj["controller"] == survivor].sort_values("policy_step")
    peer_df = traj[traj["controller"] == peer].sort_values("policy_step")

    fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)

    ax = axes[0]
    ax.plot(surv["simulation_time"], surv["route_progress"], color=COLOR_SURVIVOR, linewidth=2, label="survivor")
    ax.plot(peer_df["simulation_time"], peer_df["route_progress"], color=COLOR_PEER, linewidth=2, label="peer")
    ax.axvline(peer_exit_step * 0.2, color="grey", linestyle="--", linewidth=1)
    ax.set_ylabel("route progress (m)")
    ax.legend(loc="upper left", frameon=False)
    ax.set_title(
        f"seed={example['seed']} checkpoint={example['checkpoint_step']} "
        f"survivor={survivor} (dashed line = peer exit)"
    )

    ax = axes[1]
    if "front_gap" in surv:
        ax.plot(surv["simulation_time"], surv["front_gap"], color=COLOR_SURVIVOR, linewidth=1.5)
    ax.axvline(peer_exit_step * 0.2, color="grey", linestyle="--", linewidth=1)
    ax.set_ylabel("front gap (m)")

    ax = axes[2]
    if "minimum_TTC" in surv:
        ax.plot(surv["simulation_time"], surv["minimum_TTC"], color=COLOR_SURVIVOR, linewidth=1.5)
    ax.axvline(peer_exit_step * 0.2, color="grey", linestyle="--", linewidth=1)
    ax.set_ylabel("time-to-collision (s)")

    ax = axes[3]
    action_code = surv["commanded_action_name"].map(
        {"maintain": 0, "accelerate": 1, "decelerate": -1}
    )
    ax.step(surv["simulation_time"], action_code, color=COLOR_SURVIVOR, linewidth=1.5, where="post")
    ax.axvline(peer_exit_step * 0.2, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["decelerate", "maintain", "accelerate"])
    ax.set_ylabel("survivor action")
    ax.set_xlabel("simulation time (s)")

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    ep_path = RESULTS_ROOT / "raw" / "evaluation_episodes.csv"
    if not ep_path.is_file():
        print(f"ABORT: missing {ep_path}", file=sys.stderr)
        return 1
    ep = pd.read_csv(ep_path)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_summary = build_checkpoint_summary(ep)
    ckpt_summary.to_csv(OUT_ROOT / "checkpoint_summary.csv", index=False)
    print("=== Per-seed checkpoint summary (n=2 seeds; not bootstrap-CI'd) ===")
    print(ckpt_summary.to_string(index=False))

    bundle = load_final_locks()
    a_comfort = float(bundle.comfort.a_comfort)
    a_hard = float(bundle.comfort.a_hard)

    traj_dir = RESULTS_ROOT / "raw" / "trajectories"
    diag, examples = build_downstream_failure_diagnostics(
        ep, traj_dir, a_comfort=a_comfort, a_hard=a_hard
    )
    diag.to_csv(OUT_ROOT / "downstream_failure_trajectory_diagnostics.csv", index=False)
    print(f"\n=== downstream_failure trajectory diagnostics: {len(diag)} episodes ===")
    if not diag.empty:
        print(diag.describe().to_string())

    n_examples = min(3, len(examples))
    for i in range(n_examples):
        out_path = FIG_DIR / f"downstream_failure_example_{i + 1}.png"
        plot_example_episode(examples[i], out_path)
        print(f"wrote {out_path}")

    # Plain-language note.
    note_lines = [
        "# Stage 8 arm0 — Diagnostic Note",
        "",
        "**Status: diagnostic only. No PASS/FAIL claim. No reward or algorithm change.**",
        "",
        f"Downstream-failure episodes analysed: {len(diag)}.",
        "",
    ]
    if not diag.empty:
        mean_gain = diag["post_exit_route_progress_gain"].mean()
        mean_hb = diag["post_exit_hard_braking_cost_sum"].mean()
        mean_switch = diag["post_exit_action_switch_count"].mean()
        mean_dur = diag["post_exit_duration_steps"].mean()
        note_lines += [
            f"- Mean post-peer-exit duration: {mean_dur:.1f} policy steps "
            f"({mean_dur * 0.2:.1f} s).",
            f"- Mean post-peer-exit route-progress gain: {mean_gain:.3f} m.",
            f"- Mean post-peer-exit hard-braking cost sum: {mean_hb:.3f}.",
            f"- Mean post-peer-exit action-switch count: {mean_switch:.1f}.",
            "",
            "Interpretation guide (fill in after inspecting the numbers/figures above):",
            "the oscillation hypothesis from paper/STAGE8_PLAN_DRAFT.md SS1 predicts low",
            "route-progress gain, a non-trivial hard-braking cost sum, and a high",
            "action-switch count relative to duration, concentrated near a small",
            "front_gap / low minimum_TTC. If instead progress gain is high or hard-braking",
            "cost is near zero, the oscillation hypothesis is refuted and a different",
            "mechanism should be investigated.",
        ]
    else:
        note_lines.append(
            "No downstream_failure episodes were observed in this arm0 run "
            "(2 seeds x 100K is a much smaller sample than Stage 7C-Q1's 20 x 400K, "
            "so this alone does not refute the hypothesis)."
        )
    (OUT_ROOT / "STAGE8_ARM0_NOTE.md").write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_ROOT / 'STAGE8_ARM0_NOTE.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
