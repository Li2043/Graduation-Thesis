#!/usr/bin/env python3
"""Stage 8 arm1 analysis: descriptive checkpoint table + frozen_stall vs
moving_with_switches mode split + direct comparison against arm0.

Structurally a copy of analyze_stage8_arm0.py (checkpoint summary +
downstream_failure trajectory diagnostics + example figures), with two
additions specific to arm1:
  (a) each downstream_failure episode is classified into "frozen_stall"
      (survivor stuck at speed=0, no action switches, no progress -- the
      mechanism arm1's exploration-schedule change targets) or
      "moving_with_switches" (survivor still moving, real action switching --
      the Q-value-flatness mechanism arm1 does NOT target), using the same
      threshold rule used in the arm0 deep-dive
      (post_exit_action_switch_count <= 1 and post_exit_route_progress_gain
      < 1.0 => frozen_stall);
  (b) arm1's per-seed-checkpoint frozen_stall counts and success rates are
      compared directly against arm0's (read from
      analysis/stage8_arm0/v1/{checkpoint_summary.csv,
      downstream_failure_trajectory_diagnostics.csv}) -- this comparison is
      arm1's actual pilot-stage success signal (see STAGE8_PLAN_DRAFT.md
      SS4: "100K checkpoint 的 frozen_stall 计数相对 arm0 的两个 seed
      （65001=0, 65002=2）是否下降/不上升"), not a PASS/FAIL gate.
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

RESULTS_ROOT = Path(__file__).resolve().parents[1] / ".." / ".." / ".." / "results" / "stage8_arm1" / "v1"
RESULTS_ROOT = RESULTS_ROOT.resolve()
OUT_ROOT = Path(__file__).resolve().parents[1] / ".." / ".." / ".." / "analysis" / "stage8_arm1" / "v1"
OUT_ROOT = OUT_ROOT.resolve()
FIG_DIR = OUT_ROOT / "figures"
ARM0_ANALYSIS_ROOT = Path(__file__).resolve().parents[1] / ".." / ".." / ".." / "analysis" / "stage8_arm0" / "v1"
ARM0_ANALYSIS_ROOT = ARM0_ANALYSIS_ROOT.resolve()

# Two-color categorical pair (survivor vs peer), fixed assignment, colorblind-
# separated (blue / orange), used consistently across all figures -- carried
# over unchanged from arm0 so the two arms' figures are visually comparable.
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


def _classify_mode(switch_count: int, progress_gain: float) -> str:
    """Same threshold used in the arm0 post-hoc deep-dive
    (analysis/stage8_arm0/v1/deep_dive_qvalue_summary.csv): a survivor with
    essentially no action switching and no route-progress gain after the peer
    exits is frozen at rest (speed stays at 0 for the entire post-exit window
    in every arm0 case checked), not oscillating."""
    if switch_count <= 1 and progress_gain < 1.0:
        return "frozen_stall"
    return "moving_with_switches"


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
                "mode": _classify_mode(switch_count, progress_gain),
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


def _arm0_comparison(arm1_diag: pd.DataFrame, arm1_ckpt: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Load arm0's already-written outputs and compare frozen_stall counts /
    success rates per seed x checkpoint against arm1's. This is arm1's actual
    pilot-stage success signal (STAGE8_PLAN_DRAFT.md SS4), not a gate."""
    notes: list[str] = []
    arm0_ckpt_path = ARM0_ANALYSIS_ROOT / "checkpoint_summary.csv"
    arm0_diag_path = ARM0_ANALYSIS_ROOT / "downstream_failure_trajectory_diagnostics.csv"
    if not arm0_ckpt_path.is_file() or not arm0_diag_path.is_file():
        notes.append(
            f"WARNING: arm0 analysis outputs not found under {ARM0_ANALYSIS_ROOT} "
            "-- skipping arm0/arm1 comparison."
        )
        return pd.DataFrame(), notes

    arm0_ckpt = pd.read_csv(arm0_ckpt_path)
    arm0_diag = pd.read_csv(arm0_diag_path)
    if not arm0_diag.empty:
        if "mode" not in arm0_diag.columns:
            arm0_diag["mode"] = [
                _classify_mode(int(sw), float(pg))
                for sw, pg in zip(
                    arm0_diag["post_exit_action_switch_count"],
                    arm0_diag["post_exit_route_progress_gain"],
                )
            ]
        arm0_frozen = (
            arm0_diag[arm0_diag["mode"] == "frozen_stall"]
            .groupby("checkpoint_step")
            .size()
            .rename("arm0_frozen_stall_count")
        )
    else:
        arm0_frozen = pd.Series(dtype=int, name="arm0_frozen_stall_count")

    if not arm1_diag.empty:
        arm1_frozen = (
            arm1_diag[arm1_diag["mode"] == "frozen_stall"]
            .groupby("checkpoint_step")
            .size()
            .rename("arm1_frozen_stall_count")
        )
    else:
        arm1_frozen = pd.Series(dtype=int, name="arm1_frozen_stall_count")

    checkpoints = sorted(set(arm0_ckpt["checkpoint_step"]) | set(arm1_ckpt["checkpoint_step"]))
    rows = []
    for ckpt in checkpoints:
        arm0_success = arm0_ckpt.loc[arm0_ckpt["checkpoint_step"] == ckpt, "success_rate"]
        arm1_success = arm1_ckpt.loc[arm1_ckpt["checkpoint_step"] == ckpt, "success_rate"]
        rows.append(
            {
                "checkpoint_step": int(ckpt),
                "arm0_mean_success_rate": float(arm0_success.mean()) if len(arm0_success) else float("nan"),
                "arm1_mean_success_rate": float(arm1_success.mean()) if len(arm1_success) else float("nan"),
                "arm0_frozen_stall_count": int(arm0_frozen.get(ckpt, 0)),
                "arm1_frozen_stall_count": int(arm1_frozen.get(ckpt, 0)),
            }
        )
    comparison = pd.DataFrame(rows)

    final = comparison[comparison["checkpoint_step"] == 100_000]
    if not final.empty:
        a0 = int(final["arm0_frozen_stall_count"].iloc[0])
        a1 = int(final["arm1_frozen_stall_count"].iloc[0])
        if a1 < a0:
            notes.append(
                f"At the 100K checkpoint, frozen_stall count DROPPED from {a0} (arm0) "
                f"to {a1} (arm1) -- consistent with the exploration-coverage hypothesis."
            )
        elif a1 > a0:
            notes.append(
                f"At the 100K checkpoint, frozen_stall count ROSE from {a0} (arm0) "
                f"to {a1} (arm1) -- refutes the exploration-coverage hypothesis as the "
                "primary lever; the epsilon-decay change did not help and may have hurt "
                "late-training convergence instead."
            )
        else:
            notes.append(
                f"At the 100K checkpoint, frozen_stall count is UNCHANGED ({a0} both arms) "
                "-- inconclusive at this sample size (2 seeds); does not confirm or refute "
                "the exploration-coverage hypothesis on its own."
            )
    return comparison, notes


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
        print(diag["mode"].value_counts().to_string())
        print(diag.describe().to_string())

    n_examples = min(3, len(examples))
    for i in range(n_examples):
        out_path = FIG_DIR / f"downstream_failure_example_{i + 1}.png"
        plot_example_episode(examples[i], out_path)
        print(f"wrote {out_path}")

    comparison, comparison_notes = _arm0_comparison(diag, ckpt_summary)
    if not comparison.empty:
        comparison.to_csv(OUT_ROOT / "comparison_vs_arm0.csv", index=False)
        print("\n=== arm0 vs arm1 comparison ===")
        print(comparison.to_string(index=False))
    for n in comparison_notes:
        print(n)

    # Plain-language note.
    note_lines = [
        "# Stage 8 arm1 — Exploration-Schedule Fix, Diagnostic Note",
        "",
        "**Status: diagnostic pilot only. No PASS/FAIL claim.**",
        "",
        "Single variable changed vs arm0: `epsilon_decay_environment_steps`",
        "50,000 -> 75,000 (epsilon_start/epsilon_end/epsilon_after_decay unchanged;",
        "reward and algorithm unchanged).",
        "",
        f"Downstream-failure episodes analysed: {len(diag)}.",
        "",
    ]
    if not diag.empty:
        mode_counts = diag["mode"].value_counts()
        note_lines.append("Mode split:")
        for mode, count in mode_counts.items():
            note_lines.append(f"- {mode}: {count} ({count / len(diag):.1%})")
        note_lines.append("")
    if comparison_notes:
        note_lines.append("## Comparison vs arm0")
        note_lines.append("")
        note_lines.extend(f"- {n}" for n in comparison_notes)
        note_lines.append("")
    note_lines += [
        "Interpretation guide: arm1 targets `frozen_stall` specifically (survivor",
        "already at speed=0 before the peer exits, never resumes). It does NOT",
        "target `moving_with_switches` (Q-value flatness causing real oscillation)",
        "-- that mechanism, if still present at a similar rate, is arm2's target",
        "(training-stability bundle), not evidence against arm1.",
    ]
    (OUT_ROOT / "STAGE8_ARM1_NOTE.md").write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_ROOT / 'STAGE8_ARM1_NOTE.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
