#!/usr/bin/env python3
"""Stage 8 arm2a analysis: descriptive checkpoint table + frozen_stall vs
moving_with_switches mode split + training-stability comparison against
arm0 and arm1.

Structurally a copy of analyze_stage8_arm1.py, with the comparison section
extended in two ways specific to arm2a/arm2b's target mechanism (seed-level
training instability, not a specific failure sub-mode):
  (a) per-seed success_rate volatility (max - min success_rate across the
      5 checkpoints) is computed for every arm and compared -- this is
      arm2a's actual target (soft target update is meant to reduce
      checkpoint-to-checkpoint swings), not just the 100K-checkpoint
      frozen_stall count that arm1 targeted;
  (b) the comparison pulls in arm1's results too (not just arm0's), since
      arm2a is a sibling single-variable arm to arm1, not a replacement for
      it -- STAGE8_PLAN_DRAFT.md SS4 frames all three (arm1, arm2a, arm2b)
      as separate tests against the same arm0 baseline.
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

ARM_NAME = "stage8_arm2a"
RESULTS_ROOT = Path(__file__).resolve().parents[1] / ".." / ".." / ".." / "results" / ARM_NAME / "v1"
RESULTS_ROOT = RESULTS_ROOT.resolve()
OUT_ROOT = Path(__file__).resolve().parents[1] / ".." / ".." / ".." / "analysis" / ARM_NAME / "v1"
OUT_ROOT = OUT_ROOT.resolve()
FIG_DIR = OUT_ROOT / "figures"
ANALYSIS_BASE = Path(__file__).resolve().parents[1] / ".." / ".." / ".." / "analysis"
ANALYSIS_BASE = ANALYSIS_BASE.resolve()
ARM0_ANALYSIS_ROOT = ANALYSIS_BASE / "stage8_arm0" / "v1"
ARM1_ANALYSIS_ROOT = ANALYSIS_BASE / "stage8_arm1" / "v1"

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
            traj_cache[key] = pd.read_csv(traj_path) if traj_path.is_file() else pd.DataFrame()
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


def _load_prior_arm(analysis_root: Path, arm_label: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    ckpt_path = analysis_root / "checkpoint_summary.csv"
    diag_path = analysis_root / "downstream_failure_trajectory_diagnostics.csv"
    if not ckpt_path.is_file() or not diag_path.is_file():
        print(f"WARNING: {arm_label} analysis outputs not found under {analysis_root} -- skipping.")
        return None
    ckpt = pd.read_csv(ckpt_path)
    diag = pd.read_csv(diag_path)
    if not diag.empty and "mode" not in diag.columns:
        diag["mode"] = [
            _classify_mode(int(sw), float(pg))
            for sw, pg in zip(diag["post_exit_action_switch_count"], diag["post_exit_route_progress_gain"])
        ]
    return ckpt, diag


def _volatility(ckpt: pd.DataFrame) -> pd.DataFrame:
    """Per-seed success_rate volatility: max - min across the 5 checkpoints.
    This is arm2a/arm2b's actual target mechanism (seed-level training
    instability), not just the 100K-checkpoint frozen_stall count."""
    rows = []
    for seed, g in ckpt.groupby("master_seed"):
        rows.append(
            {
                "master_seed": int(seed),
                "success_rate_min": float(g["success_rate"].min()),
                "success_rate_max": float(g["success_rate"].max()),
                "success_rate_volatility": float(g["success_rate"].max() - g["success_rate"].min()),
                "success_rate_at_100k": float(g.loc[g["checkpoint_step"] == 100_000, "success_rate"].iloc[0])
                if (g["checkpoint_step"] == 100_000).any()
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _cross_arm_comparison(
    this_ckpt: pd.DataFrame, this_diag: pd.DataFrame, this_label: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    notes: list[str] = []
    arms = {"arm0": ARM0_ANALYSIS_ROOT, "arm1": ARM1_ANALYSIS_ROOT}
    frozen_by_ckpt: dict[str, pd.Series] = {}
    volatility_rows: list[dict] = []

    for label, root in arms.items():
        loaded = _load_prior_arm(root, label)
        if loaded is None:
            continue
        ckpt, diag = loaded
        if not diag.empty:
            frozen_by_ckpt[label] = (
                diag[diag["mode"] == "frozen_stall"].groupby("checkpoint_step").size()
            )
        else:
            frozen_by_ckpt[label] = pd.Series(dtype=int)
        vol = _volatility(ckpt)
        vol["arm"] = label
        volatility_rows.append(vol)

    if not this_diag.empty:
        frozen_by_ckpt[this_label] = this_diag[this_diag["mode"] == "frozen_stall"].groupby("checkpoint_step").size()
    else:
        frozen_by_ckpt[this_label] = pd.Series(dtype=int)
    this_vol = _volatility(this_ckpt)
    this_vol["arm"] = this_label
    volatility_rows.append(this_vol)

    checkpoints = sorted(set().union(*[s.index for s in frozen_by_ckpt.values()]) | {0, 25_000, 50_000, 75_000, 100_000})
    frozen_rows = []
    for ckpt_step in checkpoints:
        row = {"checkpoint_step": int(ckpt_step)}
        for label, series in frozen_by_ckpt.items():
            row[f"{label}_frozen_stall_count"] = int(series.get(ckpt_step, 0))
        frozen_rows.append(row)
    frozen_comparison = pd.DataFrame(frozen_rows)

    volatility_comparison = pd.concat(volatility_rows, ignore_index=True) if volatility_rows else pd.DataFrame()
    if not volatility_comparison.empty:
        mean_vol = volatility_comparison.groupby("arm")["success_rate_volatility"].mean()
        if "arm0" in mean_vol.index and this_label in mean_vol.index:
            v0 = mean_vol["arm0"]
            v_this = mean_vol[this_label]
            if v_this < v0:
                notes.append(
                    f"Mean per-seed success_rate volatility (max-min across checkpoints) DROPPED "
                    f"from {v0:.3f} (arm0) to {v_this:.3f} ({this_label}) -- consistent with the "
                    "training-stability hypothesis."
                )
            elif v_this > v0:
                notes.append(
                    f"Mean per-seed success_rate volatility ROSE from {v0:.3f} (arm0) to "
                    f"{v_this:.3f} ({this_label}) -- refutes the training-stability hypothesis "
                    "as tested here."
                )
            else:
                notes.append(
                    f"Mean per-seed success_rate volatility is UNCHANGED ({v0:.3f}) -- inconclusive "
                    "at this sample size (2 seeds)."
                )
    return frozen_comparison, volatility_comparison, notes


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

    n_examples = min(3, len(examples))
    for i in range(n_examples):
        out_path = FIG_DIR / f"downstream_failure_example_{i + 1}.png"
        plot_example_episode(examples[i], out_path)
        print(f"wrote {out_path}")

    frozen_comparison, volatility_comparison, notes = _cross_arm_comparison(ckpt_summary, diag, "arm2a")
    if not frozen_comparison.empty:
        frozen_comparison.to_csv(OUT_ROOT / "comparison_frozen_stall_vs_arm0_arm1.csv", index=False)
        print("\n=== frozen_stall comparison (arm0 / arm1 / arm2a) ===")
        print(frozen_comparison.to_string(index=False))
    if not volatility_comparison.empty:
        volatility_comparison.to_csv(OUT_ROOT / "comparison_success_volatility.csv", index=False)
        print("\n=== success_rate volatility comparison (arm0 / arm1 / arm2a) ===")
        print(volatility_comparison.to_string(index=False))
    for n in notes:
        print(n)

    note_lines = [
        "# Stage 8 arm2a — Soft Target-Update Fix, Diagnostic Note",
        "",
        "**Status: diagnostic pilot only. No PASS/FAIL claim.**",
        "",
        "Single variable changed vs arm0: `target_update_mode` hard -> soft,",
        "`target_soft_tau=0.005` (epsilon schedule, reward, and algorithm unchanged).",
        "",
        f"Downstream-failure episodes analysed: {len(diag)}.",
        "",
    ]
    if not diag.empty:
        note_lines.append("Mode split:")
        for mode, count in diag["mode"].value_counts().items():
            note_lines.append(f"- {mode}: {count} ({count / len(diag):.1%})")
        note_lines.append("")
    if notes:
        note_lines.append("## Comparison vs arm0 (and arm1 for context)")
        note_lines.append("")
        note_lines.extend(f"- {n}" for n in notes)
        note_lines.append("")
    note_lines += [
        "Interpretation guide: arm2a targets seed-level training instability",
        "(large non-monotonic swings in success_rate/downstream_failure across",
        "checkpoints), not a specific failure sub-mode -- the primary metric is",
        "success_rate volatility (max-min across the 5 checkpoints), not the",
        "100K-checkpoint frozen_stall count alone (that was arm1's target).",
    ]
    (OUT_ROOT / "STAGE8_ARM2A_NOTE.md").write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_ROOT / 'STAGE8_ARM2A_NOTE.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
