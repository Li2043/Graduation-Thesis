"""Tables, figures, and root-cause helpers for Stage 7A-0."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def seed_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, g in episodes.groupby("master_seed"):
        succ = float(g["success"].mean())
        rows.append(
            {
                "master_seed": int(seed),
                "success_rate": succ,
                "collision_rate": float(g["collision"].mean()),
                "truncation_rate": float(g["truncated"].mean()),
                "mean_episode_length": float(g["episode_length"].mean()),
                "median_episode_length": float(g["episode_length"].median()),
                "performance_band": (
                    "high" if succ >= 0.75 else "intermediate" if succ >= 0.25 else "low"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("master_seed")


def role_diagnostics(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, g in episodes.groupby("master_seed"):
        for role_col, label in (
            ("controller_A_role", "A"),
            ("controller_B_role", "B"),
        ):
            for role, gg in g.groupby(role_col):
                rows.append(
                    {
                        "master_seed": int(seed),
                        "controller": label,
                        "role": role,
                        "n": len(gg),
                        "success_rate": float(gg["success"].mean()),
                        "collision_rate": float(gg["collision"].mean()),
                        "truncation_rate": float(gg["truncated"].mean()),
                        "mean_episode_length": float(gg["episode_length"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def block_diagnostics(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, block), g in episodes.groupby(["master_seed", "validation_block_id"]):
        rows.append(
            {
                "master_seed": int(seed),
                "validation_block_id": block,
                "success_rate": float(g["success"].mean()),
                "collision_rate": float(g["collision"].mean()),
                "truncation_rate": float(g["truncated"].mean()),
                "assignment_asymmetry": abs(
                    float(g.loc[g["assignment"] == 0, "success"].mean() if (g["assignment"] == 0).any() else 0)
                    - float(g.loc[g["assignment"] == 1, "success"].mean() if (g["assignment"] == 1).any() else 0)
                ),
                "mean_episode_length": float(g["episode_length"].mean()),
            }
        )
    return pd.DataFrame(rows)


def action_summary(steps: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    ep_key = episodes.set_index(["master_seed", "validation_block_id", "assignment"])["success"]
    rows = []
    for (seed, block, asn, ctrl), g in steps.groupby(
        ["master_seed", "validation_block_id", "assignment", "controller"]
    ):
        succ = bool(ep_key.get((seed, block, asn), False))
        acts = g["commanded_action"].astype(int)
        switches = (acts.diff().fillna(0) != 0).sum()
        rows.append(
            {
                "master_seed": int(seed),
                "validation_block_id": block,
                "assignment": int(asn),
                "controller": ctrl,
                "success": succ,
                "maintain_frac": float((acts == 0).mean()),
                "accelerate_frac": float((acts == 1).mean()),
                "decelerate_frac": float((acts == 2).mean()),
                "action_switch_rate": float(switches / max(1, len(acts) - 1)),
            }
        )
    return pd.DataFrame(rows)


def q_summary(steps: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    ep = episodes.set_index(["master_seed", "validation_block_id", "assignment"])
    rows = []
    for (seed, block, asn, ctrl), g in steps.groupby(
        ["master_seed", "validation_block_id", "assignment", "controller"]
    ):
        key = (seed, block, asn)
        outcome = "truncated"
        if key in ep.index:
            row = ep.loc[key]
            if bool(row["success"]):
                outcome = "success"
            elif bool(row["collision"]):
                outcome = "collision"
        qm = g["Q_margin"].dropna()
        rows.append(
            {
                "master_seed": int(seed),
                "validation_block_id": block,
                "assignment": int(asn),
                "controller": ctrl,
                "outcome": outcome,
                "mean_abs_Q": float(np.mean(np.abs(g[["Q_maintain", "Q_accelerate", "Q_decelerate"]].values))),
                "Q_margin_mean": float(qm.mean()) if len(qm) else np.nan,
                "Q_margin_median": float(qm.median()) if len(qm) else np.nan,
                "Q_margin_q10": float(qm.quantile(0.10)) if len(qm) else np.nan,
                "Q_margin_q90": float(qm.quantile(0.90)) if len(qm) else np.nan,
                "frac_margin_lt_1e-3": float((qm < 1e-3).mean()) if len(qm) else np.nan,
                "frac_margin_lt_1e-2": float((qm < 1e-2).mean()) if len(qm) else np.nan,
                "frac_margin_lt_5e-2": float((qm < 5e-2).mean()) if len(qm) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def competence_gate_grid(seed_df: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90]
    rows = []
    for thr in thresholds:
        meet = int((seed_df["success_rate"] >= thr).sum())
        rows.append(
            {
                "safe_resolution_threshold": thr,
                "seeds_meeting": meet,
                "proportion_seeds_meeting": meet / max(1, len(seed_df)),
                "condition_mean_success": float(episodes["success"].mean()),
                "condition_median_success": float(seed_df["success_rate"].median()),
                "collision_rate": float(episodes["collision"].mean()),
                "truncation_rate": float(episodes["truncated"].mean()),
                "note": "PROVISIONAL DIAGNOSTIC GATE — NOT YET PREREGISTERED",
            }
        )
    return pd.DataFrame(rows)


def root_cause_matrix(
    *,
    seed_df: pd.DataFrame,
    taxonomy: pd.DataFrame,
    reward_sep: dict[str, Any] | None,
    continuation_status: str,
    mismatch_count: int,
) -> pd.DataFrame:
    n_seed = len(seed_df)
    high = int((seed_df["performance_band"] == "high").sum())
    low = int((seed_df["performance_band"] == "low").sum())
    tax_counts = (
        taxonomy["primary_failure_label"].value_counts().to_dict() if len(taxonomy) else {}
    )
    weak_reward = bool((reward_sep or {}).get("weak_reward_separation_any", False))

    def row(hyp, status, support, contradict, aff_s, aff_e, conf, nxt):
        return {
            "hypothesis": hyp,
            "status": status,
            "supporting_metrics": support,
            "contradicting_metrics": contradict,
            "affected_seed_count": aff_s,
            "affected_episode_count": aff_e,
            "confidence": conf,
            "recommended_next_test": nxt,
        }

    rows = [
        row(
            "insufficient_100K_budget",
            "NOT IDENTIFIABLE" if continuation_status == "BLOCKED" else "PARTIALLY SUPPORTED",
            f"continuation={continuation_status}; intermediate ckpts missing",
            "cannot test 100K→200K without resumable checkpoints",
            n_seed,
            0,
            "low",
            "recover full ckpts or retrain Baseline-only budget pilot with new seeds",
        ),
        row(
            "stable_learning_plateau",
            "NOT IDENTIFIABLE",
            "no intermediate greedy reconstructions available",
            "10K-75K weights unpublished",
            0,
            0,
            "low",
            "publish or recover intermediate weight exports",
        ),
        row(
            "mutual_yielding",
            "SUPPORTED" if tax_counts.get("mutual_yielding", 0) > 0 else "NOT SUPPORTED",
            f"primary_count={tax_counts.get('mutual_yielding', 0)}",
            "",
            int(taxonomy[taxonomy['primary_failure_label']=='mutual_yielding']['master_seed'].nunique()) if len(taxonomy) else 0,
            int(tax_counts.get("mutual_yielding", 0)),
            "moderate",
            "base-task deadlock resolution pilot if dominant",
        ),
        row(
            "unilateral_stall",
            "SUPPORTED" if tax_counts.get("unilateral_stall", 0) > 0 else "PARTIALLY SUPPORTED",
            f"primary_count={tax_counts.get('unilateral_stall', 0)}",
            "",
            int(taxonomy[taxonomy['primary_failure_label']=='unilateral_stall']['master_seed'].nunique()) if len(taxonomy) else 0,
            int(tax_counts.get("unilateral_stall", 0)),
            "moderate",
            "inspect role-conditioned stall",
        ),
        row(
            "post_exit_survivor_stall",
            "SUPPORTED" if tax_counts.get("post_exit_survivor_stall", 0) > 0 else "NOT SUPPORTED",
            f"primary_count={tax_counts.get('post_exit_survivor_stall', 0)}",
            "",
            int(taxonomy[taxonomy['primary_failure_label']=='post_exit_survivor_stall']['master_seed'].nunique()) if len(taxonomy) else 0,
            int(tax_counts.get("post_exit_survivor_stall", 0)),
            "moderate",
            "single-controller completion curriculum",
        ),
        row(
            "action_oscillation",
            "SUPPORTED" if tax_counts.get("oscillatory_control", 0) > 0 else "NOT SUPPORTED",
            f"primary_count={tax_counts.get('oscillatory_control', 0)}",
            "",
            0,
            int(tax_counts.get("oscillatory_control", 0)),
            "moderate",
            "Q-margin / Double-DQN sensitivity if co-occurs with low margin",
        ),
        row(
            "seed_bifurcation",
            "SUPPORTED" if high >= 1 and low >= 1 else "PARTIALLY SUPPORTED",
            f"high={high}, low={low}, intermediate={n_seed-high-low}",
            "",
            n_seed,
            0,
            "high",
            "increase independent training seeds before treatment comparison",
        ),
        row(
            "base_reward_weak_separation",
            "SUPPORTED" if weak_reward else "NOT SUPPORTED",
            str(reward_sep or {}),
            "",
            n_seed if weak_reward else 0,
            0,
            "moderate" if weak_reward else "low",
            "base-task incentive redesign with new experiment version",
        ),
        row(
            "environment_or_exit_anomaly",
            "SUPPORTED" if tax_counts.get("environment_or_exit_anomaly", 0) > 0 else "NOT SUPPORTED",
            f"primary_count={tax_counts.get('environment_or_exit_anomaly', 0)}",
            "",
            0,
            int(tax_counts.get("environment_or_exit_anomaly", 0)),
            "high" if tax_counts.get("environment_or_exit_anomaly", 0) else "moderate",
            "stop new training and fix environment if any anomalies",
        ),
        row(
            "evaluation_reconstruction_error",
            "NOT SUPPORTED" if mismatch_count == 0 else "SUPPORTED",
            f"h1_mismatch_count={mismatch_count}",
            "",
            0,
            mismatch_count,
            "high",
            "none if zero mismatches",
        ),
    ]
    return pd.DataFrame(rows)


def save_figures(
    *,
    seed_df: pd.DataFrame,
    episodes: pd.DataFrame,
    taxonomy: pd.DataFrame,
    action_df: pd.DataFrame,
    q_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    fig_dir: Path,
) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir = fig_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    def _save(name: str, df: pd.DataFrame, plot_fn):
        csv_path = data_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        plot_fn(ax)
        fig.tight_layout()
        png = fig_dir / f"{name}.png"
        fig.savefig(png, dpi=140)
        plt.close(fig)
        paths.append(png.as_posix())

    _save(
        "fig_baseline_100k_success_by_seed",
        seed_df,
        lambda ax: ax.scatter(seed_df["master_seed"], seed_df["success_rate"], c="#1f4e79")
        or ax.set(xlabel="master_seed", ylabel="success_rate", title="Baseline 100K success by seed (exploratory)"),
    )
    _save(
        "fig_baseline_100k_truncation_by_seed",
        seed_df,
        lambda ax: ax.scatter(seed_df["master_seed"], seed_df["truncation_rate"], c="#1f4e79")
        or ax.set(xlabel="master_seed", ylabel="truncation_rate", title="Baseline 100K truncation by seed"),
    )
    _save(
        "fig_baseline_100k_collision_by_seed",
        seed_df,
        lambda ax: ax.scatter(seed_df["master_seed"], seed_df["collision_rate"], c="#1f4e79")
        or ax.set(xlabel="master_seed", ylabel="collision_rate", title="Baseline 100K collision by seed"),
    )
    if len(taxonomy):
        counts = taxonomy["primary_failure_label"].value_counts().rename_axis("label").reset_index(name="n")
        _save(
            "fig_baseline_100k_failure_taxonomy",
            counts,
            lambda ax: ax.barh(counts["label"], counts["n"], color="#1f4e79")
            or ax.set(xlabel="episode count", title="Baseline 100K truncation failure taxonomy"),
        )
    if len(action_df):
        agg = action_df.groupby("success")[["maintain_frac", "accelerate_frac", "decelerate_frac"]].mean().reset_index()
        _save(
            "fig_baseline_action_distribution_success_vs_truncation",
            agg,
            lambda ax: agg.set_index("success")[["maintain_frac", "accelerate_frac", "decelerate_frac"]].plot(
                kind="bar", ax=ax, color=["#4c78a8", "#f58518", "#54a24b"]
            )
            or ax.set(title="Action mix: success vs non-success", ylabel="fraction"),
        )
    if len(q_df):
        def _plot_q(ax):
            data = [
                q_df.loc[q_df["outcome"] == "success", "Q_margin_median"].dropna().tolist(),
                q_df.loc[q_df["outcome"] == "truncated", "Q_margin_median"].dropna().tolist(),
            ]
            ax.boxplot(data)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["success", "truncated"])
            ax.set(title="Episode median Q-margin", ylabel="Q_margin")

        _save(
            "fig_baseline_q_margin_success_vs_truncation",
            q_df,
            _plot_q,
        )
    _save(
        "fig_baseline_competence_gate_grid",
        gate_df,
        lambda ax: ax.plot(gate_df["safe_resolution_threshold"], gate_df["seeds_meeting"], marker="o", color="#1f4e79")
        or ax.set(
            xlabel="threshold",
            ylabel="seeds meeting",
            title="Provisional competence gate grid (diagnostic only)",
        ),
    )
    # placeholders noting missing continuation / learning curves
    for name in (
        "fig_baseline_continuation_success",
        "fig_baseline_continuation_truncation",
        "fig_baseline_success_by_checkpoint_and_seed",
    ):
        note = pd.DataFrame([{"status": "unavailable", "reason": "intermediate_or_continuation_blocked"}])
        note.to_csv(data_dir / f"{name}.csv", index=False)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Unavailable / BLOCKED\n(missing full checkpoints)", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(fig_dir / f"{name}.png", dpi=120)
        plt.close(fig)
        paths.append((fig_dir / f"{name}.png").as_posix())
    return paths
