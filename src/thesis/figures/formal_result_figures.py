"""Publication figure constructors for Stage 6C (Matplotlib OO API only)."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from thesis.figures.figure_data_validation import deterministic_jitter
from thesis.figures.publication_style import (
    CONDITION_DISPLAY,
    CONDITION_ORDER,
    CONDITION_STYLE,
    CONTRAST_DISPLAY,
    CONTRAST_ORDER,
    ENDPOINT_DISPLAY,
    FORMAL_SEEDS,
    PRIMARY_NON_CONVENTION,
    PRIMARY_STEP,
    PROB_YTICKS,
    WIDTH_FULL,
    WIDTH_ONE_HALF,
)


def _panel_label(ax: Axes, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _format_p_holm(p: float | None) -> str:
    if p is None or (isinstance(p, float) and (np.isnan(p))):
        return "p_Holm = NA"
    p = float(p)
    if p < 0.001:
        return "p_Holm < 0.001"
    return f"p_Holm = {p:.3f}"


def fig_primary_endpoint_seed_distributions(
    seed_values: pd.DataFrame,
) -> tuple[Figure, np.ndarray, pd.DataFrame]:
    """Figure 4.1 — seed-level primary non-convention endpoints."""
    endpoints = list(PRIMARY_NON_CONVENTION)
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_FULL, 5.6), sharey=False)
    axes_flat = axes.ravel()
    plotted_rows: list[dict[str, Any]] = []
    x_base = np.arange(len(CONDITION_ORDER), dtype=float)

    for ax, ep, lab in zip(axes_flat, endpoints, ("(a)", "(b)", "(c)", "(d)")):
        sub = seed_values[seed_values["endpoint"] == ep].copy()
        # paired lines
        for seed in FORMAL_SEEDS:
            xs, ys = [], []
            for ci, cond in enumerate(CONDITION_ORDER):
                row = sub[(sub["condition"] == cond) & (sub["master_seed"] == seed)]
                if row.empty:
                    continue
                val = float(row["value"].iloc[0])
                x = x_base[ci] + deterministic_jitter(seed, ci)
                xs.append(x)
                ys.append(val)
                plotted_rows.append(
                    {
                        "endpoint": ep,
                        "condition": cond,
                        "master_seed": seed,
                        "value": val,
                        "x_jittered": x,
                        "checkpoint_step": PRIMARY_STEP,
                    }
                )
            if len(xs) == 3:
                ax.plot(xs, ys, color="#B0B0B0", lw=0.7, zorder=1, solid_capstyle="round")
        # points and means
        for ci, cond in enumerate(CONDITION_ORDER):
            style = CONDITION_STYLE[cond]
            vals = []
            for seed in FORMAL_SEEDS:
                row = sub[(sub["condition"] == cond) & (sub["master_seed"] == seed)]
                if row.empty:
                    continue
                val = float(row["value"].iloc[0])
                vals.append(val)
                ax.scatter(
                    [x_base[ci] + deterministic_jitter(seed, ci)],
                    [val],
                    c=style["color"],
                    marker=style["marker"],
                    s=22,
                    zorder=3,
                    edgecolors="white",
                    linewidths=0.3,
                )
            if vals:
                ax.scatter(
                    [x_base[ci]],
                    [float(np.mean(vals))],
                    c=style["color"],
                    marker=style["marker"],
                    s=55,
                    zorder=4,
                    edgecolors="black",
                    linewidths=0.6,
                    label=CONDITION_DISPLAY[cond] if lab == "(a)" else None,
                )
        ax.set_xticks(x_base)
        ax.set_xticklabels([CONDITION_DISPLAY[c] for c in CONDITION_ORDER], rotation=0)
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks(PROB_YTICKS)
        ax.set_ylabel(ENDPOINT_DISPLAY[ep])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
        ax.grid(axis="y", color="#E6E6E6", lw=0.6)
        _panel_label(ax, lab)
        ax.annotate(
            f"n = 10 seeds; step {PRIMARY_STEP}",
            xy=(0.98, 0.02),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=7,
            color="#333333",
        )
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, axes, pd.DataFrame(plotted_rows)


def fig_primary_endpoint_paired_contrasts(
    contrasts: pd.DataFrame,
) -> tuple[Figure, np.ndarray, pd.DataFrame]:
    """Figure 4.2 — forest plot of preregistered paired contrasts."""
    endpoints = list(PRIMARY_NON_CONVENTION)
    fig, axes = plt.subplots(len(endpoints), 1, figsize=(WIDTH_FULL, 7.2), sharex=True)
    if len(endpoints) == 1:
        axes = np.array([axes])
    plotted: list[dict[str, Any]] = []
    y_pos = np.arange(len(CONTRAST_ORDER), dtype=float)

    for ax, ep, lab in zip(axes, endpoints, ("(a)", "(b)", "(c)", "(d)")):
        sub = contrasts[contrasts["endpoint"] == ep]
        for yi, contrast in enumerate(CONTRAST_ORDER):
            row = sub[sub["contrast"] == contrast].iloc[0]
            mean = float(row["mean_diff"])
            lo = float(row["ci95_low"])
            hi = float(row["ci95_high"])
            n_c = int(row["n_complete"])
            p_h = row["wilcoxon_p_holm"]
            dz = float(row["cohens_dz"]) if pd.notna(row["cohens_dz"]) else float("nan")
            ax.errorbar(
                mean,
                yi,
                xerr=[[mean - lo], [hi - mean]],
                fmt="D",
                color="#222222",
                ecolor="#222222",
                elinewidth=1.2,
                markersize=4.5,
                capsize=3,
            )
            ax.text(
                1.02,
                yi,
                f"n={n_c}; {_format_p_holm(None if pd.isna(p_h) else float(p_h))}; dz={dz:.2f}",
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="left",
                fontsize=7,
            )
            plotted.append(
                {
                    "endpoint": ep,
                    "contrast": contrast,
                    "mean_diff": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_complete": n_c,
                    "pvalue_holm": None if pd.isna(p_h) else float(p_h),
                    "cohens_dz": dz,
                    "checkpoint_step": PRIMARY_STEP,
                }
            )
        ax.axvline(0.0, color="#666666", lw=0.9, zorder=0)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([CONTRAST_DISPLAY[c] for c in CONTRAST_ORDER])
        ax.set_ylabel("")
        ax.grid(axis="x", color="#E6E6E6", lw=0.6)
        _panel_label(ax, lab)
        ax.set_title(ENDPOINT_DISPLAY[ep], loc="left", fontsize=9)
    axes[-1].set_xlabel("Mean paired seed-level difference")
    fig.tight_layout()
    return fig, axes, pd.DataFrame(plotted)


def fig_convention_selection_and_consistency(
    episodes: pd.DataFrame,
    convention_availability: pd.DataFrame,
    seed_endpoints: pd.DataFrame,
) -> tuple[Figure, np.ndarray, pd.DataFrame]:
    """Figure 4.3 — convention composition, consistency, missingness."""
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_FULL, 3.4))
    plotted: list[dict[str, Any]] = []

    # (a) episode composition within locked 16-episode sets (descriptive)
    ax = axes[0]
    cats = ["mainline_first", "ramp_first", "simultaneous", "non_success"]
    cat_labels = ["Mainline-first", "Ramp-first", "Simultaneous", "Non-success"]
    x = np.arange(len(CONDITION_ORDER), dtype=float)
    bottom = np.zeros(len(CONDITION_ORDER))
    colors = ["#0072B2", "#D55E00", "#999999", "#E0E0E0"]
    for cat, lab, col in zip(cats, cat_labels, colors):
        heights = []
        for cond in CONDITION_ORDER:
            sub = episodes[episodes["condition"] == cond]
            if cat == "non_success":
                h = float((~sub["success"].astype(bool)).mean())
            elif cat == "simultaneous":
                h = float((sub["convention"] == "simultaneous").mean())
            else:
                h = float((sub["convention"] == cat).mean())
            heights.append(h)
            plotted.append(
                {
                    "panel": "a",
                    "condition": cond,
                    "category": cat,
                    "proportion": h,
                    "encoding": "episode_share_descriptive",
                }
            )
        ax.bar(x, heights, bottom=bottom, color=col, edgecolor="white", width=0.7, label=lab)
        bottom = bottom + np.asarray(heights)
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_DISPLAY[c] for c in CONDITION_ORDER])
    ax.set_ylim(0, 1)
    ax.set_yticks(PROB_YTICKS)
    ax.set_ylabel("Episode share")
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    _panel_label(ax, "(a)")

    # (b) available seed-level consistency only (no zero-fill; connect complete pairs only)
    ax = axes[1]
    x_base = np.arange(len(CONDITION_ORDER), dtype=float)
    avail_map: dict[tuple[str, int], float] = {}
    for _, row in convention_availability.iterrows():
        missing = (
            bool(row["convention_missing"])
            if not isinstance(row["convention_missing"], str)
            else str(row["convention_missing"]).lower() in {"true", "1"}
        )
        if missing or pd.isna(row["convention_consistency"]):
            continue
        avail_map[(str(row["condition"]), int(row["master_seed"]))] = float(
            row["convention_consistency"]
        )
    for seed in FORMAL_SEEDS:
        xs, ys = [], []
        for ci, cond in enumerate(CONDITION_ORDER):
            key = (cond, seed)
            if key not in avail_map:
                continue
            x = x_base[ci] + deterministic_jitter(seed, ci)
            xs.append(x)
            ys.append(avail_map[key])
        # Connect only consecutive observed pairs within the displayed conditions.
        if len(xs) >= 2:
            ax.plot(xs, ys, color="#B0B0B0", lw=0.7, zorder=1)
    for ci, cond in enumerate(CONDITION_ORDER):
        style = CONDITION_STYLE[cond]
        vals = []
        for seed in FORMAL_SEEDS:
            key = (cond, seed)
            if key not in avail_map:
                continue
            val = avail_map[key]
            vals.append(val)
            xj = x_base[ci] + deterministic_jitter(seed, ci)
            ax.scatter(
                [xj],
                [val],
                c=style["color"],
                marker=style["marker"],
                s=28,
                edgecolors="white",
                linewidths=0.3,
                zorder=3,
            )
            plotted.append(
                {
                    "panel": "b",
                    "condition": cond,
                    "master_seed": seed,
                    "convention_consistency": val,
                    "x_jittered": xj,
                }
            )
        n_av = len(vals)
        if vals:
            ax.scatter(
                [x_base[ci]],
                [float(np.mean(vals))],
                c=style["color"],
                marker=style["marker"],
                s=60,
                edgecolors="black",
                linewidths=0.6,
                zorder=4,
                label=CONDITION_DISPLAY[cond],
            )
        ax.text(x_base[ci], -0.08, f"n={n_av}", ha="center", va="top", fontsize=7)
    ax.set_xticks(x_base)
    ax.set_xticklabels([CONDITION_DISPLAY[c] for c in CONDITION_ORDER])
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks(PROB_YTICKS)
    ax.set_ylabel("Convention consistency")
    ax.grid(axis="y", color="#E6E6E6", lw=0.6)
    _panel_label(ax, "(b)")

    # (c) missingness taxonomy
    ax = axes[2]
    reasons = {
        "available": 0,
        "no_success": 0,
        "missing_with_success": 0,
    }
    for _, row in convention_availability.iterrows():
        missing = bool(row["convention_missing"]) if not isinstance(row["convention_missing"], str) else str(row["convention_missing"]).lower() in {"true", "1"}
        n_success = int(row["n_success"])
        if not missing:
            reasons["available"] += 1
            label = "available"
        elif n_success == 0:
            reasons["no_success"] += 1
            label = "no_success"
        else:
            reasons["missing_with_success"] += 1
            label = "missing_with_success"
        plotted.append(
            {
                "panel": "c",
                "condition": row["condition"],
                "master_seed": int(row["master_seed"]),
                "missing_reason": label,
                "n_success": n_success,
            }
        )
    missing_total = reasons["no_success"] + reasons["missing_with_success"]
    if missing_total != 11:
        raise ValueError(f"convention missing plotted count {missing_total} != 11")
    labels = ["Available", "No success", "Missing w/ success"]
    vals = [reasons["available"], reasons["no_success"], reasons["missing_with_success"]]
    ax.barh(np.arange(len(labels)), vals, color=["#4D4D4D", "#D55E00", "#0072B2"], height=0.6)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Condition × seed count")
    ax.set_xlim(0, max(vals) + 2)
    for i, v in enumerate(vals):
        ax.text(v + 0.2, i, str(v), va="center", fontsize=8)
    ax.annotate("missing total = 11", xy=(0.98, 0.02), xycoords="axes fraction", ha="right", fontsize=7)
    _panel_label(ax, "(c)")
    fig.tight_layout()
    return fig, axes, pd.DataFrame(plotted)


def fig_stakeholder_utility_by_role(secondary: pd.DataFrame) -> tuple[Figure, np.ndarray, pd.DataFrame]:
    """Figure 4.5 — secondary per-stakeholder utilities at step 100000."""
    roles = [
        ("mean_A_utility", "A"),
        ("mean_B_utility", "B"),
        ("mean_B_front_utility", "B_front"),
        ("mean_B_rear_utility", "B_rear"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_FULL, 5.4))
    plotted: list[dict[str, Any]] = []
    x_base = np.arange(len(CONDITION_ORDER), dtype=float)
    for ax, (col, name), lab in zip(axes.ravel(), roles, ("(a)", "(b)", "(c)", "(d)")):
        for seed in FORMAL_SEEDS:
            xs, ys = [], []
            for ci, cond in enumerate(CONDITION_ORDER):
                row = secondary[(secondary["condition"] == cond) & (secondary["master_seed"] == seed)]
                if row.empty:
                    continue
                val = float(row[col].iloc[0])
                x = x_base[ci] + deterministic_jitter(seed, ci)
                xs.append(x)
                ys.append(val)
                plotted.append(
                    {
                        "stakeholder": name,
                        "condition": cond,
                        "master_seed": seed,
                        "utility": val,
                        "checkpoint_step": PRIMARY_STEP,
                        "endpoint_class": "secondary",
                    }
                )
            if len(xs) == 3:
                ax.plot(xs, ys, color="#B0B0B0", lw=0.7, zorder=1)
        for ci, cond in enumerate(CONDITION_ORDER):
            style = CONDITION_STYLE[cond]
            vals = secondary.loc[secondary["condition"] == cond, col].astype(float).tolist()
            for seed in FORMAL_SEEDS:
                row = secondary[(secondary["condition"] == cond) & (secondary["master_seed"] == seed)]
                if row.empty:
                    continue
                ax.scatter(
                    [x_base[ci] + deterministic_jitter(seed, ci)],
                    [float(row[col].iloc[0])],
                    c=style["color"],
                    marker=style["marker"],
                    s=22,
                    zorder=3,
                    edgecolors="white",
                    linewidths=0.3,
                )
            ax.scatter(
                [x_base[ci]],
                [float(np.mean(vals))],
                c=style["color"],
                marker=style["marker"],
                s=55,
                zorder=4,
                edgecolors="black",
                linewidths=0.6,
                label=CONDITION_DISPLAY[cond] if lab == "(a)" else None,
            )
        ax.set_xticks(x_base)
        ax.set_xticklabels([CONDITION_DISPLAY[c] for c in CONDITION_ORDER])
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks(PROB_YTICKS)
        ax.set_ylabel(f"{name} utility")
        ax.grid(axis="y", color="#E6E6E6", lw=0.6)
        _panel_label(ax, lab)
        ax.annotate("secondary; n=10", xy=(0.98, 0.02), xycoords="axes fraction", ha="right", fontsize=7)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, axes, pd.DataFrame(plotted)


def fig_safety_comfort_diagnostics(episodes: pd.DataFrame) -> tuple[Figure, np.ndarray, pd.DataFrame]:
    """Figure 4.6 — seed-level safety/comfort secondary diagnostics from validated episodes."""
    # Aggregate episodes -> seed (not treating episodes as inferential replicates for CIs)
    rows = []
    for (cond, seed), g in episodes.groupby(["condition", "master_seed"]):
        rows.append(
            {
                "condition": cond,
                "master_seed": int(seed),
                "minimum_bumper_gap": float(g["minimum_bumper_gap"].dropna().min()) if g["minimum_bumper_gap"].notna().any() else np.nan,
                "minimum_TTC": float(g["minimum_TTC"].dropna().min()) if g["minimum_TTC"].notna().any() else np.nan,
                "hard_braking_rate": float(g["hard_braking_rate"].mean()),
                "background_maximum_braking": float(g["background_maximum_braking"].mean()),
            }
        )
    seed_df = pd.DataFrame(rows)
    panels = [
        ("minimum_bumper_gap", "Minimum bumper gap", "Larger preferred"),
        ("minimum_TTC", "Minimum TTC", "Larger preferred"),
        ("hard_braking_rate", "Hard-braking rate", "Smaller preferred"),
        ("background_maximum_braking", "Max background braking", "Smaller preferred"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_FULL, 5.4))
    plotted: list[dict[str, Any]] = []
    x_base = np.arange(len(CONDITION_ORDER), dtype=float)
    for ax, (col, ylab, pref), lab in zip(axes.ravel(), panels, ("(a)", "(b)", "(c)", "(d)")):
        for seed in FORMAL_SEEDS:
            xs, ys = [], []
            for ci, cond in enumerate(CONDITION_ORDER):
                row = seed_df[(seed_df["condition"] == cond) & (seed_df["master_seed"] == seed)]
                if row.empty or pd.isna(row[col].iloc[0]):
                    continue
                val = float(row[col].iloc[0])
                x = x_base[ci] + deterministic_jitter(seed, ci)
                xs.append(x)
                ys.append(val)
                plotted.append(
                    {
                        "metric": col,
                        "condition": cond,
                        "master_seed": seed,
                        "value": val,
                        "endpoint_class": "secondary",
                    }
                )
            if len(xs) >= 2:
                ax.plot(xs, ys, color="#B0B0B0", lw=0.7, zorder=1)
        for ci, cond in enumerate(CONDITION_ORDER):
            style = CONDITION_STYLE[cond]
            vals = seed_df.loc[seed_df["condition"] == cond, col].dropna().astype(float)
            for seed in FORMAL_SEEDS:
                row = seed_df[(seed_df["condition"] == cond) & (seed_df["master_seed"] == seed)]
                if row.empty or pd.isna(row[col].iloc[0]):
                    continue
                ax.scatter(
                    [x_base[ci] + deterministic_jitter(seed, ci)],
                    [float(row[col].iloc[0])],
                    c=style["color"],
                    marker=style["marker"],
                    s=22,
                    zorder=3,
                    edgecolors="white",
                    linewidths=0.3,
                )
            if len(vals):
                ax.scatter(
                    [x_base[ci]],
                    [float(vals.mean())],
                    c=style["color"],
                    marker=style["marker"],
                    s=55,
                    zorder=4,
                    edgecolors="black",
                    linewidths=0.6,
                    label=CONDITION_DISPLAY[cond] if lab == "(a)" else None,
                )
        ax.set_xticks(x_base)
        ax.set_xticklabels([CONDITION_DISPLAY[c] for c in CONDITION_ORDER])
        ax.set_ylabel(ylab)
        ax.grid(axis="y", color="#E6E6E6", lw=0.6)
        _panel_label(ax, lab)
        ax.annotate(pref, xy=(0.98, 0.02), xycoords="axes fraction", ha="right", fontsize=7)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, axes, pd.DataFrame(plotted)


def fig_run_completion_and_integrity(accounting: pd.DataFrame, integrity: pd.DataFrame) -> tuple[Figure, Axes, pd.DataFrame]:
    """Supplementary S1."""
    fig, ax = plt.subplots(figsize=(WIDTH_ONE_HALF, 2.8))
    counts = accounting["status"].value_counts()
    labels = list(counts.index.astype(str))
    vals = [int(counts[k]) for k in labels]
    ax.bar(np.arange(len(labels)), vals, color="#4D4D4D", width=0.6)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Run count")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.2, str(v), ha="center", fontsize=8)
    plotted = pd.DataFrame({"status": labels, "count": vals})
    fig.tight_layout()
    return fig, ax, plotted


def fig_worst_off_stakeholder(episodes: pd.DataFrame) -> tuple[Figure, Axes, pd.DataFrame]:
    """Supplementary S3 — seed-modal worst-off identity frequencies."""
    rows = []
    for (cond, seed), g in episodes.groupby(["condition", "master_seed"]):
        mode = g["worst_off_stakeholder_identity"].mode(dropna=True)
        identity = str(mode.iloc[0]) if len(mode) else "NA"
        rows.append({"condition": cond, "master_seed": int(seed), "worst_off_mode": identity})
    df = pd.DataFrame(rows)
    identities = ["A", "B", "B_front", "B_rear"]
    fig, ax = plt.subplots(figsize=(WIDTH_ONE_HALF, 3.0))
    x = np.arange(len(CONDITION_ORDER), dtype=float)
    width = 0.18
    plotted = []
    for i, ident in enumerate(identities):
        heights = []
        for cond in CONDITION_ORDER:
            h = float(((df["condition"] == cond) & (df["worst_off_mode"] == ident)).mean())
            heights.append(h)
            plotted.append({"condition": cond, "identity": ident, "seed_share": h})
        ax.bar(x + (i - 1.5) * width, heights, width=width, label=ident)
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_DISPLAY[c] for c in CONDITION_ORDER])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of seeds (modal identity)")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    return fig, ax, pd.DataFrame(plotted)


def fig_collision_type_composition(episodes: pd.DataFrame) -> tuple[Figure, Axes, pd.DataFrame]:
    """Supplementary S4 — descriptive collision-type shares among collision episodes."""
    coll = episodes[episodes["collision"].astype(bool)].copy()
    fig, ax = plt.subplots(figsize=(WIDTH_ONE_HALF, 3.0))
    plotted = []
    if coll.empty:
        ax.text(0.5, 0.5, "No collision episodes", ha="center", va="center")
        ax.set_axis_off()
        return fig, ax, pd.DataFrame(plotted)
    types = sorted(coll["collision_type"].fillna("unknown").astype(str).unique().tolist())
    x = np.arange(len(CONDITION_ORDER), dtype=float)
    bottom = np.zeros(len(CONDITION_ORDER))
    cmap = plt.get_cmap("cividis")
    for i, ctype in enumerate(types):
        heights = []
        for cond in CONDITION_ORDER:
            sub = coll[coll["condition"] == cond]
            h = float((sub["collision_type"].fillna("unknown") == ctype).mean()) if len(sub) else 0.0
            heights.append(h)
            plotted.append({"condition": cond, "collision_type": ctype, "share_among_collisions": h})
        ax.bar(x, heights, bottom=bottom, color=cmap(i / max(1, len(types) - 1)), width=0.7, label=ctype)
        bottom = bottom + np.asarray(heights)
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_DISPLAY[c] for c in CONDITION_ORDER])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share among collision episodes")
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    return fig, ax, pd.DataFrame(plotted)


def fig_seed_level_primary_endpoint_matrix(seed_values: pd.DataFrame) -> tuple[Figure, Axes, pd.DataFrame]:
    """Supplementary S5 — descriptive standardised heatmap (not inferential)."""
    eps = list(PRIMARY_NON_CONVENTION)
    mat = []
    index = []
    for cond in CONDITION_ORDER:
        for seed in FORMAL_SEEDS:
            row = []
            for ep in eps:
                val = seed_values[
                    (seed_values["endpoint"] == ep)
                    & (seed_values["condition"] == cond)
                    & (seed_values["master_seed"] == seed)
                ]["value"]
                row.append(float(val.iloc[0]) if len(val) else np.nan)
            mat.append(row)
            index.append(f"{CONDITION_DISPLAY[cond]}:{seed}")
    arr = np.asarray(mat, dtype=float)
    # standardise within endpoint column for display only
    z = arr.copy()
    for j in range(z.shape[1]):
        col = z[:, j]
        mu, sd = np.nanmean(col), np.nanstd(col, ddof=1)
        z[:, j] = (col - mu) / sd if sd > 0 else 0.0
    fig, ax = plt.subplots(figsize=(WIDTH_FULL, 7.5))
    im = ax.imshow(z, aspect="auto", cmap="cividis")
    ax.set_xticks(np.arange(len(eps)))
    ax.set_xticklabels([ENDPOINT_DISPLAY[e] for e in eps], rotation=25, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(index)))
    ax.set_yticklabels(index, fontsize=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Within-endpoint z-score (descriptive)")
    plotted = []
    for i, idx in enumerate(index):
        for j, ep in enumerate(eps):
            plotted.append(
                {
                    "row": idx,
                    "endpoint": ep,
                    "raw_value": float(arr[i, j]),
                    "z_score": float(z[i, j]),
                }
            )
    fig.tight_layout()
    return fig, ax, pd.DataFrame(plotted)


__all__ = [
    "fig_collision_type_composition",
    "fig_convention_selection_and_consistency",
    "fig_primary_endpoint_paired_contrasts",
    "fig_primary_endpoint_seed_distributions",
    "fig_run_completion_and_integrity",
    "fig_safety_comfort_diagnostics",
    "fig_seed_level_primary_endpoint_matrix",
    "fig_stakeholder_utility_by_role",
    "fig_worst_off_stakeholder",
]
