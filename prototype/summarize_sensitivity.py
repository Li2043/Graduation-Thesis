"""
Aggregate and plot v0.6.3 Rawlsian sensitivity results.

Reads sensitivity_raw_results.csv and writes summary tables plus matplotlib plots.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import (
    SENSITIVITY_PLOTS_DIR,
    SENSITIVITY_RAW_CSV,
    SENSITIVITY_RESULTS_DIR,
    SENSITIVITY_SUMMARY_CSV,
    SENSITIVITY_VS_DEFAULT_CSV,
)
from sensitivity_utils import DEFAULT_PARAMETER_VALUES, RESULT_ROW_METRICS

PROJECT_ROOT = Path(__file__).resolve().parent

PLOT_METRICS = [
    "mean_min_experience",
    "total_collision_count",
    "mean_risk_penalty",
    "mean_mobility_score",
    "steps",
]

HIGHER_IS_BETTER = {
    "mean_min_experience",
    "final_min_experience",
    "mean_vehicle_experience",
    "least_advantaged_ego_ratio",
    "mean_mobility_score",
    "total_reward",
    "mean_reward",
    "steps",
}

LOWER_IS_BETTER = {
    "mean_gini_experience",
    "total_collision_count",
    "mean_risk_penalty",
    "mean_collision_penalty",
    "reason_risk_steps",
    "reason_low_mobility_steps",
}


def _values_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def load_raw_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Error: raw results not found: {path}")
        print("Run: python run_sensitivity.py")
        sys.exit(1)
    return pd.read_csv(path)


def build_summary_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["variant_id", "parameter_changed", "parameter_value"]

    for keys, group in raw_df.groupby(group_cols, sort=False):
        variant_id, parameter_changed, parameter_value = keys
        row = {
            "variant_id": variant_id,
            "parameter_changed": parameter_changed,
            "parameter_value": parameter_value,
            "n_seeds": int(group["seed"].nunique()),
        }
        if "xi" in group.columns:
            row["xi"] = float(group["xi"].iloc[0])
            row["ego_neighbourhood_radius"] = float(group["ego_neighbourhood_radius"].iloc[0])
            row["W_RISK"] = float(group["W_RISK"].iloc[0])
            row["rawlsian_scope"] = str(group["rawlsian_scope"].iloc[0])

        for metric in RESULT_ROW_METRICS:
            if metric not in group.columns:
                continue
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        rows.append(row)

    return pd.DataFrame(rows)


def _is_better(metric: str, candidate: float, reference: float) -> bool:
    if metric in HIGHER_IS_BETTER:
        return candidate > reference
    if metric in LOWER_IS_BETTER:
        return candidate < reference
    return False


def build_vs_default_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for parameter_changed, default_value in DEFAULT_PARAMETER_VALUES.items():
        subset = raw_df[raw_df["parameter_changed"] == parameter_changed].copy()
        if subset.empty:
            continue

        ref_rows = subset[
            subset["parameter_value"].apply(lambda v: _values_equal(v, default_value))
        ]
        if ref_rows.empty:
            print(
                f"Warning: no default row for {parameter_changed}={default_value}; "
                "skipping vs-default comparisons for this factor."
            )
            continue

        other_values = sorted(
            v
            for v in subset["parameter_value"].unique()
            if not _values_equal(v, default_value)
        )
        seeds = sorted(subset["seed"].unique())

        for metric in RESULT_ROW_METRICS:
            if metric not in subset.columns:
                continue

            ref_by_seed = {
                int(r.seed): float(getattr(r, metric))
                for r in ref_rows.itertuples()
            }

            for parameter_value in other_values:
                var_rows = subset[subset["parameter_value"] == parameter_value]
                better_count = 0
                comparable = 0
                diffs = []

                for seed in seeds:
                    if seed not in ref_by_seed:
                        continue
                    var_seed = var_rows[var_rows["seed"] == seed]
                    if var_seed.empty:
                        continue
                    ref_val = ref_by_seed[seed]
                    cand_val = float(var_seed[metric].iloc[0])
                    comparable += 1
                    diffs.append(cand_val - ref_val)
                    if _is_better(metric, cand_val, ref_val):
                        better_count += 1

                rows.append(
                    {
                        "parameter_changed": parameter_changed,
                        "parameter_value": parameter_value,
                        "default_parameter_value": default_value,
                        "metric": metric,
                        "mean_difference_vs_default": float(pd.Series(diffs).mean())
                        if diffs
                        else float("nan"),
                        "better_than_default_seed_count": better_count,
                        "n_seeds_compared": comparable,
                    }
                )

    return pd.DataFrame(rows)


def save_factor_plot(
    summary_df: pd.DataFrame,
    parameter_changed: str,
    metric: str,
    plots_dir: Path,
) -> Path | None:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in summary_df.columns:
        return None

    subset = summary_df[summary_df["parameter_changed"] == parameter_changed].copy()
    if subset.empty:
        return None

    subset = subset.sort_values("parameter_value")
    x_labels = [str(v) for v in subset["parameter_value"]]
    means = subset[mean_col].tolist()
    stds = subset[std_col].tolist() if std_col in subset.columns else [0.0] * len(means)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x_labels, means, yerr=stds, capsize=5)
    ax.set_xlabel(parameter_changed)
    ax.set_ylabel(metric)
    ax.set_title(f"v0.6.3 Sensitivity: {metric} vs {parameter_changed}")
    fig.tight_layout()

    filename = f"{metric}_{parameter_changed}.png"
    out_path = plots_dir / filename
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_all_plots(summary_df: pd.DataFrame, plots_dir: Path) -> list[Path]:
    saved: list[Path] = []
    factors = summary_df["parameter_changed"].unique()
    for metric in PLOT_METRICS:
        for factor in factors:
            path = save_factor_plot(summary_df, factor, metric, plots_dir)
            if path is not None:
                saved.append(path)
    return saved


def main() -> None:
    raw_path = PROJECT_ROOT / SENSITIVITY_RAW_CSV
    raw_df = load_raw_results(raw_path)

    summary_df = build_summary_table(raw_df)
    vs_default_df = build_vs_default_table(raw_df)

    results_dir = PROJECT_ROOT / SENSITIVITY_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_path = PROJECT_ROOT / SENSITIVITY_SUMMARY_CSV
    vs_default_path = PROJECT_ROOT / SENSITIVITY_VS_DEFAULT_CSV

    summary_df.to_csv(summary_path, index=False)
    vs_default_df.to_csv(vs_default_path, index=False)

    plots_dir = PROJECT_ROOT / SENSITIVITY_PLOTS_DIR
    saved_plots = save_all_plots(summary_df, plots_dir)

    print("=== v0.6.3 sensitivity summary (per variant, mean across seeds) ===")
    display_cols = [
        "variant_id",
        "parameter_changed",
        "parameter_value",
        "mean_min_experience_mean",
        "total_collision_count_mean",
        "mean_risk_penalty_mean",
        "steps_mean",
    ]
    display_cols = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[display_cols].to_string(index=False))

    print(f"\nSaved {raw_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {vs_default_path}")
    for plot_path in saved_plots:
        print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
