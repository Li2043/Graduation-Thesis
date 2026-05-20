"""Compare baseline vs Rawlsian on reward and fairness metrics (random policy)."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"
BASELINE_CSV = RESULTS_DIR / "random_baseline.csv"
RAWLSIAN_CSV = RESULTS_DIR / "random_rawlsian.csv"
SUMMARY_CSV = RESULTS_DIR / "random_summary.csv"

COMMON_METRICS = [
    "total_reward",
    "mean_reward",
    "mean_min_experience",
    "final_min_experience",
    "mean_vehicle_experience",
    "mean_gini_experience",
    "final_gini_experience",
    "total_collision_count",
    "final_collision_count",
    "mean_n_vehicles",
    "steps",
]

RAWLSIAN_EXTRA_METRICS = [
    "mean_original_reward",
    "mean_rawlsian_reward",
    "rawlsian_reward_sum",
]

PLOT_CONFIG = [
    ("total_reward", "random_comparison_total_reward.png", "Average total reward"),
    ("mean_min_experience", "random_comparison_min_experience.png", "Average mean min experience"),
    ("mean_gini_experience", "random_comparison_gini.png", "Average mean Gini (experience)"),
    ("total_collision_count", "random_comparison_collision.png", "Average total collision count"),
]


def print_summary(name: str, df: pd.DataFrame, columns: list[str]) -> None:
    print(f"\n=== {name} ===")
    for col in columns:
        if col in df.columns:
            print(f"  {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")


def save_bar_plot(metric: str, title: str, baseline_mean: float, rawlsian_mean: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Baseline", "Rawlsian"], [baseline_mean, rawlsian_mean])
    ax.set_ylabel(metric)
    ax.set_title(f"Random policy: {title}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_summary_table(baseline: pd.DataFrame, rawlsian: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in COMMON_METRICS:
        if metric not in baseline.columns or metric not in rawlsian.columns:
            continue
        b_mean = baseline[metric].mean()
        b_std = baseline[metric].std()
        r_mean = rawlsian[metric].mean()
        r_std = rawlsian[metric].std()
        rows.append(
            {
                "metric": metric,
                "baseline_mean": b_mean,
                "baseline_std": b_std,
                "rawlsian_mean": r_mean,
                "rawlsian_std": r_std,
                "difference_rawlsian_minus_baseline": r_mean - b_mean,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    baseline = pd.read_csv(BASELINE_CSV)
    rawlsian = pd.read_csv(RAWLSIAN_CSV)

    print_summary("Random baseline", baseline, COMMON_METRICS)
    print_summary("Random + Rawlsian", rawlsian, COMMON_METRICS)
    print_summary("Random + Rawlsian (reward decomposition)", rawlsian, RAWLSIAN_EXTRA_METRICS)

    summary_df = build_summary_table(baseline, rawlsian)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved summary to {SUMMARY_CSV}")
    print(summary_df.to_string(index=False))

    for metric, filename, title in PLOT_CONFIG:
        if metric not in baseline.columns or metric not in rawlsian.columns:
            continue
        out_path = RESULTS_DIR / filename
        save_bar_plot(
            metric,
            title,
            baseline[metric].mean(),
            rawlsian[metric].mean(),
            out_path,
        )
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
