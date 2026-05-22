"""Compare random baseline vs Rawlsian random runs; plot average total reward."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"
BASELINE_CSV = RESULTS_DIR / "random_baseline.csv"
RAWLSIAN_CSV = RESULTS_DIR / "random_rawlsian.csv"
OUTPUT_PNG = RESULTS_DIR / "random_comparison.png"


def print_summary(name: str, df: pd.DataFrame, columns: list[str]) -> None:
    print(f"\n=== {name} ===")
    for col in columns:
        if col in df.columns:
            print(f"  {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")


def main() -> None:
    baseline = pd.read_csv(BASELINE_CSV)
    rawlsian = pd.read_csv(RAWLSIAN_CSV)

    common_cols = ["total_reward", "mean_reward", "steps"]
    print_summary("Random baseline", baseline, common_cols)
    print_summary("Random + Rawlsian", rawlsian, common_cols)

    if "mean_min_experience" in rawlsian.columns:
        print_summary(
            "Random + Rawlsian (min experience)",
            rawlsian,
            ["mean_min_experience"],
        )

    labels = ["Baseline", "Rawlsian"]
    means = [baseline["total_reward"].mean(), rawlsian["total_reward"].mean()]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, means, color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("Average total reward")
    ax.set_title("Random policy: baseline vs Rawlsian wrapper")
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150)
    plt.close(fig)
    print(f"\nSaved comparison plot to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
