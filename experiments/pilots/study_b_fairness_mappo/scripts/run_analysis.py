#!/usr/bin/env python3
"""Study B RQ2 confirmatory analysis (+ optional RQ1 sensitivity) --
new_research_plan.md's "Primary confirmatory contrasts" section.

Expects ``evaluate_policy.py`` output files named
``eval_<BANK>_seed_<N>_<condition>.csv`` in ``--results-dir`` (e.g.
``eval_H1_seed_1_baseline.csv``, ``eval_H1_seed_1_mean_pbrs.csv``,
``eval_H1_seed_1_min_pbrs.csv``, ... for every formal seed N and, if
generated, matching ``eval_H0_seed_<N>_baseline.csv`` files for the RQ1
sensitivity check). Only seeds present under ALL THREE conditions are used
for the RQ2 contrasts (matched-seed-block requirement); the script warns,
does not silently drop, on any mismatch.

Writes ``analysis_summary.json`` (all numbers) and, if matplotlib figures
are wanted, a forest plot -- see ``--skip-plots``."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(REPO_SRC))

from thesis.study_b.analysis.behaviour import mean_hard_brake_rate, worse_off_frequency_by_class  # noqa: E402
from thesis.study_b.analysis.bootstrap import BootstrapResult, holm_correction, paired_bootstrap_contrast  # noqa: E402
from thesis.study_b.analysis.plots import plot_bootstrap_forest  # noqa: E402
from thesis.study_b.analysis.welfare import read_eval_csv, seed_level_summary  # noqa: E402

FILENAME_RE = re.compile(r"eval_(?P<bank>\w+)_seed_(?P<seed>\d+)_(?P<condition>baseline|mean_pbrs|min_pbrs)\.csv$")

# new_research_plan.md's prespecified safety non-inferiority margins.
COLLISION_MARGIN = 0.03
COMPLETION_MARGIN = -0.05


def _discover(results_dir: Path, bank: str) -> dict[str, dict[int, Path]]:
    """Returns ``{condition: {seed: path}}`` for the given bank."""
    out: dict[str, dict[int, Path]] = {"baseline": {}, "mean_pbrs": {}, "min_pbrs": {}}
    for path in results_dir.glob(f"eval_{bank}_seed_*_*.csv"):
        m = FILENAME_RE.match(path.name)
        if m is None or m.group("bank") != bank:
            continue
        out[m.group("condition")][int(m.group("seed"))] = path
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--n-bootstrap", type=int, default=10_000)
    p.add_argument("--skip-plots", action="store_true")
    args = p.parse_args(argv)

    output_path = args.output or (args.results_dir / "analysis_summary.json")

    h1_files = _discover(args.results_dir, "H1")
    common_seeds = sorted(set(h1_files["baseline"]) & set(h1_files["mean_pbrs"]) & set(h1_files["min_pbrs"]))
    if not common_seeds:
        print("ERROR: no seed is present under all three conditions in --results-dir (H1 bank).", file=sys.stderr)
        return 1
    for condition, seeds in h1_files.items():
        missing = set(common_seeds) ^ set(seeds)
        if missing:
            print(f"WARNING: {condition} seeds {sorted(set(seeds) - set(common_seeds))} excluded (not present in all conditions)")

    summaries = {
        condition: {seed: seed_level_summary(h1_files[condition][seed]) for seed in common_seeds}
        for condition in ("baseline", "mean_pbrs", "min_pbrs")
    }

    def series(condition: str, metric: str) -> list[float]:
        return [summaries[condition][seed][metric] for seed in common_seeds]

    # --- RQ2 primary confirmatory contrasts ---
    contrast_h2a = paired_bootstrap_contrast(series("min_pbrs", "min_U"), series("baseline", "min_U"), n_replicates=args.n_bootstrap, seed=1)
    contrast_h2b = paired_bootstrap_contrast(series("mean_pbrs", "mean_U"), series("baseline", "mean_U"), n_replicates=args.n_bootstrap, seed=2)
    holm = holm_correction({"H2a_min_pbrs_vs_baseline_on_U_min": contrast_h2a.p_value, "H2b_mean_pbrs_vs_baseline_on_U_mean": contrast_h2b.p_value})

    # --- Safety non-inferiority (secondary, per-condition) ---
    safety = {}
    for condition in ("mean_pbrs", "min_pbrs"):
        coll = paired_bootstrap_contrast(series(condition, "collision_rate"), series("baseline", "collision_rate"), n_replicates=args.n_bootstrap, seed=3)
        comp = paired_bootstrap_contrast(series(condition, "completion_rate"), series("baseline", "completion_rate"), n_replicates=args.n_bootstrap, seed=4)
        safety[condition] = {
            "collision_delta_ci_upper": coll.ci_upper,
            "collision_non_inferior": coll.ci_upper < COLLISION_MARGIN,
            "completion_delta_ci_lower": comp.ci_lower,
            "completion_non_inferior": comp.ci_lower > COMPLETION_MARGIN,
        }

    # --- RQ1 (optional): Mean-trained H0 vs H1 Gini, using baseline condition only ---
    rq1 = None
    h0_files = _discover(args.results_dir, "H0")
    h0_seeds = sorted(set(h0_files["baseline"]) & set(common_seeds))
    if h0_seeds:
        h0_summaries = [seed_level_summary(h0_files["baseline"][s]) for s in h0_seeds]
        h1_summaries = [summaries["baseline"][s] for s in h0_seeds]
        rq1 = {
            "n_seeds": len(h0_seeds),
            "gini_H0_mean": sum(s["gini"] for s in h0_summaries if s["gini"] is not None) / len(h0_summaries),
            "gini_H1_mean": sum(s["gini"] for s in h1_summaries if s["gini"] is not None) / len(h1_summaries),
            "C_max_H0_mean": sum(s["C_max"] for s in h0_summaries) / len(h0_summaries),
            "C_max_H1_mean": sum(s["C_max"] for s in h1_summaries) / len(h1_summaries),
        }

    # --- Behavioural (baseline condition, worse-off class distribution) ---
    baseline_rows = [row for seed in common_seeds for row in read_eval_csv(h1_files["baseline"][seed])]
    behaviour = {
        "worse_off_frequency_by_class": worse_off_frequency_by_class(baseline_rows),
        "hard_brake_rate": mean_hard_brake_rate(baseline_rows),
    }

    summary = {
        "n_seeds": len(common_seeds),
        "seeds": common_seeds,
        "seed_level_summaries": summaries,
        "rq2_primary_contrasts": {
            "H2a_min_pbrs_vs_baseline_on_U_min": vars(contrast_h2a),
            "H2b_mean_pbrs_vs_baseline_on_U_mean": vars(contrast_h2b),
            "holm_correction": holm,
        },
        "safety_non_inferiority": safety,
        "rq1_homogeneous_vs_heterogeneous": rq1,
        "behaviour": behaviour,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"analysis summary written to {output_path}")
    print(f"n_seeds={len(common_seeds)}")
    print(f"H2a (min_pbrs U_min - baseline U_min): {contrast_h2a.point_estimate:+.4f} "
          f"[{contrast_h2a.ci_lower:+.4f}, {contrast_h2a.ci_upper:+.4f}] "
          f"p_holm_reject={holm['H2a_min_pbrs_vs_baseline_on_U_min']['reject_null']}")
    print(f"H2b (mean_pbrs U_mean - baseline U_mean): {contrast_h2b.point_estimate:+.4f} "
          f"[{contrast_h2b.ci_lower:+.4f}, {contrast_h2b.ci_upper:+.4f}] "
          f"p_holm_reject={holm['H2b_mean_pbrs_vs_baseline_on_U_mean']['reject_null']}")

    if not args.skip_plots:
        forest_results: dict[str, BootstrapResult] = {
            "min_pbrs - baseline (U_min)": contrast_h2a,
            "mean_pbrs - baseline (U_mean)": contrast_h2b,
        }
        plot_bootstrap_forest(forest_results, output_path=output_path.parent / "rq2_forest_plot.png")
        print(f"forest plot written to {output_path.parent / 'rq2_forest_plot.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
