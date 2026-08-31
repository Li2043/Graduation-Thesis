#!/usr/bin/env python3
"""Frozen DWS (Dense Welfare Shaping) statistical analysis pipeline.

Implements DWS_STATISTICAL_PROTOCOL_V1.md exactly. Read-only with respect to
all training/evaluation/checkpoint/manifest/log files and all existing WSC
analysis scripts -- this script only reads CSVs and writes new files under
its own dedicated output directory. Deterministic (fixed bootstrap RNG seed).

Four Maximin cells:
  Cell 1: Maximin                (Original 18D obs, terminal-only welfare)
  Cell 2: Maximin + DWS          (Original 18D obs, terminal + step-wise DWS)
  Cell 3: Maximin + WSC          (WSC 22D obs, terminal-only welfare)
  Cell 4: Maximin + WSC + DWS    (WSC 22D obs, terminal + step-wise DWS)

Cell 1 and Cell 3 come from the pre-existing, already-completed WSC v2 formal
seed-level CSV (a different drive/evaluation run than this bundle's own DWS
evaluations). Cell 2 and Cell 4 come from this bundle's own
evaluate_dense_interim.py episode-level output CSVs, aggregated here to
seed level using the SAME formula as the pre-existing pipeline (verified:
byte-identical thesis/study_b/utility.py; same H1 bank; same 256-episode
count; same ensemble_window_for_stage_end(2_000_000) rule; same
seed-level-U_min-and-gini-as-mean-of-per-episode-values convention).

Run with --self-test to execute the Section-12 validation suite against
synthetic data (no real files touched). Run without --self-test to attempt
the real four-cell analysis; it fails loudly (non-zero exit, no output
files written) if Cell 2 is not yet complete for all 12 formal seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = BUNDLE_ROOT / "outputs" / "dws_statistical_analysis"

FORMAL_SEEDS: tuple[int, ...] = (
    900101, 900102, 900103, 900104, 910101, 910102,
    920101, 920102, 920103, 920104, 920105, 920106,
)

CELL4_CSV_DEFAULT = BUNDLE_ROOT / "outputs" / "welfare_analysis" / "dense_interim_evaluation_12seed_full.csv"
CELL2_CSV_DEFAULT = BUNDLE_ROOT / "outputs" / "welfare_analysis" / "dense_interim_evaluation_maximin_dense_12seed_full.csv"
CELL13_CSV_DEFAULT = Path(__file__).resolve().parent.parent / "analysis" / "wsc_v2_formal" / "outputs" / "wsc_v2_formal_seed_level.csv"

N_BOOTSTRAP = 10_000
BOOTSTRAP_RNG_SEED = 0
CI_LEVEL = 0.95

PRIMARY_OUTCOMES = ("U_min", "Gini")
SECONDARY_TASK_SAFETY = ("mean_U", "completion", "collision", "timeout")


# ---------------------------------------------------------------------------
# Loading + seed-level aggregation
# ---------------------------------------------------------------------------

def _aggregate_episode_csv_to_seed_level(csv_path: Path, seeds: tuple[int, ...]) -> dict[int, dict[str, float]]:
    """Cell 2 / Cell 4 style: episode-level CSV -> per-seed mean of
    min_U / gini / mean_U / completion / collision / timeout, matching
    wsc_v2_formal_analysis.py's _seed_level_from_rows() convention exactly
    (mean of per-episode min_U and gini, not recomputed from raw utilities).
    The exact-256-rows check below also rules out duplicate seed-condition
    rows by construction (more than 256 rows for a seed fails just as loudly
    as fewer than 256)."""
    if not csv_path.exists():
        raise FileNotFoundError(f"required episode-level CSV not found: {csv_path}")
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: dict[int, dict[str, float]] = {}
    for seed in seeds:
        sub = [r for r in rows if int(r["seed"]) == seed]
        if len(sub) != 256:
            raise ValueError(f"{csv_path.name}: seed {seed} has {len(sub)} episode rows, expected 256")
        def col(name: str) -> list[float]:
            return [float(r[name]) for r in sub]
        vals = {
            "U_min": col("min_U"), "Gini": col("gini"), "mean_U": col("mean_U"),
            "completion": col("completion"), "collision": col("collision"), "timeout": col("timeout"),
        }
        for name, arr in vals.items():
            if not all(math.isfinite(v) for v in arr):
                raise ValueError(f"{csv_path.name}: seed {seed} has non-finite {name} values")
        out[seed] = {name: sum(arr) / len(arr) for name, arr in vals.items()}
    return out


def _load_cell1_cell3(csv_path: Path, seeds: tuple[int, ...]) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    """Cell 1 (orig_*) and Cell 3 (wsc_*) are already seed-level in this CSV
    (one row per seed x condition); no episode-level file is read here."""
    if not csv_path.exists():
        raise FileNotFoundError(f"required Cell 1/3 seed-level CSV not found: {csv_path}")
    with open(csv_path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["condition"] == "maximin"]
    by_seed = {int(r["seed"]): r for r in rows}
    missing = [s for s in seeds if s not in by_seed]
    if missing:
        raise ValueError(f"{csv_path.name}: missing maximin rows for seeds {missing}")
    cell1: dict[int, dict[str, float]] = {}
    cell3: dict[int, dict[str, float]] = {}
    for seed in seeds:
        r = by_seed[seed]
        cell1[seed] = {
            "U_min": float(r["orig_U_min"]), "Gini": float(r["orig_gini"]),
            "completion": float(r["orig_completion"]), "collision": float(r["orig_collision"]),
            # mean_U / timeout not present in this pre-existing seed-level CSV -- NOT AVAILABLE for Cell 1/3.
        }
        cell3[seed] = {
            "U_min": float(r["wsc_U_min"]), "Gini": float(r["wsc_gini"]),
            "completion": float(r["wsc_completion"]), "collision": float(r["wsc_collision"]),
        }
        for cell_name, d in (("Cell1", cell1[seed]), ("Cell3", cell3[seed])):
            for k, v in d.items():
                if not math.isfinite(v):
                    raise ValueError(f"{csv_path.name}: seed {seed} {cell_name}.{k} is non-finite")
    return cell1, cell3


# ---------------------------------------------------------------------------
# Contrasts (Section 2 / 3)
# ---------------------------------------------------------------------------

def per_seed_contrast(cell_a: dict[int, dict[str, float]], cell_b: dict[int, dict[str, float]],
                        outcome: str, seeds: tuple[int, ...]) -> dict[int, float]:
    """Y(cell_a) - Y(cell_b) per seed."""
    return {s: cell_a[s][outcome] - cell_b[s][outcome] for s in seeds}


def compute_primary_contrasts(cells: dict[int, dict[int, dict[str, float]]], seeds: tuple[int, ...]) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for outcome in PRIMARY_OUTCOMES:
        out[f"Delta_DWS_Original__{outcome}"] = per_seed_contrast(cells[2], cells[1], outcome, seeds)
        out[f"Delta_DWS_WSC__{outcome}"] = per_seed_contrast(cells[4], cells[3], outcome, seeds)
    return out


def compute_interaction(contrasts: dict[str, dict[int, float]], seeds: tuple[int, ...]) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for outcome in PRIMARY_OUTCOMES:
        orig = contrasts[f"Delta_DWS_Original__{outcome}"]
        wsc = contrasts[f"Delta_DWS_WSC__{outcome}"]
        out[f"I_DWSxWSC__{outcome}"] = {s: wsc[s] - orig[s] for s in seeds}
    return out


# ---------------------------------------------------------------------------
# Paired bootstrap (Section 4 / 5)
# ---------------------------------------------------------------------------

def paired_bootstrap(seed_values: dict[int, float], seeds: tuple[int, ...], *,
                       n_boot: int = N_BOOTSTRAP, rng_seed: int = BOOTSTRAP_RNG_SEED) -> dict:
    import random
    n = len(seeds)
    values = [seed_values[s] for s in seeds]
    observed_mean = sum(values) / n
    rng = random.Random(rng_seed)
    boot_means = []
    for _ in range(n_boot):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(resample) / n)
    boot_means.sort()
    lo_idx = int((1 - CI_LEVEL) / 2 * n_boot)
    hi_idx = int((1 + CI_LEVEL) / 2 * n_boot) - 1
    ci_lo, ci_hi = boot_means[lo_idx], boot_means[hi_idx]
    # Two-sided percentile-bootstrap p-value: 2 * min(P(boot<=0), P(boot>=0)),
    # capped at 1.0. Documented explicitly per Section 5's requirement.
    p_le = sum(1 for b in boot_means if b <= 0) / n_boot
    p_ge = sum(1 for b in boot_means if b >= 0) / n_boot
    p_value = min(1.0, 2 * min(p_le, p_ge))
    n_pos = sum(1 for v in values if v > 0)
    n_neg = sum(1 for v in values if v < 0)
    n_zero = sum(1 for v in values if v == 0)
    sorted_vals = sorted(values)
    median = sorted_vals[n // 2] if n % 2 == 1 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    return {
        "mean_effect": observed_mean, "median_effect": median,
        "ci_lower": ci_lo, "ci_upper": ci_hi, "p_value_raw": p_value,
        "n_positive": n_pos, "n_negative": n_neg, "n_zero": n_zero, "n_seeds": n,
    }


def holm_correction(p_values: list[float]) -> list[float]:
    """Standard Holm step-down. p_values order defines the family; returns
    adjusted p-values in the SAME order as input (not sorted)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, p_values[idx] * (m - rank))
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted


# ---------------------------------------------------------------------------
# Leave-one-seed-out (Section 8)
# ---------------------------------------------------------------------------

def leave_one_seed_out(seed_values: dict[int, float], seeds: tuple[int, ...]) -> dict:
    full_estimate = sum(seed_values[s] for s in seeds) / len(seeds)
    loo = {}
    for held_out in seeds:
        remaining = [s for s in seeds if s != held_out]
        loo[held_out] = sum(seed_values[s] for s in remaining) / len(remaining)
    min_seed = min(loo, key=lambda s: loo[s])
    max_seed = max(loo, key=lambda s: loo[s])
    return {
        "full_n12_estimate": full_estimate, "loo_by_seed": loo,
        "min_estimate": loo[min_seed], "min_seed": min_seed,
        "max_estimate": loo[max_seed], "max_seed": max_seed,
    }


# ---------------------------------------------------------------------------
# Self-test suite (Section 12)
# ---------------------------------------------------------------------------

def _make_synthetic_cells(seeds: tuple[int, ...]) -> dict[int, dict[int, dict[str, float]]]:
    cells = {1: {}, 2: {}, 3: {}, 4: {}}
    for i, s in enumerate(seeds):
        base = 0.5 + 0.01 * i
        cells[1][s] = {"U_min": base, "Gini": 0.2 - 0.005 * i, "mean_U": base, "completion": base, "collision": 1 - base, "timeout": 0.0}
        cells[2][s] = {"U_min": base + 0.05, "Gini": 0.2 - 0.005 * i - 0.02, "mean_U": base, "completion": base, "collision": 1 - base, "timeout": 0.0}
        cells[3][s] = {"U_min": base + 0.02, "Gini": 0.2 - 0.005 * i - 0.01, "mean_U": base, "completion": base, "collision": 1 - base, "timeout": 0.0}
        cells[4][s] = {"U_min": base + 0.10, "Gini": 0.2 - 0.005 * i - 0.05, "mean_U": base, "completion": base, "collision": 1 - base, "timeout": 0.0}
    return cells


def run_self_test() -> bool:
    seeds = FORMAL_SEEDS
    ok = True

    def check(name: str, cond: bool):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            ok = False

    print("== Section 12 validation suite (synthetic data only, no real files touched) ==")

    cells = _make_synthetic_cells(seeds)
    contrasts = compute_primary_contrasts(cells, seeds)
    interaction = compute_interaction(contrasts, seeds)

    # 1. tiny synthetic paired-difference calculation
    d = per_seed_contrast(cells[2], cells[1], "U_min", seeds)
    expect0 = cells[2][seeds[0]]["U_min"] - cells[1][seeds[0]]["U_min"]
    check("paired-difference calculation on tiny synthetic dataset", math.isclose(d[seeds[0]], expect0, rel_tol=1e-12))

    # 2. swapping cell labels flips the sign
    d_swapped = per_seed_contrast(cells[1], cells[2], "U_min", seeds)
    check("swapping cell labels flips sign", math.isclose(d[seeds[0]], -d_swapped[seeds[0]], rel_tol=1e-12))

    # 3. U_min favourable direction is positive (Cell 2 > Cell 1 in synthetic data -> positive)
    check("U_min favourable direction is positive", all(v > 0 for v in contrasts["Delta_DWS_Original__U_min"].values()))

    # 4. Gini favourable direction is negative (Cell 2 has lower gini than Cell 1 in synthetic data)
    check("Gini favourable direction is negative", all(v < 0 for v in contrasts["Delta_DWS_Original__Gini"].values()))

    # 5. Holm correction on a known synthetic pair: p=[0.01, 0.04] -> [0.02, 0.04]
    holm_out = holm_correction([0.01, 0.04])
    check("Holm correction on known synthetic pair", math.isclose(holm_out[0], 0.02, rel_tol=1e-9) and math.isclose(holm_out[1], 0.04, rel_tol=1e-9))

    # 6. bootstrap reproducibility with RNG seed 0
    seed_vals = contrasts["Delta_DWS_Original__U_min"]
    boot1 = paired_bootstrap(seed_vals, seeds, n_boot=2000, rng_seed=0)
    boot2 = paired_bootstrap(seed_vals, seeds, n_boot=2000, rng_seed=0)
    check("bootstrap reproducibility with RNG seed 0", boot1 == boot2)

    # 7. missing one seed causes a hard failure
    try:
        bad_cells = {k: {s: v for s, v in d.items() if s != seeds[0]} for k, d in cells.items()}
        for cell_id, data in bad_cells.items():
            if set(data.keys()) != set(seeds):
                raise ValueError(f"Cell {cell_id}: seed set mismatch (test)")
        check("missing one seed causes a hard failure", False)
    except ValueError:
        check("missing one seed causes a hard failure", True)

    # 8. interaction equals WSC-effect minus Original-effect
    for outcome in PRIMARY_OUTCOMES:
        orig = contrasts[f"Delta_DWS_Original__{outcome}"]
        wsc = contrasts[f"Delta_DWS_WSC__{outcome}"]
        inter = interaction[f"I_DWSxWSC__{outcome}"]
        matches = all(math.isclose(inter[s], wsc[s] - orig[s], rel_tol=1e-12) for s in seeds)
        check(f"interaction == WSC-effect minus Original-effect ({outcome})", matches)

    print(f"\n{'ALL CHECKS PASSED' if ok else 'AT LEAST ONE CHECK FAILED'}")
    return ok


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def run_formal_analysis(cell2_csv: Path, cell4_csv: Path, cell13_csv: Path) -> int:
    seeds = FORMAL_SEEDS
    try:
        cell1, cell3 = _load_cell1_cell3(cell13_csv, seeds)
        cell4 = _aggregate_episode_csv_to_seed_level(cell4_csv, seeds)
    except (FileNotFoundError, ValueError) as e:
        print("ANALYSIS BLOCKED (Cell 1/3/4 data problem, not Cell 2)")
        print(f"Reason: {e}")
        return 1
    try:
        cell2 = _aggregate_episode_csv_to_seed_level(cell2_csv, seeds)
    except (FileNotFoundError, ValueError) as e:
        print("WAITING FOR CELL 2")
        print(f"Reason: {e}")
        print("Cell 1, Cell 3, and Cell 4 data are all available and were successfully loaded/validated; "
              "only Cell 2 (Maximin + DWS, no WSC) is blocking the formal four-cell analysis.")
        return 1
    cells = {1: cell1, 2: cell2, 3: cell3, 4: cell4}
    for cell_id, data in cells.items():
        present, expected = set(data.keys()), set(seeds)
        if present != expected:
            print("ANALYSIS BLOCKED (seed-set mismatch)")
            print(f"Reason: Cell {cell_id} seed set {sorted(present)} != expected {sorted(expected)}")
            return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # dws_seed_level_metrics.csv
    seed_level_rows = []
    for cell_id in (1, 2, 3, 4):
        for s in seeds:
            row = {"seed": s, "cell": cell_id, **cells[cell_id][s]}
            seed_level_rows.append(row)
    all_fields = sorted({k for r in seed_level_rows for k in r})
    write_csv(OUT_DIR / "dws_seed_level_metrics.csv", ["seed", "cell"] + [f for f in all_fields if f not in ("seed", "cell")], seed_level_rows)

    contrasts = compute_primary_contrasts(cells, seeds)
    interaction = compute_interaction(contrasts, seeds)

    # dws_primary_contrasts.csv
    contrast_rows = []
    for name, per_seed in contrasts.items():
        for s in seeds:
            contrast_rows.append({"contrast": name, "seed": s, "value": per_seed[s]})
    write_csv(OUT_DIR / "dws_primary_contrasts.csv", ["contrast", "seed", "value"], contrast_rows)

    # bootstrap + p-values + Holm, per Section 4/5/6
    summary_rows = []
    family_a_names = ["Delta_DWS_Original__U_min", "Delta_DWS_WSC__U_min"]
    family_b_names = ["Delta_DWS_Original__Gini", "Delta_DWS_WSC__Gini"]
    boot_results = {name: paired_bootstrap(contrasts[name], seeds) for name in contrasts}
    holm_a = holm_correction([boot_results[n]["p_value_raw"] for n in family_a_names])
    holm_b = holm_correction([boot_results[n]["p_value_raw"] for n in family_b_names])
    holm_adjusted = dict(zip(family_a_names, holm_a)) | dict(zip(family_b_names, holm_b))
    for name, res in boot_results.items():
        outcome, contrast_kind = name.split("__")[1], name.split("__")[0]
        summary_rows.append({
            "outcome": outcome, "contrast": contrast_kind,
            "mean_effect": res["mean_effect"], "median_effect": res["median_effect"],
            "bootstrap_ci_lower": res["ci_lower"], "bootstrap_ci_upper": res["ci_upper"],
            "p_value_raw": res["p_value_raw"], "p_value_holm": holm_adjusted[name],
            "n_positive": res["n_positive"], "n_negative": res["n_negative"],
            "n_zero": res["n_zero"], "n_seeds": res["n_seeds"],
        })
    write_csv(OUT_DIR / "dws_primary_summary.csv",
              ["outcome", "contrast", "mean_effect", "median_effect", "bootstrap_ci_lower", "bootstrap_ci_upper",
               "p_value_raw", "p_value_holm", "n_positive", "n_negative", "n_zero", "n_seeds"], summary_rows)

    # dws_interaction_summary.csv (secondary, no Holm, no confirmatory p adjustment beyond raw bootstrap CI)
    interaction_rows = []
    for name, per_seed in interaction.items():
        outcome = name.split("__")[1]
        res = paired_bootstrap(per_seed, seeds)
        interaction_rows.append({
            "outcome": outcome, "mean_effect": res["mean_effect"], "median_effect": res["median_effect"],
            "bootstrap_ci_lower": res["ci_lower"], "bootstrap_ci_upper": res["ci_upper"],
            "n_positive": res["n_positive"], "n_negative": res["n_negative"], "n_zero": res["n_zero"],
        })
    write_csv(OUT_DIR / "dws_interaction_summary.csv",
              ["outcome", "mean_effect", "median_effect", "bootstrap_ci_lower", "bootstrap_ci_upper",
               "n_positive", "n_negative", "n_zero"], interaction_rows)

    # dws_leave_one_seed_out.csv -- 4 primary contrasts + 2 interactions
    loo_rows = []
    all_quantities = {**contrasts, **interaction}
    for name, per_seed in all_quantities.items():
        loo = leave_one_seed_out(per_seed, seeds)
        loo_rows.append({
            "quantity": name, "full_n12_estimate": loo["full_n12_estimate"],
            "min_estimate": loo["min_estimate"], "min_omitted_seed": loo["min_seed"],
            "max_estimate": loo["max_estimate"], "max_omitted_seed": loo["max_seed"],
        })
    write_csv(OUT_DIR / "dws_leave_one_seed_out.csv",
              ["quantity", "full_n12_estimate", "min_estimate", "min_omitted_seed", "max_estimate", "max_omitted_seed"], loo_rows)

    # Section 7: task/safety descriptive contrasts (no p-values, no Holm)
    task_safety_rows = []
    for outcome in SECONDARY_TASK_SAFETY:
        if outcome not in cells[1][seeds[0]] or outcome not in cells[3][seeds[0]]:
            continue  # mean_U/timeout not present for Cell 1/3 (documented limitation)
        d_orig = per_seed_contrast(cells[2], cells[1], outcome, seeds)
        d_wsc = per_seed_contrast(cells[4], cells[3], outcome, seeds)
        for label, d in (("Original", d_orig), ("WSC", d_wsc)):
            res = paired_bootstrap(d, seeds)
            task_safety_rows.append({
                "outcome": outcome, "information_condition": label,
                "mean_effect": res["mean_effect"], "bootstrap_ci_lower": res["ci_lower"],
                "bootstrap_ci_upper": res["ci_upper"],
            })
    write_csv(OUT_DIR / "dws_task_safety_descriptive.csv",
              ["outcome", "information_condition", "mean_effect", "bootstrap_ci_lower", "bootstrap_ci_upper"], task_safety_rows)

    print(f"Formal DWS statistical analysis complete. Outputs written to {OUT_DIR}")
    for name, res in boot_results.items():
        print(f"  {name}: mean={res['mean_effect']:+.4f} CI=[{res['ci_lower']:+.4f},{res['ci_upper']:+.4f}] "
              f"p_raw={res['p_value_raw']:.4f} p_holm={holm_adjusted[name]:.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help="Run the Section-12 validation suite on synthetic data and exit.")
    ap.add_argument("--cell2-csv", type=Path, default=CELL2_CSV_DEFAULT)
    ap.add_argument("--cell4-csv", type=Path, default=CELL4_CSV_DEFAULT)
    ap.add_argument("--cell13-csv", type=Path, default=CELL13_CSV_DEFAULT)
    args = ap.parse_args()

    if args.self_test:
        return 0 if run_self_test() else 1

    return run_formal_analysis(args.cell2_csv, args.cell4_csv, args.cell13_csv)


if __name__ == "__main__":
    raise SystemExit(main())
