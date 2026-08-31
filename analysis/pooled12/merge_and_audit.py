"""Pool the original 6-seed formal welfare/behavioral evaluation data with the
new 6-seed independent replication data (920101-920106) into n=12 merged CSVs,
and run the full rq_audit_final.py-style statistical audit (Task 1-5) plus a
12-row task-performance table, generalized from SEEDS(6) to SEEDS(12).

Read-only against the two source bundles (F:\\正式训练\\ and
F:\\正式训练_seed_replication_v1\\). Writes only under
F:\\正式训练_seed_replication_v1\\analysis_scripts\\pooled12\\outputs\\.

This is an ADDITIVE analysis script -- it does not modify or overwrite the
original 6-seed analysis_scripts/*.py or their outputs, which remain the
frozen record of the n=6 result.
"""
from __future__ import annotations
import os

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ORIG_WELFARE = Path(__file__).resolve().parent.parent / "data" / "formal_welfare_evaluation_merged.csv"
NEW_WELFARE = Path(os.environ.get("SEED_REPL_BUNDLE", "")) / "outputs" / "seed_replication_v1" / "welfare_eval" / "replication_welfare_evaluation_merged.csv"  # not distributed with this repo
ORIG_BEHAV = Path(__file__).resolve().parent.parent / "data" / "formal_behavioral_evaluation_merged.csv"
NEW_BEHAV = Path(os.environ.get("SEED_REPL_BUNDLE", "")) / "outputs" / "seed_replication_v1" / "behavioral" / "replication_behavioral_evaluation_merged.csv"  # not distributed with this repo

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS_ORIG = [900101, 900102, 900103, 900104, 910101, 910102]
SEEDS_NEW = [920101, 920102, 920103, 920104, 920105, 920106]
SEEDS12 = SEEDS_ORIG + SEEDS_NEW
CLASS_ORDER = ["ramp-fast", "ramp-slow", "mainline-fast", "mainline-slow"]
VIDS = ["V0", "V1", "V2", "V3"]

AUDIT_LOG: list[str] = []


def log(msg: str = ""):
    print(msg)
    AUDIT_LOG.append(msg)


# ===========================================================================
# Step 0 -- merge to n=12 CSVs
# ===========================================================================
welfare_a = pd.read_csv(ORIG_WELFARE)
welfare_b = pd.read_csv(NEW_WELFARE)
assert list(welfare_a.columns) == list(welfare_b.columns), "welfare schema mismatch"
welfare12 = pd.concat([welfare_a, welfare_b], ignore_index=True)
welfare12_path = OUT_DIR / "pooled12_welfare_evaluation_merged.csv"
welfare12.to_csv(welfare12_path, index=False)
log(f"wrote {welfare12_path}  shape={welfare12.shape}  seeds={sorted(welfare12.seed.unique())}")

behav_a = pd.read_csv(ORIG_BEHAV)
behav_b = pd.read_csv(NEW_BEHAV)
assert list(behav_a.columns) == list(behav_b.columns), "behavioral schema mismatch"
behav12 = pd.concat([behav_a, behav_b], ignore_index=True)
behav12_path = OUT_DIR / "pooled12_behavioral_evaluation_merged.csv"
behav12.to_csv(behav12_path, index=False)
log(f"wrote {behav12_path}  shape={behav12.shape}  seeds={sorted(behav12.seed.unique())}")

# Reload as list-of-dict rows to reuse rq_audit_final.py's exact row-based logic
with open(welfare12_path, encoding="utf-8") as f:
    ROWS = list(csv.DictReader(f))
log(f"\nloaded {len(ROWS)} pooled rows for n=12 audit")


# ===========================================================================
# Consistency audit (identical convention to rq_audit_final.py)
# ===========================================================================
def audit_raw_fields():
    problems = []
    for r in ROWS:
        comp = int(r["completion"]); coll = int(r["collision"]); to = int(r["timeout"])
        if comp + coll + to != 1:
            problems.append(("outcome flags not one-hot", r["run_id"], r["seed"], r["scenario_id"]))
        tr = r["term_reason"]
        expect = {"success": comp, "collision": coll, "truncation": to}
        if expect.get(tr, None) != 1:
            problems.append(("term_reason/flag mismatch", r["run_id"], r["seed"], r["scenario_id"], tr))
    log(f"[audit] outcome-flag / term_reason consistency problems: {len(problems)}")
    for p in problems[:10]:
        log(f"   {p}")
    return problems


def gini(values):
    n = len(values)
    total = sum(values)
    if total == 0:
        return None
    numerator = sum(abs(a - b) for a in values for b in values)
    return float(numerator / (2.0 * n * total))


def burden_range(values):
    return float(max(values) - min(values))


def bootstrap_ci_paired(diffs: np.ndarray, n_boot: int = 10000, rng_seed: int = 0):
    rng = np.random.default_rng(rng_seed)
    n = len(diffs)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = np.mean(diffs[idx])
    return float(np.mean(diffs)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), boot


def bootstrap_ci_single(values: np.ndarray, n_boot: int = 10000, rng_seed: int = 0):
    rng = np.random.default_rng(rng_seed)
    n = len(values)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = np.mean(values[idx])
    return float(np.mean(values)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def bootstrap_p_value(boot_diffs: np.ndarray) -> float:
    p_le = float(np.mean(boot_diffs <= 0))
    p_ge = float(np.mean(boot_diffs >= 0))
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


def holm_correction(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, i in enumerate(order):
        adj = (m - rank) * pvals[i]
        running_max = max(running_max, adj)
        adjusted[i] = min(1.0, running_max)
    return adjusted


def subset(condition=None, bank=None, seed=None):
    out = ROWS
    if condition is not None:
        out = [r for r in out if r["condition"] == condition]
    if bank is not None:
        out = [r for r in out if r["bank"] == bank]
    if seed is not None:
        out = [r for r in out if int(r["seed"]) == seed]
    return out


# ===========================================================================
# Table: per-seed task performance (extends the original Table 5.2 to n=12)
# ===========================================================================
def task_performance_table():
    log("\n" + "=" * 78)
    log("TASK PERFORMANCE TABLE -- all 12 seeds x Mean(H0,H1)/GGI(H1)/Maximin(H1)")
    log("=" * 78)
    rows_out = []
    for seed in SEEDS12:
        rec = {"seed": seed}
        for cond, banks in (("mean", ("H0", "H1")), ("ggi", ("H1",)), ("maximin", ("H1",))):
            for bank in banks:
                rs = subset(cond, bank, seed)
                comp = np.mean([float(r["completion"]) for r in rs])
                coll = np.mean([float(r["collision"]) for r in rs])
                to = np.mean([float(r["timeout"]) for r in rs])
                rec[f"{cond}_{bank}_completion"] = round(comp, 4)
                rec[f"{cond}_{bank}_collision"] = round(coll, 4)
                rec[f"{cond}_{bank}_timeout"] = round(to, 4)
        rows_out.append(rec)
        log(f"  {seed}: mean_H0={rec['mean_H0_completion']:.3f}/{rec['mean_H0_collision']:.3f}  "
            f"mean_H1={rec['mean_H1_completion']:.3f}/{rec['mean_H1_collision']:.3f}  "
            f"ggi_H1={rec['ggi_H1_completion']:.3f}/{rec['ggi_H1_collision']:.3f}/{rec['ggi_H1_timeout']:.3f}  "
            f"maximin_H1={rec['maximin_H1_completion']:.3f}/{rec['maximin_H1_collision']:.3f}/{rec['maximin_H1_timeout']:.3f}")
    out_csv = OUT_DIR / "pooled12_task_performance.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    log(f"\nwrote {out_csv}")
    return rows_out


# ===========================================================================
# TASK 1 (n=12) -- RQ1 burden-inequality analysis (Mean, H0 vs H1)
# ===========================================================================
def task1():
    log("\n" + "=" * 78)
    log("TASK 1 (n=12) -- RQ1 burden-inequality analysis (Mean policy, H0 vs H1)")
    log("=" * 78)

    mean_h0 = subset("mean", "H0")
    mean_h1 = subset("mean", "H1")
    log(f"rows: mean/H0={len(mean_h0)}  mean/H1={len(mean_h1)}  (expect 3072 each = 12 seeds x 256 episodes)")

    def episode_metrics(row):
        u = [float(row[f"U_{v}"]) for v in VIDS]
        c = [float(row[f"C_{v}"]) for v in VIDS]
        return {
            "U_mean": sum(u) / 4.0, "U_min": min(u), "utility_gini": gini(u),
            "C_mean": sum(c) / 4.0, "burden_gini": gini(c), "burden_range": burden_range(c),
            "completion": float(row["completion"]), "collision": float(row["collision"]), "timeout": float(row["timeout"]),
        }

    for row in mean_h0 + mean_h1:
        row["_m"] = episode_metrics(row)

    summary_metric_names = ["U_mean", "U_min", "utility_gini", "C_mean", "burden_gini", "burden_range"]
    metric_names = summary_metric_names + ["completion", "collision", "timeout"]
    metric_labels = {
        "U_mean": "Mean utility", "U_min": "Worst-off utility", "utility_gini": "Utility Gini",
        "C_mean": "Mean burden", "burden_gini": "Burden Gini", "burden_range": "Burden range",
    }

    seed_level = {"H0": {}, "H1": {}}
    all_zero_burden_rate = {"H0": {}, "H1": {}}
    for bank, rowset in (("H0", mean_h0), ("H1", mean_h1)):
        for seed in SEEDS12:
            srows = [r for r in rowset if int(r["seed"]) == seed]
            n_zero = sum(1 for r in srows if r["_m"]["burden_gini"] is None)
            all_zero_burden_rate[bank][seed] = n_zero / len(srows)
            d = {}
            for m in metric_names:
                vals = [r["_m"][m] for r in srows if r["_m"][m] is not None]
                d[m] = float(np.mean(vals)) if vals else None
            seed_level[bank][seed] = d

    log("\nAll-zero-burden episode rate per seed (H0/H1):")
    for bank in ("H0", "H1"):
        row_str = "  ".join(f"{s}:{all_zero_burden_rate[bank][s]:.3f}" for s in SEEDS12)
        log(f"  {bank}: {row_str}")

    n_zero_h0 = sum(1 for s in SEEDS12 if seed_level["H0"][s]["burden_gini"] is None)
    log(f"\nSeeds with fully-undefined (all-zero) H0 burden Gini: {n_zero_h0}/12")

    summary_rows = []
    log("\nRQ1 H0 vs H1 summary table (seed = unit of replication, paired bootstrap, 10000 resamples, seed0=0), n=12:")
    for m in summary_metric_names:
        h0_raw = [seed_level["H0"][s][m] for s in SEEDS12]
        h1_raw = [seed_level["H1"][s][m] for s in SEEDS12]
        undefined_seeds = [SEEDS12[i] for i in range(12) if h0_raw[i] is None or h1_raw[i] is None]
        if undefined_seeds:
            used_seeds = [s for s in SEEDS12 if s not in undefined_seeds]
            log(f"{metric_labels[m]:20s}: UNDEFINED for seed(s) {undefined_seeds}; falling back to n={len(used_seeds)} subset: {used_seeds}")
            h0v = np.array([seed_level["H0"][s][m] for s in used_seeds])
            h1v = np.array([seed_level["H1"][s][m] for s in used_seeds])
            n_used = len(used_seeds)
        else:
            h0v = np.array(h0_raw); h1v = np.array(h1_raw); n_used = 12; used_seeds = SEEDS12
        diffs = h1v - h0v
        if n_used >= 3:
            est, lo, hi, _ = bootstrap_ci_paired(diffs)
        else:
            est, lo, hi = float(np.mean(diffs)), float("nan"), float("nan")
        log(f"{metric_labels[m]:20s}: H0(n={n_used})={np.round(h0v,4)}")
        log(f"{'':20s}  H1(n={n_used})={np.round(h1v,4)}")
        if lo == lo:
            log(f"{'':20s}  H1-H0 mean={est:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]")
        else:
            log(f"{'':20s}  H1-H0 mean={est:+.4f}  95% CI=not computed (n={n_used})")
        summary_rows.append({
            "metric": metric_labels[m], "H0": round(float(np.mean(h0v)), 6), "H1": round(float(np.mean(h1v)), 6),
            "difference": round(est, 6), "CI_low": (round(lo, 6) if lo == lo else "NA"),
            "CI_high": (round(hi, 6) if hi == hi else "NA"), "n_seeds_used": n_used,
        })

    seed_metrics_csv = OUT_DIR / "pooled12_rq1_seed_level_metrics.csv"
    with open(seed_metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "bank", "U_mean", "U_min", "utility_gini", "C_mean", "burden_gini", "burden_range", "completion", "collision", "timeout"])
        for bank in ("H0", "H1"):
            for s in SEEDS12:
                d = seed_level[bank][s]
                w.writerow([s, bank, d["U_mean"], d["U_min"], d["utility_gini"], d["C_mean"], d["burden_gini"], d["burden_range"], d["completion"], d["collision"], d["timeout"]])
    log(f"\nwrote {seed_metrics_csv}")

    summary_csv = OUT_DIR / "pooled12_rq1_h0_h1_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "H0", "H1", "difference", "CI_low", "CI_high", "n_seeds_used"])
        w.writeheader(); w.writerows(summary_rows)
    log(f"wrote {summary_csv}")
    return {"seed_level": seed_level, "summary_rows": summary_rows}


# ===========================================================================
# TASK 2 (n=12) -- successful-episodes-only burden sensitivity
# ===========================================================================
def task2():
    log("\n" + "=" * 78)
    log("TASK 2 (n=12) -- successful-episodes-only burden sensitivity (H1)")
    log("=" * 78)

    result_succ = {}
    result_uncond = {}
    class_succ = {}
    n_success = {}

    for cond in ("mean", "ggi", "maximin"):
        rowset_h1 = subset(cond, "H1")
        for seed in SEEDS12:
            srows_all = [r for r in rowset_h1 if int(r["seed"]) == seed]
            srows_succ = [r for r in srows_all if int(r["completion"]) == 1]
            n_success[(seed, cond)] = (len(srows_succ), len(srows_all))

            def agg(rows):
                c_means, c_maxes, b_ranges, b_ginis = [], [], [], []
                for r in rows:
                    c = [float(r[f"C_{v}"]) for v in VIDS]
                    c_means.append(sum(c) / 4.0); c_maxes.append(max(c)); b_ranges.append(max(c) - min(c))
                    g = gini(c)
                    if g is not None:
                        b_ginis.append(g)
                return {"C_mean": float(np.mean(c_means)) if c_means else None,
                        "C_max": float(np.mean(c_maxes)) if c_maxes else None,
                        "burden_range": float(np.mean(b_ranges)) if b_ranges else None,
                        "burden_gini": float(np.mean(b_ginis)) if b_ginis else None}

            result_succ[(seed, cond)] = agg(srows_succ)
            result_uncond[(seed, cond)] = agg(srows_all)

            c_by_class = defaultdict(list)
            for r in srows_succ:
                for v in VIDS:
                    cls = f"{r[f'role_{v}']}-{r[f'speed_class_{v}']}"
                    c_by_class[cls].append(float(r[f"C_{v}"]))
            class_succ[(seed, cond)] = {cls: (float(np.mean(c_by_class[cls])) if c_by_class[cls] else None) for cls in CLASS_ORDER}

    log("\nSuccessful-episode counts per seed x condition (n_success/n_total, H1):")
    for cond in ("mean", "ggi", "maximin"):
        line = "  ".join(f"{s}:{n_success[(s,cond)][0]}/{n_success[(s,cond)][1]}" for s in SEEDS12)
        log(f"  {cond:8s}: {line}")

    log("\nCondition-level (12-seed mean) C_mean: success-only vs unconditional:")
    seed_csv_rows = []
    condition_table = {}
    for cond in ("mean", "ggi", "maximin"):
        succ_vals = np.array([result_succ[(s, cond)]["C_mean"] for s in SEEDS12 if result_succ[(s, cond)]["C_mean"] is not None])
        uncond_vals = np.array([result_uncond[(s, cond)]["C_mean"] for s in SEEDS12])
        log(f"  {cond:8s}: success-only mean={np.mean(succ_vals):.4f} (n_seeds={len(succ_vals)})   unconditional mean={np.mean(uncond_vals):.4f}")
        condition_table[cond] = {"C_mean_success": float(np.mean(succ_vals)), "C_mean_unconditional": float(np.mean(uncond_vals))}
        for s in SEEDS12:
            d = result_succ[(s, cond)]
            seed_csv_rows.append({"seed": s, "condition": cond, "C_mean_success": d["C_mean"], "C_max_success": d["C_max"],
                                   "burden_range_success": d["burden_range"], "burden_gini_success": d["burden_gini"]})

    seed_csv = OUT_DIR / "pooled12_successful_episode_burden_by_seed.csv"
    with open(seed_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "condition", "C_mean_success", "C_max_success", "burden_range_success", "burden_gini_success"])
        w.writeheader(); w.writerows(seed_csv_rows)
    log(f"\nwrote {seed_csv}")

    class_csv_rows = []
    for cond in ("mean", "ggi", "maximin"):
        for s in SEEDS12:
            for cls in CLASS_ORDER:
                class_csv_rows.append({"seed": s, "condition": cond, "role_speed_class": cls, "mean_burden_success": class_succ[(s, cond)][cls]})
    class_csv = OUT_DIR / "pooled12_successful_episode_burden_by_class.csv"
    with open(class_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "condition", "role_speed_class", "mean_burden_success"])
        w.writeheader(); w.writerows(class_csv_rows)
    log(f"wrote {class_csv}")

    log("\nBurden by class, successful-only, pooled 12-seed mean:")
    for cond in ("mean", "ggi", "maximin"):
        line = "  ".join(f"{cls}={np.mean([class_succ[(s,cond)][cls] for s in SEEDS12 if class_succ[(s,cond)][cls] is not None]):.3f}" for cls in CLASS_ORDER)
        log(f"  {cond:8s}: {line}")

    return {"result_succ": result_succ, "result_uncond": result_uncond, "condition_table": condition_table}


# ===========================================================================
# TASK 3 (n=12) -- non-inferiority (completion/collision margins)
# ===========================================================================
def task3():
    log("\n" + "=" * 78)
    log("TASK 3 (n=12) -- non-inferiority margins (completion -0.05, collision +0.03)")
    log("=" * 78)
    mean_h1 = subset("mean", "H1")
    results = []
    for cond in ("ggi", "maximin"):
        rowset = subset(cond, "H1")
        comp_c = np.array([np.mean([float(r["completion"]) for r in rowset if int(r["seed"]) == s]) for s in SEEDS12])
        comp_m = np.array([np.mean([float(r["completion"]) for r in mean_h1 if int(r["seed"]) == s]) for s in SEEDS12])
        coll_c = np.array([np.mean([float(r["collision"]) for r in rowset if int(r["seed"]) == s]) for s in SEEDS12])
        coll_m = np.array([np.mean([float(r["collision"]) for r in mean_h1 if int(r["seed"]) == s]) for s in SEEDS12])
        est_c, lo_c, hi_c, _ = bootstrap_ci_paired(comp_c - comp_m)
        est_k, lo_k, hi_k, _ = bootstrap_ci_paired(coll_c - coll_m)
        comp_cleared = lo_c > -0.05
        coll_cleared = hi_k < 0.03
        log(f"\n{cond.upper()} - Mean, completion: est={est_c:+.4f} CI=[{lo_c:+.4f},{hi_c:+.4f}]  cleared={comp_cleared}")
        log(f"{cond.upper()} - Mean, collision : est={est_k:+.4f} CI=[{lo_k:+.4f},{hi_k:+.4f}]  cleared={coll_cleared}")
        results.append({"contrast": f"{cond.upper()} - Mean", "outcome": "completion", "point_estimate": round(est_c, 6),
                         "CI_low": round(lo_c, 6), "CI_high": round(hi_c, 6), "margin": -0.05, "margin_cleared": "Yes" if comp_cleared else "No"})
        results.append({"contrast": f"{cond.upper()} - Mean", "outcome": "collision", "point_estimate": round(est_k, 6),
                         "CI_low": round(lo_k, 6), "CI_high": round(hi_k, 6), "margin": 0.03, "margin_cleared": "Yes" if coll_cleared else "No"})
    csv_path = OUT_DIR / "pooled12_noninferiority.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["contrast", "outcome", "point_estimate", "CI_low", "CI_high", "margin", "margin_cleared"])
        w.writeheader(); w.writerows(results)
    log(f"\nwrote {csv_path}")
    return results


# ===========================================================================
# TASK 4 (n=12) -- Holm-corrected U_min comparisons
# ===========================================================================
def task4():
    log("\n" + "=" * 78)
    log("TASK 4 (n=12) -- Holm-corrected U_min comparisons (GGI-Mean, Maximin-Mean)")
    log("=" * 78)
    mean_h1 = subset("mean", "H1")
    umin_mean = np.array([np.mean([float(r["min_U"]) for r in mean_h1 if int(r["seed"]) == s]) for s in SEEDS12])
    rows = []
    raw_p = {}; est_ci = {}
    for cond in ("ggi", "maximin"):
        rowset = subset(cond, "H1")
        umin_c = np.array([np.mean([float(r["min_U"]) for r in rowset if int(r["seed"]) == s]) for s in SEEDS12])
        est, lo, hi, boot = bootstrap_ci_paired(umin_c - umin_mean)
        p = bootstrap_p_value(boot)
        raw_p[cond] = p; est_ci[cond] = (est, lo, hi)
        log(f"\n{cond.upper()} - Mean, U_min: est={est:+.4f} CI=[{lo:+.4f},{hi:+.4f}]  raw bootstrap p={p:.4f}")
    order = ["ggi", "maximin"]
    pvals = [raw_p[c] for c in order]
    adj = holm_correction(pvals)
    log("\nHolm step-down correction (m=2 primary comparisons), n=12:")
    for cond, p, a in zip(order, pvals, adj):
        est, lo, hi = est_ci[cond]
        sig_before = p < 0.05; sig_after = a < 0.05
        log(f"  {cond.upper()}-Mean: raw_p={p:.4f}  holm_adjusted_p={a:.4f}  sig_before={sig_before}  sig_after={sig_after}")
        rows.append({"contrast": f"{cond.upper()} - Mean", "outcome": "U_min", "point_estimate": round(est, 6),
                     "CI_low": round(lo, 6), "CI_high": round(hi, 6), "raw_p_bootstrap": round(p, 6),
                     "holm_adjusted_p": round(a, 6), "significant_before_holm_alpha05": sig_before, "significant_after_holm_alpha05": sig_after})
    csv_path = OUT_DIR / "pooled12_umin_holm.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["contrast", "outcome", "point_estimate", "CI_low", "CI_high", "raw_p_bootstrap",
                                           "holm_adjusted_p", "significant_before_holm_alpha05", "significant_after_holm_alpha05"])
        w.writeheader(); w.writerows(rows)
    log(f"\nwrote {csv_path}")
    return rows


# ===========================================================================
# TASK 5 (n=12) -- worst-off vehicle tie-corrected identity
# ===========================================================================
def task5():
    log("\n" + "=" * 78)
    log("TASK 5 (n=12) -- worst-off vehicle tie-corrected identity")
    log("=" * 78)

    def tie_breakdown(rowset):
        tie_size_counts = defaultdict(int)
        frac = defaultdict(float)
        n_nondegenerate = 0
        for r in rowset:
            us = {v: float(r[f"U_{v}"]) for v in VIDS}
            m = min(us.values())
            tied = [v for v, u in us.items() if abs(u - m) < 1e-9]
            tie_size_counts[len(tied)] += 1
            if len(tied) == 4:
                continue
            n_nondegenerate += 1
            w = 1.0 / len(tied)
            for v in tied:
                cls = f"{r['role_' + v]}-{r['speed_class_' + v]}"
                frac[cls] += w
        return tie_size_counts, frac, n_nondegenerate

    class_rows = []
    for cond in ("mean", "ggi", "maximin"):
        rowset = subset(cond, "H1")
        tie_counts, frac, n_nd = tie_breakdown(rowset)
        n_total = len(rowset)
        n_tied_any = sum(v for k, v in tie_counts.items() if k >= 2)
        log(f"\n{cond.upper()} (H1, 12 seeds pooled, n_episodes={n_total}):")
        log(f"  episodes with ANY tie at the minimum: {n_tied_any}/{n_total} ({100*n_tied_any/n_total:.1f}%)")
        log(f"  4-way degenerate ties: {tie_counts[4]}/{n_total} ({100*tie_counts[4]/n_total:.1f}%)")
        log(f"  non-degenerate episodes used: n={n_nd}")
        for cls in CLASS_ORDER:
            pct = 100 * frac[cls] / n_nd if n_nd else float("nan")
            log(f"    {cls:16s}: pct={pct:.1f}%")
            class_rows.append({"condition": cond, "role_speed_class": cls, "fractional_count": round(frac[cls], 4),
                               "n_nondegenerate_episodes": n_nd, "percent": round(pct, 4)})
    csv_path = OUT_DIR / "pooled12_worst_off_tie_corrected.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "role_speed_class", "fractional_count", "n_nondegenerate_episodes", "percent"])
        w.writeheader(); w.writerows(class_rows)
    log(f"\nwrote {csv_path}")
    return class_rows


# ===========================================================================
# STANDALONE NEW-6 CHECK -- recompute from raw episodes (cross-check against
# the pre-existing new_seed_umin_contrasts.csv), independent of pooling
# ===========================================================================
def new6_standalone_check():
    log("\n" + "=" * 78)
    log("STANDALONE NEW-6 CHECK (920101-920106 only) -- ΔU_min GGI/Maximin vs Mean")
    log("=" * 78)
    mean_h1 = [r for r in subset("mean", "H1") if int(r["seed"]) in SEEDS_NEW]
    umin_mean = np.array([np.mean([float(r["min_U"]) for r in mean_h1 if int(r["seed"]) == s]) for s in SEEDS_NEW])
    rows = []
    for cond in ("ggi", "maximin"):
        rowset = [r for r in subset(cond, "H1") if int(r["seed"]) in SEEDS_NEW]
        umin_c = np.array([np.mean([float(r["min_U"]) for r in rowset if int(r["seed"]) == s]) for s in SEEDS_NEW])
        est, lo, hi, boot = bootstrap_ci_paired(umin_c - umin_mean)
        log(f"{cond.upper()} - Mean, U_min (new-6 only): est={est:+.4f} CI=[{lo:+.4f},{hi:+.4f}]  seed values GGI/Max={np.round(umin_c,3)}  Mean={np.round(umin_mean,3)}")
        rows.append({"contrast": f"{cond.upper()} - Mean", "cohort": "new6", "point_estimate": round(est, 6), "CI_low": round(lo, 6), "CI_high": round(hi, 6)})
    csv_path = OUT_DIR / "new6_standalone_umin_check.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["contrast", "cohort", "point_estimate", "CI_low", "CI_high"])
        w.writeheader(); w.writerows(rows)
    log(f"\nwrote {csv_path}")
    return rows


if __name__ == "__main__":
    audit_raw_fields()
    task_performance_table()
    r1 = task1()
    r2 = task2()
    r3 = task3()
    r4 = task4()
    r5 = task5()
    r6 = new6_standalone_check()
    with open(OUT_DIR / "audit_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(AUDIT_LOG))
    print(f"\n\nFull audit log written to {OUT_DIR / 'audit_log.txt'}")
