#!/usr/bin/env python3
"""Analyze new-seed replication eval CSVs (new_protocol.md §31-38, §45).

Training seed is the replication unit. Paired seed-level bootstrap CIs
(10,000 resamples, rng seed 0). No new p-value test. All six new seeds
retained regardless of quality.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

sys_path_setup = True
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent))
from _common import BUNDLE_ROOT, OUTPUTS  # noqa: E402
from replication_common import CONDITION_DIR, CURRICULUM_ROOT, SEEDS  # noqa: E402

NEW_SEEDS = list(SEEDS)
ORIG_SEEDS = [900101, 900102, 900103, 900104, 910101, 910102]
CLASS_ORDER = ["ramp-fast", "ramp-slow", "mainline-fast", "mainline-slow"]
VIDS = ["V0", "V1", "V2", "V3"]
N_BOOT = 10_000
BOOT_SEED = 0

EVAL_ROOT = OUTPUTS / "seed_replication_v1"
WELFARE_SHARD_DIR = EVAL_ROOT / "welfare_eval"
BEHAV_DIR = EVAL_ROOT / "behavioral"
OUT_DIR = EVAL_ROOT / "analysis"
ORIG_WELFARE_CSV = OUTPUTS / "welfare_analysis" / "formal_welfare_evaluation_merged.csv"
ORIG_HB_CSV = OUTPUTS / "welfare_analysis" / "high_burden_diagnostic_merged.csv"


def gini(values):
    n = len(values)
    total = sum(values)
    if total == 0:
        return None
    numerator = sum(abs(a - b) for a in values for b in values)
    return float(numerator / (2.0 * n * total))


def merge_shards(directory: Path, pattern: str) -> list[dict]:
    rows = []
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no shards matching {pattern} in {directory}")
    for f in files:
        with open(f, encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def subset(rows, *, condition=None, bank=None, seed=None, completion=None):
    out = rows
    if condition is not None:
        out = [r for r in out if r["condition"] == condition]
    if bank is not None:
        out = [r for r in out if r["bank"] == bank]
    if seed is not None:
        out = [r for r in out if int(r["seed"]) == int(seed)]
    if completion is not None:
        out = [r for r in out if int(r["completion"]) == completion]
    return out


def seed_mean(rows, field, seed):
    vals = [float(r[field]) for r in rows if int(r["seed"]) == seed]
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_mean_ci(values: np.ndarray, n=N_BOOT, rng_seed=BOOT_SEED):
    rng = np.random.default_rng(rng_seed)
    n_s = len(values)
    means = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_s, n_s)
        means[i] = np.mean(values[idx])
    return float(np.mean(values)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_paired_diff_ci(a: np.ndarray, b: np.ndarray, n=N_BOOT, rng_seed=BOOT_SEED):
    """CI for mean(b - a), resampling seed indices together."""
    rng = np.random.default_rng(rng_seed)
    diffs = b - a
    n_s = len(diffs)
    boot = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_s, n_s)
        boot[i] = np.mean(diffs[idx])
    return float(np.mean(diffs)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def vehicle_cs(row) -> list[float]:
    return [float(row[f"C_{v}"]) for v in VIDS]


def vehicle_us(row) -> list[float]:
    return [float(row[f"U_{v}"]) for v in VIDS]


def seed_level_metrics(rows, seeds, condition, bank) -> dict[int, dict]:
    out = {}
    for s in seeds:
        rs = subset(rows, condition=condition, bank=bank, seed=s)
        if not rs:
            continue
        u_means, u_mins, c_means, c_maxes, comps, colls, tos = [], [], [], [], [], [], []
        u_ginis, b_ginis, b_ranges = [], [], []
        for r in rs:
            u = vehicle_us(r)
            c = vehicle_cs(r)
            u_means.append(float(r["mean_U"]))
            u_mins.append(float(r["min_U"]))
            c_means.append(float(r["C_mean"]))
            c_maxes.append(float(r["C_max"]))
            comps.append(int(r["completion"]))
            colls.append(int(r["collision"]))
            tos.append(int(r["timeout"]))
            ug = gini(u)
            if ug is not None:
                u_ginis.append(ug)
            bg = gini(c)
            if bg is not None:
                b_ginis.append(bg)
            b_ranges.append(max(c) - min(c))
        all_c = [x for r in rs for x in vehicle_cs(r)]
        succ = [r for r in rs if int(r["completion"]) == 1]
        succ_c = [x for r in succ for x in vehicle_cs(r)]
        out[s] = {
            "seed": s, "condition": condition, "bank": bank,
            "U_mean": float(np.mean(u_means)),
            "U_min": float(np.mean(u_mins)),
            "utility_gini": float(np.mean(u_ginis)) if u_ginis else "",
            "C_mean": float(np.mean(c_means)),
            "C_max": float(np.mean(c_maxes)),
            "burden_gini": float(np.mean(b_ginis)) if b_ginis else "",
            "burden_range": float(np.mean(b_ranges)),
            "completion": float(np.mean(comps)),
            "collision": float(np.mean(colls)),
            "timeout": float(np.mean(tos)),
            "C95": float(np.percentile(all_c, 95)) if all_c else "",
            "C95_success_only": float(np.percentile(succ_c, 95)) if succ_c else "",
            "C_mean_success_only": float(np.mean([float(r["C_mean"]) for r in succ])) if succ else "",
            "n_episodes": len(rs),
            "n_success": len(succ),
            "n_burden_gini_undefined": sum(1 for r in rs if gini(vehicle_cs(r)) is None),
        }
    return out


def competence_pass(d: dict) -> bool:
    return d["completion"] >= 0.90 and d["collision"] <= 0.05 and d["timeout"] <= 0.05


def task_summary_from_curriculum() -> list[dict]:
    rows = []
    for seed in NEW_SEEDS:
        man = (CURRICULUM_ROOT / str(seed) / "C64_R50" /
               f"seed_{seed}_C64_R50_manifest.json")
        data = json.loads(man.read_text(encoding="utf-8"))
        ckpts = data.get("checkpoints") or []
        last = ckpts[-1]["window"] if ckpts else {}
        rows.append({
            "seed": seed,
            "status": "frozen",
            "curriculum_complete": 1,
            "final_task_step": data.get("final_step", 1_200_000),
            "C64_training_window_completion": last.get("completion_rate", ""),
            "C64_training_window_collision": last.get("collision_rate", ""),
            "C64_training_window_timeout": last.get("truncation_rate", ""),
        })
    return rows


def fmt(x, nd=4):
    if x is None or x == "":
        return "NA"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    welfare = merge_shards(WELFARE_SHARD_DIR, "replication_welfare_eval_shard*.csv")
    write_csv(WELFARE_SHARD_DIR / "replication_welfare_evaluation_merged.csv",
              list(welfare[0].keys()), welfare)

    behav = merge_shards(BEHAV_DIR, "replication_behavioral_eval_shard*.csv")
    write_csv(BEHAV_DIR / "replication_behavioral_evaluation_merged.csv",
              list(behav[0].keys()), behav)

    hb = merge_shards(BEHAV_DIR, "replication_high_burden_shard*.csv")
    write_csv(BEHAV_DIR / "replication_high_burden_merged.csv",
              list(hb[0].keys()), hb)

    write_csv(BUNDLE_ROOT / "new_seed_task_summary.csv",
              ["seed", "status", "curriculum_complete", "final_task_step",
               "C64_training_window_completion", "C64_training_window_collision",
               "C64_training_window_timeout"],
              task_summary_from_curriculum())

    # Seed-level tables
    h1_by_cond = {c: seed_level_metrics(welfare, NEW_SEEDS, c, "H1") for c in ("mean", "ggi", "maximin")}
    h0_mean = seed_level_metrics(welfare, NEW_SEEDS, "mean", "H0")

    task_rows = []
    welfare_rows = []
    tail_rows = []
    success_rows = []
    competence_rows = []
    for cond in ("mean", "ggi", "maximin"):
        banks = ("H0", "H1") if cond == "mean" else ("H1",)
        for bank in banks:
            sl = h0_mean if (cond == "mean" and bank == "H0") else h1_by_cond[cond]
            if bank == "H1":
                sl = h1_by_cond[cond]
            for s in NEW_SEEDS:
                d = sl[s]
                task_rows.append({
                    "seed": s, "condition": cond, "bank": bank,
                    "completion": d["completion"], "collision": d["collision"],
                    "timeout": d["timeout"],
                })
                welfare_rows.append({
                    "seed": s, "condition": cond, "bank": bank,
                    "U_mean": d["U_mean"], "U_min": d["U_min"],
                    "utility_gini": d["utility_gini"], "C_mean": d["C_mean"],
                    "C_max": d["C_max"], "burden_gini": d["burden_gini"],
                    "burden_range": d["burden_range"],
                    "n_burden_gini_undefined": d["n_burden_gini_undefined"],
                })
                tail_rows.append({
                    "seed": s, "condition": cond, "bank": bank,
                    "C95_unconditional": d["C95"],
                    "C95_success_only": d["C95_success_only"],
                })
                # success-only burden by class
                rs = subset(welfare, condition=cond, bank=bank, seed=s, completion=1)
                by_class = defaultdict(list)
                for r in rs:
                    for v in VIDS:
                        cls = f"{r[f'role_{v}']}-{r[f'speed_class_{v}']}"
                        by_class[cls].append(float(r[f"C_{v}"]))
                rec = {
                    "seed": s, "condition": cond, "bank": bank,
                    "n_success_episodes": len(rs),
                    "C_mean_success_only": d["C_mean_success_only"],
                    "C95_success_only": d["C95_success_only"],
                }
                for cls in CLASS_ORDER:
                    rec[f"C_mean_{cls}"] = float(np.mean(by_class[cls])) if by_class[cls] else ""
                    rec[f"C95_{cls}"] = float(np.percentile(by_class[cls], 95)) if by_class[cls] else ""
                success_rows.append(rec)
        for s in NEW_SEEDS:
            d = h1_by_cond[cond][s]
            competence_rows.append({
                "seed": s, "condition": cond, "bank": "H1",
                "completion": d["completion"], "collision": d["collision"],
                "timeout": d["timeout"],
                "pass_completion_ge_0.90": int(d["completion"] >= 0.90),
                "pass_collision_le_0.05": int(d["collision"] <= 0.05),
                "pass_timeout_le_0.05": int(d["timeout"] <= 0.05),
                "competence_pass": int(competence_pass(d)),
            })

    write_csv(BUNDLE_ROOT / "new_seed_formal_task_metrics.csv",
              ["seed", "condition", "bank", "completion", "collision", "timeout"], task_rows)
    write_csv(BUNDLE_ROOT / "new_seed_formal_welfare_metrics.csv",
              ["seed", "condition", "bank", "U_mean", "U_min", "utility_gini", "C_mean",
               "C_max", "burden_gini", "burden_range", "n_burden_gini_undefined"], welfare_rows)
    write_csv(BUNDLE_ROOT / "new_seed_tail_burden.csv",
              ["seed", "condition", "bank", "C95_unconditional", "C95_success_only"], tail_rows)
    write_csv(BUNDLE_ROOT / "new_seed_success_only_burden.csv",
              ["seed", "condition", "bank", "n_success_episodes", "C_mean_success_only",
               "C95_success_only"] + [f"{m}_{c}" for c in CLASS_ORDER for m in ("C_mean", "C95")],
              success_rows)
    write_csv(BUNDLE_ROOT / "new_seed_competence_summary.csv",
              ["seed", "condition", "bank", "completion", "collision", "timeout",
               "pass_completion_ge_0.90", "pass_collision_le_0.05", "pass_timeout_le_0.05",
               "competence_pass"], competence_rows)

    # Umin contrasts
    contrast_rows = []
    ggi_d, max_d = [], []
    for s in NEW_SEEDS:
        um = h1_by_cond["mean"][s]["U_min"]
        ug = h1_by_cond["ggi"][s]["U_min"]
        ux = h1_by_cond["maximin"][s]["U_min"]
        dg, dx = ug - um, ux - um
        ggi_d.append(dg)
        max_d.append(dx)
        contrast_rows.append({
            "seed": s, "U_min_Mean": um, "U_min_GGI": ug, "U_min_Maximin": ux,
            "delta_Umin_GGI": dg, "delta_Umin_Maximin": dx,
        })
    ggi_a, max_a = np.array(ggi_d), np.array(max_d)
    ggi_est, ggi_lo, ggi_hi = bootstrap_mean_ci(ggi_a)
    max_est, max_lo, max_hi = bootstrap_mean_ci(max_a)
    contrast_rows.append({
        "seed": "MEAN",
        "U_min_Mean": float(np.mean([h1_by_cond["mean"][s]["U_min"] for s in NEW_SEEDS])),
        "U_min_GGI": float(np.mean([h1_by_cond["ggi"][s]["U_min"] for s in NEW_SEEDS])),
        "U_min_Maximin": float(np.mean([h1_by_cond["maximin"][s]["U_min"] for s in NEW_SEEDS])),
        "delta_Umin_GGI": ggi_est, "delta_Umin_Maximin": max_est,
        "delta_Umin_GGI_CI95_low": ggi_lo, "delta_Umin_GGI_CI95_high": ggi_hi,
        "delta_Umin_Maximin_CI95_low": max_lo, "delta_Umin_Maximin_CI95_high": max_hi,
        "n_positive_GGI": int(np.sum(ggi_a > 0)), "n_negative_GGI": int(np.sum(ggi_a < 0)),
        "n_positive_Maximin": int(np.sum(max_a > 0)), "n_negative_Maximin": int(np.sum(max_a < 0)),
        "median_delta_GGI": float(np.median(ggi_a)), "median_delta_Maximin": float(np.median(max_a)),
    })
    write_csv(BUNDLE_ROOT / "new_seed_umin_contrasts.csv",
              ["seed", "U_min_Mean", "U_min_GGI", "U_min_Maximin", "delta_Umin_GGI",
               "delta_Umin_Maximin", "delta_Umin_GGI_CI95_low", "delta_Umin_GGI_CI95_high",
               "delta_Umin_Maximin_CI95_low", "delta_Umin_Maximin_CI95_high",
               "n_positive_GGI", "n_negative_GGI", "n_positive_Maximin", "n_negative_Maximin",
               "median_delta_GGI", "median_delta_Maximin"], contrast_rows)

    # Behavioural diagnostic summary (seed-level rate then mean)
    hb_rows_out = []
    for cond in ("mean", "ggi", "maximin"):
        for s in NEW_SEEDS:
            rs = [r for r in hb if r["condition"] == cond and int(r["seed"]) == s]
            n_ref = sum(int(r["has_reference_event"] or 0) for r in rs)
            usable = [r for r in rs if r.get("high_burden_goes_before_conflict") not in ("", None)]
            n_before = sum(int(r["high_burden_goes_before_conflict"]) == 1 for r in usable)
            n_after = sum(int(r["high_burden_goes_before_conflict"]) == 0 for r in usable)
            n_brake = sum(int(r.get("bystander_hard_brake_before_high_burden_exit") or 0) for r in rs
                          if r.get("has_reference_event") in (1, "1"))
            hb_rows_out.append({
                "seed": s, "condition": cond, "n_episodes": len(rs),
                "n_with_reference_event": n_ref,
                "n_order_defined": len(usable),
                "high_burden_goes_before_conflict_rate": (n_before / len(usable)) if usable else "",
                "n_goes_before": n_before, "n_goes_after": n_after,
                "bystander_hard_brake_rate": (n_brake / n_ref) if n_ref else "",
            })
    write_csv(BUNDLE_ROOT / "new_seed_behavioral_diagnostic.csv",
              ["seed", "condition", "n_episodes", "n_with_reference_event", "n_order_defined",
               "high_burden_goes_before_conflict_rate", "n_goes_before", "n_goes_after",
               "bystander_hard_brake_rate"], hb_rows_out)

    # H0 vs H1 Mean (secondary RQ1 replication)
    rq1_lines = []
    for field, key in [("U_mean", "U_mean"), ("U_min", "U_min"), ("utility_gini", "utility_gini"),
                       ("C_mean", "C_mean"), ("burden_range", "burden_range")]:
        h0v, h1v = [], []
        for s in NEW_SEEDS:
            a, b = h0_mean[s][key], h1_by_cond["mean"][s][key]
            if a == "" or b == "":
                continue
            h0v.append(float(a)); h1v.append(float(b))
        h0a, h1a = np.array(h0v), np.array(h1v)
        est, lo, hi = bootstrap_paired_diff_ci(h0a, h1a)
        rq1_lines.append((field, est, lo, hi, h0a, h1a))

    # Pooled 12
    orig = []
    with open(ORIG_WELFARE_CSV, encoding="utf-8") as f:
        orig = list(csv.DictReader(f))
    orig_h1 = {c: seed_level_metrics(orig, ORIG_SEEDS, c, "H1") for c in ("mean", "ggi", "maximin")}
    pooled_rows = []
    all12 = ORIG_SEEDS + NEW_SEEDS
    for cond in ("mean", "ggi", "maximin"):
        sl_orig, sl_new = orig_h1[cond], h1_by_cond[cond]
        for metric in ("U_min", "U_mean", "C_mean", "completion", "collision"):
            vals = np.array([sl_orig[s][metric] for s in ORIG_SEEDS] + [sl_new[s][metric] for s in NEW_SEEDS])
            est, lo, hi = bootstrap_mean_ci(vals)
            pooled_rows.append({
                "cohort": "pooled12", "condition": cond, "metric": metric,
                "n_seeds": 12, "mean": est, "median": float(np.median(vals)),
                "CI95_low": lo, "CI95_high": hi,
            })
        # C95 mean across seeds
        c95 = np.array([float(sl_orig[s]["C95"]) for s in ORIG_SEEDS] + [float(sl_new[s]["C95"]) for s in NEW_SEEDS])
        est, lo, hi = bootstrap_mean_ci(c95)
        pooled_rows.append({
            "cohort": "pooled12", "condition": cond, "metric": "C95",
            "n_seeds": 12, "mean": est, "median": float(np.median(c95)),
            "CI95_low": lo, "CI95_high": hi,
        })
        n_pass = sum(competence_pass(sl_orig[s]) for s in ORIG_SEEDS) + sum(competence_pass(sl_new[s]) for s in NEW_SEEDS)
        pooled_rows.append({
            "cohort": "pooled12", "condition": cond, "metric": "competence_pass_count",
            "n_seeds": 12, "mean": n_pass, "median": "", "CI95_low": "", "CI95_high": "",
            "n_pass": n_pass, "n_total": 12,
        })
    for name, cond in [("GGI", "ggi"), ("Maximin", "maximin")]:
        diffs = np.array(
            [orig_h1[cond][s]["U_min"] - orig_h1["mean"][s]["U_min"] for s in ORIG_SEEDS]
            + [h1_by_cond[cond][s]["U_min"] - h1_by_cond["mean"][s]["U_min"] for s in NEW_SEEDS]
        )
        est, lo, hi = bootstrap_mean_ci(diffs)
        pooled_rows.append({
            "cohort": "pooled12", "condition": f"{cond}-mean", "metric": "delta_Umin",
            "n_seeds": 12, "mean": est, "median": float(np.median(diffs)),
            "CI95_low": lo, "CI95_high": hi,
            "n_positive": int(np.sum(diffs > 0)), "n_negative": int(np.sum(diffs < 0)),
        })
    for cond in ("mean", "ggi", "maximin"):
        sl = h1_by_cond[cond]
        for metric in ("U_min", "U_mean", "C_mean", "completion", "C95"):
            vals = np.array([float(sl[s][metric]) for s in NEW_SEEDS])
            est, lo, hi = bootstrap_mean_ci(vals)
            pooled_rows.append({
                "cohort": "new6", "condition": cond, "metric": metric,
                "n_seeds": 6, "mean": est, "median": float(np.median(vals)),
                "CI95_low": lo, "CI95_high": hi,
            })
    write_csv(BUNDLE_ROOT / "pooled12_summary.csv",
              ["cohort", "condition", "metric", "n_seeds", "mean", "median",
               "CI95_low", "CI95_high", "n_pass", "n_total", "n_positive", "n_negative"],
              pooled_rows)

    # Behavioural contrasts new6
    def hb_rate(cond, seed):
        rec = next(r for r in hb_rows_out if r["condition"] == cond and r["seed"] == seed)
        v = rec["high_burden_goes_before_conflict_rate"]
        return float(v) if v != "" else float("nan")

    ggi_hb = np.array([hb_rate("ggi", s) - hb_rate("mean", s) for s in NEW_SEEDS])
    max_hb = np.array([hb_rate("maximin", s) - hb_rate("mean", s) for s in NEW_SEEDS])
    ggi_hb_est, ggi_hb_lo, ggi_hb_hi = bootstrap_mean_ci(ggi_hb)
    max_hb_est, max_hb_lo, max_hb_hi = bootstrap_mean_ci(max_hb)

    n_pass = {c: sum(competence_pass(h1_by_cond[c][s]) for s in NEW_SEEDS) for c in ("mean", "ggi", "maximin")}
    severe = []
    for s in NEW_SEEDS:
        d = h1_by_cond["mean"][s]
        if d["completion"] < 0.50 or d["collision"] > 0.40:
            severe.append((s, d["completion"], d["collision"]))

    # Outcome labels (descriptive, all seeds kept)
    if severe and (int(np.sum(ggi_a > 0)) + int(np.sum(max_a > 0))) > 0:
        outcome_note = "Outcome A-adjacent: at least one new Mean seed is a severe task failure."
    elif n_pass["mean"] == 6:
        outcome_note = "Outcome B-adjacent: all six new Mean policies meet the competence thresholds."
    else:
        outcome_note = "Mixed competence: some new Mean seeds miss the competence thresholds but are not 910102-like total failures."
    if int(np.sum(ggi_a > 0)) >= 5 or int(np.sum(max_a > 0)) >= 5:
        outcome_note += " Outcome C-adjacent: most matched U_min differences are positive."
    elif int(np.sum(ggi_a > 0)) <= 2 and int(np.sum(max_a > 0)) <= 2:
        outcome_note += " Outcome D-adjacent: no consistent GGI/Maximin U_min ordering."
    else:
        outcome_note += " U_min ordering is mixed across the six new seeds (Outcome D-leaning)."

    lines = []
    def L(s=""):
        lines.append(s)

    L("# new6_replication_summary")
    L()
    L("Independent-seed replication (protocol/new_protocol.md, VERSION 1).")
    L("New seeds 920101–920106 analysed separately from the original six.")
    L("Training seed is the statistical unit. Bootstrap: 10,000 resamples, rng seed 0.")
    L("No seed was dropped. No new p-value test was introduced.")
    L()
    L("## Task competence (H1)")
    L()
    L("| seed | Mean completion | Mean collision | GGI completion | Maximin completion | Mean competence pass |")
    L("|---:|---:|---:|---:|---:|:---:|")
    for s in NEW_SEEDS:
        m, g, x = h1_by_cond["mean"][s], h1_by_cond["ggi"][s], h1_by_cond["maximin"][s]
        L(f"| {s} | {m['completion']:.4f} | {m['collision']:.4f} | {g['completion']:.4f} | {x['completion']:.4f} | {competence_pass(m)} |")
    L()
    L(f"Competence-pass counts (completion≥0.90, collision≤0.05, timeout≤0.05): "
      f"Mean {n_pass['mean']}/6, GGI {n_pass['ggi']}/6, Maximin {n_pass['maximin']}/6.")
    if severe:
        L()
        L("Severe Mean-policy failures (completion<0.50 or collision>0.40):")
        for s, c, k in severe:
            L(f"- seed {s}: completion={c:.4f}, collision={k:.4f}")
    else:
        L()
        L("No new Mean seed showed a 910102-like severe task failure on H1.")
    L()
    L("## Matched U_min contrasts (H1)")
    L()
    L("| seed | U_min Mean | U_min GGI | U_min Maximin | Δ GGI | Δ Maximin |")
    L("|---:|---:|---:|---:|---:|---:|")
    for s, dg, dx in zip(NEW_SEEDS, ggi_a, max_a):
        m, g, x = h1_by_cond["mean"][s], h1_by_cond["ggi"][s], h1_by_cond["maximin"][s]
        L(f"| {s} | {m['U_min']:.4f} | {g['U_min']:.4f} | {x['U_min']:.4f} | {dg:+.4f} | {dx:+.4f} |")
    L()
    L(f"GGI−Mean U_min: mean={ggi_est:+.4f}, median={float(np.median(ggi_a)):+.4f}, "
      f"positive {int(np.sum(ggi_a>0))}/6, negative {int(np.sum(ggi_a<0))}/6, "
      f"95% CI [{ggi_lo:+.4f}, {ggi_hi:+.4f}].")
    L(f"Maximin−Mean U_min: mean={max_est:+.4f}, median={float(np.median(max_a)):+.4f}, "
      f"positive {int(np.sum(max_a>0))}/6, negative {int(np.sum(max_a<0))}/6, "
      f"95% CI [{max_lo:+.4f}, {max_hi:+.4f}].")
    L()
    L("## Secondary Mean H0 vs H1 (RQ1-style, new seeds only)")
    L()
    for field, est, lo, hi, h0a, h1a in rq1_lines:
        L(f"- {field}: H1−H0 mean={est:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}] "
          f"(H0 seed-means={np.round(h0a,3)}, H1={np.round(h1a,3)})")
    L()
    L("## Burden tail C95 and success-only C_mean (H1 seed-means)")
    L()
    for cond in ("mean", "ggi", "maximin"):
        c95 = np.array([float(h1_by_cond[cond][s]["C95"]) for s in NEW_SEEDS])
        cs = [h1_by_cond[cond][s]["C_mean_success_only"] for s in NEW_SEEDS]
        cs = np.array([float(x) for x in cs if x != ""])
        L(f"- {cond}: mean C95={np.mean(c95):.4f}; success-only C_mean={np.mean(cs):.4f}")
    L()
    L("## Behavioural diagnostic (high-burden vehicle goes before conflict, H1)")
    L()
    L(f"GGI−Mean rate difference: {ggi_hb_est:+.4f} (95% CI [{ggi_hb_lo:+.4f}, {ggi_hb_hi:+.4f}]).")
    L(f"Maximin−Mean rate difference: {max_hb_est:+.4f} (95% CI [{max_hb_lo:+.4f}, {max_hb_hi:+.4f}]).")
    L("A positive association would not mean the policy observes burden (it does not).")
    L()
    L("## Diagnostic conclusions")
    L()
    L(f"**Does the strong training-seed dependence reproduce?** "
      f"See the seed-level tables above. {outcome_note}")
    L()
    ggi_ci_excl0 = (ggi_lo > 0) or (ggi_hi < 0)
    max_ci_excl0 = (max_lo > 0) or (max_hi < 0)
    L(f"**Do GGI or Maximin show a consistent worst-off welfare advantage over Mean?** "
      f"GGI−Mean mean ΔU_min={ggi_est:+.4f} (CI [{ggi_lo:+.4f}, {ggi_hi:+.4f}], "
      f"{'excludes' if ggi_ci_excl0 else 'includes'} 0). "
      f"Maximin−Mean mean ΔU_min={max_est:+.4f} (CI [{max_lo:+.4f}, {max_hi:+.4f}], "
      f"{'excludes' if max_ci_excl0 else 'includes'} 0). "
      f"Signs: GGI positive {int(np.sum(ggi_a>0))}/6, Maximin positive {int(np.sum(max_a>0))}/6.")
    L()
    L("**When welfare changes, what moves?** Compare U_min (floor), utility Gini, "
      "C_mean / burden range / C95 (burden and tail), completion/collision (competence), "
      "and the high-burden merge-order diagnostic. No condition is labelled simply 'fairer' "
      "from a single metric.")
    L()
    L("Pooled 12-seed numbers are in `pooled12_summary.csv` and are secondary: they must not "
      "hide whether the original pattern reproduced in these six new seeds.")
    L()

    (BUNDLE_ROOT / "new6_replication_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_DIR / "new6_replication_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote analysis files under {BUNDLE_ROOT} and {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
