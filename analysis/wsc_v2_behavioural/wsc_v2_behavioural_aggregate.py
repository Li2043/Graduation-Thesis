"""Aggregate raw behavioural-event counts (wsc_v2_behavioural_run.py output)
into seed-level statistics and run the formal paired seed-level bootstrap
analysis. Evaluation-only; reads CSVs produced by wsc_v2_behavioural_run.py,
writes nothing back to any frozen artifact.

Bootstrap machinery reused verbatim (same citation as the outcome-level
formal report) from
F:\\正式训练_seed_replication_v1\\analysis_scripts\\pooled12\\merge_and_audit.py:
gini(), bootstrap_ci_paired(), bootstrap_p_value(), holm_correction().

Primary confirmatory behavioural metrics (pre-specified, matches
outputs/wsc_behavioural_metric_definitions.json):
    1. RY        -- welfare-responsive yielding contrast
    2. P_priority_worse -- P(worse-off vehicle receives merge priority)
    3. BC        -- burden-transfer / cooperative-sacrifice contrast
    4. GapClosure_k25 -- worst-off gap closure at the medium (25-step) horizon
GapClosure_k10 and GapClosure_k50 are reported as secondary/exploratory
(same underlying definition, different pre-specified horizon; not included
in the primary Holm family to avoid inflating it post hoc).

Holm families: one per primary metric, across {mean, ggi, maximin} (m=3
each) -- mirrors the outcome-level report's convention of one family per
outcome across the three non-reference conditions. NOT pooled across
metrics.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

SEEDS_12 = [900101, 900102, 900103, 900104, 910101, 910102,
            920101, 920102, 920103, 920104, 920105, 920106]
SENSITIVITY_EXCLUDE = 910102
CONDITIONS = ["baseline", "mean", "ggi", "maximin"]
WELFARE_CONDITIONS = ["mean", "ggi", "maximin"]
REGIMES = ["original", "wsc"]
RECOVERY_HORIZONS = [10, 25, 50]
PRIMARY_RECOVERY_K = 25
N_BOOT = 10000
BOOT_RNG_SEED = 0
ALPHA = 0.05

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def gini(values):
    n = len(values)
    total = sum(values)
    if total == 0:
        return None
    numerator = sum(abs(a - b) for a in values for b in values)
    return float(numerator / (2.0 * n * total))


def bootstrap_ci_paired(diffs: np.ndarray, n_boot: int = N_BOOT, rng_seed: int = BOOT_RNG_SEED):
    rng = np.random.default_rng(rng_seed)
    n = len(diffs)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = np.mean(diffs[idx])
    return float(np.mean(diffs)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), boot


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


def load_events(event_csv_paths: list[Path]) -> list[dict]:
    rows = []
    for p in event_csv_paths:
        with open(p, encoding="utf-8") as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def load_episode_counts(ep_csv_paths: list[Path]) -> dict[tuple[int, str, str], dict]:
    out = {}
    for p in ep_csv_paths:
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = (int(r["seed"]), r["condition"], r["regime"])
                if key in out:
                    raise ValueError(f"duplicate seed-condition-regime episode row: {key}")
                out[key] = {"n_episodes": int(r["n_episodes"]), "n_completion": int(r["n_completion"]), "n_collision": int(r["n_collision"])}
    return out


def safe_div(num, den):
    return (num / den) if den > 0 else float("nan")


def build_seed_metrics(event_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Returns (all_group_rows, primary_ALL_only_rows)."""
    by_key_group = {}
    for r in event_rows:
        key = (int(r["seed"]), r["condition"], r["regime"], r["group"])
        by_key_group[key] = r

    present_combos = {(int(r["seed"]), r["condition"], r["regime"]) for r in event_rows}
    expected = {(s, c, rg) for s in SEEDS_12 for c in CONDITIONS for rg in REGIMES}
    missing = expected - present_combos
    if missing:
        raise SystemExit(f"FATAL: missing {len(missing)} seed-condition-regime combos in event data: {sorted(missing)[:10]}...")

    all_rows = []
    for (seed, cond, regime, group), r in by_key_group.items():
        f = {k: float(v) for k, v in r.items() if k not in ("seed", "condition", "regime", "group")}
        p_yield_worse = safe_div(f["yield_worse"], f["opp_worse"])
        p_yield_better = safe_div(f["yield_better"], f["opp_better"])
        RY = safe_div(p_yield_worse, p_yield_better)
        p_priority_worse = safe_div(f["priority_to_worse_off"], f["priority_pairs"])
        rate_burden_worse = safe_div(f["burden_event_worse"], f["burden_opp_worse"])
        rate_burden_better = safe_div(f["burden_event_better"], f["burden_opp_better"])
        BC = safe_div(rate_burden_worse, rate_burden_better)
        row = {
            "seed": seed, "condition": cond, "regime": regime, "group": group,
            "opp_worse": f["opp_worse"], "opp_better": f["opp_better"],
            "yield_worse": f["yield_worse"], "yield_better": f["yield_better"],
            "P_yield_given_worse": p_yield_worse, "P_yield_given_better": p_yield_better, "RY": RY,
            "priority_pairs": f["priority_pairs"], "priority_to_worse_off": f["priority_to_worse_off"],
            "P_priority_worse": p_priority_worse,
            "burden_opp_worse": f["burden_opp_worse"], "burden_opp_better": f["burden_opp_better"],
            "burden_event_worse": f["burden_event_worse"], "burden_event_better": f["burden_event_better"],
            "rate_burden_worse": rate_burden_worse, "rate_burden_better": rate_burden_better, "BC": BC,
            "worst_off_samples": f["worst_off_samples"],
        }
        for k in RECOVERY_HORIZONS:
            n = f[f"recovery_n_k{k}"]
            row[f"recovery_n_k{k}"] = n
            row[f"Recovery_k{k}"] = safe_div(f[f"recovery_sum_k{k}"], n)
            row[f"GapClosure_k{k}"] = safe_div(f[f"gapclosure_sum_k{k}"], n)
        all_rows.append(row)

    primary_rows = [r for r in all_rows if r["group"] == "ALL"]
    assert len(primary_rows) == len(SEEDS_12) * len(CONDITIONS) * len(REGIMES), \
        f"expected {len(SEEDS_12)*len(CONDITIONS)*len(REGIMES)} ALL-group rows, got {len(primary_rows)}"
    return all_rows, primary_rows


def wide_by_regime(primary_rows: list[dict]) -> dict[tuple[int, str], dict]:
    out = {}
    for r in primary_rows:
        key = (r["seed"], r["condition"])
        out.setdefault(key, {})[r["regime"]] = r
    return out


PRIMARY_METRICS = ["RY", "P_priority_worse", "BC", f"GapClosure_k{PRIMARY_RECOVERY_K}"]
SECONDARY_METRICS = [f"GapClosure_k{k}" for k in RECOVERY_HORIZONS if k != PRIMARY_RECOVERY_K] + \
                     [f"Recovery_k{k}" for k in RECOVERY_HORIZONS]


def main() -> int:
    event_paths = sorted(OUT_DIR.glob("wsc_behavioural_events_*.csv"))
    ep_paths = sorted(OUT_DIR.glob("wsc_behavioural_episode_counts_*.csv"))
    if not event_paths:
        raise SystemExit("no wsc_behavioural_events_*.csv shards found -- run wsc_v2_behavioural_run.py first")
    print(f"loading {len(event_paths)} event shard(s): {[p.name for p in event_paths]}")
    event_rows = load_events(event_paths)
    episode_counts = load_episode_counts(ep_paths)

    all_rows, primary_rows = build_seed_metrics(event_rows)

    # non-finite guard on primary metrics only (secondary/group metrics may
    # legitimately be NaN under sparse events -- reported, not silently dropped)
    n_nonfinite_primary = 0
    for r in primary_rows:
        for m in PRIMARY_METRICS:
            if not math.isfinite(r[m]):
                n_nonfinite_primary += 1
    print(f"non-finite primary-metric cells (sparse-event NaNs, reported not dropped): {n_nonfinite_primary}")

    seed_summary_csv = OUT_DIR / "wsc_behavioural_seed_summary.csv"
    with open(seed_summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(primary_rows[0].keys()))
        w.writeheader(); w.writerows(primary_rows)
    print(f"wrote {seed_summary_csv} ({len(primary_rows)} rows)")

    group_csv = OUT_DIR / "wsc_behavioural_group_analysis.csv"
    group_rows = [r for r in all_rows if r["group"] != "ALL"]
    with open(group_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(group_rows[0].keys()))
        w.writeheader(); w.writerows(group_rows)
    print(f"wrote {group_csv} ({len(group_rows)} rows)")

    wide = wide_by_regime(primary_rows)

    # ---------------- primary effects: paired WSC-Original delta per condition ----------------
    primary_effects_rows = []
    boot_results = {"primary_effects": {}, "interactions": {}, "leave_one_out": {}, "event_counts": {}}
    for metric in PRIMARY_METRICS + SECONDARY_METRICS:
        for cond in CONDITIONS:
            pairs = []
            for seed in SEEDS_12:
                o = wide[(seed, cond)]["original"][metric]
                w_ = wide[(seed, cond)]["wsc"][metric]
                pairs.append((seed, o, w_))
            finite_pairs = [(s, o, w_) for s, o, w_ in pairs if math.isfinite(o) and math.isfinite(w_)]
            n_used = len(finite_pairs)
            is_primary = metric in PRIMARY_METRICS
            entry = {"metric": metric, "condition": cond, "n_seeds_total": len(pairs), "n_seeds_finite": n_used,
                     "is_primary": is_primary}
            if n_used >= 3:
                diffs = np.array([w_ - o for _, o, w_ in finite_pairs])
                est, lo, hi, boot = bootstrap_ci_paired(diffs)
                p = bootstrap_p_value(boot)
                entry.update({"mean_delta_wsc_minus_original": est, "median_delta": float(np.median(diffs)),
                              "sd": float(np.std(diffs, ddof=1)) if n_used > 1 else float("nan"),
                              "n_positive": int(np.sum(diffs > 0)), "n_negative": int(np.sum(diffs < 0)),
                              "CI95_low": lo, "CI95_high": hi, "raw_p_bootstrap": p})
            else:
                entry.update({"mean_delta_wsc_minus_original": float("nan"), "note": "insufficient finite seeds (<3) -- underpowered/descriptive only"})
            primary_effects_rows.append(entry)
            boot_results["primary_effects"][f"{metric}|{cond}"] = entry

    effects_csv = OUT_DIR / "wsc_behavioural_primary_effects.csv"
    fieldnames = sorted({k for r in primary_effects_rows for k in r.keys()})
    for r in primary_effects_rows:
        for k in fieldnames:
            r.setdefault(k, "")
    with open(effects_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(primary_effects_rows)
    print(f"wrote {effects_csv}")

    # ---------------- interactions: primary metrics only, Holm family per metric across {mean,ggi,maximin} ----------------
    interaction_rows = []
    for metric in PRIMARY_METRICS:
        for label, seed_set in (("primary_n12", SEEDS_12), ("sensitivity_n11_excl_910102", [s for s in SEEDS_12 if s != SENSITIVITY_EXCLUDE])):
            raw_p, est_ci, n_used_map = {}, {}, {}
            valid = True
            per_cond_vals = {}
            for cond in WELFARE_CONDITIONS:
                vals_list = []
                for seed in seed_set:
                    bo, bw = wide[(seed, "baseline")]["original"][metric], wide[(seed, "baseline")]["wsc"][metric]
                    co, cw = wide[(seed, cond)]["original"][metric], wide[(seed, cond)]["wsc"][metric]
                    if all(math.isfinite(x) for x in (bo, bw, co, cw)):
                        vals_list.append((cw - bw) - (co - bo))
                per_cond_vals[cond] = vals_list
                n_used_map[cond] = len(vals_list)
                if len(vals_list) < 3:
                    valid = False
            if not valid:
                for cond in WELFARE_CONDITIONS:
                    interaction_rows.append({"metric": metric, "analysis": label, "condition": cond,
                                              "n_used": n_used_map[cond], "note": "insufficient finite seeds for interaction -- descriptive only"})
                continue
            for cond in WELFARE_CONDITIONS:
                arr = np.array(per_cond_vals[cond])
                est, lo, hi, boot = bootstrap_ci_paired(arr)
                p = bootstrap_p_value(boot)
                raw_p[cond] = p
                est_ci[cond] = (est, lo, hi, arr)
            pvals = [raw_p[c] for c in WELFARE_CONDITIONS]
            adj = holm_correction(pvals)
            for cond, p, a in zip(WELFARE_CONDITIONS, pvals, adj):
                est, lo, hi, arr = est_ci[cond]
                interaction_rows.append({
                    "metric": metric, "analysis": label, "condition": cond, "n_used": len(arr),
                    "mean": est, "median": float(np.median(arr)), "sd": float(np.std(arr, ddof=1)),
                    "n_positive": int(np.sum(arr > 0)), "n_negative": int(np.sum(arr < 0)),
                    "CI95_low": lo, "CI95_high": hi, "raw_p_bootstrap": p, "holm_adjusted_p": a,
                    "reject_at_alpha05": bool(a < ALPHA),
                })
    inter_csv = OUT_DIR / "wsc_behavioural_interactions.csv"
    fieldnames = sorted({k for r in interaction_rows for k in r.keys()})
    for r in interaction_rows:
        for k in fieldnames:
            r.setdefault(k, "")
    with open(inter_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(interaction_rows)
    print(f"wrote {inter_csv}")
    boot_results["interactions_table"] = interaction_rows

    # ---------------- leave-one-seed-out (primary metrics, WSC-Original delta) ----------------
    loo_rows = []
    for metric in PRIMARY_METRICS:
        for cond in CONDITIONS:
            full_vals = {}
            for seed in SEEDS_12:
                o = wide[(seed, cond)]["original"][metric]
                w_ = wide[(seed, cond)]["wsc"][metric]
                if math.isfinite(o) and math.isfinite(w_):
                    full_vals[seed] = w_ - o
            if len(full_vals) < 3:
                continue
            full_est = float(np.mean(list(full_vals.values())))
            for excl in list(full_vals.keys()):
                remaining = [v for s, v in full_vals.items() if s != excl]
                loo_est = float(np.mean(remaining))
                loo_rows.append({
                    "metric": metric, "condition": cond, "excluded_seed": excl,
                    "leave_one_out_estimate": loo_est, "full_estimate": full_est,
                    "shift_from_full": loo_est - full_est,
                    "is_seed_910102": excl == 910102, "is_seed_920102": excl == 920102,
                })
    if loo_rows:
        loo_csv = OUT_DIR / "wsc_behavioural_leave_one_seed_out.csv"
        with open(loo_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(loo_rows[0].keys()))
            w.writeheader(); w.writerows(loo_rows)
        print(f"wrote {loo_csv}")

    # ---------------- event-count sufficiency ----------------
    event_count_rows = []
    for seed in SEEDS_12:
        for cond in CONDITIONS:
            for regime in REGIMES:
                r = wide[(seed, cond)][regime]
                event_count_rows.append({
                    "seed": seed, "condition": cond, "regime": regime,
                    "opp_worse": r["opp_worse"], "opp_better": r["opp_better"],
                    "priority_pairs": r["priority_pairs"],
                    "burden_opp_worse": r["burden_opp_worse"], "burden_opp_better": r["burden_opp_better"],
                    **{f"recovery_n_k{k}": r[f"recovery_n_k{k}"] for k in RECOVERY_HORIZONS},
                })
    boot_results["event_counts"] = event_count_rows

    provenance = {
        "seeds_12": SEEDS_12, "conditions": CONDITIONS, "regimes": REGIMES,
        "primary_metrics": PRIMARY_METRICS, "secondary_metrics": SECONDARY_METRICS,
        "recovery_horizons": RECOVERY_HORIZONS, "primary_recovery_k": PRIMARY_RECOVERY_K,
        "bootstrap": {"n_boot": N_BOOT, "rng_seed": BOOT_RNG_SEED,
                      "source": str(Path(r"F:\正式训练_seed_replication_v1\analysis_scripts\pooled12\merge_and_audit.py"))},
        "holm_family_definition": "one family per primary metric, across {mean, ggi, maximin} (m=3), NOT pooled across metrics",
        "event_shards_used": [str(p) for p in event_paths],
        "episode_counts_shards_used": [str(p) for p in ep_paths],
        "episode_counts_by_combo": {f"{s}|{c}|{r}": v for (s, c, r), v in episode_counts.items()},
    }
    prov_path = OUT_DIR / "wsc_behavioural_data_provenance.json"
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, default=str)
    print(f"wrote {prov_path}")

    boot_json_path = OUT_DIR / "wsc_behavioural_bootstrap_results.json"
    with open(boot_json_path, "w", encoding="utf-8") as f:
        json.dump(boot_results, f, indent=2, default=str)
    print(f"wrote {boot_json_path}")

    print("\n=== HEADLINE: primary effects (n=12, WSC - Original, ALL group) ===")
    for r in primary_effects_rows:
        if r["metric"] in PRIMARY_METRICS and r.get("mean_delta_wsc_minus_original") not in ("", None):
            try:
                md = float(r["mean_delta_wsc_minus_original"])
                print(f"  {r['metric']:20} {r['condition']:8} delta={md:+.4f} "
                      f"CI=[{r.get('CI95_low','')},{r.get('CI95_high','')}] p={r.get('raw_p_bootstrap','')}")
            except (TypeError, ValueError):
                pass

    print("\n=== HEADLINE: interactions (n=12 primary) ===")
    for r in interaction_rows:
        if r["analysis"] == "primary_n12" and "mean" in r and r["mean"] != "":
            print(f"  {r['metric']:20} {r['condition']:8} mean={r['mean']:+.4f} holm_p={r.get('holm_adjusted_p','')}")

    print("\nDONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
