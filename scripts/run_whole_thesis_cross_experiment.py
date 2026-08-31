"""Whole-thesis synthesis -- Sections 3/6/12: cross-experiment analyses
from the already-built seed-level evidence table plus official WSC/DWS
summary CSVs. Inferential unit = training seed. Does not retrain, does
not overwrite prior analysis. Writes only under
outputs/whole_thesis_evidence_synthesis_v1/.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dws_stats_lib import holm_correction, leave_one_out, paired_bootstrap

OUT = Path(__file__).resolve().parent.parent / "outputs" / "whole_thesis_evidence_synthesis_v1"
EVID = OUT / "whole_thesis_seed_level_evidence.csv"
WSC_FAIR = Path(__file__).resolve().parent.parent / "analysis" / "wsc_v2_formal" / "outputs" / "wsc_v2_formal_fairness_summary.csv"
DWS_PRIMARY = Path(__file__).resolve().parent.parent / "outputs" / "dws_final_reevaluation_v1" / "dws_primary_fairness_summary.csv"
DWS_INTER = Path(__file__).resolve().parent.parent / "outputs" / "dws_final_reevaluation_v1" / "dws_information_timing_interaction.csv"
DWS_TASK = Path(__file__).resolve().parent.parent / "outputs" / "dws_final_reevaluation_v1" / "dws_task_safety_summary.csv"

SEEDS = ["900101", "900102", "900103", "900104", "910101", "910102",
         "920101", "920102", "920103", "920104", "920105", "920106"]

CANONICAL = {
    "Baseline": ("Original", "baseline", "No"),
    "Mean": ("Original", "mean", "No"),
    "GGI": ("Original", "ggi", "No"),
    "Maximin": ("Original", "maximin", "No"),
    "Baseline+WSC": ("WSC", "baseline", "No"),
    "Mean+WSC": ("WSC", "mean", "No"),
    "GGI+WSC": ("WSC", "ggi", "No"),
    "Maximin+WSC": ("WSC", "maximin", "No"),
    "Maximin+DWS": ("Original", "maximin", "Yes"),
    "Maximin+WSC+DWS": ("WSC", "maximin", "Yes"),
}

ORDER = ["Baseline", "Mean", "GGI", "Maximin"]


def load_evidence() -> list[dict]:
    rows = list(csv.DictReader(open(EVID, encoding="utf-8")))
    return [r for r in rows if r["condition_label"] in CANONICAL]


def pivot(rows: list[dict], metric: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        v = r[metric]
        if v in ("", "None", "nan"):
            continue
        out[r["condition_label"]][r["seed"]] = float(v)
    return out


def contrast_table(rows: list[dict], pairs: list[tuple[str, str, str]], metrics: list[str]) -> list[dict]:
    out = []
    for a, b, name in pairs:
        for metric in metrics:
            pa, pb = pivot(rows, metric)[a], pivot(rows, metric)[b]
            common = [s for s in SEEDS if s in pa and s in pb]
            effects = [pa[s] - pb[s] for s in common]
            boot = paired_bootstrap(effects)
            loo = leave_one_out(common, effects)
            fav = "higher" if metric != "gini" else "lower"
            n_fav = sum(1 for e in effects if (e > 0 if metric != "gini" else e < 0))
            out.append({
                "contrast": name, "metric": metric, "n_seeds": len(common),
                "mean_effect": boot["mean_effect"], "median_effect": boot["median_effect"],
                "ci_lower": boot["ci_lower"], "ci_upper": boot["ci_upper"],
                "raw_p": boot["raw_p"],
                "n_favourable": n_fav, "n_unfavourable": len(common) - n_fav - boot["n_zero"],
                "n_zero": boot["n_zero"],
                "favourable_direction": fav,
                "loo_min": loo["loo_min"], "loo_max": loo["loo_max"],
                "loo_min_omitted": loo["loo_min_omitted_seed"],
                "loo_max_omitted": loo["loo_max_omitted_seed"],
                "direction_changes_loo": int(loo["direction_changes"]),
                "seed_effects": "|".join(f"{s}:{e:.6f}" for s, e in zip(common, effects)),
            })
    return out


def objective_ordering(rows: list[dict], regime: str) -> list[dict]:
    """Section 6.1: Baseline -> Mean -> GGI -> Maximin rank checks.
    Not treated as an interval scale. Reports across-seed mean order and
    within-seed rank agreement with the hypothesised fairness-improving
    order (U_min: Baseline < Mean < GGI < Maximin; Gini reversed).
    """
    labels = {
        "Original": ["Baseline", "Mean", "GGI", "Maximin"],
        "WSC": ["Baseline+WSC", "Mean+WSC", "GGI+WSC", "Maximin+WSC"],
    }[regime]
    hyp_umin = labels  # increasing U_min
    hyp_gini = list(reversed(labels))  # decreasing Gini
    out = []
    for metric, hyp in (("u_min", hyp_umin), ("gini", hyp_gini),
                        ("completion", hyp_umin), ("collision", list(reversed(labels))),
                        ("mean_u", hyp_umin)):
        pv = pivot(rows, metric)
        means = {lab: float(np.mean([pv[lab][s] for s in SEEDS if s in pv[lab]])) for lab in labels}
        mean_order = sorted(labels, key=lambda x: means[x], reverse=(metric not in ("gini", "collision")))
        n_agree = 0
        n_total = 0
        n_full_mono = 0
        for s in SEEDS:
            if any(s not in pv[lab] for lab in labels):
                continue
            n_total += 1
            vals = [pv[lab][s] for lab in labels]
            # pairwise adjacent comparisons vs hypothesised adjacent order
            adj_ok = 0
            for i in range(3):
                a, b = labels[i], labels[i + 1]
                if metric in ("gini", "collision"):
                    adj_ok += int(pv[b][s] <= pv[a][s])
                else:
                    adj_ok += int(pv[b][s] >= pv[a][s])
            if adj_ok == 3:
                n_full_mono += 1
            n_agree += adj_ok
        out.append({
            "regime": regime, "metric": metric,
            "across_seed_mean_order": " > ".join(mean_order),
            "hypothesised_fairness_order": " > ".join(hyp if metric not in ("gini", "collision") else hyp),
            "mean_Baseline": means[labels[0]], "mean_Mean": means[labels[1]],
            "mean_GGI": means[labels[2]], "mean_Maximin": means[labels[3]],
            "n_seeds": n_total,
            "n_seeds_full_monotonic": n_full_mono,
            "adjacent_pair_agreements": n_agree,
            "adjacent_pair_total": n_total * 3,
            "adjacent_agreement_rate": (n_agree / (n_total * 3)) if n_total else None,
        })
    return out


def task_fairness_assoc(rows: list[dict]) -> list[dict]:
    """Section 6.4: descriptive associations only. One row per seed-condition
    cell among the 10 canonical conditions. Spearman-like rank correlation
    computed as Pearson on ranks (no causal language).
    """
    cells = []
    for r in rows:
        if r["condition_label"] not in CANONICAL:
            continue
        if r["u_min"] in ("", "None"):
            continue
        cells.append({
            "condition": r["condition_label"], "seed": r["seed"],
            "u_min": float(r["u_min"]), "gini": float(r["gini"]) if r["gini"] not in ("", "None") else None,
            "completion": float(r["completion"]), "collision": float(r["collision"]),
            "timeout": float(r["timeout"]), "mean_u": float(r["mean_u"]),
        })

    def _rank_corr(xs, ys):
        a = np.asarray(xs, dtype=float)
        b = np.asarray(ys, dtype=float)
        ra = a.argsort().argsort().astype(float)
        rb = b.argsort().argsort().astype(float)
        if ra.std() == 0 or rb.std() == 0:
            return None
        return float(np.corrcoef(ra, rb)[0, 1])

    pairs = [
        ("u_min", "completion"), ("u_min", "collision"), ("u_min", "timeout"),
        ("gini", "completion"), ("gini", "collision"),
        ("u_min", "mean_u"),
    ]
    out = []
    for x, y in pairs:
        xs, ys = [], []
        for c in cells:
            if c[x] is None or c[y] is None:
                continue
            xs.append(c[x]); ys.append(c[y])
        out.append({
            "x": x, "y": y, "n_cells": len(xs),
            "spearman_rank_corr": _rank_corr(xs, ys),
            "note": "descriptive association across seed-condition cells; not causal",
        })

    # high-completion unfair / fair counts
    high_comp = [c for c in cells if c["completion"] >= 0.90]
    unfair = [c for c in high_comp if c["gini"] is not None and c["gini"] >= 0.05]
    fair = [c for c in high_comp if c["gini"] is not None and c["gini"] < 0.02]
    out.append({
        "x": "completion>=0.90", "y": "gini>=0.05",
        "n_cells": len(high_comp),
        "spearman_rank_corr": None,
        "note": f"high-completion cells={len(high_comp)}; of these, gini>=0.05: {len(unfair)}; gini<0.02: {len(fair)} -- high-completion unfair policies exist",
    })
    return out


def seed_matrix(rows: list[dict]) -> list[dict]:
    """Section 12: seed x intervention standardized effects."""
    interventions = [
        ("Mean-Baseline", "Mean", "Baseline"),
        ("GGI-Baseline", "GGI", "Baseline"),
        ("Maximin-Baseline", "Maximin", "Baseline"),
        ("WSC-on-Baseline", "Baseline+WSC", "Baseline"),
        ("WSC-on-Mean", "Mean+WSC", "Mean"),
        ("WSC-on-GGI", "GGI+WSC", "GGI"),
        ("WSC-on-Maximin", "Maximin+WSC", "Maximin"),
        ("DWS-on-Maximin", "Maximin+DWS", "Maximin"),
        ("DWS-on-Maximin+WSC", "Maximin+WSC+DWS", "Maximin+WSC"),
    ]
    metrics = ["u_min", "gini", "completion", "collision"]
    pv = {m: pivot(rows, m) for m in metrics}
    raw = {m: defaultdict(dict) for m in metrics}
    for name, a, b in interventions:
        for m in metrics:
            for s in SEEDS:
                if s in pv[m].get(a, {}) and s in pv[m].get(b, {}):
                    raw[m][name][s] = pv[m][a][s] - pv[m][b][s]
    # standardize within intervention (across seeds)
    out = []
    for name, a, b in interventions:
        for m in metrics:
            vals = [raw[m][name][s] for s in SEEDS if s in raw[m][name]]
            mu = float(np.mean(vals)) if vals else None
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
            for s in SEEDS:
                if s not in raw[m][name]:
                    continue
                z = (raw[m][name][s] - mu) / sd if sd not in (None, 0.0) else 0.0
                out.append({
                    "seed": s, "intervention": name, "metric": m,
                    "raw_effect": raw[m][name][s],
                    "z_within_intervention": z,
                    "intervention_mean": mu, "intervention_sd": sd,
                })
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path.name}")


def evidence_inventory() -> list[dict]:
    return [
        {"source_file": str(EVID), "metric_definition": "seed-level U_min/Gini/task from existing episode CSVs (H1 only)",
         "condition_comparison": "10 canonical cells", "inferential_unit": "training seed",
         "status": "descriptive master table", "confirmatory_or": "descriptive"},
        {"source_file": r"F:\正式训练\outputs\welfare_analysis\taskonly_evaluation_merged.csv",
         "metric_definition": "thesis.study_b.utility episode_utilities / gini_coefficient",
         "condition_comparison": "Original Baseline", "inferential_unit": "training seed (after episode aggregation)",
         "status": "confirmatory RQ1/RQ2 source", "confirmatory_or": "confirmatory"},
        {"source_file": r"F:\正式训练_seed_replication_v1\analysis_scripts\pooled12\outputs\pooled12_welfare_evaluation_merged.csv",
         "metric_definition": "same utility.py", "condition_comparison": "Original Mean/GGI/Maximin vs Baseline",
         "inferential_unit": "training seed", "status": "confirmatory RQ2 source", "confirmatory_or": "confirmatory"},
        {"source_file": str(WSC_FAIR),
         "metric_definition": "I = (WSC_c - WSC_baseline) - (Orig_c - Orig_baseline)",
         "condition_comparison": "WSC x {Mean,GGI,Maximin} interactions",
         "inferential_unit": "training seed", "status": "confirmatory WSC", "confirmatory_or": "confirmatory"},
        {"source_file": str(DWS_PRIMARY),
         "metric_definition": "Cell2-Cell1 and Cell4-Cell3 on U_min/Gini; Holm within 2-test families",
         "condition_comparison": "DWS vs terminal Maximin, Original and WSC",
         "inferential_unit": "training seed", "status": "confirmatory DWS", "confirmatory_or": "confirmatory"},
        {"source_file": str(DWS_INTER),
         "metric_definition": "(C4-C3)-(C2-C1)", "condition_comparison": "DWS x WSC",
         "inferential_unit": "training seed", "status": "secondary mechanism interaction", "confirmatory_or": "secondary"},
        {"source_file": r"F:\正式训练_seed_replication_v1\analysis_scripts\wsc_v2_behavioural\outputs\wsc_behavioural_primary_effects.csv",
         "metric_definition": "RY, merge-priority, burden transfer, GapClosure_k25 as in wsc_v2_behavioural_run.py",
         "condition_comparison": "WSC-Original within condition",
         "inferential_unit": "training seed (finite-seed subset)", "status": "secondary WSC mechanisms",
         "confirmatory_or": "secondary"},
        {"source_file": r"C:\dense reward\outputs\dws_final_reevaluation_v1\dws_behavioural_mechanisms_summary.csv",
         "metric_definition": "same four mechanisms, DWS contrasts",
         "condition_comparison": "DWS vs terminal Maximin",
         "inferential_unit": "training seed", "status": "secondary DWS mechanisms",
         "confirmatory_or": "secondary"},
    ]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_evidence()
    print(f"loaded {len(rows)} canonical seed-level rows")

    pairs = [
        ("Mean", "Baseline", "Original: Mean-Baseline"),
        ("GGI", "Baseline", "Original: GGI-Baseline"),
        ("Maximin", "Baseline", "Original: Maximin-Baseline"),
        ("Mean+WSC", "Baseline+WSC", "WSC: Mean-Baseline"),
        ("GGI+WSC", "Baseline+WSC", "WSC: GGI-Baseline"),
        ("Maximin+WSC", "Baseline+WSC", "WSC: Maximin-Baseline"),
        ("Maximin+DWS", "Maximin", "Original: DWS-Maximin"),
        ("Maximin+WSC+DWS", "Maximin+WSC", "WSC: DWS-Maximin"),
        ("GGI", "Mean", "Original: GGI-Mean"),
        ("Maximin", "Mean", "Original: Maximin-Mean"),
        ("Maximin", "GGI", "Original: Maximin-GGI"),
    ]
    metrics = ["u_min", "gini", "mean_u", "completion", "collision", "timeout"]
    contrasts = contrast_table(rows, pairs, metrics)
    # Holm only for the three Original RQ2 U_min contrasts and three Gini contrasts (separate families)
    for family, mets, names in (
        ("RQ2_Umin_vs_Baseline", ["u_min"], ["Original: Mean-Baseline", "Original: GGI-Baseline", "Original: Maximin-Baseline"]),
        ("RQ2_Gini_vs_Baseline", ["gini"], ["Original: Mean-Baseline", "Original: GGI-Baseline", "Original: Maximin-Baseline"]),
    ):
        idxs = [i for i, r in enumerate(contrasts) if r["metric"] in mets and r["contrast"] in names]
        ps = [contrasts[i]["raw_p"] for i in idxs]
        adj = holm_correction(ps)
        for i, p in zip(idxs, adj):
            contrasts[i]["holm_p"] = p
            contrasts[i]["holm_family"] = family
    for r in contrasts:
        r.setdefault("holm_p", "")
        r.setdefault("holm_family", "not in a confirmatory family (descriptive/secondary)")

    write_csv(OUT / "cross_experiment_contrasts.csv", contrasts)
    write_csv(OUT / "objective_strength_ordering.csv",
              objective_ordering(rows, "Original") + objective_ordering(rows, "WSC"))
    write_csv(OUT / "task_vs_fairness_associations.csv", task_fairness_assoc(rows))
    write_csv(OUT / "seed_intervention_matrix.csv", seed_matrix(rows))
    write_csv(OUT / "evidence_inventory.csv", evidence_inventory())

    # absolute condition means for the report
    abs_rows = []
    for lab in CANONICAL:
        sub = [r for r in rows if r["condition_label"] == lab]
        if not sub:
            continue
        rec = {"condition_label": lab, "n_seeds": len(sub)}
        for m in ("u_min", "gini", "mean_u", "completion", "collision", "timeout",
                  "fast_u", "slow_u", "ramp_u", "mainline_u", "mobility_burden_mean"):
            vals = [float(r[m]) for r in sub if r[m] not in ("", "None", "nan")]
            rec[m] = float(np.mean(vals)) if vals else None
        abs_rows.append(rec)
    write_csv(OUT / "condition_absolute_means.csv", abs_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
