"""DWS final re-evaluation -- Sections 5-8 (primary fairness, interaction,
task/safety descriptive contrasts). Reads only dws_final_episode_level.csv
(no trajectory data needed for these sections). Read-only, deterministic.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dws_stats_lib import holm_correction, leave_one_out, paired_bootstrap  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "outputs" / "dws_final_reevaluation_v1"
EPISODE_CSV = OUT / "dws_final_episode_level.csv"
SEEDS = ["900101", "900102", "900103", "900104", "910101", "910102",
         "920101", "920102", "920103", "920104", "920105", "920106"]
CELLS = ["cell1", "cell2", "cell3", "cell4"]


def load_seed_level() -> dict:
    rows = list(csv.DictReader(open(EPISODE_CSV, encoding="utf-8")))
    out = {}
    for cell in CELLS:
        for seed in SEEDS:
            sub = [r for r in rows if r["cell"] == cell and r["seed"] == seed]
            assert len(sub) == 256, f"{cell} {seed}: expected 256 episodes, got {len(sub)}"
            n = len(sub)
            out[(cell, seed)] = dict(
                u_min=sum(float(r["min_U"]) for r in sub) / n,
                gini=sum(float(r["gini"]) for r in sub) / n,
                mean_u=sum(float(r["mean_U"]) for r in sub) / n,
                completion=sum(int(r["completion"]) for r in sub) / n,
                collision=sum(int(r["collision"]) for r in sub) / n,
                timeout=sum(int(r["timeout"]) for r in sub) / n,
                episode_length=sum(int(r["episode_length"]) for r in sub) / n,
            )
    return out


def write_seed_level_csv(sl: dict) -> None:
    rows = []
    for cell in CELLS:
        for seed in SEEDS:
            v = sl[(cell, seed)]
            rows.append({"cell": cell, "seed": seed, **v})
    with open(OUT / "dws_final_seed_level_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    assert len(rows) == 48
    print(f"[Section 5] wrote dws_final_seed_level_metrics.csv ({len(rows)} rows)")


def contrast(sl: dict, cell_a: str, cell_b: str, metric: str) -> list[float]:
    """Y(cell_a) - Y(cell_b) per seed, in SEEDS order."""
    return [sl[(cell_a, s)][metric] - sl[(cell_b, s)][metric] for s in SEEDS]


def section6_primary(sl: dict) -> tuple[dict, dict]:
    """Returns (summary_rows, seed_effects_rows)."""
    contrasts = {
        ("U_min", "Original"): contrast(sl, "cell2", "cell1", "u_min"),
        ("U_min", "WSC"): contrast(sl, "cell4", "cell3", "u_min"),
        ("Gini", "Original"): contrast(sl, "cell2", "cell1", "gini"),
        ("Gini", "WSC"): contrast(sl, "cell4", "cell3", "gini"),
    }
    boot = {k: paired_bootstrap(v) for k, v in contrasts.items()}

    family_a = [boot[("U_min", "Original")]["raw_p"], boot[("U_min", "WSC")]["raw_p"]]
    family_b = [boot[("Gini", "Original")]["raw_p"], boot[("Gini", "WSC")]["raw_p"]]
    holm_a = holm_correction(family_a)
    holm_b = holm_correction(family_b)
    holm_map = {
        ("U_min", "Original"): holm_a[0], ("U_min", "WSC"): holm_a[1],
        ("Gini", "Original"): holm_b[0], ("Gini", "WSC"): holm_b[1],
    }

    summary_rows = []
    seed_effect_rows = []
    for (outcome, info), b in boot.items():
        summary_rows.append({
            "outcome": outcome, "contrast": f"{info} DWS effect",
            "mean_effect": b["mean_effect"], "median_effect": b["median_effect"],
            "ci_lower": b["ci_lower"], "ci_upper": b["ci_upper"],
            "raw_p": b["raw_p"], "holm_p": holm_map[(outcome, info)],
            "n_positive": b["n_positive"], "n_negative": b["n_negative"], "n_zero": b["n_zero"],
            "n_seeds": b["n_seeds"],
            "holm_family": "Family A (U_min)" if outcome == "U_min" else "Family B (Gini)",
        })
        for seed, val in zip(SEEDS, b["seed_effects"]):
            seed_effect_rows.append({"outcome": outcome, "contrast": f"{info} DWS effect", "seed": seed, "effect": val})

    with open(OUT / "dws_primary_fairness_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    with open(OUT / "dws_primary_fairness_seed_effects.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(seed_effect_rows[0].keys()))
        w.writeheader(); w.writerows(seed_effect_rows)
    print(f"[Section 6] wrote dws_primary_fairness_summary.csv ({len(summary_rows)} rows), "
          f"dws_primary_fairness_seed_effects.csv ({len(seed_effect_rows)} rows)")
    return contrasts, boot


def section7_interaction(contrasts: dict) -> None:
    rows = []
    for outcome in ("U_min", "Gini"):
        orig = contrasts[(outcome, "Original")]
        wsc = contrasts[(outcome, "WSC")]
        interaction_seed = [w - o for w, o in zip(wsc, orig)]
        b = paired_bootstrap(interaction_seed)
        loo = leave_one_out(SEEDS, interaction_seed)
        rows.append({
            "outcome": outcome, "mean_interaction": b["mean_effect"], "median_interaction": b["median_effect"],
            "ci_lower": b["ci_lower"], "ci_upper": b["ci_upper"],
            "n_positive": b["n_positive"], "n_negative": b["n_negative"], "n_zero": b["n_zero"],
            "loo_min": loo["loo_min"], "loo_min_omitted_seed": loo["loo_min_omitted_seed"],
            "loo_max": loo["loo_max"], "loo_max_omitted_seed": loo["loo_max_omitted_seed"],
            "direction_changes_on_loo": loo["direction_changes"],
            **{f"seed_{s}": v for s, v in zip(SEEDS, interaction_seed)},
        })
    with open(OUT / "dws_information_timing_interaction.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[Section 7] wrote dws_information_timing_interaction.csv ({len(rows)} rows)")


def section8_task_safety(sl: dict) -> None:
    rows = []
    for metric in ("completion", "collision", "timeout", "mean_u", "episode_length"):
        for label, (a, b) in (("Original: Cell2-Cell1", ("cell2", "cell1")), ("WSC: Cell4-Cell3", ("cell4", "cell3"))):
            eff = contrast(sl, a, b, metric)
            boot = paired_bootstrap(eff)
            rows.append({
                "metric": metric, "contrast": label,
                "mean_effect": boot["mean_effect"], "median_effect": boot["median_effect"],
                "ci_lower": boot["ci_lower"], "ci_upper": boot["ci_upper"],
                "n_positive": boot["n_positive"], "n_negative": boot["n_negative"], "n_zero": boot["n_zero"],
            })
    # absolute per-cell descriptive levels too
    abs_rows = []
    for cell in CELLS:
        vals = [sl[(cell, s)] for s in SEEDS]
        n = len(vals)
        abs_rows.append({
            "cell": cell,
            "mean_completion": sum(v["completion"] for v in vals) / n,
            "mean_collision": sum(v["collision"] for v in vals) / n,
            "mean_timeout": sum(v["timeout"] for v in vals) / n,
            "mean_mean_u": sum(v["mean_u"] for v in vals) / n,
            "mean_episode_length": sum(v["episode_length"] for v in vals) / n,
        })
    with open(OUT / "dws_task_safety_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "dws_task_safety_absolute_levels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(abs_rows[0].keys()))
        w.writeheader()
        w.writerows(abs_rows)
    print(f"[Section 8] wrote dws_task_safety_summary.csv ({len(rows)} contrast rows) "
          f"+ dws_task_safety_absolute_levels.csv ({len(abs_rows)} rows, companion file not in Section 24's list)")


def section17_primary_loo(contrasts: dict) -> list[dict]:
    rows = []
    for (outcome, info), eff in contrasts.items():
        loo = leave_one_out(SEEDS, eff)
        rows.append({"family": "primary", "metric": f"{info} DWS effect on {outcome}", **loo})
    return rows


def main() -> int:
    sl = load_seed_level()
    write_seed_level_csv(sl)
    contrasts, boot = section6_primary(sl)
    section7_interaction(contrasts)
    section8_task_safety(sl)
    loo_rows = section17_primary_loo(contrasts)

    print("\n=== HEADLINE PRIMARY RESULTS ===")
    for (outcome, info), b in boot.items():
        print(f"{info} DWS effect on {outcome}: mean={b['mean_effect']:+.4f} CI=[{b['ci_lower']:+.4f},{b['ci_upper']:+.4f}] "
              f"raw_p={b['raw_p']:.4f} n+/n-/n0={b['n_positive']}/{b['n_negative']}/{b['n_zero']}")

    return 0, loo_rows


if __name__ == "__main__":
    ret, loo_rows = main()
    import json
    (Path(__file__).resolve().parent.parent / "outputs" / "dws_final_reevaluation_v1" / "_primary_loo_rows.json").write_text(json.dumps(loo_rows), encoding="utf-8")
    raise SystemExit(ret)
