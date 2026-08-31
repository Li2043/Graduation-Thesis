"""Whole-thesis evidence synthesis -- Section 3: build the master 10-condition
seed-level evidence table. Read-only: aggregates existing episode-level and
seed-level CSVs from their ORIGINAL analysis pipelines (RQ1/RQ2 pooled12 +
taskonly, WSC v2 formal + behavioural, and this session's own DWS final
re-evaluation). Does not recompute any metric definition from memory --
every column traces to a specific existing source file.

Genuine gaps (left blank, not fabricated), and why:
  - "braking burden" (the thesis's Table 5.7 / Figure 5.7 acceleration-based
    measure) was not found as a stored per-seed CSV column in any located
    source (F:\\正式训练*, F:\\正式训练_seed_replication_v1\\analysis_scripts\\*) --
    it appears to be computed inline inside a plotting/analysis script not
    identified as a reusable per-seed output. Left blank for ALL conditions
    rather than approximated from a different quantity (hard-brake event
    counts measure something related but not the same thing).
  - DWS diagnostics (Phi, DeltaPhi, event rates) exist ONLY for the four
    Maximin cells (this thesis's own DWS follow-up scope) -- never computed
    for Baseline/Mean/GGI under either regime, because the DWS study never
    ran or evaluated those conditions. Left blank for those 8 rows.
  - Merge-priority allocation is present but was flagged in the WSC formal
    thesis text itself (05_results.md Sec 5.6.4) as too sparse for Baseline
    and Mean specifically -- kept as whatever wsc_behavioural_seed_summary.csv
    actually contains (may be NaN/undefined for those cells), not forced.
"""
from __future__ import annotations

import csv
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "outputs" / "whole_thesis_evidence_synthesis_v1"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = ["900101", "900102", "900103", "900104", "910101", "910102",
         "920101", "920102", "920103", "920104", "920105", "920106"]

_REPO_ROOT = Path(__file__).resolve().parent.parent
TASKONLY_CSV = _REPO_ROOT / "analysis" / "data" / "taskonly_evaluation_merged.csv"
POOLED12_CSV = _REPO_ROOT / "analysis" / "pooled12" / "outputs" / "pooled12_welfare_evaluation_merged.csv"
WSC_ALL12_CSV = _REPO_ROOT / "analysis" / "ch5_baseline" / "outputs" / "wsc_interim_v2" / "wsc_interim_v2_evaluation_all12.csv"
WSC_BEHAV_SUMMARY = _REPO_ROOT / "analysis" / "wsc_v2_behavioural" / "outputs" / "wsc_behavioural_seed_summary.csv"
DWS_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "dws_final_reevaluation_v1"

FIELDS = [
    "condition_label", "information_regime", "welfare_objective", "dws", "seed",
    "u_min", "gini", "mean_u", "completion", "collision", "timeout",
    "fast_u", "slow_u", "ramp_u", "mainline_u",
    "worst_off_role_fast_share", "worst_off_role_ramp_share",
    "mobility_burden_mean",
    "braking_burden_mean",
    "ry", "priority_share_worse_off", "burden_rate_worse", "recovery_gapclosure_k25",
    "phi_mean", "delta_phi_mean", "frac_positive_event", "frac_negative_event", "frac_neutral_event",
    "source_note",
]


def episode_rows_for(path: Path, condition_filter: str | None = None) -> list[dict]:
    """Filters to bank=='H1' -- the thesis's formal held-out evaluation bank
    throughout (05_results.md Sec 5.3.2 etc). taskonly_evaluation_merged.csv
    and pooled12_welfare_evaluation_merged.csv both contain H0+H1 mixed
    (512 rows/seed); including H0 silently would double-count episodes from
    a different, non-formal bank and does not match the thesis's own
    reported Table 5.2/5.5 numbers (verified: this filter is required to
    reproduce them, see the whole-thesis synthesis report's conflict note)."""
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    rows = [r for r in rows if r["bank"] == "H1"]
    if condition_filter:
        rows = [r for r in rows if r["condition"] == condition_filter]
    return rows


ROLE_KEY = {"V0": None, "V1": None, "V2": None, "V3": None}  # placeholder, roles read per-row


def _mean_of(sub: list[dict], key: str) -> float | None:
    """Skip blank/undefined values (e.g. Gini is undefined, not zero, for a
    degenerate all-zero-utility episode -- matches the thesis's own Section
    5.7.3 convention of reporting 'undefined' rather than defaulting to 0)."""
    vals = [float(r[key]) for r in sub if r[key] not in ("", "nan", "None")]
    return (sum(vals) / len(vals)) if vals else None


def seed_level_from_episode_rows(rows: list[dict], seed: str) -> dict:
    sub = [r for r in rows if r["seed"] == seed]
    n = len(sub)
    if n == 0:
        return {}
    u_min = _mean_of(sub, "min_U")
    n_gini_defined = sum(1 for r in sub if r["gini"] not in ("", "nan", "None"))
    gini = _mean_of(sub, "gini")
    if n_gini_defined < n:
        print(f"  [note] seed={seed}: Gini undefined (degenerate all-zero-utility episode) for "
              f"{n - n_gini_defined}/{n} episodes -- averaged over the remaining {n_gini_defined}, not defaulted to 0")
    mean_u = _mean_of(sub, "mean_U")
    completion = sum(int(r["completion"]) for r in sub) / n
    collision = sum(int(r["collision"]) for r in sub) / n
    timeout = sum(int(r["timeout"]) for r in sub) / n

    fast_u, slow_u, ramp_u, mainline_u = [], [], [], []
    mobility_burden = []
    worst_off_fast, worst_off_ramp, worst_off_total = 0, 0, 0
    for r in sub:
        for vid in ("V0", "V1", "V2", "V3"):
            role = r[f"role_{vid}"]; spd = r[f"speed_class_{vid}"]
            u = float(r[f"U_{vid}"]); c = float(r[f"C_{vid}"])
            mobility_burden.append(c)
            if spd == "fast":
                fast_u.append(u)
            elif spd == "slow":
                slow_u.append(u)
            if role == "ramp":
                ramp_u.append(u)
            elif role == "mainline":
                mainline_u.append(u)
        worst_vid = r["min_U_vehicle"]
        if worst_vid:
            worst_off_total += 1
            if r["min_U_speed_class"] == "fast":
                worst_off_fast += 1
            if r["min_U_role"] == "ramp":
                worst_off_ramp += 1

    return dict(
        u_min=u_min, gini=gini, mean_u=mean_u, completion=completion, collision=collision, timeout=timeout,
        fast_u=sum(fast_u) / len(fast_u) if fast_u else None,
        slow_u=sum(slow_u) / len(slow_u) if slow_u else None,
        ramp_u=sum(ramp_u) / len(ramp_u) if ramp_u else None,
        mainline_u=sum(mainline_u) / len(mainline_u) if mainline_u else None,
        worst_off_role_fast_share=(worst_off_fast / worst_off_total) if worst_off_total else None,
        worst_off_role_ramp_share=(worst_off_ramp / worst_off_total) if worst_off_total else None,
        mobility_burden_mean=sum(mobility_burden) / len(mobility_burden) if mobility_burden else None,
    )


def behav_lookup(condition: str, regime: str) -> dict:
    rows = list(csv.DictReader(open(WSC_BEHAV_SUMMARY, encoding="utf-8")))
    out = {}
    for r in rows:
        if r["condition"] == condition and r["regime"] == regime and r["group"] == "ALL":
            out[r["seed"]] = dict(
                ry=r["RY"] if r["RY"] not in ("", "nan", "None") else None,
                priority_share_worse_off=r["P_priority_worse"] if r["P_priority_worse"] not in ("", "nan", "None") else None,
                burden_rate_worse=r["rate_burden_worse"] if r["rate_burden_worse"] not in ("", "nan", "None") else None,
                recovery_gapclosure_k25=r["GapClosure_k25"] if r["GapClosure_k25"] not in ("", "nan", "None") else None,
            )
    return out


def dws_signal_lookup(cell: str) -> dict:
    rows = list(csv.DictReader(open(DWS_ROOT / "dws_signal_diagnostics_seed_level.csv", encoding="utf-8")))
    return {r["seed"]: r for r in rows if r["cell"] == cell}


def dws_seed_level_lookup(cell: str) -> dict:
    rows = list(csv.DictReader(open(DWS_ROOT / "dws_final_seed_level_metrics.csv", encoding="utf-8")))
    return {r["seed"]: r for r in rows if r["cell"] == cell}


def dws_mech_lookup(cell: str) -> dict:
    rows = list(csv.DictReader(open(DWS_ROOT / "dws_behavioural_mechanisms_seed_level.csv", encoding="utf-8")))
    return {r["seed"]: r for r in rows if r["cell"] == cell}


def dws_class_lookup(cell: str) -> dict:
    """dws_class_distribution_summary.csv has one row per (cell,seed,role,speed_class);
    pivot to seed -> {fast_u, slow_u, ramp_u, mainline_u, worst_off_fast_share, worst_off_ramp_share}."""
    rows = list(csv.DictReader(open(DWS_ROOT / "dws_class_distribution_summary.csv", encoding="utf-8")))
    rows = [r for r in rows if r["cell"] == cell]
    out = {}
    for seed in SEEDS:
        srows = [r for r in rows if r["seed"] == seed]
        if not srows:
            continue
        def u_for(role=None, spd=None):
            vals = [float(r["mean_U"]) for r in srows if r["mean_U"] not in ("", "None")
                     and (role is None or r["role"] == role) and (spd is None or r["speed_class"] == spd)]
            return sum(vals) / len(vals) if vals else None
        def worst_off_share(role=None, spd=None):
            vals = [float(r["worst_off_share"]) for r in srows if r["worst_off_share"] not in ("", "None")
                     and (role is None or r["role"] == role) and (spd is None or r["speed_class"] == spd)]
            return sum(vals) if vals else None
        out[seed] = dict(
            fast_u=u_for(spd="fast"), slow_u=u_for(spd="slow"),
            ramp_u=u_for(role="ramp"), mainline_u=u_for(role="mainline"),
            worst_off_role_fast_share=worst_off_share(spd="fast"),
            worst_off_role_ramp_share=worst_off_share(role="ramp"),
            mobility_burden_mean=(lambda vs: sum(vs) / len(vs) if vs else None)(
                [float(r["mean_C"]) for r in srows if r["mean_C"] not in ("", "None")]),
        )
    return out


def main() -> int:
    all_rows: list[dict] = []

    # ---- Original Baseline/Mean/GGI/Maximin, WSC Baseline/Mean/GGI/Maximin ----
    orig_sources = {
        "baseline": episode_rows_for(TASKONLY_CSV),
        "mean": episode_rows_for(POOLED12_CSV, "mean"),
        "ggi": episode_rows_for(POOLED12_CSV, "ggi"),
        "maximin": episode_rows_for(POOLED12_CSV, "maximin"),
    }
    wsc_source = episode_rows_for(WSC_ALL12_CSV)

    label_map = {"baseline": "Baseline", "mean": "Mean", "ggi": "GGI", "maximin": "Maximin"}

    for cond, rows in orig_sources.items():
        behav = behav_lookup(cond, "original")
        for seed in SEEDS:
            sl = seed_level_from_episode_rows(rows, seed)
            if not sl:
                continue
            b = behav.get(seed, {})
            all_rows.append({
                "condition_label": label_map[cond], "information_regime": "Original",
                "welfare_objective": cond, "dws": "No", "seed": seed,
                **sl, "braking_burden_mean": None,
                **{k: b.get(k) for k in ("ry", "priority_share_worse_off", "burden_rate_worse", "recovery_gapclosure_k25")},
                "phi_mean": None, "delta_phi_mean": None,
                "frac_positive_event": None, "frac_negative_event": None, "frac_neutral_event": None,
                "source_note": f"episode-level: {TASKONLY_CSV.name if cond=='baseline' else POOLED12_CSV.name}; behavioural: {WSC_BEHAV_SUMMARY.name}",
            })

    for cond in ("baseline", "mean", "ggi", "maximin"):
        rows = episode_rows_for(WSC_ALL12_CSV, cond)
        behav = behav_lookup(cond, "wsc")
        for seed in SEEDS:
            sl = seed_level_from_episode_rows(rows, seed)
            if not sl:
                continue
            b = behav.get(seed, {})
            all_rows.append({
                "condition_label": f"{label_map[cond]}+WSC", "information_regime": "WSC",
                "welfare_objective": cond, "dws": "No", "seed": seed,
                **sl, "braking_burden_mean": None,
                **{k: b.get(k) for k in ("ry", "priority_share_worse_off", "burden_rate_worse", "recovery_gapclosure_k25")},
                "phi_mean": None, "delta_phi_mean": None,
                "frac_positive_event": None, "frac_negative_event": None, "frac_neutral_event": None,
                "source_note": f"episode-level: {WSC_ALL12_CSV.name}; behavioural: {WSC_BEHAV_SUMMARY.name}",
            })

    # ---- Maximin (re-derived, cell1/cell3) and Maximin+DWS/Maximin+WSC+DWS (cell2/cell4) from this session's own DWS re-evaluation ----
    # NOTE: cell1/cell3 here are the SAME scientific condition as the "maximin"/"maximin+WSC" rows
    # above but from the FRESH DWS re-evaluation pipeline (different bundle, byte-identical utility.py,
    # cross-checked exact-match against the old Cell 4 CSV in DWS Stage 1) -- included as separate
    # condition_label values ("Maximin (DWS-study re-eval)") rather than overwriting the canonical
    # RQ2/WSC rows above, so a reader can see both without ambiguity about which is the thesis's
    # primary RQ2/WSC source (the pooled12/wsc_v2_formal rows above ARE primary; these are the
    # DWS-study's own baseline reference points for its own contrasts).
    dws_cells = {
        "cell1": ("Maximin (DWS-study re-eval)", "Original", "No"),
        "cell2": ("Maximin+DWS", "Original", "Yes"),
        "cell3": ("Maximin+WSC (DWS-study re-eval)", "WSC", "No"),
        "cell4": ("Maximin+WSC+DWS", "WSC", "Yes"),
    }
    for cell, (label, regime, dws) in dws_cells.items():
        sl_lookup = dws_seed_level_lookup(cell)
        signal_lookup = dws_signal_lookup(cell)
        mech_lookup = dws_mech_lookup(cell)
        class_lookup = dws_class_lookup(cell)
        for seed in SEEDS:
            sl = sl_lookup.get(seed)
            if not sl:
                continue
            sig = signal_lookup.get(seed, {})
            mech = mech_lookup.get(seed, {})
            cls = class_lookup.get(seed, {})
            all_rows.append({
                "condition_label": label, "information_regime": regime,
                "welfare_objective": "maximin", "dws": dws, "seed": seed,
                "u_min": sl["u_min"], "gini": sl["gini"], "mean_u": sl["mean_u"],
                "completion": sl["completion"], "collision": sl["collision"], "timeout": sl["timeout"],
                "fast_u": cls.get("fast_u"), "slow_u": cls.get("slow_u"),
                "ramp_u": cls.get("ramp_u"), "mainline_u": cls.get("mainline_u"),
                "worst_off_role_fast_share": cls.get("worst_off_role_fast_share"),
                "worst_off_role_ramp_share": cls.get("worst_off_role_ramp_share"),
                "mobility_burden_mean": cls.get("mobility_burden_mean"), "braking_burden_mean": None,
                "ry": mech.get("RY"), "priority_share_worse_off": mech.get("priority_share_worse_off"),
                "burden_rate_worse": mech.get("burden_share_worse_opp"),
                "recovery_gapclosure_k25": mech.get("recovery_gapclosure_k25"),
                "phi_mean": None, "delta_phi_mean": None,
                "frac_positive_event": sig.get("frac_positive"), "frac_negative_event": sig.get("frac_negative"),
                "frac_neutral_event": sig.get("frac_neutral"),
                "source_note": "dws_final_reevaluation_v1 (this session's own trajectory-rich re-evaluation, Stage 2)",
            })

    with open(OUT / "whole_thesis_seed_level_evidence.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {len(all_rows)} rows, {len(FIELDS)} columns -> whole_thesis_seed_level_evidence.csv")

    # completeness accounting
    from collections import defaultdict
    populated = defaultdict(lambda: defaultdict(int))
    total = defaultdict(int)
    for r in all_rows:
        cond = r["condition_label"]
        total[cond] += 1
        for f in FIELDS:
            if f in ("condition_label", "information_regime", "welfare_objective", "dws", "seed", "source_note"):
                continue
            if r[f] not in (None, "", "nan"):
                populated[cond][f] += 1
    print("\n=== populated-cell accounting (n seeds with a value / 12) ===")
    for cond in total:
        missing = [f for f in FIELDS if f not in ("condition_label", "information_regime", "welfare_objective", "dws", "seed", "source_note")
                   and populated[cond].get(f, 0) == 0]
        print(f"{cond}: {total[cond]} seed-rows; fully-empty columns: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
