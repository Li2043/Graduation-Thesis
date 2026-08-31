"""DWS final re-evaluation -- Sections 10-13 (the four pre-defined
confirmatory behavioural mechanisms: welfare-responsive yielding,
merge-priority allocation, cooperative burden transfer, worst-off recovery)
plus Section 18 (outcome decomposition, needs only the episode CSV) and the
groundwork counters reused by Sections 14-16 in a follow-up script.

Definitions reused VERBATIM (constants and logic) from
F:\\正式训练_seed_replication_v1\\analysis_scripts\\wsc_v2_behavioural\\wsc_v2_behavioural_run.py,
computed here from the already-recorded per-step trajectory data (active,
x, M, action, accel, hard_brake_start, exit_step, pair_first_state) instead
of a live rollout -- same event definitions, same thresholds, different
(equivalent) data source.

Read-only, deterministic. One pass per (cell, seed) trajectory shard.
"""
from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dws_stats_lib import holm_correction, leave_one_out, paired_bootstrap  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "outputs" / "dws_final_reevaluation_v1"
TRAJ_DIR = OUT / "trajectories"
SEEDS = ["900101", "900102", "900103", "900104", "910101", "910102",
         "920101", "920102", "920103", "920104", "920105", "920106"]
CELLS = ["cell1", "cell2", "cell3", "cell4"]
VIDS = ("V0", "V1", "V2", "V3")
X_CONVERGE_START = 220.0
X_MERGE_END = 380.0
R_OBS = 50.0
DECELERATE = 2
TIE_TOL = 1e-9
RECOVERY_HORIZONS = (10, 25, 50)


def process_shard(cell: str, seed: str) -> dict:
    """One pass over one (cell,seed) shard's 256 episodes. Returns per-seed
    aggregated counters for all mechanisms that need trajectory data."""
    path = TRAJ_DIR / f"{cell}_{seed}.jsonl.gz"
    c = dict(
        opp_worse=0, opp_better=0, yield_worse=0, yield_better=0,
        priority_pairs=0, priority_to_worse_off=0,
        burden_opp_worse=0, burden_opp_better=0, burden_event_worse=0, burden_event_better=0,
        recovery_n={k: 0 for k in RECOVERY_HORIZONS}, recovery_sum={k: 0.0 for k in RECOVERY_HORIZONS},
        gapclosure_sum={k: 0.0 for k in RECOVERY_HORIZONS},
        n_episodes=0,
    )
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            ep = json.loads(line)
            c["n_episodes"] += 1
            steps = ep["steps"]
            exit_step = ep["exit_step"]
            pair_first_state = ep["pair_first_state"]

            m_history = [s["M"] for s in steps]
            active_history = [s["active"] for s in steps]
            worst_off_log: list[tuple[int, str, float, dict]] = []

            for s in steps:
                t = s["t"]
                xs = {v: x for v, x in s["x"].items() if x is not None}
                ms = s["M"]
                actions = s["action"]

                # yielding opportunities (directed i -> j)
                for i in xs:
                    if xs[i] >= X_MERGE_END:
                        continue
                    for j in xs:
                        if j == i:
                            continue
                        if abs(xs[i] - xs[j]) > R_OBS:
                            continue
                        worse_off_j = ms[j] < ms[i]
                        yielded = actions.get(i) == DECELERATE
                        if worse_off_j:
                            c["opp_worse"] += 1
                            c["burden_opp_worse"] += 1
                            if yielded:
                                c["yield_worse"] += 1
                        else:
                            c["opp_better"] += 1
                            c["burden_opp_better"] += 1
                            if yielded:
                                c["yield_better"] += 1

                # hard-brake burden classification (event start already flagged in trajectory)
                for v in xs:
                    if not s["hard_brake_start"].get(v):
                        continue
                    has_worse_opp = any((w != v and w in xs and abs(xs[v] - xs[w]) <= R_OBS and ms[w] < ms[v]) for w in xs)
                    has_better_only_opp = (not has_worse_opp) and any(
                        (w != v and w in xs and abs(xs[v] - xs[w]) <= R_OBS) for w in xs)
                    if has_worse_opp:
                        c["burden_event_worse"] += 1
                    elif has_better_only_opp:
                        c["burden_event_better"] += 1

                # worst-off sampling (tie-tolerant, in the merge-relevant window)
                if len(xs) >= 2 and any(X_CONVERGE_START <= x < X_MERGE_END for x in xs.values()):
                    ms_active = {v: ms[v] for v in xs}
                    m_min = min(ms_active.values())
                    tied = [v for v in ms_active if abs(ms_active[v] - m_min) < TIE_TOL]
                    if len(tied) < len(ms_active):
                        w_rep = sorted(tied)[0]
                        worst_off_log.append((t, w_rep, ms_active[w_rep], {v: m for v, m in ms_active.items() if v != w_rep}))

            # merge-priority resolution (episode-level, after the step loop)
            for pair_key, (ma0, mb0, _t0) in pair_first_state.items():
                a, b = pair_key.split("-")
                if exit_step.get(a) is None or exit_step.get(b) is None:
                    continue
                if ma0 is None or mb0 is None or ma0 != ma0 or mb0 != mb0:
                    continue
                if ma0 < mb0:
                    worse = a
                elif mb0 < ma0:
                    worse = b
                else:
                    continue
                c["priority_pairs"] += 1
                other = b if worse == a else a
                if exit_step[worse] < exit_step[other]:
                    c["priority_to_worse_off"] += 1

            # worst-off recovery
            for (t0, w, mw0, others0) in worst_off_log:
                for k in RECOVERY_HORIZONS:
                    tk = t0 + k
                    if tk >= len(m_history):
                        continue
                    if not active_history[tk].get(w, False):
                        continue
                    mw_k = m_history[tk].get(w)
                    if mw_k is None:
                        continue
                    others_k = {v: m for v, m in m_history[tk].items() if v != w and active_history[tk].get(v, False)}
                    if not others0 or not others_k:
                        continue
                    mean_other_0 = sum(others0.values()) / len(others0)
                    mean_other_k = sum(others_k.values()) / len(others_k)
                    gapclosure = (mean_other_0 - mw0) - (mean_other_k - mw_k)
                    c["recovery_n"][k] += 1
                    c["recovery_sum"][k] += (mw_k - mw0)
                    c["gapclosure_sum"][k] += gapclosure
    return c


def ry_metric(c: dict) -> float | None:
    """RY = P(BRAKE | worse-off opportunity) / P(BRAKE | not-worse-off opportunity).
    None if either denominator is zero (finite-seed sparsity)."""
    if c["opp_worse"] == 0 or c["opp_better"] == 0:
        return None
    p_worse = c["yield_worse"] / c["opp_worse"]
    p_better = c["yield_better"] / c["opp_better"]
    if p_better == 0:
        return None
    return p_worse / p_better


def priority_share(c: dict) -> float | None:
    if c["priority_pairs"] == 0:
        return None
    return c["priority_to_worse_off"] / c["priority_pairs"]


def burden_share_worse(c: dict) -> float | None:
    """Share of hard-brake events that occur when the vehicle has a worse-off opportunity
    (vs. only a better-off/none opportunity) -- descriptive rate, per seed."""
    denom = c["burden_event_worse"] + c["burden_event_better"]
    if denom == 0:
        return None
    return c["burden_event_worse"] / denom


def recovery_metric(c: dict, k: int) -> float | None:
    """Mean gap-closure at horizon k (positive = worst-off vehicle's disadvantage
    relative to others narrowed after k steps)."""
    if c["recovery_n"][k] == 0:
        return None
    return c["gapclosure_sum"][k] / c["recovery_n"][k]


def main() -> int:
    print("[dws_analyze_mechanisms] processing 48 trajectory shards ...")
    counters: dict[tuple[str, str], dict] = {}
    for cell in CELLS:
        for seed in SEEDS:
            counters[(cell, seed)] = process_shard(cell, seed)
            print(f"  {cell} {seed}: n_episodes={counters[(cell, seed)]['n_episodes']}")

    mech_seed_rows = []
    for cell in CELLS:
        for seed in SEEDS:
            c = counters[(cell, seed)]
            mech_seed_rows.append({
                "cell": cell, "seed": seed,
                "RY": ry_metric(c), "opp_worse": c["opp_worse"], "opp_better": c["opp_better"],
                "yield_worse": c["yield_worse"], "yield_better": c["yield_better"],
                "priority_share_worse_off": priority_share(c), "priority_pairs": c["priority_pairs"],
                "burden_share_worse_opp": burden_share_worse(c),
                "burden_event_worse": c["burden_event_worse"], "burden_event_better": c["burden_event_better"],
                "recovery_gapclosure_k10": recovery_metric(c, 10),
                "recovery_gapclosure_k25": recovery_metric(c, 25),
                "recovery_gapclosure_k50": recovery_metric(c, 50),
                "recovery_n_k25": c["recovery_n"][25],
            })
    with open(OUT / "dws_behavioural_mechanisms_seed_level.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(mech_seed_rows[0].keys()))
        w.writeheader(); w.writerows(mech_seed_rows)
    print(f"[Section 10-13] wrote dws_behavioural_mechanisms_seed_level.csv ({len(mech_seed_rows)} rows)")

    def get(cell, seed, key):
        return next(r[key] for r in mech_seed_rows if r["cell"] == cell and r["seed"] == seed)

    def matched_contrast(cell_a, cell_b, key) -> tuple[list[str], list[float]]:
        finite_seeds, vals = [], []
        for s in SEEDS:
            va, vb = get(cell_a, s, key), get(cell_b, s, key)
            if va is None or vb is None:
                continue
            finite_seeds.append(s); vals.append(va - vb)
        return finite_seeds, vals

    families = {
        "welfare_responsive_yielding (RY)": "RY",
        "merge_priority_allocation": "priority_share_worse_off",
        "cooperative_burden_transfer": "burden_share_worse_opp",
        "worst_off_recovery_k25": "recovery_gapclosure_k25",
    }

    summary_rows = []
    loo_rows = []
    for mech_name, key in families.items():
        family_p = []
        family_meta = []
        for info, (a, b) in (("Original", ("cell2", "cell1")), ("WSC", ("cell4", "cell3"))):
            finite_seeds, vals = matched_contrast(a, b, key)
            if len(vals) < 3:
                summary_rows.append({
                    "mechanism": mech_name, "information_regime": info, "n_finite_seeds": len(vals),
                    "mean_effect": None, "ci_lower": None, "ci_upper": None, "raw_p": None, "holm_p": None,
                    "note": "too sparse for bootstrap (Section 11 rule: report sparsity, do not force it)",
                })
                family_p.append(None); family_meta.append((info, finite_seeds, vals))
                continue
            b_boot = paired_bootstrap(vals)
            family_p.append(b_boot["raw_p"])
            family_meta.append((info, finite_seeds, vals))
            summary_rows.append({
                "mechanism": mech_name, "information_regime": info, "n_finite_seeds": len(vals),
                "mean_effect": b_boot["mean_effect"], "ci_lower": b_boot["ci_lower"], "ci_upper": b_boot["ci_upper"],
                "raw_p": b_boot["raw_p"], "holm_p": None, "note": "",
            })
        # Holm within this mechanism's 2-test family only (Section 20), skipping if either side sparse
        valid_idx = [i for i, p in enumerate(family_p) if p is not None]
        if len(valid_idx) == 2:
            holm = holm_correction([family_p[0], family_p[1]])
            summary_rows[-2]["holm_p"] = holm[0]
            summary_rows[-1]["holm_p"] = holm[1]
        for info, finite_seeds, vals in family_meta:
            if len(vals) >= 3:
                loo = leave_one_out(finite_seeds, vals)
                loo_rows.append({"family": "behavioural", "metric": f"{mech_name} ({info})", **loo})

    with open(OUT / "dws_behavioural_mechanisms_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print(f"[Section 10-13 summary] wrote dws_behavioural_mechanisms_summary.csv ({len(summary_rows)} rows)")

    print("\n=== MECHANISM HEADLINE RESULTS ===")
    for r in summary_rows:
        print(r)

    import json as _json
    Path(OUT / "_behavioural_loo_rows.json").write_text(_json.dumps(loo_rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
