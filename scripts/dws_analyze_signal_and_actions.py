"""DWS final re-evaluation -- Section 14 (Dense welfare-signal dynamics,
exploratory/descriptive) and Section 16 (action-policy rates by welfare
relation, exploratory/descriptive). Second pass over the same 48 trajectory
shards (Phi/DeltaPhi/F_t and action+M fields already recorded per step).
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "outputs" / "dws_final_reevaluation_v1"
TRAJ_DIR = OUT / "trajectories"
SEEDS = ["900101", "900102", "900103", "900104", "910101", "910102",
         "920101", "920102", "920103", "920104", "920105", "920106"]
CELLS = ["cell1", "cell2", "cell3", "cell4"]
EPS = 1e-6
N_BINS = 10  # normalized episode-progress bins for the Phi trajectory profile


def process_shard(cell: str, seed: str) -> tuple[dict, list[dict], dict]:
    path = TRAJ_DIR / f"{cell}_{seed}.jsonl.gz"
    n_pos = n_neg = n_neutral = n_total = 0
    cum_scores = []
    phi_bins = [[] for _ in range(N_BINS)]
    min_m_bins = [[] for _ in range(N_BINS)]

    action_counts = {"neighbour_worse": [0, 0, 0], "ego_worse": [0, 0, 0], "equal": [0, 0, 0]}  # [HOLD, ACCEL, BRAKE]
    # action indices per this project's meta_speed action_representation: 0=HOLD,1=ACCELERATE,2=DECELERATE (confirmed via DECELERATE=2 in wsc_v2_behavioural_run.py)

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            ep = json.loads(line)
            steps = ep["steps"]
            n_steps = len(steps)
            cum = 0.0
            for s in steps:
                f_t = s["F_t"]
                n_total += 1
                cum += f_t
                if f_t > 0:
                    n_pos += 1
                elif f_t < 0:
                    n_neg += 1
                else:
                    n_neutral += 1

                prog = s["t"] / max(1, n_steps - 1)
                b = min(N_BINS - 1, int(prog * N_BINS))
                phi_bins[b].append(s["Phi"])
                active_m = [m for v, m in s["M"].items() if s["active"].get(v)]
                if active_m:
                    min_m_bins[b].append(min(active_m))

                xs = {v: x for v, x in s["x"].items() if x is not None}
                ms = s["M"]
                for i in xs:
                    others_m = [ms[w] for w in xs if w != i]
                    if not others_m:
                        continue
                    min_other = min(others_m)
                    a = s["action"].get(i)
                    if a is None:
                        continue
                    if ms[i] < min_other - EPS:
                        cat = "ego_worse"
                    elif min_other < ms[i] - EPS:
                        cat = "neighbour_worse"
                    else:
                        cat = "equal"
                    action_counts[cat][a if a in (0, 1, 2) else 0] += 1
            cum_scores.append(cum)

    n = max(1, n_total)
    diag = {
        "cell": cell, "seed": seed,
        "frac_positive": n_pos / n, "frac_negative": n_neg / n, "frac_neutral": n_neutral / n,
        "net_event_balance": (n_pos - n_neg) / n,
        "mean_cumulative_F_per_episode": sum(cum_scores) / len(cum_scores),
        "n_steps_total": n_total,
    }
    timing_rows = []
    for b in range(N_BINS):
        timing_rows.append({
            "cell": cell, "seed": seed, "progress_bin": b,
            "mean_Phi": (sum(phi_bins[b]) / len(phi_bins[b])) if phi_bins[b] else None,
            "mean_min_running_M": (sum(min_m_bins[b]) / len(min_m_bins[b])) if min_m_bins[b] else None,
            "n_samples": len(phi_bins[b]),
        })

    action_rows = {}
    for cat, counts in action_counts.items():
        total = sum(counts)
        action_rows[cat] = {
            "hold_rate": counts[0] / total if total else None,
            "accelerate_rate": counts[1] / total if total else None,
            "brake_rate": counts[2] / total if total else None,
            "n_obs": total,
        }
    return diag, timing_rows, action_rows


def main() -> int:
    diag_rows = []
    timing_rows_all = []
    action_rows_all = []
    for cell in CELLS:
        for seed in SEEDS:
            diag, timing_rows, action_rows = process_shard(cell, seed)
            diag_rows.append(diag)
            timing_rows_all.extend(timing_rows)
            for cat, vals in action_rows.items():
                action_rows_all.append({"cell": cell, "seed": seed, "welfare_relation": cat, **vals})
            print(f"  {cell} {seed} done")

    with open(OUT / "dws_signal_diagnostics_seed_level.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()))
        w.writeheader(); w.writerows(diag_rows)
    print(f"[Section 14] wrote dws_signal_diagnostics_seed_level.csv ({len(diag_rows)} rows)")

    with open(OUT / "dws_signal_timing_profile.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(timing_rows_all[0].keys()))
        w.writeheader(); w.writerows(timing_rows_all)
    print(f"[Section 14] wrote dws_signal_timing_profile.csv ({len(timing_rows_all)} rows)")

    with open(OUT / "dws_action_policy_by_welfare_relation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(action_rows_all[0].keys()))
        w.writeheader(); w.writerows(action_rows_all)
    print(f"[Section 16] wrote dws_action_policy_by_welfare_relation.csv ({len(action_rows_all)} rows) "
          f"(companion name; Section 16 did not mandate an exact filename)")

    # descriptive Cell2-Cell1 / Cell4-Cell3 summary of signal diagnostics (no p-values, exploratory)
    def get(cell, seed, key):
        return next(r[key] for r in diag_rows if r["cell"] == cell and r["seed"] == seed)

    summary = []
    for key in ("frac_positive", "frac_negative", "net_event_balance", "mean_cumulative_F_per_episode"):
        for info, (a, b) in (("Original", ("cell2", "cell1")), ("WSC", ("cell4", "cell3"))):
            vals = [get(a, s, key) - get(b, s, key) for s in SEEDS]
            summary.append({"metric": key, "contrast": info, "mean_diff": sum(vals) / len(vals),
                             "min": min(vals), "max": max(vals)})
    with open(OUT / "dws_signal_diagnostics_contrast_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    print("descriptive signal contrast summary:")
    for r in summary:
        print(" ", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
