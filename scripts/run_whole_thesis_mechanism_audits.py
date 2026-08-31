"""Whole-thesis synthesis -- Sections 7-11 mechanism audits.

Reads the 48 DWS-study trajectory shards already produced by the isolated
re-evaluation (cells 1-4 = Maximin / Maximin+DWS / Maximin+WSC /
Maximin+WSC+DWS). Does not retrain, does not overwrite those shards.

Section 7 aliasing is a PREDECLARED PROXY, not the requested 18D/22D
policy-observation analysis:

  The stored trajectories do not dump observation vectors (dws_eval_worker.py
  writes x, M, action, accel, Phi, DeltaPhi, F_t only). Re-rolling 12,288
  episodes solely to dump observations is not required to finish the
  synthesis and would duplicate the frozen evaluation. Instead this script
  compares two reconstructed global states, declared before looking at
  results:
    traffic proxy (4D):  standardized [x_V0, x_V1, x_V2, x_V3]
    welfare-augmented (8D): standardized [x's + M's]
  k in {10, 25, 50} is fixed. Neighbours are found within the same
  (cell, seed, ego-action) pool among merge-window steps
  (any active vehicle with 220 <= x <= 380). This is labelled
  reward-state aliasing in reconstructed kinematics, NOT observation-space
  aliasing of the trained policy. The 18D vs 22D comparison is
  NOT ESTIMABLE from stored files.

Action codes (meta_speed): 0=HOLD, 1=ACCELERATE, 2=DECELERATE/BRAKE.
"""
from __future__ import annotations

import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dws_stats_lib import paired_bootstrap

OUT = Path(__file__).resolve().parent.parent / "outputs" / "whole_thesis_evidence_synthesis_v1"
TRAJ = Path(__file__).resolve().parent.parent / "outputs" / "dws_final_reevaluation_v1" / "trajectories"
EPISODE = Path(__file__).resolve().parent.parent / "outputs" / "dws_final_reevaluation_v1" / "dws_final_episode_level.csv"
SEED_MET = Path(__file__).resolve().parent.parent / "outputs" / "dws_final_reevaluation_v1" / "dws_final_seed_level_metrics.csv"

SEEDS = ["900101", "900102", "900103", "900104", "910101", "910102",
         "920101", "920103", "920102", "920104", "920105", "920106"]
# keep listed order but iterate sorted for determinism
SEEDS = ["900101", "900102", "900103", "900104", "910101", "910102",
         "920101", "920102", "920103", "920104", "920105", "920106"]
CELLS = ["cell1", "cell2", "cell3", "cell4"]
VIDS = ("V0", "V1", "V2", "V3")
X_MERGE_LO, X_MERGE_HI = 220.0, 380.0
EPS = 1e-6
K_LIST = (10, 25, 50)
HOLD, ACCEL, BRAKE = 0, 1, 2
PERCENTILES = (0.25, 0.50, 0.75, 0.90)


def sign_of(f: float) -> int:
    if f > 0:
        return 1
    if f < 0:
        return -1
    return 0


def iter_episodes(cell: str, seed: str):
    path = TRAJ / f"{cell}_{seed}.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        print(f"skip empty {path.name}")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path.name}")


def process_pair(cell: str, seed: str) -> dict:
    """One pass over a shard: concession, shared-credit, running-vs-terminal,
    action rates, and aliasing sample collection."""
    # concession counters
    n_act = {0: 0, 1: 0, 2: 0}
    n_fneg_act = {0: 0, 1: 0, 2: 0}
    n_fpos = 0
    n_fneg = 0
    n_brake_fpos = 0
    n_brake_fneg = 0
    n_act_success = {0: 0, 1: 0, 2: 0}
    n_fneg_act_success = {0: 0, 1: 0, 2: 0}
    n_act_coll = {0: 0, 1: 0, 2: 0}
    n_fneg_act_coll = {0: 0, 1: 0, 2: 0}
    n_hb = 0
    n_hb_fneg = 0
    n_merge_act = {0: 0, 1: 0, 2: 0}
    n_merge_fneg_act = {0: 0, 1: 0, 2: 0}

    # shared credit (negative DeltaPhi steps)
    n_neg_dphi = 0
    n_single_decliner = 0
    n_multi_decliner = 0
    n_no_decliner = 0  # numerical noise
    n_mismatch_actor = 0  # largest |dM| vehicle != strongest control (most neg accel / BRAKE)
    n_all_same_penalty = 0  # always true by construction; counted for the report
    decliner_counts = []

    # running vs terminal per episode
    run_rows = []

    # action rates
    n_steps_merge = 0
    n_hold = n_acc = n_brk = 0
    n_hb_events = 0
    ep_lens = []

    # aliasing samples: list of (x4, m4, action, f_sign, dphi)
    alias_x = []
    alias_m = []
    alias_action = []
    alias_sign = []
    alias_dphi = []

    n_ep = 0
    for ep in iter_episodes(cell, seed):
        n_ep += 1
        steps = ep["steps"]
        n = len(steps)
        ep_lens.append(n)
        success = ep["term_reason"] == "success"
        collision = ep["term_reason"] == "collision"
        # running welfare at percentiles
        snap = {}
        for p in PERCENTILES:
            idx = min(n - 1, max(0, int(round(p * (n - 1)))))
            s = steps[idx]
            snap[p] = (s["Phi"], min(s["M"].values()))
        run_rows.append({
            "cell": cell, "seed": seed, "scenario_id": ep["scenario_id"],
            "term_reason": ep["term_reason"],
            **{f"phi_p{int(p*100)}": snap[p][0] for p in PERCENTILES},
            **{f"minM_p{int(p*100)}": snap[p][1] for p in PERCENTILES},
        })

        prev_m = None
        for s in steps:
            f = s["F_t"]
            dphi = s["DeltaPhi"]
            sg = sign_of(f)
            xs = s["x"]
            ms = s["M"]
            acts = s["action"]
            accels = s["accel"]
            in_merge = any(
                xs.get(v) is not None and X_MERGE_LO <= xs[v] <= X_MERGE_HI for v in VIDS
            )
            if in_merge:
                n_steps_merge += 1

            # aliasing sample: require all 4 x present (or fill last/mid)
            xvec = []
            mvec = []
            ok = True
            for v in VIDS:
                xv = xs.get(v)
                if xv is None:
                    ok = False
                    break
                xvec.append(float(xv))
                mvec.append(float(ms[v]))
            if ok and in_merge:
                # one row per active vehicle
                for v, a in acts.items():
                    if a not in (0, 1, 2):
                        continue
                    alias_x.append(xvec)
                    alias_m.append(mvec)
                    alias_action.append(int(a))
                    alias_sign.append(sg)
                    alias_dphi.append(float(dphi))

            for v, a in acts.items():
                if a not in (0, 1, 2):
                    continue
                n_act[a] += 1
                if f < 0:
                    n_fneg_act[a] += 1
                if success:
                    n_act_success[a] += 1
                    if f < 0:
                        n_fneg_act_success[a] += 1
                if collision:
                    n_act_coll[a] += 1
                    if f < 0:
                        n_fneg_act_coll[a] += 1
                if in_merge:
                    n_merge_act[a] += 1
                    if f < 0:
                        n_merge_fneg_act[a] += 1
                    if a == HOLD:
                        n_hold += 1
                    elif a == ACCEL:
                        n_acc += 1
                    else:
                        n_brk += 1
            if f > 0:
                n_fpos += 1
                n_brake_fpos += sum(1 for a in acts.values() if a == BRAKE)
            elif f < 0:
                n_fneg += 1
                n_brake_fneg += sum(1 for a in acts.values() if a == BRAKE)

            for v in VIDS:
                if s["hard_brake_start"].get(v):
                    n_hb += 1
                    n_hb_events += 1
                    if f < 0:
                        n_hb_fneg += 1

            if dphi < -EPS:
                n_neg_dphi += 1
                n_all_same_penalty += 1  # F_t is shared
                decliners = []
                dms = {}
                if prev_m is not None:
                    for v in VIDS:
                        dms[v] = ms[v] - prev_m[v]
                        if dms[v] < -EPS:
                            decliners.append(v)
                else:
                    for v in VIDS:
                        dms[v] = 0.0
                n_dec = len(decliners)
                decliner_counts.append(n_dec)
                if n_dec == 1:
                    n_single_decliner += 1
                elif n_dec > 1:
                    n_multi_decliner += 1
                else:
                    n_no_decliner += 1
                if dms:
                    worst_v = min(VIDS, key=lambda v: dms[v])
                    # strongest control: most negative accel among active, else BRAKE
                    active = [v for v in VIDS if accels.get(v) is not None]
                    if active:
                        control_v = min(active, key=lambda v: accels[v])
                        if control_v != worst_v:
                            n_mismatch_actor += 1
            prev_m = {v: ms[v] for v in VIDS}

    def _rate(num, den):
        return (num / den) if den else None

    n_agent_steps = sum(n_act.values())
    concession = {
        "cell": cell, "seed": seed,
        "p_fneg_given_brake": _rate(n_fneg_act[BRAKE], n_act[BRAKE]),
        "p_fneg_given_hold": _rate(n_fneg_act[HOLD], n_act[HOLD]),
        "p_fneg_given_accel": _rate(n_fneg_act[ACCEL], n_act[ACCEL]),
        "p_brake_given_fneg": _rate(n_brake_fneg, n_fneg) if n_fneg else None,
        "p_brake_given_fpos": _rate(n_brake_fpos, n_fpos) if n_fpos else None,
        "p_fneg_given_brake_success": _rate(n_fneg_act_success[BRAKE], n_act_success[BRAKE]),
        "p_fneg_given_brake_collision": _rate(n_fneg_act_coll[BRAKE], n_act_coll[BRAKE]),
        "p_fneg_given_hardbrake": _rate(n_hb_fneg, n_hb),
        "p_fneg_given_brake_merge": _rate(n_merge_fneg_act[BRAKE], n_merge_act[BRAKE]),
        "n_brake": n_act[BRAKE], "n_hold": n_act[HOLD], "n_accel": n_act[ACCEL],
        "n_fneg_steps": n_fneg, "n_fpos_steps": n_fpos, "n_hardbrake_events": n_hb,
    }
    credit = {
        "cell": cell, "seed": seed,
        "n_negative_deltaphi": n_neg_dphi,
        "frac_single_decliner": _rate(n_single_decliner, n_neg_dphi),
        "frac_multi_decliner": _rate(n_multi_decliner, n_neg_dphi),
        "frac_no_decliner": _rate(n_no_decliner, n_neg_dphi),
        "frac_shared_penalty_all_four": 1.0 if n_neg_dphi else None,
        "frac_largest_decline_ne_strongest_control": _rate(n_mismatch_actor, n_neg_dphi),
        "mean_n_decliners": float(np.mean(decliner_counts)) if decliner_counts else None,
    }
    actions = {
        "cell": cell, "seed": seed,
        "hold_rate_merge": _rate(n_hold, n_hold + n_acc + n_brk),
        "accel_rate_merge": _rate(n_acc, n_hold + n_acc + n_brk),
        "brake_rate_merge": _rate(n_brk, n_hold + n_acc + n_brk),
        "hardbrake_events_per_episode": n_hb_events / max(1, n_ep),
        "mean_episode_length": float(np.mean(ep_lens)) if ep_lens else None,
        "n_episodes": n_ep,
    }
    alias_pack = (np.asarray(alias_x, dtype=float),
                  np.asarray(alias_m, dtype=float),
                  np.asarray(alias_action, dtype=int),
                  np.asarray(alias_sign, dtype=int),
                  np.asarray(alias_dphi, dtype=float))
    return concession, credit, actions, run_rows, alias_pack


def aliasing_for_seed(x, m, action, signs, dphi, cell, seed) -> list[dict]:
    """Predeclared k-NN aliasing on reconstructed state. k not tuned."""
    rows = []
    if len(x) < 60:
        return rows
    # standardize within this seed (declared)
    x_mu, x_sd = x.mean(axis=0), x.std(axis=0)
    x_sd = np.where(x_sd < 1e-9, 1.0, x_sd)
    xs = (x - x_mu) / x_sd
    m_mu, m_sd = m.mean(axis=0), m.std(axis=0)
    m_sd = np.where(m_sd < 1e-9, 1.0, m_sd)
    ms = (m - m_mu) / m_sd
    states = {
        "traffic_4d": xs,
        "welfare_aug_8d": np.concatenate([xs, ms], axis=1),
    }
    for state_name, Z in states.items():
        for a in (0, 1, 2):
            mask = action == a
            if mask.sum() < 60:
                continue
            Z_a = Z[mask]
            s_a = signs[mask]
            d_a = dphi[mask]
            # Cap at 4000 points per (seed, action) with a frozen RNG so k-NN
            # stays tractable without scipy. Cap and seed are predeclared, not
            # tuned to the result.
            rng = np.random.default_rng(0)
            if len(Z_a) > 4000:
                take = rng.choice(len(Z_a), size=4000, replace=False)
                Z_a = Z_a[take]
                s_a = s_a[take]
                d_a = d_a[take]
            # brute-force k-NN on the (possibly capped) set
            # pairwise distances in chunks to bound memory
            n_a = len(Z_a)
            all_idx = np.empty((n_a, max(K_LIST) + 1), dtype=int)
            chunk = 250
            for start in range(0, n_a, chunk):
                sl = Z_a[start:start + chunk]
                dmat = ((sl[:, None, :] - Z_a[None, :, :]) ** 2).sum(axis=2)
                all_idx[start:start + chunk] = np.argpartition(dmat, max(K_LIST), axis=1)[:, : max(K_LIST) + 1]
                # order the selected columns
                part = all_idx[start:start + chunk]
                sub = np.take_along_axis(dmat, part, axis=1)
                order = np.argsort(sub, axis=1)
                all_idx[start:start + chunk] = np.take_along_axis(part, order, axis=1)
            for k in K_LIST:
                kk = min(k + 1, n_a)
                if kk <= 1:
                    continue
                neigh = all_idx[:, 1:kk]
                # sign disagreement
                neigh_signs = s_a[neigh]
                disagree = (neigh_signs != s_a[:, None]).mean()
                # 3-way entropy of neighbourhood majority? use query-level mean entropy
                entropies = []
                dphi_vars = []
                pos_f = neg_f = neu_f = []
                pos_fracs, neg_fracs, neu_fracs = [], [], []
                for i in range(len(Z_a)):
                    ns = neigh_signs[i]
                    p_pos = float(np.mean(ns == 1))
                    p_neg = float(np.mean(ns == -1))
                    p_neu = float(np.mean(ns == 0))
                    pos_fracs.append(p_pos); neg_fracs.append(p_neg); neu_fracs.append(p_neu)
                    ps = np.array([p_pos, p_neg, p_neu])
                    ps = ps[ps > 0]
                    entropies.append(float(-(ps * np.log(ps)).sum()))
                    dphi_vars.append(float(np.var(d_a[neigh[i]])))
                rows.append({
                    "cell": cell, "seed": seed, "state": state_name, "action": a,
                    "k": k, "n_points": int(mask.sum()),
                    "sign_disagreement_rate": float(disagree),
                    "mean_neigh_pos_frac": float(np.mean(pos_fracs)),
                    "mean_neigh_neg_frac": float(np.mean(neg_fracs)),
                    "mean_neigh_neu_frac": float(np.mean(neu_fracs)),
                    "mean_sign_entropy": float(np.mean(entropies)),
                    "mean_neigh_deltaphi_var": float(np.mean(dphi_vars)),
                    "note": "PROXY reconstructed (x,M) state; not 18D/22D policy observation",
                })
    return rows


def attach_terminal(run_rows: list[dict], episode_index: dict) -> list[dict]:
    out = []
    for r in run_rows:
        key = (r["cell"], str(r["seed"]), r["scenario_id"])
        ep = episode_index.get(key)
        if ep is None:
            continue
        r = dict(r)
        r["final_u_min"] = float(ep["min_U"])
        r["final_gini"] = float(ep["gini"]) if ep["gini"] not in ("", "None") else None
        r["final_completion"] = int(ep["completion"])
        r["final_collision"] = int(ep["collision"])
        out.append(r)
    return out


def seed_level_running(run_rows: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for r in run_rows:
        by[(r["cell"], str(r["seed"]))].append(r)
    out = []
    for (cell, seed), rs in by.items():
        rec = {"cell": cell, "seed": seed, "n_episodes": len(rs)}
        for p in (25, 50, 75, 90):
            rec[f"mean_phi_p{p}"] = float(np.mean([r[f"phi_p{p}"] for r in rs]))
            rec[f"mean_minM_p{p}"] = float(np.mean([r[f"minM_p{p}"] for r in rs]))
        rec["mean_final_u_min"] = float(np.mean([r["final_u_min"] for r in rs]))
        ginis = [r["final_gini"] for r in rs if r["final_gini"] is not None]
        rec["mean_final_gini"] = float(np.mean(ginis)) if ginis else None
        rec["completion"] = float(np.mean([r["final_completion"] for r in rs]))
        rec["collision"] = float(np.mean([r["final_collision"] for r in rs]))
        # Pearson of mid-episode Phi vs final U_min (descriptive)
        phi50 = np.array([r["phi_p50"] for r in rs], dtype=float)
        umin = np.array([r["final_u_min"] for r in rs], dtype=float)
        rec["corr_phi50_umin"] = float(np.corrcoef(phi50, umin)[0, 1]) if phi50.std() > 0 and umin.std() > 0 else None
        rec["corr_phi90_umin"] = float(np.corrcoef(
            np.array([r["phi_p90"] for r in rs], dtype=float), umin
        )[0, 1]) if umin.std() > 0 else None
        out.append(rec)
    return out


def contrast_seed_dicts(rows: list[dict], metrics: list[str], c_treat: str, c_ctrl: str, name: str) -> list[dict]:
    by = defaultdict(dict)
    for r in rows:
        by[str(r["seed"])][r["cell"]] = r
    out = []
    for m in metrics:
        effects, seeds = [], []
        for s in SEEDS:
            if c_treat in by[s] and c_ctrl in by[s]:
                a, b = by[s][c_treat].get(m), by[s][c_ctrl].get(m)
                if a in (None, "") or b in (None, ""):
                    continue
                effects.append(float(a) - float(b))
                seeds.append(s)
        if len(effects) < 3:
            out.append({"contrast": name, "metric": m, "n_seeds": len(effects),
                        "mean_effect": None, "ci_lower": None, "ci_upper": None, "raw_p": None,
                        "note": "too few finite seeds"})
            continue
        boot = paired_bootstrap(effects)
        out.append({
            "contrast": name, "metric": m, "n_seeds": len(effects),
            "mean_effect": boot["mean_effect"], "median_effect": boot["median_effect"],
            "ci_lower": boot["ci_lower"], "ci_upper": boot["ci_upper"],
            "raw_p": boot["raw_p"],
            "n_positive": boot["n_positive"], "n_negative": boot["n_negative"],
            "seed_effects": "|".join(f"{s}:{e:.6f}" for s, e in zip(seeds, effects)),
        })
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("indexing episode-level outcomes...")
    episode_index = {}
    with open(EPISODE, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            episode_index[(r["cell"], r["seed"], r["scenario_id"])] = r

    concessions, credits, actions, all_run, alias_rows = [], [], [], [], []
    for cell in CELLS:
        for seed in SEEDS:
            print(f"  {cell} {seed}")
            conc, cred, act, run_rows, alias_pack = process_pair(cell, seed)
            concessions.append(conc)
            credits.append(cred)
            actions.append(act)
            all_run.extend(attach_terminal(run_rows, episode_index))
            x, m, a, sg, dp = alias_pack
            alias_rows.extend(aliasing_for_seed(x, m, a, sg, dp, cell, seed))

    write_csv(OUT / "dws_concession_signal_analysis.csv", concessions)
    write_csv(OUT / "dws_shared_credit_diagnostics.csv", credits)
    write_csv(OUT / "dws_action_strategy_seed_level.csv", actions)
    write_csv(OUT / "reward_observation_aliasing_seed_level.csv", alias_rows)

    run_seed = seed_level_running(all_run)
    write_csv(OUT / "running_vs_terminal_welfare.csv", run_seed)

    # summaries / contrasts
    conc_metrics = [
        "p_fneg_given_brake", "p_fneg_given_hold", "p_fneg_given_accel",
        "p_brake_given_fneg", "p_brake_given_fpos",
        "p_fneg_given_brake_success", "p_fneg_given_hardbrake",
    ]
    cred_metrics = [
        "frac_single_decliner", "frac_multi_decliner",
        "frac_largest_decline_ne_strongest_control", "mean_n_decliners",
    ]
    act_metrics = ["hold_rate_merge", "accel_rate_merge", "brake_rate_merge",
                   "hardbrake_events_per_episode", "mean_episode_length"]
    run_metrics = ["mean_phi_p50", "mean_phi_p90", "mean_minM_p50", "corr_phi50_umin", "corr_phi90_umin"]

    summaries = []
    summaries += contrast_seed_dicts(concessions, conc_metrics, "cell2", "cell1", "Original DWS")
    summaries += contrast_seed_dicts(concessions, conc_metrics, "cell4", "cell3", "WSC DWS")
    write_csv(OUT / "dws_concession_signal_summary.csv", summaries)

    cred_sum = []
    cred_sum += contrast_seed_dicts(credits, cred_metrics, "cell2", "cell1", "Original DWS")
    cred_sum += contrast_seed_dicts(credits, cred_metrics, "cell4", "cell3", "WSC DWS")
    write_csv(OUT / "dws_shared_credit_summary.csv", cred_sum)

    act_sum = []
    act_sum += contrast_seed_dicts(actions, act_metrics, "cell2", "cell1", "Original DWS")
    act_sum += contrast_seed_dicts(actions, act_metrics, "cell4", "cell3", "WSC DWS")
    write_csv(OUT / "dws_action_strategy_contrasts.csv", act_sum)

    run_sum = []
    run_sum += contrast_seed_dicts(run_seed, run_metrics, "cell2", "cell1", "Original DWS")
    run_sum += contrast_seed_dicts(run_seed, run_metrics, "cell4", "cell3", "WSC DWS")
    write_csv(OUT / "running_vs_terminal_contrasts.csv", run_sum)

    # aliasing summary: mean across seeds of disagreement, by cell x state x k (action-pooled)
    alias_sum = []
    by = defaultdict(list)
    for r in alias_rows:
        by[(r["cell"], r["state"], r["k"])].append(r)
    for (cell, state, k), rs in sorted(by.items()):
        alias_sum.append({
            "cell": cell, "state": state, "k": k, "n_seed_action_rows": len(rs),
            "mean_sign_disagreement": float(np.mean([r["sign_disagreement_rate"] for r in rs])),
            "mean_sign_entropy": float(np.mean([r["mean_sign_entropy"] for r in rs])),
            "mean_neigh_deltaphi_var": float(np.mean([r["mean_neigh_deltaphi_var"] for r in rs])),
            "mean_neigh_pos_frac": float(np.mean([r["mean_neigh_pos_frac"] for r in rs])),
            "mean_neigh_neg_frac": float(np.mean([r["mean_neigh_neg_frac"] for r in rs])),
            "note": "PROXY; k not tuned; 18D/22D NOT ESTIMABLE",
        })
    write_csv(OUT / "reward_observation_aliasing_summary.csv", alias_sum)
    print("mechanism audits done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
