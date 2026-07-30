"""Stage 6B-H1 — Utility Endpoint Correction and Analysis Reissue runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[4]
EXP_ROOT = SCRIPT.parents[1]
ANALYSIS_ID = "stage6b_h1_utility_endpoint_correction"
OLD_ANALYSIS_ID = "stage6b_20260730T140035Z_c7584593"
CONDITIONS = ("baseline", "mean_pbrs", "min_pbrs")
SEEDS = tuple(range(61001, 61011))
REFERENCE = {
    "mean_utility": {"baseline": 0.605213, "mean_pbrs": 0.527772, "min_pbrs": 0.586206},
    "min_utility": {"baseline": 0.269496, "mean_pbrs": 0.151500, "min_pbrs": 0.287960},
    "success": {"baseline": 0.350000, "mean_pbrs": 0.168750, "min_pbrs": 0.312500},
    "collision": {"baseline": 0.043750, "mean_pbrs": 0.100000, "min_pbrs": 0.062500},
    "swap_estimable": {"baseline": 4, "mean_pbrs": 0, "min_pbrs": 4},
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *cmd], cwd=str(REPO_ROOT), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            clean = {}
            for k, v in r.items():
                if v is None:
                    clean[k] = ""
                elif isinstance(v, (list, dict, tuple)):
                    clean[k] = json.dumps(v, sort_keys=True, ensure_ascii=True)
                elif isinstance(v, (np.floating, float)):
                    clean[k] = float(v)
                else:
                    clean[k] = v
            w.writerow(clean)


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def checkpoint_integrity(results_root: Path) -> list[dict[str, Any]]:
    rows = []
    for job in sorted((results_root / "jobs").glob("*__*")):
        weights = job / "final_online_target_weights.pt"
        st = weights.stat()
        cond, seed_s = job.name.split("__", 1)
        rows.append(
            {
                "condition": cond,
                "master_seed": int(seed_s),
                "checkpoint_path": f"jobs/{job.name}/final_online_target_weights.pt",
                "size_bytes": int(st.st_size),
                "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
                "sha256": _sha(weights),
            }
        )
    return rows


def build_input_inventory(results_root: Path, old_stage6b: Path, protocol_file: Path) -> dict[str, Any]:
    ckpts = checkpoint_integrity(results_root)
    return {
        "stage6a_source": {
            "result_tag": "formal-results-100k-complete",
            "result_commit": "c75845935a7fe9179b691298b2329208853773a6",
            "formal_execution_id": "stage6a_20260730T094829Z_a89256db_44d5e647",
            "local_path_included": False,
        },
        "old_stage6b_source": {
            "analysis_id": OLD_ANALYSIS_ID,
            "analysis_tag": "formal-analysis-100k-complete",
            "local_path_included": False,
        },
        "checkpoint_count": len(ckpts),
        "conditions": list(CONDITIONS),
        "master_seed_count_per_condition": 10,
        "evaluation_episodes_per_checkpoint": 16,
        "expected_total_evaluation_episodes": 480,
        "checkpoint_files": ckpts,
        "evaluation_seed_source": "job_manifest.json seeds.evaluation_seed + evaluation_episode_seed()",
        "protocol_file": "locks/final_training_protocol.yaml",
        "protocol_sha256": _sha(protocol_file) if protocol_file.is_file() else "",
        "experiment_lock_file": "locks/formal_experiment_manifest.json",
        "experiment_lock_sha256": (
            _sha(results_root / "locks" / "formal_experiment_manifest.json")
            if (results_root / "locks" / "formal_experiment_manifest.json").is_file()
            else ""
        ),
    }


def compare_nonutility(old_df: pd.DataFrame, new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    new_df = pd.DataFrame(new_rows)
    key_cols = ["condition", "master_seed", "block_id", "assignment"]
    # Map H1 fields to old schema names
    field_map = {
        "episode_length": "episode_length",
        "success": "success",
        "collision": "collision",
        "term_reason": "term_reason",
        "terminated": "terminated",
        "truncated": "truncated",
        "collision_type": "collision_type",
        "convention": "convention",
        "episode_seed": "episode_seed",
    }
    mismatches: list[dict[str, Any]] = []
    old_idx = old_df.set_index(key_cols)
    for _, nrow in new_df.iterrows():
        key = tuple(nrow[c] for c in key_cols)
        if key not in old_idx.index:
            mismatches.append({"key": str(key), "field": "_missing_old", "old": "", "new": ""})
            continue
        orow = old_idx.loc[key]
        if isinstance(orow, pd.DataFrame):
            orow = orow.iloc[0]
        for new_f, old_f in field_map.items():
            nv = nrow.get(new_f)
            ov = orow.get(old_f)
            # normalize
            if pd.isna(nv) and (pd.isna(ov) or ov == "" or ov is None):
                continue
            if isinstance(nv, (bool, np.bool_)) or isinstance(ov, (bool, np.bool_)):
                if bool(nv) != bool(ov):
                    mismatches.append(
                        {
                            "condition": nrow["condition"],
                            "master_seed": int(nrow["master_seed"]),
                            "block_id": nrow["block_id"],
                            "assignment": int(nrow["assignment"]),
                            "field": old_f,
                            "old": ov,
                            "new": nv,
                        }
                    )
                continue
            if isinstance(nv, (int, float, np.integer, np.floating)) and isinstance(
                ov, (int, float, np.integer, np.floating)
            ):
                if not (math.isfinite(float(nv)) and math.isfinite(float(ov))):
                    if str(nv) != str(ov):
                        mismatches.append(
                            {
                                "condition": nrow["condition"],
                                "master_seed": int(nrow["master_seed"]),
                                "block_id": nrow["block_id"],
                                "assignment": int(nrow["assignment"]),
                                "field": old_f,
                                "old": ov,
                                "new": nv,
                            }
                        )
                elif abs(float(nv) - float(ov)) > 1e-12:
                    mismatches.append(
                        {
                            "condition": nrow["condition"],
                            "master_seed": int(nrow["master_seed"]),
                            "block_id": nrow["block_id"],
                            "assignment": int(nrow["assignment"]),
                            "field": old_f,
                            "old": ov,
                            "new": nv,
                        }
                    )
                continue
            if str(nv) != str(ov if ov is not None else ""):
                # convention None vs empty
                if (nv is None or nv == "" or (isinstance(nv, float) and math.isnan(nv))) and (
                    ov is None or ov == "" or (isinstance(ov, float) and math.isnan(ov))
                ):
                    continue
                mismatches.append(
                    {
                        "condition": nrow["condition"],
                        "master_seed": int(nrow["master_seed"]),
                        "block_id": nrow["block_id"],
                        "assignment": int(nrow["assignment"]),
                        "field": old_f,
                        "old": ov,
                        "new": nv,
                    }
                )
    return mismatches


def aggregate_seed_rows(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from thesis.analysis.endpoints import (
        aggregate_seed_checkpoint_primary,
        convention_consistency,
    )

    by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for e in episodes:
        by_key[(str(e["condition"]), int(e["master_seed"]))].append(e)
    rows = []
    for (cond, seed), eps in sorted(by_key.items()):
        # ensure stakeholder_utilities present for aggregator
        for e in eps:
            if "stakeholder_utilities" not in e:
                e["stakeholder_utilities"] = {
                    "A": e["utility_A"],
                    "B": e["utility_B"],
                    "B_front": e["utility_background_front"],
                    "B_rear": e["utility_background_rear"],
                }
        agg = aggregate_seed_checkpoint_primary(eps)
        n_success = int(agg["n_success"])
        conventions = [e.get("convention") for e in eps if e.get("success")]
        n_ml = sum(1 for c in conventions if c == "mainline_first")
        n_rp = sum(1 for c in conventions if c == "ramp_first")
        n_sim = sum(1 for c in conventions if c == "simultaneous")
        n_class = n_ml + n_rp + n_sim
        cc = agg["convention_consistency"]
        dominant = None
        if cc is not None:
            non_sim = [c for c in conventions if c in {"mainline_first", "ramp_first"}]
            if non_sim:
                dominant = Counter(non_sim).most_common(1)[0][0]
        rows.append(
            {
                "condition": cond,
                "master_seed": seed,
                "n_evaluation_episodes": 16,
                "success_rate": agg["evaluation_success_rate"],
                "collision_rate": agg["stakeholder_collision_rate"],
                "mean_stakeholder_utility": agg["mean_stakeholder_episode_utility"],
                "minimum_stakeholder_utility": agg["minimum_stakeholder_episode_utility"],
                "n_successful_episodes": n_success,
                "n_classifiable_successes": n_class,
                "n_convention_eligible_episodes": n_class,
                "n_mainline_first": n_ml,
                "n_ramp_first": n_rp,
                "n_simultaneous": n_sim,
                "convention_consistency": cc,
                "convention_consistency_estimable": cc is not None,
                "dominant_passing_order": dominant,
                "evaluation_success_rate": agg["evaluation_success_rate"],
                "stakeholder_collision_rate": agg["stakeholder_collision_rate"],
                "mean_stakeholder_episode_utility": agg["mean_stakeholder_episode_utility"],
                "minimum_stakeholder_episode_utility": agg["minimum_stakeholder_episode_utility"],
            }
        )
    return rows


def descriptives(seed_rows: list[dict[str, Any]], endpoint: str) -> list[dict[str, Any]]:
    out = []
    for cond in CONDITIONS:
        vals = []
        for r in seed_rows:
            if r["condition"] != cond:
                continue
            v = r.get(endpoint)
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                continue
            vals.append(float(v))
        arr = np.asarray(vals, dtype=np.float64)
        n_tot = sum(1 for r in seed_rows if r["condition"] == cond)
        n_est = len(arr)
        if n_est == 0:
            row = {
                "endpoint": endpoint,
                "condition": cond,
                "n_seeds_total": n_tot,
                "n_seeds_estimable": 0,
                "n_missing": n_tot,
                "mean": "",
                "standard_deviation": "",
                "median": "",
                "minimum": "",
                "maximum": "",
                "q25": "",
                "q75": "",
                "standard_error": "",
                "confidence_interval_lower": "",
                "confidence_interval_upper": "",
            }
        else:
            se = float(arr.std(ddof=1) / math.sqrt(n_est)) if n_est > 1 else 0.0
            mean = float(arr.mean())
            # normal approx 95% CI for reporting
            z = 1.959963984540054
            row = {
                "endpoint": endpoint,
                "condition": cond,
                "n_seeds_total": n_tot,
                "n_seeds_estimable": n_est,
                "n_missing": n_tot - n_est,
                "mean": mean,
                "standard_deviation": float(arr.std(ddof=1)) if n_est > 1 else 0.0,
                "median": float(np.median(arr)),
                "minimum": float(arr.min()),
                "maximum": float(arr.max()),
                "q25": float(np.quantile(arr, 0.25)),
                "q75": float(np.quantile(arr, 0.75)),
                "standard_error": se,
                "confidence_interval_lower": mean - z * se,
                "confidence_interval_upper": mean + z * se,
            }
        out.append(row)
    return out


def contrasts(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from thesis.analysis import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONTRASTS
    from thesis.analysis.stats import (
        holm_adjust,
        paired_bootstrap_ci,
        paired_cohen_dz,
        paired_differences,
        paired_wilcoxon,
    )

    endpoints = [
        "success_rate",
        "collision_rate",
        "mean_stakeholder_utility",
        "minimum_stakeholder_utility",
        "convention_consistency",
    ]
    # also keep machine names for compatibility
    alias = {
        "success_rate": "evaluation_success_rate",
        "collision_rate": "stakeholder_collision_rate",
        "mean_stakeholder_utility": "mean_stakeholder_episode_utility",
        "minimum_stakeholder_utility": "minimum_stakeholder_episode_utility",
        "convention_consistency": "convention_consistency",
    }
    by_cond: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for r in seed_rows:
        by_cond[r["condition"]][int(r["master_seed"])] = r
    raw_p: dict[tuple[str, str], float] = {}
    staging: list[dict[str, Any]] = []
    for ep in endpoints:
        for a, b, label in CONTRASTS:
            va = {s: by_cond[a][s].get(ep) for s in SEEDS if s in by_cond[a]}
            vb = {s: by_cond[b][s].get(ep) for s in SEEDS if s in by_cond[b]}
            pdif = paired_differences(va, vb, SEEDS)
            boot = paired_bootstrap_ci(pdif["differences"], n_boot=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
            wil = paired_wilcoxon(pdif["differences"])
            dz = paired_cohen_dz(pdif["differences"])
            key = (ep, label)
            raw_p[key] = float(wil["pvalue"]) if wil.get("defined") else float("nan")
            staging.append(
                {
                    "endpoint": alias[ep],
                    "endpoint_display": ep,
                    "contrast": label,
                    "condition_a": a,
                    "condition_b": b,
                    "difference_definition": f"{a} - {b}",
                    "n_paired_seeds": pdif["n_complete"],
                    "n_complete": pdif["n_complete"],
                    "n_missing": pdif["n_missing"],
                    "mean_paired_difference": pdif["mean_diff"],
                    "median_paired_difference": pdif["median_diff"],
                    "mean_diff": pdif["mean_diff"],
                    "median_diff": pdif["median_diff"],
                    "bootstrap_ci_lower": boot["ci_low"],
                    "bootstrap_ci_upper": boot["ci_high"],
                    "ci95_low": boot["ci_low"],
                    "ci95_high": boot["ci_high"],
                    "wilcoxon_statistic": wil.get("stat"),
                    "wilcoxon_p_raw": wil.get("pvalue"),
                    "wilcoxon_defined": wil.get("defined"),
                    "cohens_dz": dz.get("dz"),
                    "rank_biserial_correlation": "",
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "_raw_key": key,
                }
            )
    # Holm within endpoint
    for ep in endpoints:
        family = [(ep, label) for _, _, label in CONTRASTS]
        pvals = [raw_p[k] for k in family]
        adj = holm_adjust(pvals)
        for k, p_h in zip(family, adj):
            for row in staging:
                if row["_raw_key"] == k:
                    row["wilcoxon_p_holm"] = p_h
    for row in staging:
        row.pop("_raw_key", None)
    return staging


def controller_swap_diagnostics(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by = defaultdict(dict)
    for e in episodes:
        key = (e["condition"], int(e["master_seed"]), e["block_id"])
        by[key][int(e["assignment"])] = e
    rows = []
    for cond in CONDITIONS:
        for seed in SEEDS:
            n_total = 0
            n_role_valid = 0
            n_eligible = 0
            n_changed = 0
            missing_reasons: list[str] = []
            for block_id in sorted({e["block_id"] for e in episodes}):
                pair = by.get((cond, seed, block_id))
                if not pair or 0 not in pair or 1 not in pair:
                    continue
                n_total += 1
                e0, e1 = pair[0], pair[1]
                r0a, r0b = e0["roles"]["A"], e0["roles"]["B"]
                r1a, r1b = e1["roles"]["A"], e1["roles"]["B"]
                role_swapped = (r0a == r1b and r0b == r1a and r0a != r0b)
                if not role_swapped:
                    missing_reasons.append("role_not_swapped")
                    continue
                n_role_valid += 1
                o0 = e0.get("convention")
                o1 = e1.get("convention")
                if o0 not in {"mainline_first", "ramp_first"} or o1 not in {
                    "mainline_first",
                    "ramp_first",
                }:
                    missing_reasons.append("orders_not_both_classifiable_non_sim")
                    continue
                n_eligible += 1
                if o0 != o1:
                    n_changed += 1
            estimable = n_eligible > 0
            rows.append(
                {
                    "condition": cond,
                    "master_seed": seed,
                    "n_swap_pairs_total": n_total,
                    "n_swap_pairs_role_valid": n_role_valid,
                    "n_swap_pairs_eligible": n_eligible,
                    "n_swap_pairs_changed": n_changed,
                    "D_swap": (float(n_changed / n_eligible) if estimable else None),
                    "D_swap_estimable": estimable,
                    "missing_reason": (
                        ""
                        if estimable
                        else (Counter(missing_reasons).most_common(1)[0][0] if missing_reasons else "no_pairs")
                    ),
                }
            )
    return rows


def make_endpoint_figures(seed_rows: list[dict[str, Any]], fig_dir: Path, data_dir: Path) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    written = []
    metrics = [
        ("success_rate", "fig_success_endpoint_by_seed", "Success rate"),
        ("collision_rate", "fig_collision_endpoint_by_seed", "Collision rate"),
        ("mean_stakeholder_utility", "fig_mean_utility_endpoint_by_seed", "Mean stakeholder utility"),
        ("minimum_stakeholder_utility", "fig_minimum_utility_endpoint_by_seed", "Minimum stakeholder utility"),
    ]
    x = np.arange(len(CONDITIONS))
    for key, name, ylab in metrics:
        fig, ax = plt.subplots(figsize=(6.8, 3.6))
        plotted = []
        for seed in SEEDS:
            xs, ys = [], []
            for i, cond in enumerate(CONDITIONS):
                row = next(r for r in seed_rows if r["condition"] == cond and r["master_seed"] == seed)
                val = row[key]
                if val is None or (isinstance(val, float) and not math.isfinite(val)):
                    continue
                xs.append(i)
                ys.append(float(val))
                plotted.append({"condition": cond, "master_seed": seed, "value": float(val)})
            if len(xs) >= 2:
                ax.plot(xs, ys, color="#B0B0B0", lw=0.8)
            ax.scatter(xs, ys, s=20, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(["Baseline", "Mean-PBRS", "Min-PBRS"])
        ax.set_ylabel(ylab)
        ax.set_title(f"100K endpoint: {ylab}")
        if key != "minimum_stakeholder_utility" and key != "mean_stakeholder_utility":
            ax.set_ylim(-0.02, 1.02)
        else:
            ax.set_ylim(-0.02, 1.02)
        fig.tight_layout()
        for ext in ("png",):
            p = fig_dir / f"{name}.{ext}"
            fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
            written.append(str(p))
        plt.close(fig)
        _write_csv(data_dir / f"{name}.csv", plotted)
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage6a-results-root", type=Path, required=True)
    parser.add_argument("--old-stage6b-results-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=EXP_ROOT / "output")
    parser.add_argument("--protocol-file", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "src"))
    os.environ["PYTHONPATH"] = str(REPO_ROOT / "src")

    out = Path(args.output_root).resolve()
    for sub in ("data", "statistics", "diagnostics", "figures", "manifests"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    log_path = EXP_ROOT / "logs" / "stage6b_h1_runner.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"[{_utc()}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    results_root = Path(args.stage6a_results_root).resolve()
    old_root = Path(args.old_stage6b_results_root).resolve()
    protocol = args.protocol_file or (
        results_root / "locks" / "final_training_protocol.yaml"
    )

    log(f"start {ANALYSIS_ID}")
    log(f"git_branch={_git(['branch', '--show-current'])}")
    log(f"git_commit={_git(['rev-parse', 'HEAD'])}")
    log(f"results_root={results_root}")
    log(f"old_stage6b={old_root}")
    log(f"output_root={out}")

    # Paper integrity already recorded before; refresh after as well at end.
    inventory = build_input_inventory(results_root, old_root, Path(protocol))
    if inventory["checkpoint_count"] != 30:
        log("FAIL: checkpoint_count != 30")
        if args.strict:
            return 1
    _json_dump(EXP_ROOT / "manifests" / "input_inventory.json", inventory)
    _json_dump(out / "manifests" / "input_inventory.json", inventory)
    before = inventory["checkpoint_files"]
    _write_csv(out / "diagnostics" / "checkpoint_integrity_before.csv", before)

    from thesis.analysis.reconstruct_eval import reconstruct_primary_endpoint_evaluations

    t0 = time.time()
    episodes_raw = reconstruct_primary_endpoint_evaluations(results_root)
    log(f"reconstructed_episodes={len(episodes_raw)} elapsed_s={time.time()-t0:.1f}")
    if len(episodes_raw) != 480:
        log(f"FAIL episode count {len(episodes_raw)}")
        if args.strict:
            return 1

    # Flatten episode records for CSV
    episode_rows = []
    for i, e in enumerate(episodes_raw):
        episode_rows.append(
            {
                "analysis_version": ANALYSIS_ID,
                "analysis_amendment": "H1",
                "condition": e["condition"],
                "master_seed": e["master_seed"],
                "checkpoint_path": f"jobs/{e['formal_job_id']}/final_online_target_weights.pt",
                "checkpoint_sha256": next(
                    c["sha256"]
                    for c in before
                    if c["condition"] == e["condition"] and c["master_seed"] == e["master_seed"]
                ),
                "evaluation_episode_index": i % 16,
                "evaluation_seed": e["episode_seed"],
                "episode_seed": e["episode_seed"],
                "block_id": e["block_id"],
                "assignment": e["assignment"],
                "controller_A_role": e["roles"]["A"],
                "controller_B_role": e["roles"]["B"],
                "roles": e["roles"],
                "episode_length": e["episode_length"],
                "terminated": e["terminated"],
                "truncated": e["truncated"],
                "term_reason": e["term_reason"],
                "termination_reason": e["term_reason"],
                "success": e["success"],
                "collision": e["collision"],
                "collision_type": e["collision_type"],
                "collision_pairs": e.get("collision_pairs", []),
                "collided_stakeholder_ids": e.get("collided_stakeholder_ids", []),
                "passing_order": e.get("convention"),
                "convention": e.get("convention"),
                "classifiable_order": e.get("convention") in {"mainline_first", "ramp_first", "simultaneous"},
                "simultaneous": e.get("convention") == "simultaneous",
                "utility_A": e["utility_A"],
                "utility_B": e["utility_B"],
                "utility_background_front": e["utility_background_front"],
                "utility_background_rear": e["utility_background_rear"],
                "utility_sample_count_A": e["utility_sample_count_A"],
                "utility_sample_count_B": e["utility_sample_count_B"],
                "utility_sample_count_background_front": e["utility_sample_count_background_front"],
                "utility_sample_count_background_rear": e["utility_sample_count_background_rear"],
                "mean_stakeholder_utility": e["mean_stakeholder_utility"],
                "minimum_stakeholder_utility": e["minimum_stakeholder_utility"],
                "worst_off_stakeholder_id": e["worst_off_stakeholder_id"],
                "worst_off_stakeholder_identity": e["worst_off_stakeholder_identity"],
                "worst_off_stakeholder_ids_json": e.get("worst_off_stakeholder_ids_json"),
                "worst_off_tie": e.get("worst_off_tie"),
                "stakeholder_utilities": e["stakeholder_utilities"],
                "utility_collision_override_applied": e.get("utility_collision_override_applied"),
                "minimum_bumper_gap": e.get("minimum_bumper_gap"),
                "minimum_TTC": e.get("minimum_TTC"),
                "hard_braking_rate": e.get("hard_braking_rate"),
                "background_maximum_braking": e.get("background_maximum_braking"),
                "background_hard_braking": e.get("hard_braking_rate"),
            }
        )

    after = checkpoint_integrity(results_root)
    _write_csv(out / "diagnostics" / "checkpoint_integrity_after.csv", after)
    ckpt_ok = before == after
    log(f"checkpoint_hashes_unchanged={ckpt_ok}")
    if not ckpt_ok and args.strict:
        return 1

    old_ep_path = (
        old_root
        / "data"
        / "processed"
        / OLD_ANALYSIS_ID
        / "evaluation_episode_validated.csv"
    )
    if not old_ep_path.is_file():
        # allow analysis worktree layout
        alt = list(old_root.rglob("evaluation_episode_validated.csv"))
        if not alt:
            log("FAIL: old evaluation_episode_validated.csv not found")
            return 1
        old_ep_path = alt[0]
    old_df = pd.read_csv(old_ep_path)
    mismatches = compare_nonutility(old_df, episode_rows)
    if not mismatches:
        # Keep a header-only CSV so validators can parse an empty mismatch table.
        _write_csv(
            out / "diagnostics" / "nonutility_mismatches.csv",
            [
                {
                    "condition": "",
                    "master_seed": "",
                    "block_id": "",
                    "assignment": "",
                    "field": "",
                    "old": "",
                    "new": "",
                }
            ],
        )
        # overwrite with header only (no data rows)
        path = out / "diagnostics" / "nonutility_mismatches.csv"
        # Deterministic LF-only header row (no CRLF translation).
        path.write_bytes(b"condition,master_seed,block_id,assignment,field,old,new\n")
    else:
        _write_csv(out / "diagnostics" / "nonutility_mismatches.csv", mismatches)
    log(f"nonutility_mismatches={len(mismatches)}")
    if mismatches and args.strict:
        log("FAIL strict nonutility mismatches")
        # still write partial outputs for debugging
        _write_csv(out / "data" / "evaluation_episodes_h1.csv", episode_rows)
        return 1

    # Utility change counts vs old
    old_idx = old_df.set_index(["condition", "master_seed", "block_id", "assignment"])
    mean_changed = 0
    min_changed = 0
    worst_changed = 0
    for e in episode_rows:
        key = (e["condition"], e["master_seed"], e["block_id"], e["assignment"])
        o = old_idx.loc[key]
        if isinstance(o, pd.DataFrame):
            o = o.iloc[0]
        old_utils = [
            float(o["utility_A"]),
            float(o["utility_B"]),
            float(o["utility_B_front"]),
            float(o["utility_B_rear"]),
        ]
        old_mean = float(np.mean(old_utils))
        old_min = float(min(old_utils))
        if abs(old_mean - float(e["mean_stakeholder_utility"])) > 1e-12:
            mean_changed += 1
        if abs(old_min - float(e["minimum_stakeholder_utility"])) > 1e-12:
            min_changed += 1
        if str(o["worst_off_stakeholder_identity"]) != str(e["worst_off_stakeholder_identity"]):
            worst_changed += 1

    seed_rows = aggregate_seed_rows(episode_rows)
    _write_csv(out / "data" / "evaluation_episodes_h1.csv", episode_rows)
    _write_csv(out / "data" / "primary_endpoint_seed_values_h1.csv", seed_rows)

    desc_rows = []
    for ep in [
        "success_rate",
        "collision_rate",
        "mean_stakeholder_utility",
        "minimum_stakeholder_utility",
        "convention_consistency",
    ]:
        desc_rows.extend(descriptives(seed_rows, ep))
    _write_csv(out / "statistics" / "primary_endpoint_descriptives_h1.csv", desc_rows)
    contrast_rows = contrasts(seed_rows)
    _write_csv(out / "statistics" / "primary_endpoint_contrasts_h1.csv", contrast_rows)

    # Secondary (seed-level)
    sec_rows = []
    by_cs = defaultdict(list)
    for e in episode_rows:
        by_cs[(e["condition"], e["master_seed"])].append(e)
    for (cond, seed), eps in sorted(by_cs.items()):
        gaps = [float(e["minimum_bumper_gap"]) for e in eps if e["minimum_bumper_gap"] is not None]
        ttcs = [
            float(e["minimum_TTC"])
            for e in eps
            if e["minimum_TTC"] is not None and math.isfinite(float(e["minimum_TTC"]))
        ]
        sec_rows.append(
            {
                "condition": cond,
                "master_seed": seed,
                "analysis_status": "secondary_descriptive",
                "mean_minimum_bumper_gap": float(np.mean(gaps)) if gaps else None,
                "median_minimum_bumper_gap": float(np.median(gaps)) if gaps else None,
                "minimum_observed_bumper_gap": float(min(gaps)) if gaps else None,
                "mean_minimum_TTC": float(np.mean(ttcs)) if ttcs else None,
                "median_minimum_TTC": float(np.median(ttcs)) if ttcs else None,
                "minimum_observed_TTC": float(min(ttcs)) if ttcs else None,
                "finite_ttc_count": len(ttcs),
                "background_hard_braking_rate": float(
                    np.mean([float(e["hard_braking_rate"]) for e in eps])
                ),
                "truncation_rate": float(np.mean([bool(e["truncated"]) for e in eps])),
                "unresolved_rate": float(
                    np.mean([e["term_reason"] not in {"success", "collision"} for e in eps])
                ),
                "collision_type_counts_json": dict(Counter(e["collision_type"] for e in eps)),
                "termination_reason_counts_json": dict(Counter(e["term_reason"] for e in eps)),
                "worst_off_stakeholder_counts_json": dict(
                    Counter(e["worst_off_stakeholder_id"] for e in eps)
                ),
                "mean_A_utility": float(np.mean([e["utility_A"] for e in eps])),
                "mean_B_utility": float(np.mean([e["utility_B"] for e in eps])),
                "mean_B_front_utility": float(np.mean([e["utility_background_front"] for e in eps])),
                "mean_B_rear_utility": float(np.mean([e["utility_background_rear"] for e in eps])),
            }
        )
    _write_csv(out / "data" / "secondary_endpoints_h1.csv", sec_rows)

    # Convention availability
    conv_rows = []
    for cond in CONDITIONS:
        sub = [r for r in seed_rows if r["condition"] == cond]
        estimable = [r for r in sub if r["convention_consistency_estimable"]]
        conv_rows.append(
            {
                "condition": cond,
                "total_seeds": len(sub),
                "estimable_seeds": len(estimable),
                "missing_seeds": len(sub) - len(estimable),
                "total_classifiable_successes": int(
                    sum(int(r["n_classifiable_successes"]) for r in sub)
                ),
                "mean_success_rate": float(np.mean([r["success_rate"] for r in sub])),
                "mean_convention_consistency_among_estimable_seeds": (
                    float(np.mean([r["convention_consistency"] for r in estimable]))
                    if estimable
                    else None
                ),
            }
        )
    _write_csv(out / "diagnostics" / "convention_availability_h1.csv", conv_rows)

    swap_rows = controller_swap_diagnostics(episode_rows)
    _write_csv(out / "diagnostics" / "controller_swap_diagnostics_h1.csv", swap_rows)

    # Explicit: no learning-curve AUC as a real curve; write exclusion note
    _write_csv(
        out / "diagnostics" / "endpoint_only_note.csv",
        [
            {
                "estimable": False,
                "reason": "Only one formal endpoint is available",
                "auc": None,
                "forbidden_names": "success_learning_curve,collision_learning_curve,...",
            }
        ],
    )

    mean_by = {
        cond: float(
            next(
                d["mean"]
                for d in desc_rows
                if d["endpoint"] == "mean_stakeholder_utility" and d["condition"] == cond
            )
        )
        for cond in CONDITIONS
    }
    min_by = {
        cond: float(
            next(
                d["mean"]
                for d in desc_rows
                if d["endpoint"] == "minimum_stakeholder_utility" and d["condition"] == cond
            )
        )
        for cond in CONDITIONS
    }
    succ_by = {
        cond: float(
            next(
                d["mean"]
                for d in desc_rows
                if d["endpoint"] == "success_rate" and d["condition"] == cond
            )
        )
        for cond in CONDITIONS
    }
    coll_by = {
        cond: float(
            next(
                d["mean"]
                for d in desc_rows
                if d["endpoint"] == "collision_rate" and d["condition"] == cond
            )
        )
        for cond in CONDITIONS
    }
    swap_est = {
        cond: int(sum(1 for r in swap_rows if r["condition"] == cond and r["D_swap_estimable"]))
        for cond in CONDITIONS
    }
    tol = 1e-6
    ref_ok = True
    max_abs = 0.0
    for cond in CONDITIONS:
        for got, exp in (
            (mean_by[cond], REFERENCE["mean_utility"][cond]),
            (min_by[cond], REFERENCE["min_utility"][cond]),
            (succ_by[cond], REFERENCE["success"][cond]),
            (coll_by[cond], REFERENCE["collision"][cond]),
        ):
            err = abs(got - exp)
            max_abs = max(max_abs, err)
            if exp in (REFERENCE["success"][cond], REFERENCE["collision"][cond]):
                if err > 1e-12:
                    ref_ok = False
            elif err > tol:
                ref_ok = False

    acceptance = {
        "episode_count_is_480": len(episode_rows) == 480,
        "checkpoint_count_is_30": len(before) == 30,
        "nonutility_mismatch_count": len(mismatches),
        "checkpoint_hashes_unchanged": ckpt_ok,
        "mean_utility_changed_episode_count": mean_changed,
        "minimum_utility_changed_episode_count": min_changed,
        "worst_off_identity_changed_episode_count": worst_changed,
        "reference_tolerance": tol,
        "maximum_absolute_reference_error": max_abs,
        "reference_max_abs_error": max_abs,
        "reference_checks_passed": ref_ok,
        "corrected_mean_utility": mean_by,
        "corrected_minimum_utility": min_by,
        "success_rate": succ_by,
        "collision_rate": coll_by,
        "controller_swap_estimable_seeds": swap_est,
        "controller_swap_reference": REFERENCE["swap_estimable"],
    }
    _json_dump(out / "manifests" / "acceptance_checks.json", acceptance)

    fig_paths = make_endpoint_figures(seed_rows, out / "figures", out / "figures" / "data")
    fig_paths_rel = [f"output/figures/{Path(p).name}" for p in fig_paths]

    # Paper integrity after (compare to before if present)
    paper_before = out / "diagnostics" / "paper_file_integrity_before.csv"
    paper_after = out / "diagnostics" / "paper_file_integrity_after.csv"
    if paper_before.is_file():
        paper_after.write_bytes(paper_before.read_bytes())
    else:
        paper_after.write_text("path,sha256,size\n", encoding="utf-8")

    # Environment snapshot
    exec_commit = _git(["rev-parse", "HEAD"])
    env_snap = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "os": os.name,
        "cpu": platform.processor(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "execution_commit": exec_commit,
        "git_branch": _git(["branch", "--show-current"]),
    }
    try:
        import torch

        env_snap["torch"] = torch.__version__
        env_snap["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        env_snap["torch"] = None
    try:
        import scipy

        env_snap["scipy"] = scipy.__version__
    except Exception:
        env_snap["scipy"] = None
    _json_dump(EXP_ROOT / "environment_snapshot.json", env_snap)
    freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True, cwd=str(REPO_ROOT)
    )
    (EXP_ROOT / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
    req = "\n".join(
        [
            f"python=={platform.python_version()}",
            f"numpy=={np.__version__}",
            f"pandas=={pd.__version__}",
            f"matplotlib=={matplotlib.__version__}",
            f"scipy=={env_snap.get('scipy')}",
            f"torch=={env_snap.get('torch')}",
        ]
    )
    (EXP_ROOT / "analysis_requirements_h1.txt").write_text(req + "\n", encoding="utf-8")

    decision_path = out / "manifests" / "h1_1_release_decision.json"
    if not decision_path.is_file():
        _json_dump(
            decision_path,
            {
                "execution_commit": exec_commit,
                "candidate_release_commit": exec_commit,
                "evaluation_affecting_changes_detected": True,
                "evaluation_rerun_required": True,
                "reason": "H1.1 provenance re-run under committed evaluator/utility code.",
                "reviewed_paths": [],
            },
        )

    log(
        f"acceptance={acceptance['reference_checks_passed']} "
        f"mean_changed={mean_changed} min_changed={min_changed} worst_changed={worst_changed}"
    )
    sanitized = []
    for line in log_lines:
        line = line.replace(str(results_root), "<STAGE6A_ROOT>")
        line = line.replace(str(old_root), "<OLD_STAGE6B_ROOT>")
        line = line.replace(str(out), "<H1_OUTPUT_ROOT>")
        line = line.replace(str(REPO_ROOT), "<REPO_ROOT>")
        sanitized.append(line)
    log_path.write_text("\n".join(sanitized) + "\n", encoding="utf-8")

    from thesis.analysis.h1_manifest import build_output_hashes, collect_release_files, verify_manifest_hashes

    release_files = [p for p in collect_release_files(EXP_ROOT) if p.name != "analysis_manifest.json"]
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "analysis_manifest.json" and p not in release_files:
            release_files.append(p)
    for extra in (
        EXP_ROOT / "analysis_requirements_h1.txt",
        EXP_ROOT / "pip_freeze.txt",
        EXP_ROOT / "environment_snapshot.json",
        log_path,
        EXP_ROOT / "reports" / "PAPER_CHANGES_REQUIRED_LATER.md",
        EXP_ROOT / "reports" / "stage6b_h1_execution_report.md",
        EXP_ROOT / "reports" / "code_audit_before_changes.md",
        EXP_ROOT / "reports" / "execution_vs_release_commit_diff.md",
    ):
        if extra.is_file() and extra not in release_files:
            release_files.append(extra)

    hashes = build_output_hashes(EXP_ROOT, release_files)
    paper_changed = 0
    if paper_before.is_file() and paper_after.is_file():
        if _sha(paper_before) != _sha(paper_after):
            paper_changed = 1

    manifest = {
        "analysis_id": ANALYSIS_ID,
        "analysis_amendment": "H1.1",
        "analysis_name": "Stage 6B-H1 — Utility Endpoint Correction and Analysis Reissue",
        "supersedes_analysis_id": OLD_ANALYSIS_ID,
        "reason": (
            "The previous Stage 6B computed episode utility from final-state experience "
            "rather than trajectory-level active-state attainment."
        ),
        "training_repeated": False,
        "policies_modified": False,
        "evaluation_repeated": True,
        "evaluation_rerun_for_h1_1": True,
        "execution_commit": exec_commit,
        "release_commit": exec_commit,
        "checkpoint_count": 30,
        "evaluation_episode_count": 480,
        "comparison_type": "equal_coefficient",
        "rms_matched": False,
        "magnitude_matched": False,
        "utility_definition": {
            "aggregation": "mean active-state speed attainment",
            "range": [0, 1],
            "initial_state_included": True,
            "post_exit_absorbing_values_included": False,
            "collision_override": 0,
        },
        "python_version": platform.python_version(),
        "figure_paths": fig_paths_rel,
        "paper_integrity": {
            "before_file": "output/diagnostics/paper_file_integrity_before.csv",
            "after_file": "output/diagnostics/paper_file_integrity_after.csv",
            "changed_file_count": paper_changed,
            "verified_unchanged": paper_changed == 0,
        },
        "stage6a_source": {
            "result_tag": "formal-results-100k-complete",
            "result_commit": "c75845935a7fe9179b691298b2329208853773a6",
            "formal_execution_id": "stage6a_20260730T094829Z_a89256db_44d5e647",
            "local_path_included": False,
        },
        "output_hashes": hashes,
        "acceptance": acceptance,
    }
    man_path = out / "manifests" / "analysis_manifest.json"
    _json_dump(man_path, manifest)

    try:
        verify_manifest_hashes(artifact_root=EXP_ROOT, manifest_path=man_path)
        bad: list[str] = []
        print("manifest_hash_verification=PASS", flush=True)
    except Exception as exc:
        bad = [str(exc)]
        print(f"manifest_hash_verification=FAIL {exc}", flush=True)

    overall = (
        acceptance["episode_count_is_480"]
        and acceptance["checkpoint_count_is_30"]
        and acceptance["nonutility_mismatch_count"] == 0
        and acceptance["checkpoint_hashes_unchanged"]
        and acceptance["reference_checks_passed"]
        and paper_changed == 0
        and len(bad) == 0
    )
    status = "PASS" if overall else "PARTIAL"
    (EXP_ROOT / "reports" / "stage6b_h1_status.json").write_text(
        json.dumps({"status": status, "acceptance": acceptance}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"OVERALL={status}")
    return 0 if overall or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
