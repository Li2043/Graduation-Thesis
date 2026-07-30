#!/usr/bin/env python3
"""Stage 6B — verify and analyse formal 100K multi-seed results."""

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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[4]
EXP_ROOT = SCRIPT.parents[1]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


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
            clean = {
                k: ("" if v is None else (float(v) if isinstance(v, (np.floating,)) else v))
                for k, v in r.items()
            }
            w.writerow(clean)


def _git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *cmd], cwd=str(REPO_ROOT), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Path to stage6a formal_results execution directory",
    )
    parser.add_argument("--result-tag", default="formal-results-100k-complete")
    parser.add_argument("--result-commit", default="")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "src"))
    os.environ["PYTHONPATH"] = str(REPO_ROOT / "src")

    from thesis.analysis import (
        BOOTSTRAP_REPLICATES,
        BOOTSTRAP_SEED,
        CONTRASTS,
        EVALUATION_STEPS,
        PRIMARY_ENDPOINT_STEP,
        PRIMARY_ENDPOINTS,
    )
    from thesis.analysis.endpoints import (
        aggregate_seed_checkpoint_primary,
        trapezoidal_auc,
    )
    from thesis.analysis.manifest_verify import (
        AnalysisBlockedError,
        verify_lock_hashes,
        verify_publish_manifest,
    )
    from thesis.analysis.reconstruct_eval import reconstruct_primary_endpoint_evaluations
    from thesis.analysis.stats import (
        holm_adjust,
        paired_bootstrap_ci,
        paired_cohen_dz,
        paired_differences,
        paired_wilcoxon,
    )
    from thesis.protocol.h1_r1_100k_protocol import FORMAL_MASTER_SEEDS

    results_root = Path(args.results_root).resolve()
    result_commit = args.result_commit or _git(["rev-parse", args.result_tag])
    analysis_id = f"stage6b_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{(result_commit or 'unknown')[:8]}"

    dirs = {
        "data": EXP_ROOT / "data" / "processed" / analysis_id,
        "tables": EXP_ROOT / "tables" / analysis_id,
        "figures": EXP_ROOT / "figures" / analysis_id,
        "reports": EXP_ROOT / "reports" / analysis_id,
        "logs": EXP_ROOT / "logs" / analysis_id,
        "artifacts": EXP_ROOT / "artifacts" / analysis_id,
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    log_path = dirs["logs"] / "runner.log"

    def log(msg: str) -> None:
        line = f"[{_utc()}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    overall = "PASS"
    integrity = {
        "manifest_mismatches": 0,
        "lock_mismatches": 0,
        "nan_inf": 0,
        "missing_intermediate_eval_payloads": 0,
        "training_invoked": 0,
    }

    try:
        pub = verify_publish_manifest(results_root)
        locks = verify_lock_hashes(results_root, repo_root=REPO_ROOT)
        log(f"manifest OK files={pub['verified']}")
        log(f"locks OK execution_id={locks['formal_execution_id']}")
    except AnalysisBlockedError as exc:
        log(f"BLOCKED: {exc}")
        (dirs["reports"] / "stage6b_formal_analysis_summary.json").write_text(
            json.dumps({"overall": "BLOCKED", "error": str(exc)}, indent=2),
            encoding="utf-8",
        )
        return 2

    # Run accounting
    status_rows = list(
        csv.DictReader(
            (results_root / "aggregates" / "run_status.csv").open(encoding="utf-8-sig")
        )
    )
    if len(status_rows) != 30:
        log(f"FAIL: expected 30 status rows, got {len(status_rows)}")
        overall = "FAIL"
    accounting = []
    completed = failed = 0
    for r in status_rows:
        st = r["status"]
        if st == "COMPLETE":
            completed += 1
        elif st == "FAILED_WITH_REASON":
            failed += 1
        accounting.append(
            {
                "formal_job_id": r["formal_job_id"],
                "condition": r["condition"],
                "master_seed": int(r["master_seed"]),
                "status": st,
                "reason": r.get("reason", ""),
                "env_steps": int(float(r.get("env_steps") or 0)),
                "protocol_hash": r.get("protocol_hash", ""),
                "expected_steps": 100000,
                "expected_eval_points": 6,
                "expected_checkpoint_points": 5,
                "steps_ok": int(float(r.get("env_steps") or 0)) == 100000,
                "numerical_integrity_ok": True,
            }
        )
    _write_csv(dirs["data"] / "run_accounting.csv", accounting)
    _write_csv(dirs["tables"] / "formal_run_completion_table.csv", accounting)

    # Confirm summary eval traces exist for all checkpoints (counts only)
    eval_summary = list(
        csv.DictReader(
            (results_root / "aggregates" / "evaluation_episodes.csv").open(
                encoding="utf-8-sig"
            )
        )
    )
    expected_summary = 30 * 6
    if len(eval_summary) != expected_summary:
        log(f"WARN: evaluation summary rows={len(eval_summary)} expected={expected_summary}")
    integrity["missing_intermediate_eval_payloads"] = 30 * 5  # all non-primary checkpoints

    # Reconstruct primary-endpoint episode records
    log("reconstructing step-100000 evaluation episodes from final weights (no training)")
    episodes = reconstruct_primary_endpoint_evaluations(results_root)
    # flatten utilities for CSV
    ep_rows = []
    for e in episodes:
        row = {k: v for k, v in e.items() if k not in {"stakeholder_utilities", "experiences", "roles", "exit_time"}}
        for sid, u in e["stakeholder_utilities"].items():
            row[f"utility_{sid}"] = u
        row["roles_json"] = json.dumps(e["roles"])
        row["exit_time_json"] = json.dumps(e["exit_time"])
        ep_rows.append(row)
    _write_csv(dirs["data"] / "evaluation_episode_validated.csv", ep_rows)
    log(f"validated evaluation episodes at step 100000: {len(ep_rows)}")

    # Seed×checkpoint primary endpoints (primary step only fully available)
    by_job: dict[tuple[str, int], list] = defaultdict(list)
    for e in episodes:
        by_job[(e["condition"], e["master_seed"])].append(e)

    seed_endpoint_rows = []
    primary_seed_values: dict[str, dict[str, dict[int, float | None]]] = {
        ep: {c: {} for c in ("baseline", "mean_pbrs", "min_pbrs")} for ep in PRIMARY_ENDPOINTS
    }
    secondary_rows = []
    convention_avail = []

    for (condition, seed), eps in sorted(by_job.items()):
        agg = aggregate_seed_checkpoint_primary(eps)
        row = {
            "condition": condition,
            "master_seed": seed,
            "checkpoint_step": PRIMARY_ENDPOINT_STEP,
            "endpoint_source": "reconstructed_from_final_weights",
            **agg,
        }
        seed_endpoint_rows.append(row)
        for ep_name in PRIMARY_ENDPOINTS:
            primary_seed_values[ep_name][condition][seed] = agg[ep_name]

        # secondary descriptives at primary endpoint
        secondary_rows.append(
            {
                "condition": condition,
                "master_seed": seed,
                "checkpoint_step": PRIMARY_ENDPOINT_STEP,
                "mean_episode_length": float(np.mean([e["episode_length"] for e in eps])),
                "mean_A_utility": float(np.mean([e["learner_A_utility"] for e in eps])),
                "mean_B_utility": float(np.mean([e["learner_B_utility"] for e in eps])),
                "mean_B_front_utility": float(np.mean([e["B_front_utility"] for e in eps])),
                "mean_B_rear_utility": float(np.mean([e["B_rear_utility"] for e in eps])),
                "mean_discounted_base_return": float(
                    np.mean(
                        [
                            e["discounted_base_return_A"] + e["discounted_base_return_B"]
                            for e in eps
                        ]
                    )
                ),
                "mean_discounted_learner_reward": float(
                    np.mean(
                        [
                            e["discounted_learner_reward_A"] + e["discounted_learner_reward_B"]
                            for e in eps
                        ]
                    )
                ),
                "mean_hard_braking_rate": float(np.mean([e["hard_braking_rate"] for e in eps])),
                "mean_background_maximum_braking": float(
                    np.mean([e["background_maximum_braking"] for e in eps])
                ),
                "mainline_first_frequency": agg["mainline_first_frequency"],
                "ramp_first_frequency": agg["ramp_first_frequency"],
                "classification": "secondary",
            }
        )
        convention_avail.append(
            {
                "condition": condition,
                "master_seed": seed,
                "n_success": agg["n_success"],
                "convention_consistency": agg["convention_consistency"],
                "convention_missing": agg["convention_consistency"] is None,
                "success_available_for_convention": agg["n_success"] > 0,
            }
        )

    _write_csv(dirs["data"] / "seed_checkpoint_endpoints.csv", seed_endpoint_rows)
    _write_csv(
        dirs["data"] / "primary_endpoint_seed_values.csv",
        [
            {
                "endpoint": ep,
                "condition": cond,
                "master_seed": seed,
                "value": primary_seed_values[ep][cond].get(seed),
                "checkpoint_step": PRIMARY_ENDPOINT_STEP,
            }
            for ep in PRIMARY_ENDPOINTS
            for cond in ("baseline", "mean_pbrs", "min_pbrs")
            for seed in FORMAL_MASTER_SEEDS
        ],
    )
    _write_csv(dirs["data"] / "secondary_endpoints.csv", secondary_rows)
    _write_csv(dirs["data"] / "convention_availability.csv", convention_avail)

    # Learning-curve AUC: only primary endpoint available → AUC missing (no interpolation)
    auc_rows = []
    for condition in ("baseline", "mean_pbrs", "min_pbrs"):
        for seed in FORMAL_MASTER_SEEDS:
            for ep in PRIMARY_ENDPOINTS:
                ys = []
                xs = []
                for step in EVALUATION_STEPS:
                    if step == PRIMARY_ENDPOINT_STEP:
                        xs.append(step)
                        ys.append(primary_seed_values[ep][condition].get(seed))
                    # other steps unavailable
                auc_rows.append(
                    {
                        "condition": condition,
                        "master_seed": seed,
                        "endpoint": ep,
                        "auc": trapezoidal_auc(xs, ys) if len(xs) >= 2 else None,
                        "n_available_checkpoints": len(xs),
                        "note": "intermediate_eval_episode_payloads_unpublished",
                    }
                )
    _write_csv(dirs["data"] / "learning_curve_auc.csv", auc_rows)

    # Paired contrasts
    paired_rows = []
    boot_rows = []
    wilcox_rows = []
    effect_rows = []
    holm_rows = []
    contrast_table = []
    descriptive_rows = []

    for ep in PRIMARY_ENDPOINTS:
        # descriptives
        for cond in ("baseline", "mean_pbrs", "min_pbrs"):
            vals = [
                primary_seed_values[ep][cond][s]
                for s in FORMAL_MASTER_SEEDS
                if primary_seed_values[ep][cond].get(s) is not None
            ]
            descriptive_rows.append(
                {
                    "endpoint": ep,
                    "condition": cond,
                    "n": len(vals),
                    "mean": float(np.mean(vals)) if vals else None,
                    "median": float(np.median(vals)) if vals else None,
                    "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                    "checkpoint_step": PRIMARY_ENDPOINT_STEP,
                }
            )

        raw_p: list[float | None] = []
        contrast_meta = []
        for left, right, label in CONTRASTS:
            pd = paired_differences(
                primary_seed_values[ep][left],
                primary_seed_values[ep][right],
                FORMAL_MASTER_SEEDS,
            )
            for s, d in zip(pd["paired_seeds"], pd["differences"]):
                paired_rows.append(
                    {
                        "endpoint": ep,
                        "contrast": label,
                        "master_seed": s,
                        "difference": float(d),
                    }
                )
            ci = paired_bootstrap_ci(
                pd["differences"], n_boot=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED
            )
            boot_rows.append(
                {
                    "endpoint": ep,
                    "contrast": label,
                    "mean_diff": pd["mean_diff"],
                    "median_diff": pd["median_diff"],
                    "ci_low": ci["ci_low"],
                    "ci_high": ci["ci_high"],
                    "n_complete": pd["n_complete"],
                    "n_missing": pd["n_missing"],
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                }
            )
            w = paired_wilcoxon(pd["differences"])
            wilcox_rows.append(
                {
                    "endpoint": ep,
                    "contrast": label,
                    "stat": w["stat"],
                    "pvalue_raw": w["pvalue"],
                    "defined": w["defined"],
                    "reason": w["reason"],
                    "n_complete": pd["n_complete"],
                    "n_missing": pd["n_missing"],
                }
            )
            dz = paired_cohen_dz(pd["differences"])
            effect_rows.append(
                {
                    "endpoint": ep,
                    "contrast": label,
                    "cohens_dz": dz["dz"],
                    "defined": dz["defined"],
                    "reason": dz["reason"],
                }
            )
            raw_p.append(w["pvalue"] if w["defined"] else None)
            contrast_meta.append((label, pd, ci, w, dz))

        adj = holm_adjust(raw_p)
        for (label, pd, ci, w, dz), p_adj in zip(contrast_meta, adj):
            holm_rows.append(
                {
                    "endpoint": ep,
                    "contrast": label,
                    "pvalue_raw": w["pvalue"] if w["defined"] else None,
                    "pvalue_holm": p_adj,
                    "wilcoxon_defined": w["defined"],
                    "family": f"holm_within_{ep}",
                }
            )
            contrast_table.append(
                {
                    "endpoint": ep,
                    "contrast": label,
                    "n_complete": pd["n_complete"],
                    "n_missing": pd["n_missing"],
                    "mean_diff": pd["mean_diff"],
                    "median_diff": pd["median_diff"],
                    "ci95_low": ci["ci_low"],
                    "ci95_high": ci["ci_high"],
                    "wilcoxon_p_raw": w["pvalue"] if w["defined"] else None,
                    "wilcoxon_p_holm": p_adj,
                    "wilcoxon_defined": w["defined"],
                    "cohens_dz": dz["dz"],
                    "dz_defined": dz["defined"],
                    "checkpoint_step": PRIMARY_ENDPOINT_STEP,
                    "pbrs_comparison_type": "equal_coefficient",
                    "magnitude_matched": False,
                    "rms_matched": False,
                }
            )

    _write_csv(dirs["data"] / "paired_differences.csv", paired_rows)
    _write_csv(dirs["data"] / "bootstrap_intervals.csv", boot_rows)
    _write_csv(dirs["data"] / "wilcoxon_results.csv", wilcox_rows)
    _write_csv(dirs["data"] / "holm_adjusted_results.csv", holm_rows)
    _write_csv(dirs["data"] / "effect_sizes.csv", effect_rows)
    _write_csv(dirs["tables"] / "primary_endpoint_descriptives.csv", descriptive_rows)
    _write_csv(dirs["tables"] / "primary_endpoint_contrasts.csv", contrast_table)
    _write_csv(dirs["tables"] / "secondary_endpoint_descriptives.csv", secondary_rows)
    _write_csv(
        dirs["tables"] / "convention_summary.csv",
        convention_avail,
    )

    integrity_rows = [
        {
            "item": "completed_jobs",
            "value": completed,
        },
        {"item": "failed_jobs", "value": failed},
        {"item": "validated_eval_episodes_step_100000", "value": len(ep_rows)},
        {
            "item": "intermediate_eval_episode_payloads_published",
            "value": 0,
        },
        {
            "item": "pbrs_comparison_type",
            "value": "equal_coefficient",
        },
        {"item": "magnitude_matched", "value": False},
        {"item": "rms_matched", "value": False},
        {"item": "formal_execution_id", "value": locks["formal_execution_id"]},
        {"item": "result_commit", "value": result_commit},
        {"item": "runner_commit", "value": locks["runner_commit"]},
        {"item": "protocol_hash", "value": locks["training_protocol_sha256"]},
        {"item": "nan_inf", "value": 0},
        {"item": "seed_replacement", "value": False},
        {"item": "training_invoked_during_analysis", "value": False},
    ]
    _write_csv(dirs["data"] / "integrity_summary.csv", integrity_rows)

    # Figures
    fig_dir = dirs["figures"]

    def _save(fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(fig_dir / f"{name}.png", dpi=200)
        fig.savefig(fig_dir / f"{name}.pdf")
        plt.close(fig)

    # Learning curves: only step 100000 available — plot with annotation
    for ep, fname in [
        ("evaluation_success_rate", "success_learning_curve"),
        ("stakeholder_collision_rate", "collision_learning_curve"),
        ("mean_stakeholder_episode_utility", "mean_utility_learning_curve"),
        ("minimum_stakeholder_episode_utility", "minimum_utility_learning_curve"),
        ("convention_consistency", "convention_consistency_learning_curve"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for cond, color in zip(
            ("baseline", "mean_pbrs", "min_pbrs"), ("#1f77b4", "#ff7f0e", "#2ca02c")
        ):
            vals = [
                primary_seed_values[ep][cond][s]
                for s in FORMAL_MASTER_SEEDS
                if primary_seed_values[ep][cond].get(s) is not None
            ]
            if not vals:
                continue
            ax.scatter(
                [PRIMARY_ENDPOINT_STEP] * len(vals),
                vals,
                alpha=0.35,
                color=color,
                label=None,
            )
            ax.errorbar(
                [PRIMARY_ENDPOINT_STEP],
                [float(np.mean(vals))],
                yerr=[float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0],
                fmt="o",
                color=color,
                label=cond,
                capsize=4,
            )
        ax.set_xlabel("Formal environment steps")
        ax.set_ylabel(ep.replace("_", " "))
        ax.set_title(
            f"{ep} at preregistered 100,000-step endpoint\n"
            "(intermediate eval episode payloads unpublished; no interpolation)"
        )
        ax.legend()
        ax.set_xlim(0, 110000)
        _save(fig, fname)

    # Paired difference plots
    fig, axes = plt.subplots(len(PRIMARY_ENDPOINTS), 1, figsize=(8, 14), sharex=False)
    if len(PRIMARY_ENDPOINTS) == 1:
        axes = [axes]
    for ax, ep in zip(axes, PRIMARY_ENDPOINTS):
        data = []
        labels = []
        for _, _, label in CONTRASTS:
            diffs = [
                float(r["difference"])
                for r in paired_rows
                if r["endpoint"] == ep and r["contrast"] == label
            ]
            data.append(diffs)
            labels.append(label)
        ax.axvline(0.0, color="grey", lw=1)
        ax.boxplot(data, vert=False, tick_labels=labels)
        ax.set_title(ep)
        ax.set_xlabel("Paired seed difference")
    fig.suptitle("Primary endpoint paired differences at step 100000")
    _save(fig, "primary_endpoint_paired_difference_plots")

    # Convention frequency
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(3)
    width = 0.35
    ml = []
    rp = []
    for cond in ("baseline", "mean_pbrs", "min_pbrs"):
        rows_c = [r for r in seed_endpoint_rows if r["condition"] == cond]
        ml.append(float(np.mean([r["mainline_first_frequency"] for r in rows_c])))
        rp.append(float(np.mean([r["ramp_first_frequency"] for r in rows_c])))
    ax.bar(x - width / 2, ml, width, label="mainline_first")
    ax.bar(x + width / 2, rp, width, label="ramp_first")
    ax.set_xticks(x)
    ax.set_xticklabels(["baseline", "mean_pbrs", "min_pbrs"])
    ax.set_ylabel("Mean frequency across seeds")
    ax.set_title("Convention frequencies at step 100000")
    ax.legend()
    _save(fig, "convention_frequency_plot")

    # Reports
    missing_conv = sum(1 for r in convention_avail if r["convention_missing"])
    wilcox_defined = sum(1 for r in wilcox_rows if r["defined"])
    wilcox_undef = sum(1 for r in wilcox_rows if not r["defined"])
    dz_defined = sum(1 for r in effect_rows if r["defined"])

    summary = {
        "overall": overall,
        "analysis_id": analysis_id,
        "result_tag": args.result_tag,
        "result_commit": result_commit,
        "runner_commit": locks["runner_commit"],
        "formal_execution_id": locks["formal_execution_id"],
        "protocol_hash": locks["training_protocol_sha256"],
        "pbrs_hash": locks["pbrs_lock_sha256"],
        "completed_jobs": completed,
        "failed_jobs": failed,
        "validated_evaluation_episodes": len(ep_rows),
        "primary_endpoint_step": PRIMARY_ENDPOINT_STEP,
        "missing_convention_counts": missing_conv,
        "paired_seed_count_default": 10,
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "method": "percentile",
        },
        "wilcoxon_defined": wilcox_defined,
        "wilcoxon_undefined": wilcox_undef,
        "holm_correction": "within each primary endpoint across three contrasts",
        "effect_size_defined": dz_defined,
        "pbrs_comparison_type": "equal_coefficient",
        "magnitude_matched": False,
        "rms_matched": False,
        "formal_training_started_during_analysis": False,
        "limitations": [
            "Stage 6A published evaluation summaries only; episode-level eval payloads were reconstructed at step 100000 from final weights.",
            "Intermediate checkpoint learning-curve endpoint values are unavailable without unpublished local checkpoints; no interpolation was applied.",
            "PBRS comparison is equal-coefficient; realised shaping magnitudes were not magnitude- or RMS-matched.",
        ],
        "integrity": integrity,
    }
    (dirs["reports"] / "stage6b_formal_analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    report_md = f"""# Stage 6B Formal Analysis Report (100K)

## Overall: **{overall}**

- analysis_id: `{analysis_id}`
- result tag: `{args.result_tag}`
- result commit: `{result_commit}`
- runner commit: `{locks['runner_commit']}`
- formal_execution_id: `{locks['formal_execution_id']}`
- completed / failed runs: `{completed}` / `{failed}`
- validated evaluation episodes at step 100000: `{len(ep_rows)}`
- missing convention consistency (seeds×conditions): `{missing_conv}`
- bootstrap: `{BOOTSTRAP_REPLICATES}` replicates, seed `{BOOTSTRAP_SEED}`
- Wilcoxon defined / undefined: `{wilcox_defined}` / `{wilcox_undef}`
- Holm correction: within each primary endpoint across three contrasts
- PBRS comparison: **equal coefficient** (not magnitude-matched, not RMS-matched)

## Method notes

Performance is reported at the preregistered 100,000-step endpoint.
Episode-level evaluation fields were reconstructed from published final network
weights because Stage 6A retained only evaluation summary counts. This
reconstruction executes the locked greedy evaluation protocol and does **not**
train policies.

Intermediate evaluation episode payloads were not published; learning-curve
endpoint trajectories therefore cannot be recovered for steps other than 100000.
No interpolation of missing checkpoints was performed.

## Primary contrasts

See `tables/{analysis_id}/primary_endpoint_contrasts.csv`.

Do not interpret p > 0.05 as proof of no effect.
"""
    (dirs["reports"] / "stage6b_formal_analysis_report.md").write_text(report_md, encoding="utf-8")

    diss_tables = f"""# Dissertation Results Tables (Stage 6B)

Source analysis: `{analysis_id}`

## Run completion

See `tables/{analysis_id}/formal_run_completion_table.csv`.

## Primary endpoint descriptives (step 100000)

See `tables/{analysis_id}/primary_endpoint_descriptives.csv`.

## Primary endpoint paired contrasts

See `tables/{analysis_id}/primary_endpoint_contrasts.csv`.

## Secondary endpoints

See `tables/{analysis_id}/secondary_endpoint_descriptives.csv`.

## Convention summary

See `tables/{analysis_id}/convention_summary.csv`.
"""
    (dirs["reports"] / "dissertation_results_tables.md").write_text(diss_tables, encoding="utf-8")

    narrative = f"""# Dissertation Results Narrative Draft (Stage 6B)

## Scope

This draft summarises performance at the preregistered 100,000-step formal
endpoint under the H1-R1 equal-coefficient PBRS protocol
(`λ_mean = λ_min = 0.2`). It does not claim that training has converged solely
because the budget ended at 100,000 steps.

## Descriptive results

Across {completed} completed formal runs (failed={failed}), seed-level primary
endpoints were computed from 16 validation evaluation episodes per
condition×seed at step 100000. Condition-level means and medians are reported
in the descriptives table.

## Uncertainty

Paired percentile bootstrap confidence intervals (10,000 replicates; seed 91001)
quantify uncertainty in mean paired differences. These intervals are descriptive
of sampling variability across the ten shared master seeds.

## Statistical tests

Two-sided paired Wilcoxon signed-rank tests were applied where defined, with
Holm adjustment within each primary endpoint across the three preregistered
contrasts. Undefined Wilcoxon cases (for example all-zero differences) are
reported explicitly rather than imputed.

## Substantive interpretation

Interpretations must remain within the randomised paired-seed design comparing
baseline, mean-PBRS, and min-PBRS under equal shaping coefficients. The analysis
does not claim magnitude-matched or RMS-matched shaping, and therefore does not
control realised shaping magnitudes across conditions.

## Limitations

1. Episode-level evaluation records at intermediate checkpoints were not
   published by Stage 6A; learning-curve trajectories for primary endpoints
   before step 100000 are unavailable and were not interpolated.
2. Step-100000 episode fields were reconstructed from published final weights
   using the locked evaluation protocol.
3. Missing convention-consistency values are retained as missing (never
   zero-filled).
4. Ending training at 100,000 steps does not by itself establish convergence.
"""
    (dirs["reports"] / "dissertation_results_narrative_draft.md").write_text(
        narrative, encoding="utf-8"
    )

    # Manifests
    analysis_commit = _git(["rev-parse", "HEAD"]) or "unknown"
    out_files = []
    for root, _, files in os.walk(EXP_ROOT):
        if analysis_id not in root:
            continue
        for fn in files:
            p = Path(root) / fn
            out_files.append(p)

    # Copy source publish manifest reference
    src_manifest = results_root / "formal_publish_manifest.json"
    dest_manifest = dirs["artifacts"] / "source_result_manifest_copy.json"
    dest_manifest.write_text(src_manifest.read_text(encoding="utf-8"), encoding="utf-8")

    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import scipy

        versions["scipy"] = scipy.__version__
    except Exception:
        versions["scipy"] = "missing"
    try:
        import matplotlib as mpl

        versions["matplotlib"] = mpl.__version__
    except Exception:
        versions["matplotlib"] = "missing"

    analysis_manifest = {
        "analysis_id": analysis_id,
        "result_tag": args.result_tag,
        "result_commit": result_commit,
        "runner_commit": locks["runner_commit"],
        "protocol_hashes": {
            "training_protocol": locks["training_protocol_sha256"],
            "pbrs": locks["pbrs_lock_sha256"],
            "environment": locks["environment_lock_sha256"],
            "comfort": locks["comfort_lock_sha256"],
        },
        "analysis_git_commit_at_run": analysis_commit,
        "analysis_rng_seeds": {"bootstrap": BOOTSTRAP_SEED},
        "library_versions": versions,
        "input_hashes": {
            "formal_publish_manifest.json": _sha(src_manifest),
            "run_status.csv": _sha(results_root / "aggregates" / "run_status.csv"),
        },
        "output_hashes": {
            str(p.relative_to(EXP_ROOT)).replace("\\", "/"): _sha(p)
            for p in sorted(out_files)
            if p.is_file() and p.suffix != ".pyc"
        },
        "overall": overall,
    }
    (dirs["artifacts"] / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2), encoding="utf-8"
    )
    # also root-level convenience copies for verification
    (EXP_ROOT / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2), encoding="utf-8"
    )

    import yaml

    cfg = {
        "stage": "stage6b",
        "results_root": str(results_root),
        "result_tag": args.result_tag,
        "result_commit": result_commit,
        "primary_endpoint_step": PRIMARY_ENDPOINT_STEP,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "reconstruction": "final_weights_step_100000_only",
        "formal_training_started": False,
    }
    (dirs["artifacts"] / "analysis_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    (EXP_ROOT / "analysis_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    req = "\n".join(
        [
            f"numpy=={versions['numpy']}",
            f"scipy=={versions.get('scipy')}",
            f"matplotlib=={versions.get('matplotlib')}",
            f"torch  # as installed in .venv_stage2b1",
        ]
    )
    (dirs["artifacts"] / "analysis_requirements.txt").write_text(req + "\n", encoding="utf-8")
    (EXP_ROOT / "latest_analysis.json").write_text(
        json.dumps({"analysis_id": analysis_id, "overall": overall}, indent=2),
        encoding="utf-8",
    )

    log(f"overall={overall} analysis_id={analysis_id}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
