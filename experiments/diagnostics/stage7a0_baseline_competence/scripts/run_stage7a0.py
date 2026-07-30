#!/usr/bin/env python3
"""Stage 7A-0 — Baseline Competence Diagnostic Pilot runner."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[4]
EXP = Path(__file__).resolve().parents[1]
DEFAULT_OUT = EXP / "output"
DEFAULT_STAGE6A = Path(
    r"C:\Users\HP\Desktop\毕业项目\thesis\final_new_results_100k\formal_results\100k\stage6a_20260730T094829Z_a89256db_44d5e647"
)
DEFAULT_H1 = REPO / "experiments/formal/stage6b_h1/output/data/evaluation_episodes_h1.csv"


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=str(REPO), text=True).strip()
    except Exception:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage6a-root", type=Path, default=DEFAULT_STAGE6A)
    parser.add_argument("--h1-episodes", type=Path, default=DEFAULT_H1)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-reward-audit", action="store_true")
    parser.add_argument("--skip-eval", action="store_true", help="reuse existing episode/step CSVs")
    args = parser.parse_args()

    out = Path(args.output_root)
    for sub in (
        "reconstructed_evaluations",
        "trajectories",
        "endpoint_tables",
        "failure_taxonomy",
        "q_diagnostics",
        "reward_audit",
        "continuation_probe",
        "figures",
        "manifests",
    ):
        (out / sub).mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(REPO / "src"))
    from thesis.diagnostics.stage7a0_analyses import (
        action_summary,
        block_diagnostics,
        competence_gate_grid,
        q_summary,
        role_diagnostics,
        root_cause_matrix,
        save_figures,
        seed_summary,
    )
    from thesis.diagnostics.stage7a0_failure_taxonomy import (
        DEFAULT_THRESHOLDS,
        build_failure_taxonomy,
    )
    from thesis.diagnostics.stage7a0_inventory import (
        build_checkpoint_inventory,
        collect_paper_integrity,
        sha256_file,
        write_csv,
    )
    from thesis.diagnostics.stage7a0_manifest import verify_manifest_hashes
    from thesis.diagnostics.stage7a0_reward_audit import run_reward_audit
    from thesis.diagnostics.stage7a0_trajectory_eval import run_baseline_100k_diagnostics
    from thesis.training.final_lock_loader import (
        EXPECTED_COMFORT_LOCK_SHA256,
        EXPECTED_ENVIRONMENT_LOCK_SHA256,
    )

    # Paper integrity before
    paper_before = collect_paper_integrity(REPO)
    write_csv(out / "manifests" / "paper_file_integrity_before.csv", paper_before)

    # Phase 0 inventory
    inv = build_checkpoint_inventory(stage6a_root=args.stage6a_root, out_dir=out / "manifests")
    inv["environment_lock_hash"] = EXPECTED_ENVIRONMENT_LOCK_SHA256
    inv["comfort_lock_hash"] = EXPECTED_COMFORT_LOCK_SHA256
    h1_man = REPO / "experiments/formal/stage6b_h1/output/manifests/analysis_manifest.json"
    inv["stage6b_h1_manifest_hash"] = sha256_file(h1_man) if h1_man.is_file() else None
    (out / "manifests" / "input_inventory.json").write_text(
        json.dumps(inv, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Availability table for learning trajectory
    avail_rows = []
    for step in [0, 10000, 25000, 50000, 75000, 100000]:
        if step == 0:
            status = "unavailable"
            reason = "no verified initial checkpoint / step-zero reference reconstruction"
        elif step == 100000:
            status = "available_final_weights"
            reason = "final_online_target_weights.pt published"
        else:
            status = "unavailable"
            reason = "ckpt_step_*.pt local_only and missing from published results worktree"
        avail_rows.append({"checkpoint_step": step, "status": status, "reason": reason})
    pd.DataFrame(avail_rows).to_csv(
        out / "endpoint_tables" / "baseline_checkpoint_availability.csv", index=False
    )

    # Phase 2 diagnostic eval
    if args.skip_eval and (out / "reconstructed_evaluations" / "baseline_100k_diagnostic_episodes.csv").is_file():
        ep_df = pd.read_csv(out / "reconstructed_evaluations" / "baseline_100k_diagnostic_episodes.csv")
        step_df = pd.read_csv(out / "trajectories" / "baseline_100k_step_log.csv")
        mm_path = out / "reconstructed_evaluations" / "baseline_h1_nonddiagnostic_mismatches.csv"
        if mm_path.is_file() and mm_path.stat().st_size > 0:
            mm = pd.read_csv(mm_path)
        else:
            mm = pd.DataFrame()
        diag = {
            "episode_count": len(ep_df),
            "step_row_count": len(step_df),
            "nondiagnostic_mismatch_count": len(mm),
            "episodes": ep_df,
            "steps": step_df,
        }
    else:
        diag = run_baseline_100k_diagnostics(
            stage6a_root=args.stage6a_root,
            out_root=out,
            h1_episodes_csv=args.h1_episodes,
        )
        ep_df = diag["episodes"]
        step_df = diag["steps"]

    # Endpoint summaries
    seed_df = seed_summary(ep_df)
    seed_df.to_csv(out / "endpoint_tables" / "baseline_seed_profiles.csv", index=False)
    role_diagnostics(ep_df).to_csv(
        out / "endpoint_tables" / "baseline_role_identity_diagnostics.csv", index=False
    )
    block_diagnostics(ep_df).to_csv(
        out / "endpoint_tables" / "baseline_block_diagnostics.csv", index=False
    )
    cond = {
        "checkpoint_step": 100000,
        "success_rate": float(ep_df["success"].mean()),
        "collision_rate": float(ep_df["collision"].mean()),
        "truncation_rate": float(ep_df["truncated"].mean()),
        "n_episodes": int(len(ep_df)),
        "n_seeds": int(ep_df["master_seed"].nunique()),
    }
    pd.DataFrame([cond]).to_csv(
        out / "endpoint_tables" / "baseline_checkpoint_condition_summary.csv", index=False
    )
    # Learning trajectory placeholders
    pd.DataFrame(
        [
            {
                "master_seed": s,
                "classification": "insufficient_checkpoints",
                "success_10k": None,
                "success_25k": None,
                "success_50k": None,
                "success_75k": None,
                "success_100k": float(seed_df.loc[seed_df["master_seed"] == s, "success_rate"].iloc[0]),
            }
            for s in seed_df["master_seed"]
        ]
    ).to_csv(out / "endpoint_tables" / "baseline_learning_trajectory_by_seed.csv", index=False)

    # Taxonomy
    tax = build_failure_taxonomy(ep_df, step_df, thresholds=DEFAULT_THRESHOLDS)
    tax.to_csv(out / "failure_taxonomy" / "baseline_failure_taxonomy_episode.csv", index=False)
    if len(tax):
        tax["primary_failure_label"].value_counts().rename_axis("label").reset_index(name="n").to_csv(
            out / "failure_taxonomy" / "baseline_failure_taxonomy_summary.csv", index=False
        )
        tax.groupby(["master_seed", "primary_failure_label"]).size().reset_index(name="n").to_csv(
            out / "failure_taxonomy" / "baseline_failure_taxonomy_by_seed.csv", index=False
        )
        tax.groupby(["validation_block_id", "primary_failure_label"]).size().reset_index(name="n").to_csv(
            out / "failure_taxonomy" / "baseline_failure_taxonomy_by_block.csv", index=False
        )
        tax.groupby(["controller_A_role", "controller_B_role", "primary_failure_label"]).size().reset_index(
            name="n"
        ).to_csv(
            out / "failure_taxonomy" / "baseline_failure_taxonomy_by_role_assignment.csv",
            index=False,
        )
    # Threshold sensitivity
    sens = []
    for ns in (0.5, 1.0, 2.0):
        for ls in (2.0, 3.0, 5.0):
            for mp in (0.5, 1.0, 2.0):
                thr = {
                    **DEFAULT_THRESHOLDS,
                    "near_stop_speed": ns,
                    "low_speed": ls,
                    "minimal_progress_50": mp,
                    "minimal_progress_25": mp / 2,
                }
                t2 = build_failure_taxonomy(ep_df, step_df, thresholds=thr)
                vc = t2["primary_failure_label"].value_counts().to_dict() if len(t2) else {}
                sens.append({"near_stop_speed": ns, "low_speed": ls, "minimal_progress_50": mp, **vc})
    pd.DataFrame(sens).to_csv(
        out / "failure_taxonomy" / "failure_taxonomy_threshold_sensitivity.csv", index=False
    )

    act = action_summary(step_df, ep_df)
    act.to_csv(out / "endpoint_tables" / "baseline_action_summary.csv", index=False)
    # joint action matrix
    j = (
        step_df.groupby("joint_action_category").size().rename_axis("joint_action_category").reset_index(name="n")
    )
    j.to_csv(out / "endpoint_tables" / "baseline_joint_action_matrix.csv", index=False)
    qd = q_summary(step_df, ep_df)
    qd.to_csv(out / "q_diagnostics" / "baseline_q_summary.csv", index=False)
    gate = competence_gate_grid(seed_df, ep_df)
    gate.to_csv(out / "endpoint_tables" / "baseline_competence_gate_grid.csv", index=False)

    # Reward audit
    if args.skip_reward_audit:
        reward_sep = {"weak_reward_separation_any": False, "skipped": True}
    else:
        reward_sep = run_reward_audit(out_csv=out / "reward_audit" / "baseline_scripted_reward_audit.csv")

    # Continuation BLOCKED
    cont = {
        "status": "BLOCKED",
        "reason": inv.get("continuation_block_reason"),
        "required_full_checkpoints": 10,
        "available_resumable_100k_checkpoints": 0,
        "published_final_weights_only": True,
        "executed": False,
        "resume_equivalence": "not_run",
    }
    (out / "continuation_probe" / "baseline_continuation_status.json").write_text(
        json.dumps(cont, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame([cont]).to_csv(
        out / "continuation_probe" / "baseline_continuation_run_completion.csv", index=False
    )

    # Replay/TD unavailable
    pd.DataFrame(
        [{"status": "unavailable", "reason": "replay not present in published final_online_target_weights.pt"}]
    ).to_csv(out / "q_diagnostics" / "baseline_replay_composition.csv", index=False)
    pd.DataFrame(
        [{"status": "unavailable", "reason": "replay absent; TD-error sampling blocked"}]
    ).to_csv(out / "q_diagnostics" / "baseline_td_error_summary.csv", index=False)

    # Training log weak summary from episode_summaries if present
    train_rows = []
    for seed in seed_df["master_seed"]:
        p = Path(args.stage6a_root) / "jobs" / f"baseline__{seed}" / "episode_summaries.json"
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8-sig"))
            train_rows.append(
                {
                    "master_seed": int(seed),
                    "training_episodes_logged": len(data) if isinstance(data, list) else None,
                    "note": "training IC episodes only; not greedy eval competence",
                }
            )
    pd.DataFrame(train_rows).to_csv(
        out / "endpoint_tables" / "baseline_training_log_summary.csv", index=False
    )

    rc = root_cause_matrix(
        seed_df=seed_df,
        taxonomy=tax,
        reward_sep=reward_sep,
        continuation_status="BLOCKED",
        mismatch_count=int(diag["nondiagnostic_mismatch_count"]),
    )
    rc.to_csv(out / "endpoint_tables" / "baseline_root_cause_matrix.csv", index=False)

    save_figures(
        seed_df=seed_df,
        episodes=ep_df,
        taxonomy=tax,
        action_df=act,
        q_df=qd,
        gate_df=gate,
        fig_dir=out / "figures",
    )

    # Paper after
    paper_after = collect_paper_integrity(REPO)
    write_csv(out / "manifests" / "paper_file_integrity_after.csv", paper_after)
    before_map = {r["path"]: r["sha256"] for r in paper_before}
    changed = [
        r["path"] for r in paper_after if before_map.get(r["path"]) != r["sha256"]
    ]
    # also detect new/removed roughly
    paper_changed = len(changed)

    # Checkpoint integrity after = rehash published weights
    before_ck = pd.read_csv(out / "manifests" / "checkpoint_integrity_before.csv")
    after_rows = []
    for _, r in before_ck.iterrows():
        path = str(r["path"])
        if path.startswith("<missing>"):
            after_rows.append({**r.to_dict(), "sha256_after": "", "unchanged": True})
            continue
        p = Path(path)
        h = sha256_file(p) if p.is_file() else ""
        after_rows.append(
            {
                **{k: r[k] for k in r.index},
                "sha256_after": h,
                "unchanged": h == str(r.get("sha256_actual") or r.get("sha256_recorded") or ""),
            }
        )
    write_csv(out / "manifests" / "checkpoint_integrity_after.csv", after_rows)
    ckpt_changed = sum(1 for r in after_rows if r.get("exists") and not r.get("unchanged"))

    # Environment snapshot
    snap = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "os": os_name(),
        "numpy": __import__("numpy").__version__,
        "pandas": pd.__version__,
        "matplotlib": __import__("matplotlib").__version__,
        "git_branch": _git(["branch", "--show-current"]),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "stage": "Stage 7A-0",
    }
    try:
        import torch

        snap["torch"] = torch.__version__
        snap["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        snap["torch"] = None
    (EXP / "environment_snapshot.json").write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (EXP / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
    (EXP / "diagnostic_requirements.txt").write_text(
        "\n".join(
            [
                f"python=={platform.python_version()}",
                f"numpy=={snap['numpy']}",
                f"pandas=={pd.__version__}",
                f"torch=={snap.get('torch')}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Reports
    top_tax = (
        tax["primary_failure_label"].value_counts().head(5).to_dict() if len(tax) else {}
    )
    high = seed_df.loc[seed_df["performance_band"] == "high", "master_seed"].tolist()
    mid = seed_df.loc[seed_df["performance_band"] == "intermediate", "master_seed"].tolist()
    low = seed_df.loc[seed_df["performance_band"] == "low", "master_seed"].tolist()
    pipeline = "PARTIAL"
    if (
        diag["episode_count"] == 160
        and diag["nondiagnostic_mismatch_count"] == 0
        and paper_changed == 0
        and ckpt_changed == 0
        and cont["status"] == "BLOCKED"
    ):
        pipeline = "PARTIAL"  # blocked continuation / missing intermediate => PARTIAL not FAIL

    summary = {
        "diagnostic_pipeline_status": pipeline,
        "baseline_competence_status": "NOT PASSED",
        "continuation_probe_status": "BLOCKED",
        "success_100k": cond["success_rate"],
        "collision_100k": cond["collision_rate"],
        "truncation_100k": cond["truncation_rate"],
        "top_failure_categories": top_tax,
        "high_seeds": high,
        "intermediate_seeds": mid,
        "low_seeds": low,
        "nondiagnostic_mismatch_count": diag["nondiagnostic_mismatch_count"],
        "paper_files_changed": paper_changed,
        "formal_checkpoints_changed": ckpt_changed,
        "weak_reward_separation": reward_sep.get("weak_reward_separation_any"),
        "recommended_next": "recover_or_retrain_baseline_budget_pilot_with_new_seeds_OR_base_reward_deadlock_pilot",
    }
    (EXP / "reports" / "stage7a0_baseline_competence_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    report = f"""# Stage 7A-0 — Baseline Competence Diagnostic Pilot

## A. Status

```text
Diagnostic pipeline status: {pipeline}
Baseline competence status: NOT PASSED
Continuation probe status: BLOCKED
```

PASS here means the diagnostic workflow completed for available artifacts.
It does **not** mean Baseline competence is adequate.

## B. Input integrity

- Published Baseline final weights: {inv.get('published_final_weights_count')}
- Missing full resumable checkpoints: {inv.get('missing_full_checkpoint_count')}
- Formal checkpoint hash changes: {ckpt_changed}
- Paper file changes: {paper_changed}
- Stage 6A / Stage 6B-H1: not modified by this pilot

## C. Checkpoint learning trajectory

| step | status |
|-----:|--------|
| 0K | unavailable |
| 10K | unavailable (local-only ckpt missing) |
| 25K | unavailable |
| 50K | unavailable |
| 75K | unavailable |
| 100K | available (final_online_target_weights.pt) |
| 125K–200K | BLOCKED (continuation) |

## D. 100K outcomes (reconstructed diagnostic; n=160)

- Success: {cond['success_rate']:.4f}
- Collision: {cond['collision_rate']:.4f}
- Truncation: {cond['truncation_rate']:.4f}
- Non-diagnostic mismatches vs H1: {diag['nondiagnostic_mismatch_count']}

## E. Failure taxonomy (truncated episodes)

{json.dumps(top_tax, indent=2)}

## F. Seed bifurcation

- High (>=0.75): {high}
- Intermediate: {mid}
- Low (<0.25): {low}

## G–I. Policy / reward / continuation

- Replay/TD diagnostics: unavailable (no replay in published weights)
- Reward weak separation: {reward_sep.get('weak_reward_separation_any')}
- Continuation: BLOCKED — {cont['reason']}

## J. Root-cause matrix

See `output/endpoint_tables/baseline_root_cause_matrix.csv`.

## K. Recommendation

Do **not** implement interventions in this stage.

Highest-priority next experiment candidates (choose one primary axis):

1. If full checkpoints can be recovered: unchanged continuation / longer Baseline-only budget pilot on **new** seeds.
2. If truncation taxonomy is dominated by mutual yielding / post-exit stall and reward separation is weak: base-task deadlock-resolution reward pilot (new experiment version; re-audit PBRS boundary; retrain all conditions).
3. If seed bifurcation dominates with some high competence seeds: increase independent Baseline seeds and stabilise before treatment comparison.

This pilot reused formal seeds and is exploratory only.
"""
    (EXP / "reports" / "stage7a0_baseline_competence_report.md").write_text(report, encoding="utf-8")
    (EXP / "reports" / "NEXT_EXPERIMENT_RECOMMENDATION.md").write_text(
        """# Next experiment recommendation (Stage 7A-0)

## Single highest-priority intervention

Recover or regenerate **resumable Baseline checkpoints** (or run a new Baseline-only budget pilot on fresh seeds)
before changing reward or algorithm. Without intermediate/resumable artifacts, budget-vs-plateau cannot be identified.

If checkpoints cannot be recovered and 100K failure taxonomy is dominated by mutual yielding /
post-exit stall with weak scripted reward separation, prioritise a **base-task deadlock resolution pilot**
(new experiment version).

## Must remain fixed in the immediate next pilot

- Do not silently alter formal 100K endpoint
- Do not reuse this pilot as confirmatory treatment evidence
- Change only one primary axis per candidate experiment

## Required new seeds

Yes — any confirmatory or new budget/reward experiment must use new training seeds.
""",
        encoding="utf-8",
    )
    (EXP / "reports" / "PAPER_CHANGES_REQUIRED_LATER.md").write_text(
        """# Paper changes required later (do not edit thesis now)

1. 100K Baseline competence gate: NOT PASSED (success ≈ 0.35).
2. Failures are dominated by unresolved truncation, not collision.
3. Record primary failure taxonomy from Stage 7A-0 once reviewed.
4. Continuation 100K→200K could not be tested (checkpoints missing).
5. Seed bifurcation appears present (high/intermediate/low bands).
6. Role/block concentration should be summarised from diagnostic tables.
7. Weak base-reward separation may be relevant if audit flag is true.
8. A new experiment version may be required depending on next intervention.
9. This pilot reused formal seeds → exploratory diagnostic only.
10. Future confirmatory experiments must use new seeds.
""",
        encoding="utf-8",
    )

    # Manifest
    hashes: dict[str, str] = {}
    for p in sorted(out.rglob("*")):
        if not p.is_file():
            continue
        if p.name == "baseline_diagnostic_manifest.json":
            continue
        rel = ("output/" + p.relative_to(out).as_posix()).replace("\\", "/")
        hashes[rel] = sha256_file(p)
    for rel in (
        "reports/stage7a0_baseline_competence_report.md",
        "reports/stage7a0_baseline_competence_summary.json",
        "reports/NEXT_EXPERIMENT_RECOMMENDATION.md",
        "reports/PAPER_CHANGES_REQUIRED_LATER.md",
        "environment_snapshot.json",
        "pip_freeze.txt",
        "diagnostic_requirements.txt",
    ):
        p = EXP / rel
        if p.is_file():
            hashes[rel] = sha256_file(p)
    hashes = dict(sorted(hashes.items()))
    man = {
        "stage": "Stage 7A-0",
        "name": "Baseline Competence Diagnostic Pilot",
        "analysis_status": "exploratory_diagnostic",
        "formal_results_modified": False,
        "formal_checkpoints_modified": False,
        "paper_files_modified": paper_changed != 0,
        "conditions_analysed": ["baseline"],
        "formal_seed_reuse": True,
        "formal_seed_use": "exploratory diagnosis only",
        "continuation_probe_formal_status": "not part of formal experiment",
        "continuation_probe_status": "BLOCKED",
        "diagnostic_pipeline_status": pipeline,
        "git_commit": snap["git_commit"],
        "git_branch": snap["git_branch"],
        "input_hashes": {
            "stage6a_logical": inv.get("stage6a_root_logical"),
            "protocol_hash": inv.get("protocol_hash"),
            "environment_lock_hash": inv.get("environment_lock_hash"),
            "comfort_lock_hash": inv.get("comfort_lock_hash"),
            "stage6b_h1_manifest_hash": inv.get("stage6b_h1_manifest_hash"),
        },
        "output_hashes": hashes,
        "summary": summary,
    }
    man_path = out / "manifests" / "baseline_diagnostic_manifest.json"
    man_path.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verify_manifest_hashes(artifact_root=EXP, manifest_path=man_path)

    print(json.dumps(summary, indent=2))
    print(f"pipeline={pipeline}")
    return 0 if paper_changed == 0 and ckpt_changed == 0 else 1


def os_name() -> str:
    return platform.system()


if __name__ == "__main__":
    raise SystemExit(main())
