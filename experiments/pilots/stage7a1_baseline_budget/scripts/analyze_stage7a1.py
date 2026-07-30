#!/usr/bin/env python3
"""Aggregate Stage 7A-1 results: summaries, taxonomy, gate, figures, reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve()
PILOT_ROOT = SCRIPT.parents[1]
REPO_ROOT = SCRIPT.parents[4]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=np.float64)
    if len(vals) == 0:
        return float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        means.append(float(np.mean(sample)))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(REPO_ROOT / "src"))
    os.environ["PYTHONPATH"] = str(REPO_ROOT / "src")

    from thesis.diagnostics.stage7a0_failure_taxonomy import (
        DEFAULT_THRESHOLDS,
        build_failure_taxonomy,
    )
    from thesis.pilots.stage7a1_config import (
        CHECKPOINT_STEPS,
        PILOT_SEEDS,
        PRIMARY_BUDGET_CHECKPOINTS,
        competence_gate_pass,
        select_stable_budget,
    )

    eval_dir = PILOT_ROOT / "output" / "evaluations"
    traj_dir = PILOT_ROOT / "output" / "trajectories"
    stats_dir = PILOT_ROOT / "output" / "statistics"
    fig_dir = PILOT_ROOT / "output" / "figures"
    reports = PILOT_ROOT / "reports"
    for d in (stats_dir, fig_dir, reports):
        d.mkdir(parents=True, exist_ok=True)

    frames = []
    for seed in PILOT_SEEDS:
        p = eval_dir / f"seed_{seed}_episodes.csv"
        if p.is_file():
            frames.append(pd.read_csv(p))
    if not frames:
        print("No evaluation CSVs found", file=sys.stderr)
        return 1
    ep = pd.concat(frames, ignore_index=True)
    ep_path = stats_dir / "baseline_budget_evaluation_episodes.csv"
    ep.to_csv(ep_path, index=False)

    # Seed × checkpoint summary
    seed_rows = []
    for (seed, step), g in ep.groupby(["master_seed", "checkpoint_step"]):
        n = len(g)
        success = g["success"].astype(bool)
        coll = g["collision"].astype(bool)
        trunc = g["truncated"].astype(bool)
        po = g["passing_order"].astype(str)
        seed_rows.append(
            {
                "master_seed": int(seed),
                "checkpoint_step": int(step),
                "n_evaluation_episodes": n,
                "success_rate": float(success.mean()),
                "collision_rate": float(coll.mean()),
                "truncation_rate": float(trunc.mean()),
                "mean_episode_length": float(g["episode_length"].mean()),
                "median_episode_length": float(g["episode_length"].median()),
                "mainline_first_count": int((po == "mainline_first").sum()),
                "ramp_first_count": int((po == "ramp_first").sum()),
                "simultaneous_count": int((po == "simultaneous").sum()),
                "unresolved_count": int((po == "unresolved").sum()),
                "mean_minimum_utility": float(g["minimum_stakeholder_utility"].mean()),
                "mean_stakeholder_utility": float(g["mean_stakeholder_utility"].mean()),
            }
        )
    seed_sum = pd.DataFrame(seed_rows)

    # Taxonomy on primary checkpoints using trajectory logs when available
    tax_eps = []
    for step in PRIMARY_BUDGET_CHECKPOINTS:
        step_frames = []
        for seed in PILOT_SEEDS:
            tp = traj_dir / f"seed_{seed}_traj_step_{step}.csv"
            if tp.is_file():
                step_frames.append(pd.read_csv(tp))
        if not step_frames:
            continue
        steps_df = pd.concat(step_frames, ignore_index=True)
        ep_step = ep[ep["checkpoint_step"] == step].copy()
        tax = build_failure_taxonomy(ep_step, steps_df, thresholds=DEFAULT_THRESHOLDS)
        tax["checkpoint_step"] = step
        tax_eps.append(tax)
    if tax_eps:
        tax_all = pd.concat(tax_eps, ignore_index=True)
        tax_all.to_csv(stats_dir / "baseline_budget_failure_taxonomy_episode.csv", index=False)
        by_ck = (
            tax_all.groupby(["checkpoint_step", "primary_failure_label"])
            .size()
            .reset_index(name="count")
        )
        by_ck.to_csv(stats_dir / "baseline_budget_failure_taxonomy_by_checkpoint.csv", index=False)
        by_seed = (
            tax_all.groupby(["master_seed", "checkpoint_step", "primary_failure_label"])
            .size()
            .reset_index(name="count")
        )
        by_seed.to_csv(stats_dir / "baseline_budget_failure_taxonomy_by_seed.csv", index=False)

        # attach stall rates to seed summary
        for idx, row in seed_sum.iterrows():
            sub = tax_all[
                (tax_all["master_seed"] == row["master_seed"])
                & (tax_all["checkpoint_step"] == row["checkpoint_step"])
            ]
            if sub.empty:
                continue
            n = max(1, len(sub))
            seed_sum.at[idx, "unilateral_stall_rate"] = float(
                (sub["primary_failure_label"] == "unilateral_stall").mean()
            )
            seed_sum.at[idx, "mutual_yielding_rate"] = float(
                (sub["primary_failure_label"] == "mutual_yielding").mean()
            )
            seed_sum.at[idx, "post_exit_stall_rate"] = float(
                (sub["primary_failure_label"] == "post_exit_survivor_stall").mean()
            )

    # Convention / swap eligibility placeholders (block-pair level simplified)
    # Use fraction of success episodes with classifiable order as proxy + swap eligible
    # from paired assignments when both succeed with opposite conventions.
    seed_sum["convention_estimable"] = seed_sum["success_rate"] > 0
    seed_sum["convention_consistency"] = np.nan
    seed_sum["swap_eligible_pairs"] = 0
    seed_sum["D_swap"] = np.nan
    seed_sum["D_swap_estimable"] = False

    for (seed, step), g in ep.groupby(["master_seed", "checkpoint_step"]):
        # eligible pairs: same block, both assignments success and opposite passing orders
        eligible = 0
        estimable = 0
        for bid, bg in g.groupby("validation_block_id"):
            if set(bg["assignment"].tolist()) != {0, 1}:
                continue
            a0 = bg[bg["assignment"] == 0].iloc[0]
            a1 = bg[bg["assignment"] == 1].iloc[0]
            if not (bool(a0["success"]) and bool(a1["success"])):
                continue
            estimable += 1
            po0, po1 = str(a0["passing_order"]), str(a1["passing_order"])
            if {po0, po1} == {"mainline_first", "ramp_first"}:
                eligible += 1
        mask = (seed_sum["master_seed"] == seed) & (seed_sum["checkpoint_step"] == step)
        seed_sum.loc[mask, "swap_eligible_pairs"] = eligible
        seed_sum.loc[mask, "D_swap_estimable"] = estimable > 0
        # 8 blocks max
        seed_sum.loc[mask, "swap_eligible_pair_proportion"] = eligible / 8.0

    seed_sum_path = stats_dir / "baseline_budget_seed_checkpoint_summary.csv"
    seed_sum.to_csv(seed_sum_path, index=False)

    # Condition-level summary
    ck_rows = []
    for step, g in seed_sum.groupby("checkpoint_step"):
        succ = g["success_rate"].to_numpy()
        lo, hi = bootstrap_ci(succ)
        ck_rows.append(
            {
                "checkpoint_step": int(step),
                "n_seeds": int(len(g)),
                "mean_success": float(succ.mean()),
                "median_success": float(np.median(succ)),
                "sd_success": float(succ.std(ddof=1)) if len(succ) > 1 else 0.0,
                "min_success": float(succ.min()),
                "max_success": float(succ.max()),
                "q25_success": float(np.quantile(succ, 0.25)),
                "q75_success": float(np.quantile(succ, 0.75)),
                "bootstrap_CI_success_lower": lo,
                "bootstrap_CI_success_upper": hi,
                "mean_collision": float(g["collision_rate"].mean()),
                "mean_truncation": float(g["truncation_rate"].mean()),
                "seeds_success_ge_0_50": int((g["success_rate"] >= 0.50).sum()),
                "seeds_success_ge_0_60": int((g["success_rate"] >= 0.60).sum()),
                "seeds_success_ge_0_70": int((g["success_rate"] >= 0.70).sum()),
                "seeds_success_ge_0_75": int((g["success_rate"] >= 0.75).sum()),
                "seeds_success_ge_0_80": int((g["success_rate"] >= 0.80).sum()),
                "seeds_success_ge_0_90": int((g["success_rate"] >= 0.90).sum()),
                "swap_eligible_pair_proportion": float(
                    g.get("swap_eligible_pair_proportion", pd.Series([0.0])).mean()
                ),
                "mean_unilateral_stall_rate": float(
                    g["unilateral_stall_rate"].mean()
                )
                if "unilateral_stall_rate" in g
                else float("nan"),
                "mean_mutual_yielding_rate": float(g["mutual_yielding_rate"].mean())
                if "mutual_yielding_rate" in g
                else float("nan"),
            }
        )
    ck_sum = pd.DataFrame(ck_rows).sort_values("checkpoint_step")
    ck_path = stats_dir / "baseline_budget_checkpoint_summary.csv"
    ck_sum.to_csv(ck_path, index=False)

    gate_rows = []
    for _, r in ck_sum.iterrows():
        g = competence_gate_pass(r.to_dict())
        gate_rows.append(
            {
                "checkpoint_step": int(r["checkpoint_step"]),
                "passed": g["passed"],
                **{f"check_{k}": v for k, v in g["checks"].items()},
            }
        )
    gate_df = pd.DataFrame(gate_rows)
    gate_df.to_csv(stats_dir / "baseline_budget_competence_gate.csv", index=False)

    selection = select_stable_budget(ck_sum.to_dict(orient="records"))
    (stats_dir / "budget_selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )

    # Classify budget response
    primary = ck_sum[ck_sum["checkpoint_step"].isin(PRIMARY_BUDGET_CHECKPOINTS)]
    s100 = float(primary.loc[primary["checkpoint_step"] == 100000, "mean_success"].values[0]) if (primary["checkpoint_step"] == 100000).any() else float("nan")
    s300 = float(primary.loc[primary["checkpoint_step"] == 300000, "mean_success"].values[0]) if (primary["checkpoint_step"] == 300000).any() else float("nan")
    low300 = int(
        (
            seed_sum[
                (seed_sum["checkpoint_step"] == 300000)
                & (seed_sum["success_rate"] < 0.5)
            ].shape[0]
        )
    )
    if selection.get("stable_sufficient_budget") is not None:
        classification = "budget-responsive and competence-qualified"
    elif s300 - s100 >= 0.10 and not competence_gate_pass(
        primary[primary["checkpoint_step"] == 300000].iloc[0].to_dict()
    )["passed"]:
        classification = "budget-responsive but not competence-qualified"
    elif low300 >= 6:
        classification = "seed-bifurcated"
    elif abs(s300 - s100) < 0.05:
        classification = "plateau-like"
    else:
        classification = selection.get("status", "plateau-like")
        if classification == "budget extension alone did not establish competence":
            if s300 > s100 + 0.05:
                classification = "budget-responsive but not competence-qualified"
            else:
                classification = "plateau-like"

    # Run completion
    completed = []
    failed = []
    for seed in PILOT_SEEDS:
        cp = PILOT_ROOT / "output" / "runs" / f"seed_{seed}" / "run_completion.json"
        if cp.is_file():
            data = json.loads(cp.read_text(encoding="utf-8"))
            if data.get("success") and int(data.get("final_step", 0)) == 300000:
                completed.append(seed)
            else:
                failed.append(seed)
        else:
            failed.append(seed)

    pipeline = "PASS" if len(completed) == 20 and not failed else ("PARTIAL" if completed else "FAIL")

    # Figures
    if not args.skip_figures:
        import matplotlib.pyplot as plt

        def savefig(name):
            fig_dir.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(fig_dir / name, dpi=150)
            plt.close()

        # spaghetti
        plt.figure(figsize=(8, 5))
        for seed, g in seed_sum.groupby("master_seed"):
            g = g.sort_values("checkpoint_step")
            plt.plot(g["checkpoint_step"], g["success_rate"], alpha=0.5, linewidth=1)
        means = ck_sum.sort_values("checkpoint_step")
        plt.plot(means["checkpoint_step"], means["mean_success"], "k-", linewidth=2, label="mean")
        plt.xlabel("Checkpoint step")
        plt.ylabel("Success rate")
        plt.title("Baseline budget pilot — seed trajectories")
        plt.legend()
        savefig("fig_seed_trajectory_spaghetti.png")

        plt.figure(figsize=(8, 5))
        for _, r in seed_sum.iterrows():
            plt.scatter(r["checkpoint_step"], r["success_rate"], alpha=0.35, s=20)
        plt.plot(means["checkpoint_step"], means["mean_success"], "k-o")
        plt.axhline(0.75, color="gray", linestyle="--")
        plt.title("Success by checkpoint and seed")
        savefig("fig_success_by_checkpoint_and_seed.png")

        plt.figure(figsize=(8, 5))
        plt.errorbar(
            means["checkpoint_step"],
            means["mean_success"],
            yerr=[
                means["mean_success"] - means["bootstrap_CI_success_lower"],
                means["bootstrap_CI_success_upper"] - means["mean_success"],
            ],
            fmt="o-",
            capsize=3,
        )
        plt.axhline(0.75, linestyle="--", color="gray")
        plt.title("Checkpoint success summary (seed bootstrap CI)")
        savefig("fig_success_checkpoint_summary.png")

        for col, name, title in [
            ("mean_collision", "fig_collision_by_checkpoint.png", "Collision by checkpoint"),
            ("mean_truncation", "fig_truncation_by_checkpoint.png", "Truncation by checkpoint"),
            ("seeds_success_ge_0_75", "fig_seeds_meeting_075_gate.png", "Seeds ≥ 0.75"),
            (
                "swap_eligible_pair_proportion",
                "fig_swap_eligibility_by_checkpoint.png",
                "Swap eligibility",
            ),
            (
                "mean_unilateral_stall_rate",
                "fig_unilateral_stall_by_checkpoint.png",
                "Unilateral stall",
            ),
            (
                "mean_mutual_yielding_rate",
                "fig_mutual_yielding_by_checkpoint.png",
                "Mutual yielding",
            ),
        ]:
            if col not in means.columns:
                continue
            plt.figure(figsize=(8, 5))
            plt.plot(means["checkpoint_step"], means[col], "o-")
            plt.title(title + " — Baseline budget pilot")
            savefig(name)

        # gate components
        plt.figure(figsize=(8, 5))
        for col, lab in [
            ("mean_success", "mean success"),
            ("mean_collision", "collision"),
            ("mean_truncation", "truncation"),
            ("swap_eligible_pair_proportion", "swap elig."),
        ]:
            plt.plot(means["checkpoint_step"], means[col], "o-", label=lab)
        plt.legend()
        plt.title("Competence gate components")
        savefig("fig_competence_gate_components.png")

        # taxonomy stacked
        tax_ck = stats_dir / "baseline_budget_failure_taxonomy_by_checkpoint.csv"
        if tax_ck.is_file():
            tdf = pd.read_csv(tax_ck)
            pivot = tdf.pivot_table(
                index="checkpoint_step",
                columns="primary_failure_label",
                values="count",
                fill_value=0,
            )
            pivot.plot(kind="bar", stacked=True, figsize=(10, 5))
            plt.title("Failure taxonomy by checkpoint — Baseline budget pilot")
            plt.ylabel("Count")
            savefig("fig_failure_taxonomy_by_checkpoint.png")

        # historical reference (descriptive only)
        plt.figure(figsize=(6, 4))
        hist = 0.35
        new100 = s100
        plt.bar(["Historical formal 100K\n(seeds 61001–61010)", "Pilot 100K\n(seeds 62001–62020)"], [hist, new100])
        plt.ylabel("Mean success")
        plt.title("100K reference (not pooled)")
        savefig("fig_100k_vs_historical_100k_reference.png")

    # Manifest
    inv = PILOT_ROOT / "output" / "manifests" / "checkpoint_inventory.csv"
    protocol = PILOT_ROOT / "configs" / "stage7a1_baseline_budget_protocol.yaml"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
    manifest = {
        "stage": "Stage 7A-1",
        "name": "Baseline-Only Unchanged-Budget Competence Pilot",
        "analysis_status": "exploratory",
        "condition": "baseline",
        "reward_shaping_enabled": False,
        "formal_experiment_modified": False,
        "paper_files_modified": False,
        "old_formal_seeds_reused": False,
        "master_seeds": list(PILOT_SEEDS),
        "maximum_training_steps": 300000,
        "training_run_count": len(completed),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "evaluation_episode_count": int(len(ep)),
        "statistical_unit": "training_seed",
        "competence_gate": selection,
        "budget_response_classification": classification,
        "pipeline_status": pipeline,
        "completed_seeds": completed,
        "failed_seeds": failed,
        "git_commit": head,
        "input_hashes": {"protocol": _sha(protocol)},
        "output_hashes": {
            _rel(ep_path): _sha(ep_path),
            _rel(seed_sum_path): _sha(seed_sum_path),
            _rel(ck_path): _sha(ck_path),
        },
    }
    man_path = PILOT_ROOT / "output" / "manifests" / "stage7a1_baseline_budget_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Reports
    recommended_budget = selection.get("stable_sufficient_budget")
    next_step = "freeze the selected budget for a new multi-condition pilot" if recommended_budget else (
        "extend Baseline-only pilot to 400K or reconsider algorithm stability"
        if classification.startswith("budget-responsive")
        else "single-axis algorithm-stability pilot (Double DQN candidate)"
    )
    summary = {
        "pipeline_status": pipeline,
        "baseline_competence_status": "PASSED" if recommended_budget else "NOT PASSED",
        "stable_sufficient_budget": recommended_budget,
        "budget_response_classification": classification,
        "checkpoint_summary": ck_sum.to_dict(orient="records"),
        "gate": gate_rows,
        "selection": selection,
        "completed_seeds": completed,
        "failed_seeds": failed,
        "recommended_next_experiment": next_step,
    }
    (reports / "stage7a1_baseline_budget_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (reports / "NEXT_EXPERIMENT_DECISION.md").write_text(
        f"# Next experiment decision\n\nClassification: `{classification}`\n\n"
        f"Recommended next step: {next_step}\n\n"
        f"Stable budget: {recommended_budget}\n\n"
        "Do not start the next experiment in this stage.\n"
        "Keep environment, base reward, and evaluation fixed.\n"
        "Do not reuse seeds 62001–62020 as confirmatory seeds.\n",
        encoding="utf-8",
    )
    (reports / "PAPER_CHANGES_REQUIRED_LATER.md").write_text(
        "# Paper changes required later\n\n"
        "No paper edits were made in Stage 7A-1.\n\n"
        "Later (only if a confirmatory multi-condition study is completed):\n"
        "- Describe Experiment Version 2 budget if frozen.\n"
        "- Do not present this pilot as confirmatory competence evidence.\n"
        "- Keep historical 100K formal results separate from this exploratory pilot.\n",
        encoding="utf-8",
    )

    # Main markdown report
    lines = [
        "# Stage 7A-1 — Baseline-Only Unchanged-Budget Competence Pilot",
        "",
        "## Pipeline status",
        f"- Pilot pipeline status: **{pipeline}**",
        f"- Baseline competence status: **{summary['baseline_competence_status']}**",
        f"- Stable sufficient budget identified: **{recommended_budget}**",
        "",
        "## Protocol integrity",
        "- Condition: baseline",
        "- Seeds: 62001–62020",
        "- Budget: 300000",
        f"- Code commit: `{head}`",
        "- No formal modifications; no paper modifications",
        "",
        "## Run completion",
        f"- Completed: {len(completed)}/20",
        f"- Failed: {failed}",
        "",
        "## Checkpoint outcomes",
    ]
    for _, r in ck_sum.iterrows():
        lines.append(
            f"- {int(r['checkpoint_step'])}: mean_success={r['mean_success']:.4f}, "
            f"collision={r['mean_collision']:.4f}, truncation={r['mean_truncation']:.4f}, "
            f"seeds≥0.75={int(r['seeds_success_ge_0_75'])}, "
            f"swap={r['swap_eligible_pair_proportion']:.3f}"
        )
    lines += [
        "",
        "## Competence gate / budget decision",
        f"- Classification: `{classification}`",
        f"- Selection: `{json.dumps(selection)}`",
        "",
        "## Recommended next experiment",
        next_step,
        "",
    ]
    (reports / "stage7a1_baseline_budget_pilot_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps({"pipeline": pipeline, "classification": classification, "budget": recommended_budget}, indent=2))
    return 0 if pipeline in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
