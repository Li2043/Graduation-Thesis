#!/usr/bin/env python3
"""Stage 6C — publication-quality formal result figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

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


def _git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *cmd], cwd=str(REPO_ROOT), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def save_figure(fig, out_stem: Path, *, formats: list[str], png_dpi: int) -> dict[str, str]:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = {}
    for fmt in formats:
        path = out_stem.with_suffix("." + fmt)
        kw: dict[str, Any] = {"bbox_inches": "tight", "pad_inches": 0.03, "facecolor": "white"}
        if fmt == "png":
            kw["dpi"] = int(png_dpi)
        fig.savefig(path, format=fmt, **kw)
        paths[fmt] = str(path)
    return paths


def write_alt(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_meta(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def contact_sheet(png_paths: list[Path], out_path: Path, *, cols: int = 2) -> None:
    if not png_paths:
        return
    n = len(png_paths)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.8, 3.2 * rows))
    axes_list = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax in axes_list:
        ax.axis("off")
    for ax, p in zip(axes_list, png_paths):
        img = plt.imread(p)
        ax.imshow(img)
        ax.set_title(p.stem, fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-tag", default="formal-analysis-100k-complete")
    parser.add_argument("--analysis-worktree", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=EXP_ROOT)
    parser.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"])
    parser.add_argument("--png-dpi", type=int, default=600)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "src"))
    os.environ["PYTHONPATH"] = str(REPO_ROOT / "src")

    from thesis.figures.figure_data_validation import (
        EXPECTED_ANALYSIS_COMMIT,
        EXPECTED_ANALYSIS_ID,
        EXPECTED_MISSING_CONVENTION,
        build_resolved_figure_inputs,
        discover_stage6b_paths,
        validate_contrasts,
        validate_primary_seed_table,
        verify_analysis_source,
        FigureDataBlockedError,
    )
    from thesis.figures.formal_result_figures import (
        fig_collision_type_composition,
        fig_convention_selection_and_consistency,
        fig_primary_endpoint_paired_contrasts,
        fig_primary_endpoint_seed_distributions,
        fig_run_completion_and_integrity,
        fig_safety_comfort_diagnostics,
        fig_seed_level_primary_endpoint_matrix,
        fig_stakeholder_utility_by_role,
        fig_worst_off_stakeholder,
    )
    from thesis.figures.publication_style import (
        apply_publication_style,
        style_manifest_dict,
    )

    out_root = Path(args.output_root).resolve()
    run_id = f"stage6c_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{EXPECTED_ANALYSIS_COMMIT[:8]}"
    dirs = {
        "main": out_root / "figures" / "main",
        "supp": out_root / "figures" / "supplementary",
        "data": out_root / "figure_data",
        "captions": out_root / "captions",
        "reports": out_root / "reports" / run_id,
        "logs": out_root / "logs" / run_id,
        "artifacts": out_root / "artifacts" / run_id,
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    log_path = dirs["logs"] / "runner.log"

    def log(msg: str) -> None:
        line = f"[{_utc()}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    # Verify immutable source worktree commit
    src_commit = _git(["-C", str(Path(args.analysis_worktree).resolve()), "rev-list", "-n", "1", "HEAD"])
    if not src_commit:
        # fallback
        try:
            src_commit = subprocess.check_output(
                ["git", "-C", str(Path(args.analysis_worktree).resolve()), "rev-list", "-n", "1", "HEAD"],
                text=True,
            ).strip()
        except Exception as exc:
            log(f"BLOCKED: cannot read worktree commit: {exc}")
            return 2
    if src_commit != EXPECTED_ANALYSIS_COMMIT:
        log(f"BLOCKED: analysis worktree commit {src_commit} != {EXPECTED_ANALYSIS_COMMIT}")
        return 2

    try:
        source = verify_analysis_source(Path(args.analysis_worktree))
        paths = {k: Path(v) for k, v in source["paths"].items() if k != "summary"}
        # rediscover typed paths
        paths = discover_stage6b_paths(Path(args.analysis_worktree), EXPECTED_ANALYSIS_ID)
    except FigureDataBlockedError as exc:
        log(f"BLOCKED: {exc}")
        return 2

    resolved = build_resolved_figure_inputs(paths)
    resolved_text = json.dumps(resolved, indent=2)
    (dirs["artifacts"] / "resolved_figure_inputs.json").write_text(resolved_text, encoding="utf-8")
    (out_root / "resolved_figure_inputs.json").write_text(resolved_text, encoding="utf-8")
    log("source inputs resolved and hashed")

    seed_values = pd.read_csv(paths["primary_endpoint_seed_values"])
    contrasts = pd.read_csv(paths["primary_endpoint_contrasts"])
    bootstrap = pd.read_csv(paths["bootstrap_intervals"])
    holm = pd.read_csv(paths["holm_adjusted_results"])
    effects = pd.read_csv(paths["effect_sizes"])
    convention = pd.read_csv(paths["convention_availability"])
    secondary = pd.read_csv(paths["secondary_endpoints"])
    episodes = pd.read_csv(paths["evaluation_episode_validated"])
    accounting = pd.read_csv(paths["run_accounting"])
    integrity = pd.read_csv(paths["integrity_summary"])
    seed_ck = pd.read_csv(paths["seed_checkpoint_endpoints"])
    auc = pd.read_csv(paths["learning_curve_auc"])

    try:
        validate_primary_seed_table(seed_values)
        validate_contrasts(contrasts, bootstrap, holm, effects)
    except FigureDataBlockedError as exc:
        log(f"BLOCKED validation: {exc}")
        return 2

    style = apply_publication_style()
    log(f"matplotlib={style.matplotlib_version} font={style.resolved_font}")

    exclusions: list[dict[str, str]] = []
    # Learning curve Case C
    ckpts = sorted(int(x) for x in seed_ck["checkpoint_step"].dropna().unique())
    if ckpts != [100000] or int(auc["n_available_checkpoints"].max()) <= 1:
        exclusions.append(
            {
                "figure": "fig_4_4_learning_curve_summaries",
                "reason": (
                    "Case C: only the 100000-step checkpoint endpoint summaries are available; "
                    "intermediate episode-level checkpoint payloads were not published. "
                    "No interpolation performed."
                ),
            }
        )
    if int(auc["auc"].notna().sum()) == 0:
        exclusions.append(
            {
                "figure": "fig_s2_learning_curve_auc",
                "reason": "All learning-curve AUC values are missing (single available checkpoint).",
            }
        )

    captions: list[str] = ["# Publication Figure Captions (Stage 6C)\n"]
    index_rows: list[dict[str, Any]] = []
    output_manifest: dict[str, Any] = {"figures": {}}
    main_pngs: list[Path] = []
    supp_pngs: list[Path] = []
    integrity_counts = {
        "source_hash_mismatches": 0,
        "validation_failures": 0,
        "excluded_figures": len(exclusions),
        "training_invoked": 0,
        "evaluation_invoked": 0,
    }

    def register(
        name: str,
        *,
        kind: str,
        fig,
        plotted: pd.DataFrame,
        alt: str,
        caption: str,
        meta_extra: dict[str, Any] | None = None,
    ) -> None:
        dest_dir = dirs["main"] if kind == "main" else dirs["supp"]
        stem = dest_dir / name
        paths_out = save_figure(fig, stem, formats=list(args.formats), png_dpi=int(args.png_dpi))
        plt.close(fig)
        csv_path = dirs["data"] / f"{name}.csv"
        plotted.to_csv(csv_path, index=False)
        meta = {
            "figure": name,
            "kind": kind,
            "input_sources": {
                k: {"path": str(paths[k]), "sha256": resolved["files"][k]["sha256"]}
                for k in resolved["files"]
                if k in paths
            },
            "statistical_unit": "formal_training_seed",
            "primary_checkpoint_step": 100000,
            "matplotlib_version": style.matplotlib_version,
            "resolved_font": style.resolved_font,
            "png_dpi": int(args.png_dpi),
            "formats": paths_out,
            "row_count_plotted": int(len(plotted)),
            **(meta_extra or {}),
        }
        write_meta(dirs["data"] / f"{name}_metadata.json", meta)
        write_alt(dirs["captions"] / f"{name}_alt_text.txt", alt)
        captions.append(caption + "\n")
        index_rows.append(
            {
                "figure": name,
                "kind": kind,
                "pdf": paths_out.get("pdf", ""),
                "svg": paths_out.get("svg", ""),
                "png": paths_out.get("png", ""),
                "companion_csv": str(csv_path),
            }
        )
        output_manifest["figures"][name] = {
            **paths_out,
            "sha256_png": _sha(Path(paths_out["png"])) if "png" in paths_out else "",
            "sha256_pdf": _sha(Path(paths_out["pdf"])) if "pdf" in paths_out else "",
        }
        if kind == "main":
            main_pngs.append(Path(paths_out["png"]))
        else:
            supp_pngs.append(Path(paths_out["png"]))
        log(f"wrote {name}")

    # Figure 4.1
    fig, _, plotted = fig_primary_endpoint_seed_distributions(seed_values)
    register(
        "fig_4_1_primary_endpoint_seed_distributions",
        kind="main",
        fig=fig,
        plotted=plotted,
        alt=(
            "Four-panel figure of seed-level primary outcomes at 100000 steps for Baseline, "
            "Mean-PBRS and Min-PBRS. Points are formal training seeds; thin lines connect identical seeds."
        ),
        caption=(
            "### Figure 4.1. Seed-level primary outcomes at the preregistered 100,000-step endpoint.\n\n"
            "Each point is one formal training seed aggregated over the locked 16-episode evaluation set. "
            "Thin lines connect identical seeds across reward conditions. Filled larger markers show "
            "condition means. Axes are fixed to [0, 1]. Statistical unit: formal training seed (n=10). "
            "Source: Stage 6B `primary_endpoint_seed_values.csv`."
        ),
        meta_extra={"uncertainty_method": "none_beyond_raw_seed_dispersion", "n_seeds": 10},
    )

    # Figure 4.2
    fig, _, plotted = fig_primary_endpoint_paired_contrasts(contrasts)
    register(
        "fig_4_2_primary_endpoint_paired_contrasts",
        kind="main",
        fig=fig,
        plotted=plotted,
        alt=(
            "Forest plot of paired seed-level mean differences with 95 percent percentile-bootstrap "
            "intervals for four primary endpoints and three fixed contrasts."
        ),
        caption=(
            "### Figure 4.2. Paired contrasts at the preregistered 100,000-step endpoint.\n\n"
            "Markers show mean paired seed-level differences with Stage 6B 95% paired percentile-bootstrap "
            "intervals (10,000 replicates; seed 91001). Annotations report complete paired n, Holm-adjusted "
            "Wilcoxon p-values, and paired Cohen's dz. For success and utility, positive values favour the "
            "first-named condition; for collision rate, negative values favour the first-named condition. "
            "Zero reference lines are shown. Source: `primary_endpoint_contrasts.csv`."
        ),
        meta_extra={"uncertainty_method": "stage6b_percentile_bootstrap_ci", "holm": True},
    )

    # Figure 4.3
    fig, _, plotted = fig_convention_selection_and_consistency(episodes, convention, seed_ck)
    if convention["convention_missing"].dtype == bool:
        miss = int(convention["convention_missing"].sum())
    else:
        miss = int(
            convention["convention_missing"]
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
            .sum()
        )
    if miss != EXPECTED_MISSING_CONVENTION:
        log(f"FAIL convention missing {miss}")
        integrity_counts["validation_failures"] += 1
    register(
        "fig_4_3_convention_selection_and_consistency",
        kind="main",
        fig=fig,
        plotted=plotted,
        alt=(
            "Three-panel convention figure: episode composition, available seed-level consistency points, "
            "and missingness counts totalling eleven condition-by-seed cells."
        ),
        caption=(
            "### Figure 4.3. Convention selection and consistency.\n\n"
            "Panel (a) shows descriptive episode shares within the locked 16-episode evaluation sets. "
            "Panel (b) shows only observed seed-level convention consistency (missing values not plotted at zero). "
            "Panel (c) summarises availability and missingness. Convention consistency is undefined when no unique "
            "non-simultaneous modal convention exists; missing values were not imputed. Expected missing "
            f"condition×seed cells: {EXPECTED_MISSING_CONVENTION} (observed {miss})."
        ),
        meta_extra={"missing_convention_cells": miss, "zero_fill": False},
    )

    # Figure 4.5
    fig, _, plotted = fig_stakeholder_utility_by_role(secondary)
    register(
        "fig_4_5_stakeholder_utility_by_role",
        kind="main",
        fig=fig,
        plotted=plotted,
        alt="Four-panel secondary figure of seed-level utilities for stakeholders A, B, B_front and B_rear.",
        caption=(
            "### Figure 4.5. Per-stakeholder utilities at the 100,000-step endpoint (secondary).\n\n"
            "Each point is a formal training seed. Thin lines connect identical seeds across conditions. "
            "This figure reports secondary endpoints and does not alter primary endpoint classification."
        ),
        meta_extra={"endpoint_class": "secondary"},
    )

    # Figure 4.6
    fig, _, plotted = fig_safety_comfort_diagnostics(episodes)
    register(
        "fig_4_6_safety_comfort_diagnostics",
        kind="main",
        fig=fig,
        plotted=plotted,
        alt=(
            "Four-panel secondary diagnostics of seed-aggregated minimum bumper gap, minimum TTC, "
            "hard-braking rate and maximum background braking."
        ),
        caption=(
            "### Figure 4.6. Safety and comfort diagnostics (secondary).\n\n"
            "Seed-level aggregates derived from Stage 6B validated evaluation episode records "
            "(not episode-level inferential intervals). Preferable directions are annotated in each panel. "
            "No dual y-axes are used."
        ),
        meta_extra={"endpoint_class": "secondary", "aggregation": "episode_to_seed_from_stage6b_table"},
    )

    # Supplementary
    fig, _, plotted = fig_run_completion_and_integrity(accounting, integrity)
    register(
        "fig_s1_run_completion_and_integrity",
        kind="supplementary",
        fig=fig,
        plotted=plotted,
        alt="Bar chart of formal run terminal statuses.",
        caption="### Figure S1. Formal run completion status counts from Stage 6B run accounting.",
    )

    fig, _, plotted = fig_worst_off_stakeholder(episodes)
    register(
        "fig_s3_worst_off_stakeholder_identity",
        kind="supplementary",
        fig=fig,
        plotted=plotted,
        alt="Grouped bars of modal worst-off stakeholder identity shares by condition.",
        caption="### Figure S3. Modal worst-off stakeholder identity by condition (descriptive secondary summary).",
    )

    fig, _, plotted = fig_collision_type_composition(episodes)
    register(
        "fig_s4_collision_type_composition",
        kind="supplementary",
        fig=fig,
        plotted=plotted,
        alt="Stacked bars of collision-type composition among collision episodes by condition.",
        caption="### Figure S4. Collision-type composition among collision episodes (descriptive).",
    )

    fig, _, plotted = fig_seed_level_primary_endpoint_matrix(seed_values)
    register(
        "fig_s5_seed_level_primary_endpoint_matrix",
        kind="supplementary",
        fig=fig,
        plotted=plotted,
        alt="Heatmap of within-endpoint z-scores for seed-level primary non-convention endpoints.",
        caption=(
            "### Figure S5. Seed-level primary endpoint matrix (descriptive z-scores).\n\n"
            "Not for inferential claims. Companion CSV retains raw values."
        ),
        meta_extra={"inferential": False},
    )

    # Contact sheets
    contact_sheet(main_pngs, dirs["main"] / "figure_contact_sheet_main.png")
    contact_sheet(supp_pngs, dirs["supp"] / "figure_contact_sheet_supplementary.png")

    # Captions and reports
    (dirs["captions"] / "publication_figure_captions.md").write_text(
        "\n".join(captions), encoding="utf-8"
    )
    with (out_root / "publication_figure_index.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        w.writeheader()
        w.writerows(index_rows)

    excl_path = dirs["artifacts"] / "figure_exclusion_log.yaml"
    excl_path.write_text(yaml.safe_dump({"excluded": exclusions}, sort_keys=False), encoding="utf-8")
    (out_root / "figure_exclusion_log.yaml").write_text(
        yaml.safe_dump({"excluded": exclusions}, sort_keys=False), encoding="utf-8"
    )

    style_path = dirs["artifacts"] / "figure_style_manifest.json"
    style_path.write_text(json.dumps(style_manifest_dict(style), indent=2), encoding="utf-8")
    (out_root / "figure_style_manifest.json").write_text(
        json.dumps(style_manifest_dict(style), indent=2), encoding="utf-8"
    )

    (dirs["artifacts"] / "figure_input_manifest.json").write_text(
        json.dumps(resolved, indent=2), encoding="utf-8"
    )
    (out_root / "figure_input_manifest.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    (dirs["artifacts"] / "figure_output_manifest.json").write_text(
        json.dumps(output_manifest, indent=2), encoding="utf-8"
    )
    (out_root / "figure_output_manifest.json").write_text(
        json.dumps(output_manifest, indent=2), encoding="utf-8"
    )

    # Copy captions to out root captions folder already done
    (out_root / "captions" / "publication_figure_captions.md").write_text(
        "\n".join(captions), encoding="utf-8"
    )

    overall = "PASS" if integrity_counts["validation_failures"] == 0 else "FAIL"
    summary = {
        "run_id": run_id,
        "overall": overall,
        "analysis_tag": args.analysis_tag,
        "analysis_commit": src_commit,
        "analysis_id": EXPECTED_ANALYSIS_ID,
        "matplotlib_version": style.matplotlib_version,
        "resolved_font": style.resolved_font,
        "main_figures": [r["figure"] for r in index_rows if r["kind"] == "main"],
        "supplementary_figures": [r["figure"] for r in index_rows if r["kind"] == "supplementary"],
        "excluded_figures": exclusions,
        "convention_missing_count": miss,
        "checkpoint_availability": ckpts,
        "formats": args.formats,
        "png_dpi": args.png_dpi,
        "integrity": integrity_counts,
        "exact_command": (
            f"{sys.executable} {SCRIPT} --analysis-tag {args.analysis_tag} "
            f"--analysis-worktree {Path(args.analysis_worktree).resolve()} "
            f"--output-root {out_root} --formats {' '.join(args.formats)} --png-dpi {args.png_dpi}"
        ),
        "formal_training_started": False,
        "evaluation_invoked": False,
    }
    (dirs["reports"] / "stage6c_figure_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report = f"""# Stage 6C Publication Figures Report

## Overall: **{overall}**

- run_id: `{run_id}`
- analysis tag: `{args.analysis_tag}`
- analysis commit: `{src_commit}`
- Matplotlib: `{style.matplotlib_version}`
- resolved font: `{style.resolved_font}`
- main figures: {summary['main_figures']}
- supplementary figures: {summary['supplementary_figures']}
- excluded: {exclusions}
- convention missing count represented: `{miss}`
- checkpoint availability: `{ckpts}`

No policies were retrained and no evaluation environments were re-executed.
"""
    (dirs["reports"] / "stage6c_figure_report.md").write_text(report, encoding="utf-8")
    (out_root / "stage6c_figure_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_root / "stage6c_figure_report.md").write_text(report, encoding="utf-8")
    (out_root / "publication_figure_captions.md").write_text("\n".join(captions), encoding="utf-8")
    (out_root / "latest_run.json").write_text(
        json.dumps({"run_id": run_id, "overall": overall}, indent=2), encoding="utf-8"
    )
    log(f"overall={overall}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
