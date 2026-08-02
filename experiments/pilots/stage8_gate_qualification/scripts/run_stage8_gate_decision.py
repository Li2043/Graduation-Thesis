#!/usr/bin/env python3
"""Stage 8 formal gate decision: PASS / FAIL / INVALID.

Reuses `thesis.pilots.stage7c_q1_gate.evaluate_competence_gate` UNCHANGED --
every GATE_* threshold, GATE_CHECKPOINTS, and LEARNING_CURVE_CHECKPOINTS
constant it reads is copied verbatim from `stage7c_q1_config.py` into
`stage8_gate_config.py` (see that module's docstring), so this is the exact
same decision function evaluated against a different training run, not a
re-derived one -- direct comparability to the Stage 7C-Q1 FAIL is the point.

Before calling the gate, this script checks basic completeness (all 20
seeds x all 17 checkpoints present, no NaN in the required columns) and
passes the result as `integrity_ok` -- an incomplete raw dataset must not
silently produce a PASS/FAIL, only INVALID.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

from thesis.pilots.stage7c_q1_gate import evaluate_competence_gate  # noqa: E402
from thesis.pilots.stage8_gate_config import (  # noqa: E402
    CHECKPOINT_STEPS,
    GATE_CHECKPOINTS,
    LEARNING_CURVE_CHECKPOINTS,
    PILOT_SEEDS,
    PROTOCOL_TAG,
)

RESULTS_ROOT = Path(__file__).resolve().parents[1] / ".." / ".." / ".." / "results" / "stage8_gate" / "v1"
RESULTS_ROOT = RESULTS_ROOT.resolve()
OUT_ROOT = Path(__file__).resolve().parents[1] / ".." / ".." / ".." / "analysis" / "stage8_gate" / "v1"
OUT_ROOT = OUT_ROOT.resolve()


def _check_integrity(df: pd.DataFrame) -> tuple[bool, list[str]]:
    errors: list[str] = []
    required_cols = {
        "master_seed",
        "checkpoint_step",
        "success_rate",
        "collision_rate",
        "truncation_rate",
        "swap_eligibility",
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        errors.append(f"missing columns: {sorted(missing_cols)}")
        return False, errors

    seeds_present = sorted(int(s) for s in df["master_seed"].unique())
    if seeds_present != sorted(PILOT_SEEDS):
        missing_seeds = sorted(set(PILOT_SEEDS) - set(seeds_present))
        extra_seeds = sorted(set(seeds_present) - set(PILOT_SEEDS))
        errors.append(f"seed set mismatch: missing={missing_seeds} extra={extra_seeds}")

    for seed in PILOT_SEEDS:
        present_ckpts = sorted(
            int(c) for c in df.loc[df["master_seed"] == seed, "checkpoint_step"].unique()
        )
        expected_ckpts = sorted(CHECKPOINT_STEPS)
        if present_ckpts != expected_ckpts:
            missing = sorted(set(expected_ckpts) - set(present_ckpts))
            if missing:
                errors.append(f"seed {seed} missing checkpoints: {missing}")

    for col in ["success_rate", "collision_rate", "truncation_rate", "swap_eligibility"]:
        if df[col].isna().any():
            errors.append(f"NaN present in column {col}")

    return (len(errors) == 0), errors


def main() -> int:
    raw_path = RESULTS_ROOT / "raw" / "seed_checkpoint_summary.csv"
    if not raw_path.is_file():
        print(f"ABORT: missing {raw_path}", file=sys.stderr)
        return 1
    df = pd.read_csv(raw_path)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    integrity_ok, integrity_errors = _check_integrity(df)
    if not integrity_ok:
        print("INTEGRITY CHECK FAILED:")
        for e in integrity_errors:
            print(f"  - {e}")

    decision = evaluate_competence_gate(
        df,
        expected_seeds=PILOT_SEEDS,
        integrity_ok=integrity_ok,
        integrity_errors=integrity_errors,
    )

    (OUT_ROOT / "STAGE8_GATE_DECISION.json").write_text(
        json.dumps(decision, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, default=str))

    lines = [
        "# Stage 8 Formal Qualification Gate — Decision",
        "",
        f"**Status: {decision['status']}**",
        "",
        f"Protocol tag: `{PROTOCOL_TAG}`. Seeds: {PILOT_SEEDS[0]}-{PILOT_SEEDS[-1]} "
        f"(n={len(PILOT_SEEDS)}). Gate checkpoints: {GATE_CHECKPOINTS}. "
        f"Learning-curve checkpoints: {LEARNING_CURVE_CHECKPOINTS}.",
        "",
        "Thresholds copied verbatim from Stage 7C-Q1 "
        "(`thesis.pilots.stage7c_q1_config`), not re-derived -- this decision "
        "is directly comparable to the Stage 7C-Q1 FAIL under the identical "
        "rule set.",
        "",
    ]
    if decision["status"] == "INVALID":
        lines += ["## Reason", "", decision.get("reason", ""), ""]
    else:
        comps = decision["components"]
        lines.append("## Per-checkpoint gate criteria")
        lines.append("")
        lines.append("| checkpoint | mean_success | collision | truncation | swap_eligibility | seeds≥threshold |")
        lines.append("|---|---|---|---|---|---|")
        for ckpt in GATE_CHECKPOINTS:
            c = comps[str(ckpt)]
            lines.append(
                f"| {ckpt} | {c['mean_success_value']:.4f} ({'OK' if c['mean_success'] else 'FAIL'}) "
                f"| {c['mean_collision_value']:.4f} ({'OK' if c['collision'] else 'FAIL'}) "
                f"| {c['mean_truncation_value']:.4f} ({'OK' if c['truncation'] else 'FAIL'}) "
                f"| {c['mean_swap_value']:.4f} ({'OK' if c['swap_eligibility'] else 'FAIL'}) "
                f"| {c['seeds_ge_threshold']} |"
            )
        lines += [
            "",
            f"**Qualified seed intersection** (success ≥ {61/64:.4f} at all 3 gate "
            f"checkpoints): {comps['qualified_seed_intersection_count']} seeds "
            f"({'OK' if comps['intersection_ok'] else 'FAIL'}, need ≥16) — "
            f"{comps['qualified_seed_intersection']}",
            "",
            f"**Learning-curve continuity** (adjacent 25K-checkpoint success drop "
            f"≤ 0.03, checkpoints 200K-400K): "
            f"{'OK' if comps['learning_curve_ok'] else 'FAIL'}",
        ]
        if comps["learning_curve_violations"]:
            lines.append("")
            lines.append("Violations:")
            for v in comps["learning_curve_violations"]:
                lines.append(f"- {v}")
        lines += [
            "",
            f"**Material regression seeds** (350K-400K drop > 0.20): "
            f"{comps['material_regression_seeds']} "
            f"(need ≤1 seed)",
            "",
            f"**Late-collapse seeds** (≥0.95 at 350K/375K, then <0.75 later): "
            f"{comps['late_collapse_seeds']} "
            f"(need ≤1 seed)",
        ]

    (OUT_ROOT / "STAGE8_GATE_DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_ROOT / 'STAGE8_GATE_DECISION.md'}")
    return 0 if decision["status"] != "INVALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
