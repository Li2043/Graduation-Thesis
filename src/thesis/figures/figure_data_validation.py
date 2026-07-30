"""Validate Stage 6B inputs for publication figures (read-only)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from thesis.figures.publication_style import (
    CONDITION_ORDER,
    FORMAL_SEEDS,
    PRIMARY_NON_CONVENTION,
    PRIMARY_STEP,
)


class FigureDataBlockedError(RuntimeError):
    pass


EXPECTED_ANALYSIS_COMMIT = "9c586754d3548b89cf0f1ecd7d3a262caf4b7cf6"
EXPECTED_RESULT_COMMIT = "c75845935a7fe9179b691298b2329208853773a6"
EXPECTED_EXECUTION_ID = "stage6a_20260730T094829Z_a89256db_44d5e647"
EXPECTED_ANALYSIS_ID = "stage6b_20260730T140035Z_c7584593"
EXPECTED_MISSING_CONVENTION = 11


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def discover_stage6b_paths(analysis_root: Path, analysis_id: str = EXPECTED_ANALYSIS_ID) -> dict[str, Path]:
    root = Path(analysis_root)
    base = root / "experiments" / "formal" / "stage6b_analysis_100k"
    if not base.is_dir():
        # allow passing the stage6b directory directly
        base = root if (root / "data" / "processed" / analysis_id).is_dir() else root
    proc = base / "data" / "processed" / analysis_id
    tables = base / "tables" / analysis_id
    required = {
        "run_accounting": proc / "run_accounting.csv",
        "evaluation_episode_validated": proc / "evaluation_episode_validated.csv",
        "seed_checkpoint_endpoints": proc / "seed_checkpoint_endpoints.csv",
        "primary_endpoint_seed_values": proc / "primary_endpoint_seed_values.csv",
        "paired_differences": proc / "paired_differences.csv",
        "bootstrap_intervals": proc / "bootstrap_intervals.csv",
        "wilcoxon_results": proc / "wilcoxon_results.csv",
        "holm_adjusted_results": proc / "holm_adjusted_results.csv",
        "effect_sizes": proc / "effect_sizes.csv",
        "convention_availability": proc / "convention_availability.csv",
        "secondary_endpoints": proc / "secondary_endpoints.csv",
        "learning_curve_auc": proc / "learning_curve_auc.csv",
        "integrity_summary": proc / "integrity_summary.csv",
        "primary_endpoint_descriptives": tables / "primary_endpoint_descriptives.csv",
        "primary_endpoint_contrasts": tables / "primary_endpoint_contrasts.csv",
        "secondary_endpoint_descriptives": tables / "secondary_endpoint_descriptives.csv",
        "convention_summary": tables / "convention_summary.csv",
        "analysis_manifest": base / "analysis_manifest.json",
        "analysis_summary": base / "reports" / analysis_id / "stage6b_formal_analysis_summary.json",
    }
    missing = [k for k, p in required.items() if not p.is_file()]
    if missing:
        raise FigureDataBlockedError(f"missing Stage 6B inputs: {missing}")
    return required


def verify_recorded_output_hashes(analysis_worktree: Path, manifest: dict[str, Any]) -> list[str]:
    """Return list of mismatch descriptions; empty means all recorded hashes match."""
    base = Path(analysis_worktree) / "experiments" / "formal" / "stage6b_analysis_100k"
    if not base.is_dir():
        base = Path(analysis_worktree)
    mismatches: list[str] = []
    for rel, expected in (manifest.get("output_hashes") or {}).items():
        path = base / rel
        # Figures and mutable runner logs are not Stage 6C authoritative inputs.
        if any(rel.endswith(ext) for ext in (".pdf", ".png", ".log")):
            continue
        if not path.is_file():
            mismatches.append(f"missing recorded output: {rel}")
            continue
        actual = sha256_file(path)
        if actual != expected:
            mismatches.append(f"{rel}: expected {expected}, got {actual}")
    return mismatches


def verify_analysis_source(analysis_worktree: Path) -> dict[str, Any]:
    wt = Path(analysis_worktree)
    paths = discover_stage6b_paths(wt)
    summary = json.loads(paths["analysis_summary"].read_text(encoding="utf-8-sig"))
    manifest = json.loads(paths["analysis_manifest"].read_text(encoding="utf-8-sig"))
    if str(summary.get("overall")) != "PASS":
        raise FigureDataBlockedError(f"Stage 6B overall is not PASS: {summary.get('overall')}")
    if str(manifest.get("overall")) != "PASS":
        raise FigureDataBlockedError(f"analysis manifest overall is not PASS: {manifest.get('overall')}")
    if str(summary.get("formal_execution_id")) != EXPECTED_EXECUTION_ID:
        raise FigureDataBlockedError("formal_execution_id mismatch")
    if str(summary.get("result_commit")) != EXPECTED_RESULT_COMMIT:
        raise FigureDataBlockedError("result_commit mismatch")
    if str(manifest.get("result_commit")) != EXPECTED_RESULT_COMMIT:
        raise FigureDataBlockedError("manifest result_commit mismatch")
    if str(summary.get("analysis_id")) != EXPECTED_ANALYSIS_ID:
        raise FigureDataBlockedError("analysis_id mismatch")
    if int(summary.get("missing_convention_counts", -1)) != EXPECTED_MISSING_CONVENTION:
        raise FigureDataBlockedError("missing convention count mismatch")
    mismatches = verify_recorded_output_hashes(wt, manifest)
    if mismatches:
        raise FigureDataBlockedError("Stage 6B output hash mismatches: " + "; ".join(mismatches[:5]))
    protocol = manifest.get("protocol_hashes") or {}
    required_protocol = ("training_protocol", "pbrs", "environment", "comfort")
    if any(k not in protocol for k in required_protocol):
        raise FigureDataBlockedError("protocol lock hashes incomplete in analysis manifest")
    return {
        "paths": {k: str(v) for k, v in paths.items()},
        "summary": summary,
        "manifest_overall": manifest.get("overall"),
        "protocol_hashes": protocol,
        "verified_output_hash_count": len(manifest.get("output_hashes") or {}),
    }


def build_resolved_figure_inputs(paths: dict[str, Path]) -> dict[str, Any]:
    out: dict[str, Any] = {"files": {}}
    for key, path in paths.items():
        if path.suffix.lower() != ".csv":
            out["files"][key] = {
                "relative_hint": key,
                "path": str(path),
                "sha256": sha256_file(path),
            }
            continue
        df = _read_csv(path)
        rec: dict[str, Any] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "row_count": int(len(df)),
            "column_names": list(df.columns),
        }
        if "endpoint" in df.columns:
            rec["endpoint_names"] = sorted(df["endpoint"].dropna().astype(str).unique().tolist())
        if "condition" in df.columns:
            rec["condition_names"] = sorted(df["condition"].dropna().astype(str).unique().tolist())
        if "master_seed" in df.columns:
            rec["seed_count"] = int(df["master_seed"].nunique())
            rec["seeds"] = sorted(int(x) for x in df["master_seed"].dropna().unique().tolist())
        if "checkpoint_step" in df.columns:
            rec["checkpoint_values"] = sorted(
                int(x) for x in df["checkpoint_step"].dropna().unique().tolist()
            )
        missing = {}
        for col in df.columns:
            missing[col] = int(df[col].isna().sum())
        rec["missing_value_counts"] = missing
        out["files"][key] = rec
    return out


def validate_primary_seed_table(df: pd.DataFrame) -> None:
    conds = set(df["condition"].astype(str))
    if conds != set(CONDITION_ORDER):
        raise FigureDataBlockedError(f"unexpected conditions: {conds}")
    seeds = set(int(x) for x in df["master_seed"].unique())
    if seeds != set(FORMAL_SEEDS):
        raise FigureDataBlockedError(f"unexpected seeds: {seeds}")
    if set(int(x) for x in df["checkpoint_step"].unique()) != {PRIMARY_STEP}:
        raise FigureDataBlockedError("primary endpoint checkpoint must be 100000 only")
    # duplicates
    keys = df.groupby(["endpoint", "condition", "master_seed"]).size()
    if (keys > 1).any():
        raise FigureDataBlockedError("duplicated condition×seed×endpoint keys")
    for ep in PRIMARY_NON_CONVENTION:
        sub = df[df["endpoint"] == ep]
        if sub["value"].isna().any():
            raise FigureDataBlockedError(f"unexpected NA in {ep}")
        vals = sub["value"].astype(float)
        if (~vals.between(0.0, 1.0)).any() or (~vals.apply(math.isfinite)).any():
            raise FigureDataBlockedError(f"out-of-range values in {ep}")
        for c in CONDITION_ORDER:
            n = int((sub["condition"] == c).sum())
            if n != 10:
                raise FigureDataBlockedError(f"{ep}/{c} expected 10 seeds, got {n}")
    conv = df[df["endpoint"] == "convention_consistency"]
    miss = int(conv["value"].isna().sum())
    if miss != EXPECTED_MISSING_CONVENTION:
        raise FigureDataBlockedError(
            f"convention missing count {miss} != {EXPECTED_MISSING_CONVENTION}"
        )
    observed = conv["value"].dropna().astype(float)
    if len(observed) and ((~observed.between(0.0, 1.0)).any() or (~observed.apply(math.isfinite)).any()):
        raise FigureDataBlockedError("convention consistency out of range")


def validate_contrasts(contrasts: pd.DataFrame, bootstrap: pd.DataFrame, holm: pd.DataFrame, effects: pd.DataFrame) -> None:
    keys = contrasts.groupby(["endpoint", "contrast"]).size()
    if (keys > 1).any():
        raise FigureDataBlockedError("duplicated contrast×endpoint keys")
    # Cross-check bootstrap intervals against contrasts table
    merged = contrasts.merge(
        bootstrap,
        on=["endpoint", "contrast"],
        suffixes=("_c", "_b"),
    )
    for _, row in merged.iterrows():
        if abs(float(row["ci95_low"]) - float(row["ci_low"])) > 1e-12:
            raise FigureDataBlockedError("bootstrap CI low mismatch vs contrasts table")
        if abs(float(row["ci95_high"]) - float(row["ci_high"])) > 1e-12:
            raise FigureDataBlockedError("bootstrap CI high mismatch vs contrasts table")
        if abs(float(row["mean_diff_c"]) - float(row["mean_diff_b"])) > 1e-12:
            raise FigureDataBlockedError("mean_diff mismatch vs bootstrap table")
    holm_m = contrasts.merge(holm, on=["endpoint", "contrast"], suffixes=("", "_h"))
    for _, row in holm_m.iterrows():
        a = row["wilcoxon_p_holm"]
        b = row["pvalue_holm"]
        if pd.isna(a) and pd.isna(b):
            continue
        if abs(float(a) - float(b)) > 1e-12:
            raise FigureDataBlockedError("Holm p-value mismatch")
    eff_m = contrasts.merge(effects, on=["endpoint", "contrast"])
    for _, row in eff_m.iterrows():
        if abs(float(row["cohens_dz_x"]) - float(row["cohens_dz_y"])) > 1e-12:
            raise FigureDataBlockedError("effect size mismatch")


def deterministic_jitter(seed: int, condition_index: int, *, scale: float = 0.08) -> float:
    """Deterministic horizontal jitter in [-scale, scale] from seed hash."""
    h = hashlib.md5(f"{seed}:{condition_index}".encode("utf-8")).hexdigest()
    u = int(h[:8], 16) / 0xFFFFFFFF
    return (u - 0.5) * 2.0 * scale


__all__ = [
    "EXPECTED_ANALYSIS_COMMIT",
    "EXPECTED_ANALYSIS_ID",
    "EXPECTED_EXECUTION_ID",
    "EXPECTED_MISSING_CONVENTION",
    "EXPECTED_RESULT_COMMIT",
    "FigureDataBlockedError",
    "build_resolved_figure_inputs",
    "deterministic_jitter",
    "discover_stage6b_paths",
    "sha256_file",
    "validate_contrasts",
    "validate_primary_seed_table",
    "verify_analysis_source",
    "verify_recorded_output_hashes",
]
