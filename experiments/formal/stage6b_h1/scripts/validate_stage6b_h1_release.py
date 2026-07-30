#!/usr/bin/env python3
"""Validate Stage 6B-H1.1 release metadata and experiment preservation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from thesis.analysis.h1_manifest import is_absolute_path_string, verify_manifest_hashes

H1 = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h1-root", type=Path, default=H1)
    args = parser.parse_args()
    root = Path(args.h1_root).resolve()
    failures: list[str] = []

    man_path = root / "output" / "manifests" / "analysis_manifest.json"
    acc_path = root / "output" / "manifests" / "acceptance_checks.json"
    if not man_path.is_file():
        print("FAIL missing analysis_manifest.json")
        return 1
    man = json.loads(man_path.read_text(encoding="utf-8"))
    acc = json.loads(acc_path.read_text(encoding="utf-8"))

    try:
        verify_manifest_hashes(artifact_root=root, manifest_path=man_path)
    except Exception as exc:
        failures.append(f"manifest hash verify: {exc}")

    for rel in man.get("output_hashes", {}):
        if is_absolute_path_string(rel) or ".." in Path(rel).parts:
            failures.append(f"bad path {rel}")
        if not re.fullmatch(r"[0-9a-f]{64}", man["output_hashes"][rel]):
            failures.append(f"bad hash for {rel}")

    for p in man.get("figure_paths", []):
        if is_absolute_path_string(p):
            failures.append(f"absolute figure path {p}")

    hashes = man.get("output_hashes", {})
    if "output/diagnostics/paper_file_integrity_before.csv" not in hashes:
        failures.append("paper before missing from manifest")
    if "output/diagnostics/paper_file_integrity_after.csv" not in hashes:
        failures.append("paper after missing from manifest")

    tol = float(acc.get("reference_tolerance", 1))
    if tol > 1e-6:
        failures.append(f"tolerance too loose: {tol}")

    ep = root / "output" / "data" / "evaluation_episodes_h1.csv"
    mm = root / "output" / "diagnostics" / "nonutility_mismatches.csv"
    if ep.is_file():
        df = pd.read_csv(ep)
        if len(df) != 480:
            failures.append(f"episodes={len(df)}")
    else:
        failures.append("missing episodes")
    if mm.is_file():
        mdf = pd.read_csv(mm)
        if len(mdf) != 0:
            failures.append(f"mismatches={len(mdf)}")
        expected_cols = [
            "condition",
            "master_seed",
            "block_id",
            "assignment",
            "field",
            "old",
            "new",
        ]
        if list(mdf.columns) != expected_cols:
            failures.append(f"mismatch columns={list(mdf.columns)}")
    else:
        failures.append("missing mismatch csv")

    if not acc.get("checkpoint_hashes_unchanged"):
        failures.append("checkpoint hashes changed")
    if not man.get("paper_integrity", {}).get("verified_unchanged", False):
        failures.append("paper integrity not verified")

    mu = acc.get("corrected_mean_utility", {})
    for cond, exp in (("baseline", 0.605213), ("mean_pbrs", 0.527772), ("min_pbrs", 0.586206)):
        if abs(float(mu.get(cond, 0)) - exp) > 1e-6:
            failures.append(f"utility drift {cond}")
    swap = acc.get("controller_swap_estimable_seeds", {})
    if swap != {"baseline": 4, "mean_pbrs": 0, "min_pbrs": 4}:
        failures.append(f"swap estimability {swap}")

    if "execution_commit" not in man or "release_commit" not in man:
        failures.append("missing execution/release commits")

    if failures:
        print("FAIL")
        for f in failures:
            print(" -", f)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
