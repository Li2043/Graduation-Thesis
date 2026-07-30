#!/usr/bin/env python3
"""Validate Stage 6B-H1 outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, default=None)
    args = parser.parse_args()
    root = Path(args.output_root)
    failures: list[str] = []

    def need(rel: str) -> Path:
        p = root / rel
        if not p.is_file():
            failures.append(f"missing {rel}")
        return p

    ep = need("data/evaluation_episodes_h1.csv")
    need("data/primary_endpoint_seed_values_h1.csv")
    need("statistics/primary_endpoint_descriptives_h1.csv")
    need("statistics/primary_endpoint_contrasts_h1.csv")
    need("data/secondary_endpoints_h1.csv")
    mm = need("diagnostics/nonutility_mismatches.csv")
    need("diagnostics/convention_availability_h1.csv")
    need("diagnostics/controller_swap_diagnostics_h1.csv")
    need("diagnostics/checkpoint_integrity_before.csv")
    need("diagnostics/checkpoint_integrity_after.csv")
    need("manifests/analysis_manifest.json")
    need("manifests/acceptance_checks.json")

    if ep.is_file():
        df = pd.read_csv(ep)
        if len(df) != 480:
            failures.append(f"episode count {len(df)} != 480")
        if df.duplicated(["condition", "master_seed", "block_id", "assignment"]).any():
            failures.append("duplicate episode keys")
        for col in ("utility_A", "utility_B", "utility_background_front", "utility_background_rear"):
            if ((df[col] < 0) | (df[col] > 1)).any():
                failures.append(f"{col} out of [0,1]")

    if mm.is_file():
        mdf = pd.read_csv(mm)
        if len(mdf) != 0:
            failures.append(f"nonutility mismatches={len(mdf)}")

    acc_path = root / "manifests" / "acceptance_checks.json"
    if acc_path.is_file():
        acc = json.loads(acc_path.read_text(encoding="utf-8"))
        if not acc.get("checkpoint_hashes_unchanged"):
            failures.append("checkpoint hashes changed")
        if not acc.get("episode_count_is_480"):
            failures.append("acceptance episode flag false")

    before = root / "diagnostics" / "checkpoint_integrity_before.csv"
    after = root / "diagnostics" / "checkpoint_integrity_after.csv"
    if before.is_file() and after.is_file():
        b = pd.read_csv(before)
        a = pd.read_csv(after)
        if len(b) != 30 or len(a) != 30:
            failures.append("checkpoint count not 30")
        if not b.equals(a):
            failures.append("checkpoint integrity before/after differ")

    if failures:
        print("FAIL")
        for f in failures:
            print(" -", f)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
