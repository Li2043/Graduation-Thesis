#!/usr/bin/env python3
"""Seed collision audit for Stage 7B-A1 (63001-63020)."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

SCRIPT = Path(__file__).resolve()
PILOT_ROOT = SCRIPT.parents[1]
REPO_ROOT = SCRIPT.parents[4]
PILOT_SEEDS = list(range(63001, 63021))
FORBIDDEN = {
    "formal": list(range(61001, 61011)),
    "stage7a1": list(range(62001, 62021)),
}


def main() -> int:
    hits = []
    # Scan CSV master_seed columns and yaml/json seed lists
    # Scan experiment outputs / manifests / protocols (not source that defines the new plan)
    scan_roots = [
        REPO_ROOT / "experiments",
        REPO_ROOT / "manifests",
    ]
    for root in scan_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            s = str(path).replace("\\", "/")
            if any(
                x in s
                for x in (
                    ".git/",
                    ".venv",
                    "__pycache__",
                    "/releases/",
                    "stage7b_a1_double_dqn/",
                    "node_modules",
                )
            ):
                continue
            if path.suffix.lower() not in {".csv", ".yaml", ".yml", ".json", ".md", ".txt"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            rel = path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
            if path.suffix.lower() == ".csv":
                try:
                    with path.open(encoding="utf-8", newline="") as f:
                        reader = csv.DictReader(f)
                        if not reader.fieldnames or "master_seed" not in reader.fieldnames:
                            continue
                        for row in reader:
                            try:
                                ms = int(row["master_seed"])
                            except Exception:
                                continue
                            if ms in PILOT_SEEDS:
                                hits.append(
                                    {
                                        "seed": ms,
                                        "path": rel,
                                        "kind": "master_seed_column",
                                    }
                                )
                except Exception:
                    continue
            else:
                # Only treat explicit master_seed assignments as collisions in non-CSV
                for seed in PILOT_SEEDS:
                    if re.search(
                        rf"master_seed\s*[:=]\s*{seed}\b|master_seeds?[^\n]*\b{seed}\b",
                        text,
                    ):
                        hits.append({"seed": seed, "path": rel, "kind": "master_seed_assignment"})

    # Deduplicate
    uniq = {(h["seed"], h["path"], h["kind"]): h for h in hits}
    hits = list(uniq.values())
    # Filter known false positives: protocol files that only list forbidden blocks elsewhere
    # Any hit is ABORT per user instruction
    status = "PASS" if not hits else "ABORT"
    report = {
        "pilot_seeds": PILOT_SEEDS,
        "forbidden_blocks": FORBIDDEN,
        "collision_seeds": sorted({h["seed"] for h in hits}),
        "hits": hits[:200],
        "hit_count": len(hits),
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    out = PILOT_ROOT / "manifests" / "seed_collision_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "hit_count": len(hits), "collision_seeds": report["collision_seeds"]}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
