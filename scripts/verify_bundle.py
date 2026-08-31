#!/usr/bin/env python3
"""Sanity-check that the bundle's directory structure is intact after
being copied to a new machine/drive -- before doing anything else."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import BUNDLE_ROOT  # noqa: E402

REQUIRED_PATHS = [
    "project/src/thesis/study_b/local_observation.py",
    "project/src/thesis/study_b/envs/highwayenv_wrapper.py",
    "project/experiments/pilots/study_b_fairness_mappo/scripts/train_curriculum_stage_highwayenv.py",
    "project/experiments/pilots/study_b_fairness_mappo/scripts/stage_q_ensemble_gate.py",
    "project/pytest.ini",
    "scenario_banks/Q.json", "scenario_banks/H0.json", "scenario_banks/H1.json",
    "scenario_banks/C4.json", "scenario_banks/C16.json",
    "configs/FROZEN_EXPERIMENT_CONFIG.json", "configs/FROZEN_EXPERIMENT_CONFIG.md",
    "experiment_records/RUNBOOK.md", "experiment_records/autonomous_highwayenv/GATE_RESULTS.json",
    "checkpoints/formal_init/900101/C64_R50",
    "checkpoints/curriculum_910101_910102/910101/M6_R50_audited",
    "environment/pip_freeze.txt", "environment/python_version.txt",
    "CHECKSUMS.sha256", "MIGRATION_MANIFEST.json", "README.md",
]


def main() -> int:
    missing = [p for p in REQUIRED_PATHS if not (BUNDLE_ROOT / p).exists()]
    if missing:
        print("[verify_bundle] MISSING:")
        for m in missing:
            print(f"  - {m}")
        return 1
    print(f"[verify_bundle] OK -- all {len(REQUIRED_PATHS)} required paths present under {BUNDLE_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
