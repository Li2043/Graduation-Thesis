"""M4-D regression test (fast subset -- the official 10,000-scenario gate
run is `experiments/pilots/study_b_fairness_mappo/scripts/
validate_matched_ttc_highwayenv.py`, result recorded in
`output/highwayenv_migration/validation/M4_D_SUMMARY.json` and
`output/autonomous_highwayenv/GATE_RESULTS.json`)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "pilots" / "study_b_fairness_mappo" / "scripts"))

from validate_matched_ttc_highwayenv import run  # noqa: E402


def test_m4d_matched_ttc_gate_passes_on_a_fast_subset(tmp_path):
    summary = run(n_scenarios=500, master_seed=1, before_merge_length=220.0, out_dir=tmp_path)
    assert summary["gate"] == "PASS"
    assert summary["pct_within_0.5s_tolerance"] >= 0.95
    assert summary["same_lane_gap_violations"] == 0
    assert summary["negative_spawn_count"] == 0
    assert (tmp_path / "matched_ttc_validation.csv").exists()
    assert (tmp_path / "spawn_validity.csv").exists()
