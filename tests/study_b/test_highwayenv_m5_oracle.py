"""M5 -- oracle feasibility gate regression test (fast subset -- the
official Q/M/H1 (64/64/256) runs are recorded in
`output/highwayenv_migration/validation/oracle_*.csv` and
`oracle_controller_highwayenv_summary.json`, all STRONG_PASS: 100%
completion, 0% collision, 0% timeout across all 384 scenarios, exactly
matching the legacy backend's own oracle result)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "pilots" / "study_b_fairness_mappo" / "scripts"))

from run_oracle_controller_highwayenv import run_oracle_controller_highwayenv  # noqa: E402

_Q_BANK = Path(__file__).resolve().parents[2] / "experiments" / "pilots" / "study_b_fairness_mappo" / "scenario_banks" / "Q.json"


def test_oracle_reaches_strong_pass_on_the_q_bank():
    if not _Q_BANK.exists():
        import pytest
        pytest.skip("requires the frozen Q scenario bank on disk")
    rows = run_oracle_controller_highwayenv(scenario_bank=_Q_BANK)
    n = len(rows)
    completion_rate = sum(r["completion"] for r in rows) / n
    collision_rate = sum(r["collision"] for r in rows) / n
    timeout_rate = sum(r["timeout"] for r in rows) / n
    assert completion_rate >= 0.98
    assert collision_rate <= 0.01
    assert timeout_rate <= 0.01
