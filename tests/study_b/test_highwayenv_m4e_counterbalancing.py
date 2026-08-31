"""M4-E regression test. Also guards against the exact bug this gate
found and fixed: ``StudyBHeterogeneousHighwayEnv._role_members()`` must
NOT reuse the same seed value that gets passed into
``generate_scenario()``'s own internal RNG -- doing so statistically
entangles physical-id<->role assignment with the generator's own
speed_class/ttc_slot permutation (measured ~69/31 skew before the fix,
~50/50 after)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "pilots" / "study_b_fairness_mappo" / "scripts"))

from validate_counterbalancing_highwayenv import run  # noqa: E402


def test_m4e_counterbalancing_gate_passes_and_is_tightly_balanced():
    summary = run(n_resets=2000, master_seed=1)
    assert summary["gate"] == "PASS"
    for report in (
        summary["role_balance_by_vehicle_id"],
        summary["speed_class_balance_by_vehicle_id"],
        summary["ttc_slot_balance_by_vehicle_id"],
    ):
        for vid, fracs in report.items():
            for cls, frac in fracs.items():
                # Tight band (not just the gate's generous [0.30,0.70]):
                # regresses to the pre-fix ~0.69/0.31 skew if the seed
                # decorrelation is ever accidentally reverted.
                assert 0.44 <= frac <= 0.56, (vid, cls, frac)
