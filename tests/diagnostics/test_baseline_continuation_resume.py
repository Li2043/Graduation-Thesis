"""Continuation probe blocked without resumable checkpoints."""

import json
from pathlib import Path


def test_continuation_status_blocked_when_present():
    p = (
        Path(__file__).resolve().parents[2]
        / "experiments/diagnostics/stage7a0_baseline_competence/output/continuation_probe/baseline_continuation_status.json"
    )
    if not p.is_file():
        return
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["status"] == "BLOCKED"
    assert payload["executed"] is False
    assert payload["available_resumable_100k_checkpoints"] == 0
