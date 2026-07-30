"""Stage 5C-0 — formal evaluation / checkpoint plan tests."""

from __future__ import annotations

from thesis.protocol.final_training_protocol import (
    FORMAL_CHECKPOINT_STEPS,
    FORMAL_EVALUATION_STEPS,
    PRIMARY_ENDPOINT_STEP,
    build_final_training_protocol,
)
from thesis.protocol.prerequisites import verify_stage5c0_prerequisites


def test_evaluation_and_checkpoint_schedules_fixed():
    assert FORMAL_CHECKPOINT_STEPS == (5000, 10000, 15000, 20000)
    assert FORMAL_EVALUATION_STEPS == (0, 5000, 10000, 15000, 20000)
    assert PRIMARY_ENDPOINT_STEP == 20000
    prereq = verify_stage5c0_prerequisites()
    lock = build_final_training_protocol(
        prereq,
        git_commit="test",
        source_hashes={},
        configuration_sha256="abc",
        pbrs_lock_sha256="x",
    )
    assert lock["evaluation_schedule"]["greedy"] is True
    assert lock["evaluation_schedule"]["epsilon"] == 0.0
    assert lock["evaluation_schedule"]["no_replay_writes"] is True
    assert lock["evaluation_schedule"]["results_alter_checkpoint_selection"] is False
    assert lock["checkpoint_schedule"]["best_checkpoint_selection"] is False
    assert lock["checkpoint_schedule"]["primary_endpoint"] == 20000
