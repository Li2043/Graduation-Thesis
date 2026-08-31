"""Checkpoint inventory tests."""

from thesis.diagnostics.stage7a0_inventory import FORMAL_BASELINE_SEEDS, FORMAL_CHECKPOINT_STEPS


def test_expected_baseline_inventory_constants():
    assert len(FORMAL_BASELINE_SEEDS) == 10
    assert FORMAL_CHECKPOINT_STEPS == [10000, 25000, 50000, 75000, 100000]
