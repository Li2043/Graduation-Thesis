"""Formal 24-episode schedule tests."""

from thesis.formal.formal_schedule import FormalICSchedule
from thesis.training.final_lock_loader import load_final_locks


def test_cycle_contains_each_pair_once():
    bundle = load_final_locks()
    sched = FormalICSchedule(bundle, schedule_seed=61001 + 600_000)
    seen: list[tuple[str, int]] = []
    for _ in range(24):
        _block, assignment, bid = sched.peek()
        seen.append((bid, assignment))
        sched.advance()
    assert len(seen) == 24
    assert len(set(seen)) == 24
    assert {(b, a) for b, a in seen} == {
        (bid, a)
        for bid in sorted(b.block_id for b in bundle.environment.calibration_blocks)
        for a in (0, 1)
    }
