"""Formal evaluation episode seed formula."""

from thesis.formal.formal_schedule import evaluation_episode_seed


def test_evaluation_seed_formula():
    assert evaluation_episode_seed(500000, checkpoint_index=0, block_index=0, assignment_index=0) == 500000
    assert evaluation_episode_seed(500000, checkpoint_index=2, block_index=3, assignment_index=1) == (
        500000 + 1000 * 2 + 2 * 3 + 1
    )
