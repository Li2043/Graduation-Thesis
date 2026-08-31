"""100K formal run matrix tests."""

from thesis.formal.formal_config import derive_formal_job_seeds
from thesis.protocol.h1_r1_100k_protocol import (
    FORMAL_CONDITIONS,
    FORMAL_MASTER_SEEDS,
    build_formal_run_matrix_100k,
)


def test_paired_seeds_and_matrix():
    rows = build_formal_run_matrix_100k(
        protocol_sha256="p", environment_lock_sha256="e", comfort_lock_sha256="c"
    )
    assert len(rows) == 30
    assert len(FORMAL_CONDITIONS) == 3
    assert len(FORMAL_MASTER_SEEDS) == 10
    for master in FORMAL_MASTER_SEEDS:
        derived = derive_formal_job_seeds(master)
        matching = [r for r in rows if int(r["master_seed"]) == master]
        assert len(matching) == 3
        for r in matching:
            assert r["environment_seed"] == derived["environment_seed"]
            assert r["learner_A_seed"] == derived["learner_A_seed"]
            assert r["replay_A_seed"] == derived["replay_A_seed"]
            assert r["replay_B_seed"] == derived["replay_B_seed"]
            assert r["expected_steps"] == 100_000
