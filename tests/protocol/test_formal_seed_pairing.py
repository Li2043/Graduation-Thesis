"""Stage 5C-0 — formal seed pairing / derivation tests."""

from __future__ import annotations

from thesis.protocol.final_training_protocol import (
    FORMAL_CONDITIONS,
    FORMAL_MASTER_SEEDS,
    build_formal_run_matrix,
    derive_formal_seeds,
)


def test_identical_seed_derivation_across_conditions():
    for master in FORMAL_MASTER_SEEDS:
        derived = derive_formal_seeds(master)
        assert derived["environment_seed"] == master
        assert derived["learner_A_seed"] == master + 100_000
        assert derived["learner_B_seed"] == master + 200_000
        assert derived["replay_A_seed"] == master + 300_000
        assert derived["replay_B_seed"] == master + 400_000
        assert derived["evaluation_seed"] == master + 500_000
        assert derived["schedule_seed"] == master + 600_000

    matrix = build_formal_run_matrix()
    by_seed: dict[int, list] = {}
    for row in matrix:
        by_seed.setdefault(row["master_seed"], []).append(row)
    for master, rows in by_seed.items():
        assert len(rows) == len(FORMAL_CONDITIONS)
        keys = (
            "environment_seed",
            "learner_A_seed",
            "learner_B_seed",
            "replay_A_seed",
            "replay_B_seed",
            "evaluation_seed",
            "schedule_seed",
        )
        ref = {k: rows[0][k] for k in keys}
        for row in rows[1:]:
            assert {k: row[k] for k in keys} == ref
