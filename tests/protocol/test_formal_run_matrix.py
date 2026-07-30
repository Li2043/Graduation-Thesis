"""Stage 5C-0 — formal run matrix tests."""

from __future__ import annotations

from thesis.protocol.final_training_protocol import (
    FORMAL_CONDITIONS,
    FORMAL_MASTER_SEEDS,
    build_formal_run_matrix,
    write_formal_run_matrix,
)


def test_exactly_thirty_paired_slots(tmp_path):
    rows = build_formal_run_matrix()
    assert len(rows) == 30
    assert len(FORMAL_CONDITIONS) == 3
    assert len(FORMAL_MASTER_SEEDS) == 10
    assert {r["condition"] for r in rows} == set(FORMAL_CONDITIONS)
    assert {r["master_seed"] for r in rows} == set(FORMAL_MASTER_SEEDS)
    for cond in FORMAL_CONDITIONS:
        assert sum(1 for r in rows if r["condition"] == cond) == 10
    path = tmp_path / "formal_run_matrix.csv"
    write_formal_run_matrix(path, rows)
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "baseline" in text and "mean_pbrs" in text and "min_pbrs" in text
