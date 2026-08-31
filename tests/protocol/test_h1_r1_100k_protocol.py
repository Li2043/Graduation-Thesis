"""H1-R1 100K protocol amendment tests."""

from __future__ import annotations

from pathlib import Path

from thesis.calibration.final_environment_trace_loader import sha256_file
from thesis.protocol.h1_r1_100k_protocol import (
    EPSILON_DECAY_STEPS,
    FORMAL_CHECKPOINT_STEPS,
    FORMAL_EVALUATION_STEPS,
    FORMAL_STEPS_PER_RUN,
    PROTOCOL_VERSION,
    SUPERSEDED_STAGE5C0_PROTOCOL_SHA256,
    build_formal_run_matrix_100k,
    build_h1_r1_training_protocol,
    write_h1_r1_artifact_bundle,
)
from thesis.protocol.prerequisites import verify_stage5c0_prerequisites


def test_budget_and_epsilon():
    assert FORMAL_STEPS_PER_RUN == 100_000
    assert 30 * FORMAL_STEPS_PER_RUN == 3_000_000
    assert EPSILON_DECAY_STEPS == 50_000
    assert FORMAL_CHECKPOINT_STEPS == (10_000, 25_000, 50_000, 75_000, 100_000)
    assert FORMAL_EVALUATION_STEPS == (0, 10_000, 25_000, 50_000, 75_000, 100_000)
    assert PROTOCOL_VERSION == "5C-0-H1-R1-100K"


def test_supersedes_stage5c0_and_writes_bundle(tmp_path):
    prereq = verify_stage5c0_prerequisites()
    proto = build_h1_r1_training_protocol(
        prereq, git_commit="t", pbrs_lock_sha256="x"
    )
    assert proto["supersedes_stage5c0_protocol_sha256"] == SUPERSEDED_STAGE5C0_PROTOCOL_SHA256
    assert proto["formal_training_started"] is False
    assert proto["pre_result_budget_amendment"] is True
    assert proto["environment"]["num_parallel_training_envs_per_run"] == 1
    assert proto["environment"]["vectorized_training"] is False
    assert proto["exploration"]["epsilon_decay_environment_steps"] == 50_000

    out = write_h1_r1_artifact_bundle(tmp_path / "art", git_commit="t")
    assert out["n_rows"] == 30
    assert (tmp_path / "art" / "final_training_protocol.yaml").is_file()
    assert (tmp_path / "art" / "final_training_protocol.sha256").is_file()
    assert (tmp_path / "art" / "protocol_amendment_record.yaml").is_file()
    assert (tmp_path / "art" / "protocol_diff_from_stage5c0.yaml").is_file()
    # Predecessor unchanged
    assert (
        sha256_file(
            Path(
                "experiments/formal/protocol/artifacts/20260730T072103Z_94767983/"
                "final_training_protocol.yaml"
            )
        )
        == SUPERSEDED_STAGE5C0_PROTOCOL_SHA256
    )


def test_matrix_rows():
    rows = build_formal_run_matrix_100k(
        protocol_sha256="p",
        environment_lock_sha256="e",
        comfort_lock_sha256="c",
    )
    assert len(rows) == 30
    assert all("replay_A_seed" in r for r in rows)
    assert all("formal_job_id" in r for r in rows)
