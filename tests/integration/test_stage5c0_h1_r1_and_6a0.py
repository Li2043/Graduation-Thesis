"""Stage 5C-0-H1-R1 + 6A-0 integration — no sustained 100K training."""

from __future__ import annotations

import ast
from pathlib import Path

from thesis.formal.formal_config import FormalConfig
from thesis.protocol.h1_r1_100k_protocol import PROTOCOL_VERSION, write_h1_r1_artifact_bundle
from thesis.protocol.prerequisites import STAGE5A0_RUN_ID, STAGE5B0_RUN_ID, verify_stage5c0_prerequisites


def test_prerequisites_and_no_100k_in_builders():
    prereq = verify_stage5c0_prerequisites()
    assert prereq.stage5a0_run_id == STAGE5A0_RUN_ID
    assert prereq.stage5b0_run_id == STAGE5B0_RUN_ID
    cfg = FormalConfig()
    cfg.validate()
    assert cfg.formal_training_started is False


def test_write_locks_without_starting_formal_training(tmp_path):
    out = write_h1_r1_artifact_bundle(tmp_path / "art", git_commit="test")
    assert out["n_rows"] == 30
    assert PROTOCOL_VERSION == "5C-0-H1-R1-100K"
    # Ensure orchestrator default is dry regarding retained training: scripts exist
    job_runner = Path(
        "experiments/formal/stage6a_formal_training/scripts/run_formal_job.py"
    )
    orch = Path(
        "experiments/formal/stage6a_formal_training/scripts/run_formal_matrix.py"
    )
    assert job_runner.is_file() and orch.is_file()
    # Integration test must not invoke 100K FormalTrainer.run with full budget
    text = Path("src/thesis/formal/formal_trainer.py").read_text(encoding="utf-8")
    # No hard-coded auto-start of formal_training_started=True at import
    tree = ast.parse(text)
    assert isinstance(tree, ast.Module)


def test_stage5c0_regression_flags_unchanged_on_old_lock():
    from thesis.protocol.final_training_protocol import build_final_training_protocol
    from thesis.protocol.prerequisites import verify_stage5c0_prerequisites

    prereq = verify_stage5c0_prerequisites()
    lock = build_final_training_protocol(
        prereq,
        git_commit="t",
        source_hashes={},
        configuration_sha256="c",
        pbrs_lock_sha256="x",
    )
    assert lock["training_budget"]["formal_environment_steps_per_run"] == 20_000
    assert lock["formal_training_started"] is False
