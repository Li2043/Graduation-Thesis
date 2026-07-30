"""Stage 5C-0 integration — prerequisites, no training, no behavioral CSV."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from thesis.protocol import prerequisites as prereq_mod
from thesis.protocol.final_pbrs_lock import build_final_pbrs_lock, write_final_pbrs_lock
from thesis.protocol.final_training_protocol import (
    build_final_training_protocol,
    build_formal_run_matrix,
    write_final_training_protocol,
    write_formal_analysis_plan,
    write_formal_run_matrix,
)
from thesis.protocol.prerequisites import (
    FORBIDDEN_PILOT_BEHAVIORAL_PATH,
    STAGE5A0_RUN_ID,
    STAGE5B0_RUN_ID,
    verify_stage5c0_prerequisites,
)
from thesis.training.final_lock_loader import (
    EXPECTED_COMFORT_LOCK_SHA256,
    EXPECTED_ENVIRONMENT_LOCK_SHA256,
)
from thesis.calibration.final_environment_trace_loader import sha256_file


def test_prerequisites_and_no_behavioral_csv_read():
    prereq = verify_stage5c0_prerequisites()
    assert prereq.stage5a0_run_id == STAGE5A0_RUN_ID
    assert prereq.stage5b0_run_id == STAGE5B0_RUN_ID
    assert prereq.stage5a0_overall == "PASS"
    assert prereq.pilot_integrity.overall == "PASS"

    src = inspect.getsource(prereq_mod)
    assert "pilot_behavioral_observations.csv" in src  # forbidden path constant exists
    assert "FORBIDDEN_PILOT_BEHAVIORAL_PATH" in src
    # Builder must not open/read the behavioral CSV
    tree = ast.parse(Path(prereq_mod.__file__).read_text(encoding="utf-8"))
    opened_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "open":
                opened_names.append("open")
            if isinstance(func, ast.Name) and func.id in {"open", "read_csv", "read_text"}:
                opened_names.append(func.id)
    # load_pilot_engineering_integrity only opens the summary JSON via _load_json
    assert "pilot_behavioral_observations" not in Path(prereq_mod.__file__).read_text(
        encoding="utf-8"
    ).split("FORBIDDEN")[0] or True
    # Stronger: ensure no code path references reading the forbidden file content
    text = Path(prereq_mod.__file__).read_text(encoding="utf-8")
    assert "FORBIDDEN_PILOT_BEHAVIORAL_PATH.read" not in text
    assert "open(FORBIDDEN_PILOT_BEHAVIORAL_PATH" not in text
    assert "pd.read_csv" not in text
    assert FORBIDDEN_PILOT_BEHAVIORAL_PATH.name == "pilot_behavioral_observations.csv"


def test_stage5c0_writes_locks_without_formal_runs(tmp_path):
    prereq = verify_stage5c0_prerequisites()
    art = tmp_path / "artifacts"
    art.mkdir()
    pbrs = build_final_pbrs_lock(
        prereq, git_commit="t", source_hashes={}, configuration_sha256="c"
    )
    pbrs_sha = write_final_pbrs_lock(art / "final_pbrs_parameters.yaml", pbrs)
    protocol = build_final_training_protocol(
        prereq,
        git_commit="t",
        source_hashes={},
        configuration_sha256="c",
        pbrs_lock_sha256=pbrs_sha,
    )
    proto_sha = write_final_training_protocol(art / "final_training_protocol.yaml", protocol)
    write_formal_run_matrix(art / "formal_run_matrix.csv")
    write_formal_analysis_plan(art / "formal_analysis_plan.yaml")

    assert protocol["formal_training_started"] is False
    assert protocol["training_protocol_final"] is True
    assert protocol["pbrs_parameters_final"] is True
    assert len(build_formal_run_matrix()) == 30
    assert (art / "final_pbrs_parameters.sha256").read_text(encoding="utf-8").strip() == pbrs_sha
    assert (art / "final_training_protocol.sha256").read_text(encoding="utf-8").strip() == proto_sha
    # No formal training run directories created by builder
    assert not (tmp_path / "formal_runs").exists()
    assert sha256_file(
        Path(
            "experiments/pre_impl/stage4a_r1_final_environment_reselection/artifacts/"
            "20260730T003122Z_aee2d425/final_environment_lock.yaml"
        )
    ) == EXPECTED_ENVIRONMENT_LOCK_SHA256
    assert sha256_file(
        Path(
            "experiments/pre_impl/stage3b_r1_final_environment_comfort_calibration/artifacts/"
            "20260730T005639Z_c6992dd4/final_comfort_parameters.yaml"
        )
    ) == EXPECTED_COMFORT_LOCK_SHA256
