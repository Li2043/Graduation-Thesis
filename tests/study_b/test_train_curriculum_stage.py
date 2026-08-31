"""Tests for train_curriculum_stage.py -- Claude_Code_Autonomous_Experiment_Runbook.md
Diagnostic 6N curriculum training script."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "pilots" / "study_b_fairness_mappo" / "scripts"
SCENARIO_BANK = Path(__file__).resolve().parents[2] / "experiments/pilots/study_b_fairness_mappo/scenario_banks/Q.json"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


curriculum = _load_script("train_curriculum_stage")


def _skip_if_no_bank():
    if not SCENARIO_BANK.exists():
        import pytest

        pytest.skip("requires the frozen Q scenario bank on disk")


def test_smoke_fresh_run_end_to_end(tmp_path):
    _skip_if_no_bank()
    argv = [
        "--scenario-bank", str(SCENARIO_BANK), "--scenario-ids", "Q_00000", "Q_00016", "Q_00032", "Q_00048",
        "--stage-name", "C4_test", "--master-seed", "1",
        "--output-root", str(tmp_path / "out"), "--checkpoint-root", str(tmp_path / "ckpt"),
        "--start-step", "0", "--max-additional-steps", "300", "--episode-max-steps", "40",
        "--checkpoint-every", "150", "--replay-warmup", "64",
    ]
    rc = curriculum.main(argv)
    assert rc == 0

    manifest = json.loads((tmp_path / "out" / "seed_1_C4_test_manifest.json").read_text(encoding="utf-8"))
    assert manifest["final_step"] == 300
    assert manifest["start_step"] == 0
    assert len(manifest["checkpoints"]) >= 2
    record = manifest["checkpoints"][-1]
    assert "q_diagnostics_policy_visited" in record
    assert "q_diagnostics_fixed_oracle_ref" in record
    assert set(record["per_scenario"]) == {"Q_00000", "Q_00016", "Q_00032", "Q_00048"}


def test_resume_continues_from_absolute_step_with_matching_q_values(tmp_path):
    _skip_if_no_bank()
    common_scenarios = ["--scenario-ids", "Q_00000", "Q_00016"]
    ckpt_root = tmp_path / "ckpt"

    argv1 = [
        "--scenario-bank", str(SCENARIO_BANK), *common_scenarios,
        "--stage-name", "C_resume_test", "--master-seed", "2",
        "--output-root", str(tmp_path / "out1"), "--checkpoint-root", str(ckpt_root),
        "--start-step", "0", "--max-additional-steps", "400", "--episode-max-steps", "40",
        "--checkpoint-every", "400", "--replay-warmup", "64",
    ]
    assert curriculum.main(argv1) == 0
    ckpt_path = ckpt_root / "seed_2_C_resume_test" / "ckpt_step_400.pt"
    assert ckpt_path.exists()

    argv2 = [
        "--scenario-bank", str(SCENARIO_BANK), *common_scenarios,
        "--stage-name", "C_resume_test", "--master-seed", "2",
        "--output-root", str(tmp_path / "out2"), "--checkpoint-root", str(tmp_path / "ckpt2"),
        "--start-step", "400", "--max-additional-steps", "200", "--episode-max-steps", "40",
        "--checkpoint-every", "200", "--replay-warmup", "64", "--resume-from", str(ckpt_path),
    ]
    assert curriculum.main(argv2) == 0

    manifest2 = json.loads((tmp_path / "out2" / "seed_2_C_resume_test_manifest.json").read_text(encoding="utf-8"))
    assert manifest2["start_step"] == 400
    assert manifest2["final_step"] == 600

    # The network state loaded from the checkpoint must exactly match what was saved.
    ckpt1 = torch.load(ckpt_path, map_location="cpu")
    resumed_online_first_checkpoint = torch.load(tmp_path / "ckpt2" / "seed_2_C_resume_test" / "ckpt_step_400.pt", map_location="cpu")
    for key in ckpt1["online"]:
        torch.testing.assert_close(ckpt1["online"][key], resumed_online_first_checkpoint["online"][key])


def test_scenario_selection_stays_within_stage_set(tmp_path):
    """Every episode's scenario must come from the passed --scenario-ids
    set -- exercised indirectly via per_scenario keys never containing an
    id outside the requested set."""
    _skip_if_no_bank()
    argv = [
        "--scenario-bank", str(SCENARIO_BANK), "--scenario-ids", "Q_00000", "Q_00001",
        "--stage-name", "C_selection_test", "--master-seed", "3",
        "--output-root", str(tmp_path / "out"), "--checkpoint-root", str(tmp_path / "ckpt"),
        "--start-step", "0", "--max-additional-steps", "500", "--episode-max-steps", "30",
        "--checkpoint-every", "500", "--replay-warmup", "64",
    ]
    assert curriculum.main(argv) == 0
    manifest = json.loads((tmp_path / "out" / "seed_3_C_selection_test_manifest.json").read_text(encoding="utf-8"))
    per_scenario = manifest["checkpoints"][-1]["per_scenario"]
    assert set(per_scenario) == {"Q_00000", "Q_00001"}
