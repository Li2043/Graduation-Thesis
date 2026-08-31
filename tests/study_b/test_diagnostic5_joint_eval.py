"""Smoke tests for the joint-DQN evaluation path added to support
Diagnostic 5 (train_joint_dqn_diagnostic.run_eval_joint,
analyze_greedy_action_distribution.tally_greedy_actions_joint, and
multi_checkpoint_eval.py's --algorithm joint_dqn dispatch). Uses a real,
short-lived joint DQN checkpoint (freshly trained for a handful of steps)
so these tests don't depend on any specific long-running background job."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "pilots" / "study_b_fairness_mappo" / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


diagnostic = _load_script("train_joint_dqn_diagnostic")
analyze = _load_script("analyze_greedy_action_distribution")
multi_eval = _load_script("multi_checkpoint_eval")


def _train_tiny_joint_checkpoint(tmp_path) -> Path:
    argv = [
        "--condition", "mean", "--master-seed", "11", "--welfare-lambda", "0.0",
        "--output-root", str(tmp_path / "out"), "--checkpoint-root", str(tmp_path / "ckpt"),
        "--max-steps", "200", "--episode-max-steps", "30", "--checkpoint-every", "100",
        "--replay-warmup", "32", "--device", "cpu",
    ]
    rc = diagnostic.main(argv)
    assert rc == 0
    ckpt = tmp_path / "ckpt" / "seed_11" / "ckpt_step_200.pt"
    assert ckpt.exists()
    return ckpt


def test_run_eval_joint_produces_evaluate_policy_compatible_rows(tmp_path):
    ckpt = _train_tiny_joint_checkpoint(tmp_path)
    scenario_bank = (
        Path(__file__).resolve().parents[2]
        / "experiments/pilots/study_b_fairness_mappo/scenario_banks/Q.json"
    )
    rows = diagnostic.run_eval_joint(checkpoint=ckpt, scenario_bank=scenario_bank, episode_max_steps=20)
    assert len(rows) == 64
    required_fields = {
        "scenario_id", "term_reason", "completion", "collision", "timeout",
        "mean_U", "min_U", "episode_length", "mean_undiscounted_return",
    }
    assert required_fields.issubset(rows[0].keys())


def test_tally_greedy_actions_joint_covers_all_four_classes(tmp_path):
    ckpt = _train_tiny_joint_checkpoint(tmp_path)
    scenario_bank = (
        Path(__file__).resolve().parents[2]
        / "experiments/pilots/study_b_fairness_mappo/scenario_banks/Q.json"
    )
    report = analyze.tally_greedy_actions_joint(checkpoint=ckpt, scenario_bank=scenario_bank, episode_max_steps=20)
    assert set(report["distributions"]) == {"ramp-fast", "ramp-slow", "mainline-fast", "mainline-slow"}
    for dist in report["distributions"].values():
        assert abs(sum(dist.values()) - 1.0) < 1e-9


def test_multi_checkpoint_eval_dispatches_joint_dqn_algorithm(tmp_path):
    ckpt = _train_tiny_joint_checkpoint(tmp_path)
    scenario_bank = (
        Path(__file__).resolve().parents[2]
        / "experiments/pilots/study_b_fairness_mappo/scenario_banks/Q.json"
    )
    report = multi_eval.evaluate_checkpoints(
        checkpoint_dir=ckpt.parent, scenario_bank=scenario_bank, output_dir=tmp_path / "eval_out",
        steps=(200,), algorithm="joint_dqn", episode_max_steps=20,
    )
    assert report["checkpoints"][0]["step"] == 200
    assert 0.0 <= report["checkpoints"][0]["completion_rate"] <= 1.0
