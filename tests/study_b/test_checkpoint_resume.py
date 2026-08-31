"""Integration tests for the --resume-from workflow in both training
scripts -- exercises the exact scenario a machine switch needs: train a
bit, stop, resume on a "different" invocation (same process here, but
config is re-parsed from scratch each call exactly like a fresh process
would), and confirm the run continues past the original max_steps with a
single continuous manifest."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "pilots" / "study_b_fairness_mappo" / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


train_mappo = _load_script("train_mappo")
train_dqn_fallback = _load_script("train_dqn_fallback")


def test_mappo_resume_continues_past_original_max_steps(tmp_path):
    output_root = tmp_path / "output"
    checkpoint_root = tmp_path / "checkpoints"
    common = [
        "--condition", "baseline", "--master-seed", "1",
        "--output-root", str(output_root), "--checkpoint-root", str(checkpoint_root),
        "--n-parallel-envs", "2", "--rollout-length", "20", "--episode-max-steps", "40",
        "--checkpoint-every", "100", "--device", "cpu", "--hidden-size", "8",
    ]
    rc = train_mappo.main(common + ["--max-steps", "100"])
    assert rc == 0

    ckpt_path = checkpoint_root / "seed_1" / "ckpt_step_100.pt"
    assert ckpt_path.exists()

    rc = train_mappo.main(common + ["--max-steps", "200", "--resume-from", str(ckpt_path)])
    assert rc == 0

    manifest_path = output_root / "seed_1_baseline_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["final_step"] == 200
    steps = [c["step"] for c in manifest["checkpoints"]]
    assert steps == sorted(steps)
    assert 0 in steps and 100 in steps and 200 in steps


def test_mappo_resume_rejects_condition_mismatch(tmp_path):
    output_root = tmp_path / "output"
    checkpoint_root = tmp_path / "checkpoints"
    base_args = [
        "--master-seed", "2", "--output-root", str(output_root), "--checkpoint-root", str(checkpoint_root),
        "--n-parallel-envs", "2", "--rollout-length", "20", "--episode-max-steps", "40",
        "--checkpoint-every", "100", "--device", "cpu", "--hidden-size", "8",
    ]
    train_mappo.main(["--condition", "baseline", "--max-steps", "100"] + base_args)
    ckpt_path = checkpoint_root / "seed_2" / "ckpt_step_100.pt"

    with pytest.raises(ValueError, match="condition"):
        train_mappo.main(
            ["--condition", "mean_pbrs", "--max-steps", "200"] + base_args + ["--resume-from", str(ckpt_path)]
        )


def test_mappo_resume_rejects_hidden_size_mismatch(tmp_path):
    output_root = tmp_path / "output"
    checkpoint_root = tmp_path / "checkpoints"
    base_args = [
        "--condition", "baseline", "--master-seed", "3",
        "--output-root", str(output_root), "--checkpoint-root", str(checkpoint_root),
        "--n-parallel-envs", "2", "--rollout-length", "20", "--episode-max-steps", "40",
        "--checkpoint-every", "100", "--device", "cpu",
    ]
    train_mappo.main(base_args + ["--max-steps", "100", "--hidden-size", "8"])
    ckpt_path = checkpoint_root / "seed_3" / "ckpt_step_100.pt"

    with pytest.raises(ValueError, match="hidden_sizes"):
        train_mappo.main(base_args + ["--max-steps", "200", "--hidden-size", "16", "--resume-from", str(ckpt_path)])


def test_dqn_fallback_resume_continues_past_original_max_steps(tmp_path):
    output_root = tmp_path / "output"
    checkpoint_root = tmp_path / "checkpoints"
    common = [
        "--condition", "min_pbrs", "--master-seed", "4",
        "--output-root", str(output_root), "--checkpoint-root", str(checkpoint_root),
        "--episode-max-steps", "40", "--checkpoint-every", "100", "--replay-warmup", "32", "--device", "cpu",
    ]
    rc = train_dqn_fallback.main(common + ["--max-steps", "100"])
    assert rc == 0

    ckpt_path = checkpoint_root / "seed_4" / "ckpt_step_100.pt"
    assert ckpt_path.exists()

    rc = train_dqn_fallback.main(common + ["--max-steps", "200", "--resume-from", str(ckpt_path)])
    assert rc == 0

    manifest_path = output_root / "seed_4_min_pbrs_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["final_step"] == 200
    steps = [c["step"] for c in manifest["checkpoints"]]
    assert 0 in steps and 100 in steps and 200 in steps


def test_dqn_fallback_resume_rejects_condition_mismatch(tmp_path):
    output_root = tmp_path / "output"
    checkpoint_root = tmp_path / "checkpoints"
    base_args = [
        "--master-seed", "5", "--output-root", str(output_root), "--checkpoint-root", str(checkpoint_root),
        "--episode-max-steps", "40", "--checkpoint-every", "100", "--replay-warmup", "32", "--device", "cpu",
    ]
    train_dqn_fallback.main(["--condition", "baseline", "--max-steps", "100"] + base_args)
    ckpt_path = checkpoint_root / "seed_5" / "ckpt_step_100.pt"

    with pytest.raises(ValueError, match="condition"):
        train_dqn_fallback.main(
            ["--condition", "min_pbrs", "--max-steps", "200"] + base_args + ["--resume-from", str(ckpt_path)]
        )
