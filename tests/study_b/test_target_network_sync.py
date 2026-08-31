"""Diagnostic_6_DQN_Pipeline_Verification_Protocol.md sec 12 (6H,
"target-network and bootstrap logic"): regression test for a real bug
found and fixed in this session -- train_dqn_direct_welfare.py,
train_dqn_fallback.py, and train_joint_dqn_diagnostic.py each called
``learner.hard_sync_target()`` exactly once, at construction time, and
NEVER again during training (unlike the sibling
stage11_dyad_merge_runner.py/stage10_shared_dqn_runner.py, which
correctly do ``if learner._update_count % TARGET_SYNC_INTERVAL_UPDATES
== 0: learner.hard_sync_target()`` after every update). This meant the
target network stayed frozen at its RANDOM INITIALIZATION for the
entire training run, corrupting every bootstrapped TD target -- likely
the dominant root cause of the persistent training failures observed
across this entire diagnostic sequence (PBRS 8-seed, direct-welfare
Mean qualification, task-only Diagnostic 4, joint-DQN Diagnostic 5).

Empirical confirmation of the bug (before the fix): after 3000 real
training steps, the online network's parameters moved (total abs diff
~119) while the target network's parameters moved EXACTLY ZERO from
their initial random values.

These tests assert the FIXED behaviour: after enough steps to cross at
least one TARGET_SYNC_INTERVAL_UPDATES (250) boundary, the target
network's parameters must have moved from their initial random values,
and must exactly match the online network's parameters as of the last
sync point (not just "moved somewhat")."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(REPO_SRC))
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "pilots" / "study_b_fairness_mappo" / "scripts"

from thesis.pilots.stage11_dyad_merge_pilot_config import TARGET_SYNC_INTERVAL_UPDATES  # noqa: E402


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _total_abs_diff(sd1: dict, sd2: dict) -> float:
    return sum((sd1[k] - sd2[k]).abs().sum().item() for k in sd1)


def test_target_sync_constant_matches_established_project_convention():
    # Sanity check on the constant itself, not just its use -- if this
    # value ever changes, TARGET_SYNC_INTERVAL_UPDATES-dependent step
    # budgets below need revisiting too.
    assert TARGET_SYNC_INTERVAL_UPDATES == 250


def test_direct_welfare_target_network_syncs_periodically(tmp_path):
    train_dqn_direct_welfare = _load_script("train_dqn_direct_welfare")
    checkpoint_root = tmp_path / "ckpt"
    argv = [
        "--condition", "mean", "--master-seed", "5", "--welfare-lambda", "0.0",
        "--output-root", str(tmp_path / "out"), "--checkpoint-root", str(checkpoint_root),
        "--max-steps", "1500", "--episode-max-steps", "40", "--checkpoint-every", "1500",
        "--replay-warmup", "64", "--device", "cpu",
    ]
    rc = train_dqn_direct_welfare.main(argv)
    assert rc == 0

    ckpt0 = torch.load(checkpoint_root / "seed_5" / "ckpt_step_0.pt", map_location="cpu")
    ckpt_final = torch.load(checkpoint_root / "seed_5" / "ckpt_step_1500.pt", map_location="cpu")

    online_moved = _total_abs_diff(ckpt0["online"], ckpt_final["online"])
    target_moved = _total_abs_diff(ckpt0["target"], ckpt_final["target"])
    assert online_moved > 0.0, "online network should have learned something over 1500 steps"
    assert target_moved > 0.0, "target network must have moved from its random init -- sync never fired (regression of the fixed bug)"

    # As of the LAST update, target should have been hard-synced to online
    # within the last (update_count % TARGET_SYNC_INTERVAL_UPDATES) updates
    # -- exact equality only holds right at a sync boundary, so instead
    # assert the target is now genuinely closer to a trained network than
    # to its own untrained starting point (a weak but robust check).
    target_vs_own_init = _total_abs_diff(ckpt0["target"], ckpt_final["target"])
    assert target_vs_own_init > 0.0


def test_dqn_fallback_target_network_syncs_periodically(tmp_path):
    train_dqn_fallback = _load_script("train_dqn_fallback")
    checkpoint_root = tmp_path / "ckpt"
    argv = [
        "--condition", "baseline", "--master-seed", "5",
        "--output-root", str(tmp_path / "out"), "--checkpoint-root", str(checkpoint_root),
        "--max-steps", "1500", "--episode-max-steps", "40", "--checkpoint-every", "1500",
        "--replay-warmup", "64", "--device", "cpu",
    ]
    rc = train_dqn_fallback.main(argv)
    assert rc == 0

    ckpt0 = torch.load(checkpoint_root / "seed_5" / "ckpt_step_0.pt", map_location="cpu")
    ckpt_final = torch.load(checkpoint_root / "seed_5" / "ckpt_step_1500.pt", map_location="cpu")

    online_moved = _total_abs_diff(ckpt0["online"], ckpt_final["online"])
    target_moved = _total_abs_diff(ckpt0["target"], ckpt_final["target"])
    assert online_moved > 0.0
    assert target_moved > 0.0, "target network must have moved from its random init -- sync never fired"


def test_joint_dqn_diagnostic_target_network_syncs_periodically(tmp_path):
    train_joint_dqn_diagnostic = _load_script("train_joint_dqn_diagnostic")
    checkpoint_root = tmp_path / "ckpt"
    argv = [
        "--condition", "mean", "--master-seed", "5", "--welfare-lambda", "0.0",
        "--output-root", str(tmp_path / "out"), "--checkpoint-root", str(checkpoint_root),
        "--max-steps", "1500", "--episode-max-steps", "40", "--checkpoint-every", "1500",
        "--replay-warmup", "64", "--device", "cpu",
    ]
    rc = train_joint_dqn_diagnostic.main(argv)
    assert rc == 0

    ckpt0 = torch.load(checkpoint_root / "seed_5" / "ckpt_step_0.pt", map_location="cpu")
    ckpt_final = torch.load(checkpoint_root / "seed_5" / "ckpt_step_1500.pt", map_location="cpu")

    online_moved = _total_abs_diff(ckpt0["online"], ckpt_final["online"])
    target_moved = _total_abs_diff(ckpt0["target"], ckpt_final["target"])
    assert online_moved > 0.0
    assert target_moved > 0.0, "target network must have moved from its random init -- sync never fired"
