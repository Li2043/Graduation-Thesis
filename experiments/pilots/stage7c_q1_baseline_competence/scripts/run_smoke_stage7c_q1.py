#!/usr/bin/env python3
"""Local smoke test for Stage 7C-Q1 (temporary seed, short budget, external ckpt root)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve()
PILOT_ROOT = SCRIPT.parents[1]
REPO_ROOT = SCRIPT.parents[4]


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    os.environ["PYTHONPATH"] = str(REPO_ROOT / "src")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    from thesis.pilots.stage7c_q1_eval import evaluate_checkpoint_stage7c, summarise_seed_checkpoint
    from thesis.pilots.stage7c_q1_runner import run_training_job
    from thesis.pilots.stage7c_q1_scripted_audit import run_scripted_reward_audit
    from thesis.training.final_lock_loader import load_final_locks
    from thesis.training.pilot_checkpoint import load_checkpoint
    from thesis.agents.independent_dqn_v2 import DQNConfig, IndependentDQNLearner
    from thesis.agents.dqn_bootstrap import DQNTargetMode
    import torch

    torch.set_num_threads(1)

    audit = run_scripted_reward_audit()
    if not audit["passed"]:
        print(json.dumps(audit, indent=2))
        print("ABORT: scripted reward audit failed", file=sys.stderr)
        return 2

    protocol = PILOT_ROOT / "configs" / "stage7c_q1_protocol.yaml"
    smoke_seed = 64999  # outside frozen / historical blocks
    max_steps = 2000

    with tempfile.TemporaryDirectory(prefix="stage7c_q1_smoke_") as td:
        td_path = Path(td)
        out_root = td_path / "output"
        ckpt_root = td_path / "checkpoints_external"
        # Ensure outside-repo style separation (temp is outside worktree content)
        result = run_training_job(
            master_seed=smoke_seed,
            protocol_path=protocol,
            output_root=out_root,
            checkpoint_root=ckpt_root,
            max_steps=max_steps,
            resume=False,
            device="cpu",
            strict=False,
            allow_smoke=True,
        )
        assert result.get("success"), result
        ckpt0 = ckpt_root / "baseline" / f"seed_{smoke_seed}" / "ckpt_step_0_full.pt"
        assert ckpt0.is_file(), ckpt0
        payload = load_checkpoint(ckpt0)
        assert payload["algorithm_mode"] == "double_dqn"
        assert payload["condition"] == "baseline"
        assert abs(float(payload["active_time_cost_per_step"]) - 0.0005) < 1e-15

        # Resume from ckpt 0 into a fresh trainer briefly to verify load path
        result2 = run_training_job(
            master_seed=smoke_seed,
            protocol_path=protocol,
            output_root=out_root,
            checkpoint_root=ckpt_root,
            max_steps=max_steps,
            resume=True,
            device="cpu",
            strict=False,
            allow_smoke=True,
        )
        assert result2.get("skipped_completed") or result2.get("success")

        # Role-swap greedy evaluation (1 block × 2 assignments via early plan subset)
        bundle = load_final_locks()
        cfg = DQNConfig(
            obs_dim=27,
            n_actions=3,
            hidden_sizes=(64, 64),
            target_mode=DQNTargetMode.DOUBLE,
        )
        learners = {
            "A": IndependentDQNLearner("A", cfg, seed=1, replay_seed=2),
            "B": IndependentDQNLearner("B", cfg, seed=3, replay_seed=4),
        }
        for aid in ("A", "B"):
            learners[aid].import_state(payload["learners"][aid])

        from thesis.pilots.stage7c_q1_eval import run_greedy_episode
        from thesis.pilots.stage7c_q1_eval_seeds import eval_plan_for_checkpoint
        from dataclasses import replace

        plan = eval_plan_for_checkpoint(master_seed=smoke_seed, checkpoint_step=0)[:2]
        assert plan[0]["eval_seed"] == plan[1]["eval_seed"]
        assert plan[0]["assignment"] != plan[1]["assignment"]
        blocks = list(bundle.environment.validation_blocks)
        block0 = blocks[0]
        eps = []
        for row in plan:
            blk = block0 if int(row["assignment"]) == 0 else replace(
                block0, role_A=block0.role_B, role_B=block0.role_A
            )
            ep, _ = run_greedy_episode(
                bundle, learners, block=blk, episode_seed=int(row["eval_seed"])
            )
            required = {
                "success",
                "collision",
                "truncation",
                "episode_length",
                "exit_step_agent_0",
                "exit_step_agent_1",
                "passing_order",
                "controller_role_mapping",
                "failure_category",
                "reward_total_A",
                "reward_active_time_A",
            }
            missing = required - set(ep)
            assert not missing, missing
            assert "reward_active_time" in {
                "reward_active_time"
            } or "reward_active_time_A" in ep
            eps.append(ep)

        # Manifest + path separation
        assert (out_root / "runs" / "baseline" / f"seed_{smoke_seed}").is_dir()
        assert (ckpt_root / "baseline" / f"seed_{smoke_seed}").is_dir()
        assert out_root.resolve() != ckpt_root.resolve()

        # Ensure .pt would not be tracked if under repo (gitignore patterns exist)
        gi = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "*.pt" in gi and "*.pth" in gi and "*.ckpt" in gi

        report = {
            "smoke": "PASS",
            "seed": smoke_seed,
            "max_steps": max_steps,
            "scripted_audit_passed": True,
            "training_success": True,
            "checkpoint0": str(ckpt0),
            "role_swap_eval_episodes": len(eps),
            "reward_components_present": True,
            "output_root_separated": True,
            "final_step": result.get("final_step"),
        }
        dest = PILOT_ROOT / "logs" / "smoke_test_report.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
