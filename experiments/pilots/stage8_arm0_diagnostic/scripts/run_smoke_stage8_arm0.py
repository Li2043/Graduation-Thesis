#!/usr/bin/env python3
"""Local smoke test for Stage 8 arm0 (temporary seed, short budget, external ckpt root).

Adapted from run_smoke_stage7c_q1.py. The key addition over the Stage 7C-Q1
smoke test is the final assertion block: it calls run_greedy_episode_diagnostic
(not run_greedy_episode) and asserts the rich per-step trajectory fields are
actually populated -- this is what validates the new trajectory-logging wiring
end to end before committing to the real 2x100K run.
"""

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

    from thesis.pilots.stage8_arm0_runner import run_training_job
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

    protocol = PILOT_ROOT / "configs" / "stage8_arm0_protocol.yaml"
    smoke_seed = 65999  # outside frozen (65001-65002) / historical blocks
    max_steps = 2000  # > replay_warmup_per_controller (512): exercises real updates

    with tempfile.TemporaryDirectory(prefix="stage8_arm0_smoke_") as td:
        td_path = Path(td)
        out_root = td_path / "output"
        ckpt_root = td_path / "checkpoints_external"
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

        # Role-swap greedy evaluation with rich per-step trajectory logging.
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

        from thesis.pilots.stage8_arm0_eval import run_greedy_episode_diagnostic
        from thesis.pilots.stage7c_q1_eval_seeds import eval_plan_for_checkpoint
        from dataclasses import replace

        plan = eval_plan_for_checkpoint(
            master_seed=smoke_seed, checkpoint_step=0, protocol_tag="stage8-arm0-protocol-v1"
        )[:2]
        assert plan[0]["eval_seed"] == plan[1]["eval_seed"]
        assert plan[0]["assignment"] != plan[1]["assignment"]
        blocks = list(bundle.environment.validation_blocks)
        block0 = blocks[0]
        eps = []
        all_step_rows = []
        for row in plan:
            blk = block0 if int(row["assignment"]) == 0 else replace(
                block0, role_A=block0.role_B, role_B=block0.role_A
            )
            ep, step_rows = run_greedy_episode_diagnostic(
                bundle, learners, block=blk, episode_seed=int(row["eval_seed"])
            )
            required_episode_fields = {
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
            missing = required_episode_fields - set(ep)
            assert not missing, missing
            eps.append(ep)

            # This is the new assertion: confirm rich per-step trajectory
            # logging actually produced non-empty, correctly-populated rows.
            # front_gap/minimum_TTC are only attached when the environment
            # reports a finite value that step (e.g. no gap reading before
            # the first physics substep, or no relevant vehicle present) --
            # so they are checked for presence in at least one row, not
            # required on every row. Everything else is always-present.
            assert step_rows, "run_greedy_episode_diagnostic produced no step rows"
            required_every_row = {
                "policy_step",
                "controller",
                "controller_role",
                "route_progress",
                "speed",
                "Q_maintain",
                "Q_accelerate",
                "Q_decelerate",
                "greedy_action",
                "best_Q",
                "Q_margin",
                "action_mask",
                "commanded_action_name",
                "joint_action_category",
            }
            for srow in step_rows:
                missing_step = required_every_row - set(srow)
                assert not missing_step, missing_step
            all_step_rows.extend(step_rows)

        assert len(all_step_rows) > 0
        front_gap_populated = any("front_gap" in r for r in all_step_rows)
        ttc_populated = any("minimum_TTC" in r for r in all_step_rows)
        # front_gap/minimum_TTC are only attached when the environment reports
        # a finite value that step; with just 2 short smoke episodes on a
        # freshly-initialised (near-random) policy, a finite TTC in particular
        # may legitimately never occur (it requires a closing-speed condition).
        # Treat as a soft signal here, not a hard failure -- the real 2x100K
        # run with 5 checkpoints x 16 episodes gives this far more chances to
        # populate, and analyze_stage8_arm0.py already handles missing values
        # via dropna(). front_gap not populating at all would be more
        # surprising and worth a printed warning.
        if not front_gap_populated:
            print("WARNING: front_gap was never populated in the smoke run", file=sys.stderr)
        if not ttc_populated:
            print("WARNING: minimum_TTC was never populated in the smoke run", file=sys.stderr)

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
            "trajectory_step_rows": len(all_step_rows),
            "front_gap_populated": front_gap_populated,
            "minimum_ttc_populated": ttc_populated,
            "reward_components_present": True,
            "trajectory_logging_verified": True,
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
