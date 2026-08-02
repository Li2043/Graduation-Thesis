#!/usr/bin/env python3
"""Local smoke test for the Stage 8 formal gate (temporary seed, short
budget, external ckpt root).

Wider scope than the pilot-arm smoke tests, because the gate introduces
genuinely new code paths none of arm0/arm1/arm2a/arm2b exercised:
  1. learning-rate decay extended to a 400,000-step window (vs arm2b's
     100,000) -- sanity-checked directly via lr_at_step().
  2. the "late" 32-scenario-block evaluation plan (checkpoint > 175,000) --
     every prior Stage 8 arm stayed at 100,000 steps and never touched this
     code path. Exercised here against the smoke-trained checkpoint with a
     spoofed checkpoint_step=300_000 (only affects eval-seed derivation and
     episode count, not the correctness of the block-cycling logic itself).
  3. BOTH evaluators the gate's evaluation driver actually calls: the rich
     diagnostic evaluator (stage8_gate_eval, for RICH_LOG_CHECKPOINTS) and
     the lightweight one reused unchanged from Stage 7C-Q1
     (stage7c_q1_eval.evaluate_checkpoint_stage7c, for every other
     checkpoint) -- both are exercised against early (8-block) AND late
     (32-block) plans.
  4. the gate-decision function itself (stage7c_q1_gate.evaluate_competence_gate),
     smoke-tested against a tiny synthetic dataframe engineered to PASS, so
     an import/signature break would be caught before real compute is spent.
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

    from thesis.pilots.stage8_gate_runner import run_training_job
    from thesis.pilots.stage8_gate_config import (
        LEARNING_RATE_END,
        LEARNING_RATE_DECAY_STEPS,
        TARGET_UPDATE_MODE,
        n_scenario_blocks,
        episodes_per_seed_checkpoint,
    )
    from thesis.formal.formal_config import lr_at_step, FormalDQNConfig
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

    # (1) LR decay math over the 400K window, before spending any training time.
    dqn_cfg = FormalDQNConfig(
        learning_rate=0.0005,
        learning_rate_end=LEARNING_RATE_END,
        learning_rate_decay_environment_steps=LEARNING_RATE_DECAY_STEPS,
    )
    assert lr_at_step(0, dqn_cfg) == 0.0005
    assert abs(lr_at_step(LEARNING_RATE_DECAY_STEPS, dqn_cfg) - LEARNING_RATE_END) < 1e-15
    mid_lr = lr_at_step(LEARNING_RATE_DECAY_STEPS // 2, dqn_cfg)
    assert LEARNING_RATE_END < mid_lr < 0.0005, mid_lr

    # Early/late scenario-block counts.
    assert n_scenario_blocks(175_000) == 8
    assert n_scenario_blocks(200_000) == 32
    assert n_scenario_blocks(400_000) == 32
    assert episodes_per_seed_checkpoint(400_000) == 64

    bundle = load_final_locks()

    protocol = PILOT_ROOT / "configs" / "stage8_gate_protocol.yaml"
    smoke_seed = 65994  # outside frozen (65021-65040) / all pilot / historical blocks
    max_steps = 2000  # > replay_warmup_per_controller (512): exercises real updates

    with tempfile.TemporaryDirectory(prefix="stage8_gate_smoke_") as td:
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

        meta0 = json.loads(
            (ckpt_root / "baseline" / f"seed_{smoke_seed}" / "checkpoint_metadata_0.json").read_text(
                encoding="utf-8"
            )
        )
        assert meta0["target_update_mode"] == TARGET_UPDATE_MODE == "hard", meta0
        assert abs(float(meta0["learning_rate_end"]) - LEARNING_RATE_END) < 1e-15, meta0
        assert int(meta0["learning_rate_decay_environment_steps"]) == LEARNING_RATE_DECAY_STEPS, meta0

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

        from thesis.pilots.stage8_gate_eval import evaluate_checkpoint_stage8_gate
        from thesis.pilots.stage7c_q1_eval import evaluate_checkpoint_stage7c, compute_swap_eligibility
        from thesis.pilots.stage8_gate_config import PROTOCOL_TAG

        # (3a) Rich evaluator, EARLY plan (checkpoint_step=0, 16 episodes).
        rich_early = evaluate_checkpoint_stage8_gate(
            bundle,
            learners,
            master_seed=smoke_seed,
            checkpoint_step=0,
            code_commit="smoke",
            protocol_tag=PROTOCOL_TAG,
            collect_trajectories=True,
        )
        assert rich_early["n_episodes"] == 16, rich_early["n_episodes"]
        assert len(rich_early["trajectories"]) > 0
        required_every_row = {
            "policy_step", "controller", "controller_role", "route_progress", "speed",
            "Q_maintain", "Q_accelerate", "Q_decelerate", "greedy_action", "best_Q",
            "Q_margin", "action_mask", "commanded_action_name", "joint_action_category",
        }
        for srow in rich_early["trajectories"]:
            missing_step = required_every_row - set(srow)
            assert not missing_step, missing_step

        # (2)+(3a) Rich evaluator, LATE plan (spoofed checkpoint_step=300_000,
        # exercises the never-before-tested 32-scenario-block cycling logic).
        rich_late = evaluate_checkpoint_stage8_gate(
            bundle,
            learners,
            master_seed=smoke_seed,
            checkpoint_step=300_000,
            code_commit="smoke",
            protocol_tag=PROTOCOL_TAG,
            collect_trajectories=True,
        )
        assert rich_late["n_episodes"] == 64, rich_late["n_episodes"]
        assert len(rich_late["trajectories"]) > 0
        # 32 scenario_block values (0-31) must each map to one of the 8
        # physical templates via %8, all 8 templates must actually appear.
        seen_templates = {int(e["scenario_block"]) % 8 for e in rich_late["episodes"]}
        assert seen_templates == set(range(8)), seen_templates

        # (3b) Lightweight evaluator (reused unchanged from Stage 7C-Q1),
        # EARLY plan.
        light_early = evaluate_checkpoint_stage7c(
            bundle,
            learners,
            master_seed=smoke_seed,
            checkpoint_step=0,
            code_commit="smoke",
            protocol_tag=PROTOCOL_TAG,
        )
        assert light_early["n_episodes"] == 16, light_early["n_episodes"]
        swap_e = compute_swap_eligibility(light_early["episodes"])
        assert 0.0 <= swap_e <= 1.0

        # (3b)+(2) Lightweight evaluator, LATE plan.
        light_late = evaluate_checkpoint_stage7c(
            bundle,
            learners,
            master_seed=smoke_seed,
            checkpoint_step=300_000,
            code_commit="smoke",
            protocol_tag=PROTOCOL_TAG,
        )
        assert light_late["n_episodes"] == 64, light_late["n_episodes"]

        # (4) Gate-decision function smoke test: tiny synthetic dataframe
        # engineered to PASS, just to catch an import/signature break early.
        import pandas as pd
        from thesis.pilots.stage7c_q1_gate import evaluate_competence_gate

        # GATE_MIN_QUALIFIED_SEEDS=16, so the synthetic set needs >=16 seeds
        # all qualifying for intersection_ok to hold -- this is exercising
        # real, unmodified gate math (stage7c_q1_gate.py), not a relaxed
        # smoke-only threshold.
        synth_seeds = tuple(range(90001, 90021))
        rows = []
        for seed in synth_seeds:
            for ckpt in range(200_000, 400_001, 25_000):
                rows.append(
                    {
                        "master_seed": seed,
                        "checkpoint_step": ckpt,
                        "success_rate": 1.0,
                        "collision_rate": 0.0,
                        "truncation_rate": 0.0,
                        "swap_eligibility": 1.0,
                    }
                )
        synth_df = pd.DataFrame(rows)
        synth_decision = evaluate_competence_gate(synth_df, expected_seeds=synth_seeds)
        assert synth_decision["status"] == "PASS", synth_decision

        gi = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "*.pt" in gi and "*.pth" in gi and "*.ckpt" in gi

        report = {
            "smoke": "PASS",
            "seed": smoke_seed,
            "max_steps": max_steps,
            "target_update_mode_confirmed": meta0["target_update_mode"],
            "learning_rate_end_confirmed": meta0["learning_rate_end"],
            "learning_rate_decay_environment_steps_confirmed": meta0["learning_rate_decay_environment_steps"],
            "rich_early_episodes": rich_early["n_episodes"],
            "rich_late_episodes": rich_late["n_episodes"],
            "light_early_episodes": light_early["n_episodes"],
            "light_late_episodes": light_late["n_episodes"],
            "late_plan_template_coverage": sorted(seen_templates),
            "gate_decision_smoke_status": synth_decision["status"],
            "scripted_audit_passed": True,
            "training_success": True,
            "final_step": result.get("final_step"),
        }
        dest = PILOT_ROOT / "logs" / "smoke_test_report.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
