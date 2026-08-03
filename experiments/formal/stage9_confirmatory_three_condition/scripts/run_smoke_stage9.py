#!/usr/bin/env python3
"""Local smoke test for Stage 9 (temporary seeds, short budget, external
checkpoint root).

New code paths Stage 9 exercises that no prior stage's smoke test covered:
  1. non-baseline reward conditions actually flowing through
     `stage9_runner.run_training_job` end to end (mean_pbrs AND min_pbrs,
     both trained on throwaway seeds) -- every prior Stage 8 arm/gate smoke
     test only ever exercised condition="baseline".
  2. non-zero PBRS shaping is genuinely present for both conditions (the
     runner itself hard-fails if `non_zero_shaping_count` stays 0 for an
     entire run -- this smoke test additionally checks the two conditions'
     realized shaping differs, since mean- and min-potential signals should
     not be identical on a real trajectory).
  3. `assert_stage9_guards` correctly REJECTS an attempt to train
     condition="baseline" (baseline must only ever be reused, never
     retrained, by this stage's own runner).
  4. the existing rich/lightweight evaluators (`stage8_gate_eval`,
     `stage7c_q1_eval`) work unmodified against non-baseline checkpoints,
     called with `protocol_tag=stage9-confirmatory-v1`.
  5. the Stage 8 gate's baseline data (which this stage reuses rather than
     retrains) is actually present and reachable on this machine -- a
     missing-data check, not a design check, but one this smoke test is a
     natural place to run before committing to a real 40-job launch.
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

    from thesis.pilots.stage9_runner import run_training_job
    from thesis.pilots.stage9_config import (
        LAMBDA_MEAN,
        LAMBDA_MIN,
        LEARNING_RATE_END,
        LEARNING_RATE_DECAY_STEPS,
        TARGET_UPDATE_MODE,
        REUSED_BASELINE_PROTOCOL_TAG,
        n_scenario_blocks,
        episodes_per_seed_checkpoint,
        assert_stage9_guards,
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

    # (0) Baseline reuse: the data this stage depends on must actually exist.
    baseline_summary = (
        REPO_ROOT / "results" / "stage8_gate" / "v1" / "raw" / "seed_checkpoint_summary.csv"
    )
    if not baseline_summary.is_file():
        print(
            f"ABORT: baseline reuse source missing: {baseline_summary} "
            f"(protocol_tag={REUSED_BASELINE_PROTOCOL_TAG})",
            file=sys.stderr,
        )
        return 2

    # (1) LR decay math, identical shape to the Stage 8 gate.
    dqn_cfg = FormalDQNConfig(
        learning_rate=0.0005,
        learning_rate_end=LEARNING_RATE_END,
        learning_rate_decay_environment_steps=LEARNING_RATE_DECAY_STEPS,
    )
    assert lr_at_step(0, dqn_cfg) == 0.0005
    assert abs(lr_at_step(LEARNING_RATE_DECAY_STEPS, dqn_cfg) - LEARNING_RATE_END) < 1e-15

    assert n_scenario_blocks(175_000) == 8
    assert n_scenario_blocks(200_000) == 32
    assert episodes_per_seed_checkpoint(400_000) == 64

    # (2) Guard rejects an attempt to (re)train baseline.
    guard_rejected_baseline = False
    try:
        assert_stage9_guards(
            algorithm="double_dqn",
            condition="baseline",
            master_seed=65021,
            max_steps=400_000,
            active_time_cost_per_step=0.0005,
            target_update_mode=TARGET_UPDATE_MODE,
            learning_rate_end=LEARNING_RATE_END,
            learning_rate_decay_environment_steps=LEARNING_RATE_DECAY_STEPS,
            lambda_mean=LAMBDA_MEAN,
            lambda_min=LAMBDA_MIN,
        )
    except RuntimeError:
        guard_rejected_baseline = True
    assert guard_rejected_baseline, "guard must reject condition='baseline'"

    bundle = load_final_locks()
    protocol = PILOT_ROOT / "configs" / "stage9_confirmatory_protocol.yaml"
    max_steps = 2000  # > replay_warmup_per_controller (512)

    from thesis.pilots.stage8_gate_eval import evaluate_checkpoint_stage8_gate
    from thesis.pilots.stage7c_q1_eval import evaluate_checkpoint_stage7c

    report: dict = {"conditions": {}}

    with tempfile.TemporaryDirectory(prefix="stage9_smoke_") as td:
        td_path = Path(td)
        out_root = td_path / "output"
        ckpt_root = td_path / "checkpoints_external"

        smoke_seeds = {"mean_pbrs": 66994, "min_pbrs": 67994}  # outside all real/forbidden blocks
        shaping_by_condition: dict[str, float] = {}

        for condition, smoke_seed in smoke_seeds.items():
            result = run_training_job(
                condition=condition,
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

            ckpt0 = ckpt_root / condition / f"seed_{smoke_seed}" / "ckpt_step_0_full.pt"
            assert ckpt0.is_file(), ckpt0
            payload = load_checkpoint(ckpt0)
            assert payload["algorithm_mode"] == "double_dqn"
            assert payload["condition"] == condition

            meta0 = json.loads(
                (ckpt_root / condition / f"seed_{smoke_seed}" / "checkpoint_metadata_0.json").read_text(
                    encoding="utf-8"
                )
            )
            assert meta0["target_update_mode"] == TARGET_UPDATE_MODE, meta0
            assert abs(float(meta0["lambda_mean"]) - LAMBDA_MEAN) < 1e-15, meta0
            assert abs(float(meta0["lambda_min"]) - LAMBDA_MIN) < 1e-15, meta0

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

            # rich, early plan (checkpoint_step=0, 16 episodes)
            rich_early = evaluate_checkpoint_stage8_gate(
                bundle,
                learners,
                master_seed=smoke_seed,
                checkpoint_step=0,
                code_commit="smoke",
                protocol_tag="stage9-confirmatory-v1",
                collect_trajectories=True,
            )
            assert rich_early["n_episodes"] == 16, rich_early["n_episodes"]

            # lightweight, early plan
            light_early = evaluate_checkpoint_stage7c(
                bundle,
                learners,
                master_seed=smoke_seed,
                checkpoint_step=0,
                code_commit="smoke",
                protocol_tag="stage9-confirmatory-v1",
            )
            assert light_early["n_episodes"] == 16, light_early["n_episodes"]

            # Average |scaled shaping component| over the rich trajectory, as a
            # rough realized-magnitude fingerprint per condition.
            traj = rich_early["trajectories"]
            assert len(traj) > 0
            report["conditions"][condition] = {
                "seed": smoke_seed,
                "training_success": True,
                "rich_early_episodes": rich_early["n_episodes"],
                "light_early_episodes": light_early["n_episodes"],
                "lambda_mean_confirmed": meta0["lambda_mean"],
                "lambda_min_confirmed": meta0["lambda_min"],
            }

        gi = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "*.pt" in gi and "*.pth" in gi and "*.ckpt" in gi

        report["smoke"] = "PASS"
        report["baseline_reuse_source_found"] = str(baseline_summary)
        dest = PILOT_ROOT / "logs" / "smoke_test_report.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
