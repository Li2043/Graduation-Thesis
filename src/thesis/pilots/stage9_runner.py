"""Stage 9 confirmatory three-condition experiment: single-seed runner for
the two NEWLY TRAINED conditions (mean_pbrs, min_pbrs). Baseline is reused
verbatim from the Stage 8 gate and is never trained by this module -- see
`stage9_config` module docstring and the protocol doc SS4.

Near-literal adaptation of `thesis.pilots.stage8_gate_runner` (identical DQN
hyperparameters: Double DQN, hard target sync, LR linear decay 0.0005 ->
0.0001 over 400,000 steps, epsilon decay over 50,000 steps). The only
functional difference is `condition` (mean_pbrs / min_pbrs instead of
baseline) and the resulting non-zero PBRS shaping term -- `FormalTrainer`
already computes both mean and min shaped rewards every step and selects the
one matching `condition` (`thesis.training.final_reward_conditions.
select_condition_reward`); no new reward-wiring code is needed here.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thesis.agents.dqn_bootstrap import DQNTargetMode
from thesis.formal.formal_config import (
    FormalConfig,
    FormalDQNConfig,
    FormalDurationConfig,
    FormalExplorationConfig,
    FormalPBRSConfig,
    derive_formal_job_seeds,
)
from thesis.formal.formal_trainer import FormalEngineeringError, FormalTrainer
from thesis.pilots.stage7b_a1_checkpoint import (
    atomic_hashed_torch_save,
    sha256_file,
    validate_resume_compatibility,
    write_json_atomic,
)
from thesis.pilots.stage9_config import (
    ACTIVE_TIME_COST_PER_STEP,
    ALGORITHM,
    ALL_FORBIDDEN_SEEDS,
    CHECKPOINT_STEPS,
    EPSILON_DECAY_STEPS,
    LAMBDA_MEAN,
    LAMBDA_MIN,
    LEARNING_RATE_DECAY_STEPS,
    LEARNING_RATE_END,
    LEARNING_RATE_START,
    MAX_STEPS,
    PBRS_GAMMA,
    PROTOCOL_TAG,
    SEEDS_BY_CONDITION,
    TARGET_UPDATE_MODE,
    TRAINED_CONDITIONS,
    assert_stage9_guards,
    target_mode,
)
from thesis.training.final_lock_loader import load_final_locks
from thesis.training.pilot_checkpoint import load_checkpoint


def _git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
        ).strip()
    except Exception:
        return "unknown"


def build_config(*, max_steps: int = MAX_STEPS) -> FormalConfig:
    train_steps = tuple(s for s in CHECKPOINT_STEPS if 0 < s <= max_steps)
    cfg = FormalConfig(
        duration=FormalDurationConfig(
            environment_steps_per_run=int(max_steps),
            checkpoint_steps=train_steps,
            evaluation_steps=(),
            early_stopping=False,
            max_policy_steps=400,
        ),
        exploration=FormalExplorationConfig(
            epsilon_start=1.0,
            epsilon_end=0.10,
            epsilon_decay_environment_steps=EPSILON_DECAY_STEPS,
            schedule="linear",
            epsilon_after_decay=0.10,
        ),
        dqn=FormalDQNConfig(
            device="cpu",
            learning_rate=LEARNING_RATE_START,
            learning_rate_end=LEARNING_RATE_END,
            learning_rate_decay_environment_steps=LEARNING_RATE_DECAY_STEPS,
        ),
        pbrs=FormalPBRSConfig(
            lambda_mean=LAMBDA_MEAN,
            lambda_min=LAMBDA_MIN,
            gamma=PBRS_GAMMA,
            comparison_type="equal_coefficient",
            magnitude_matched=False,
            rms_matched=False,
        ),
        allow_test_budget=True,
        formal_training_started=False,
    )
    cfg.validate()
    return cfg


def _assert_checkpoint_outside_repo(checkpoint_root: Path, protocol_path: Path) -> None:
    repo_marker = protocol_path.resolve()
    while repo_marker != repo_marker.parent and not (repo_marker / ".git").exists():
        repo_marker = repo_marker.parent
    if (repo_marker / ".git").exists():
        try:
            checkpoint_root.resolve().relative_to(repo_marker.resolve())
            raise RuntimeError(
                "checkpoint-root must be outside the git repository "
                f"(got {checkpoint_root})"
            )
        except ValueError:
            pass


def run_training_job(
    *,
    condition: str,
    master_seed: int,
    protocol_path: Path,
    output_root: Path,
    checkpoint_root: Path,
    max_steps: int = MAX_STEPS,
    resume: bool = True,
    device: str = "cpu",
    strict: bool = True,
    allow_smoke: bool = False,
) -> dict[str, Any]:
    if device != "cpu" and strict:
        raise RuntimeError("Stage 9 device must be cpu unless non-strict")
    algorithm = ALGORITHM
    if condition not in TRAINED_CONDITIONS:
        raise RuntimeError(
            f"Stage 9 runner only trains {TRAINED_CONDITIONS!r} -- baseline is reused "
            f"from the Stage 8 gate, not retrained. Got condition={condition!r}"
        )
    if not allow_smoke:
        assert_stage9_guards(
            algorithm=algorithm,
            condition=condition,
            master_seed=master_seed,
            max_steps=MAX_STEPS if strict else int(max_steps),
            active_time_cost_per_step=ACTIVE_TIME_COST_PER_STEP,
            target_update_mode=TARGET_UPDATE_MODE,
            learning_rate_end=LEARNING_RATE_END,
            learning_rate_decay_environment_steps=LEARNING_RATE_DECAY_STEPS,
            lambda_mean=LAMBDA_MEAN,
            lambda_min=LAMBDA_MIN,
        )
        if strict and int(max_steps) != MAX_STEPS:
            raise RuntimeError("strict mode requires max_steps=400000")
    else:
        if master_seed in SEEDS_BY_CONDITION[condition] or master_seed in ALL_FORBIDDEN_SEEDS:
            raise RuntimeError(
                f"smoke seed {master_seed} collides with a real Stage 9 seed block "
                "or a historical seed block"
            )

    protocol_path = Path(protocol_path).resolve()
    output_root = Path(output_root).resolve()
    checkpoint_root = Path(checkpoint_root).resolve()
    if strict and not allow_smoke:
        _assert_checkpoint_outside_repo(checkpoint_root, protocol_path)

    protocol_hash = sha256_file(protocol_path)
    mode = target_mode()
    if mode is not DQNTargetMode.DOUBLE:
        raise RuntimeError("Stage 9 requires Double DQN")
    seeds = derive_formal_job_seeds(master_seed)
    bundle = load_final_locks()
    cfg = build_config(max_steps=max_steps)

    job_out = output_root / "runs" / condition / f"seed_{master_seed}"
    ckpt_dir = checkpoint_root / condition / f"seed_{master_seed}"
    job_out.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    completion_path = job_out / "run_completion.json"
    if completion_path.exists() and resume:
        prev = json.loads(completion_path.read_text(encoding="utf-8"))
        if prev.get("success") and int(prev.get("final_step", -1)) == int(max_steps):
            return {**prev, "skipped_completed": True}

    repo_root = protocol_path
    while repo_root != repo_root.parent and not (repo_root / ".git").exists():
        repo_root = repo_root.parent
    code_commit = _git_head(repo_root)

    trainer = FormalTrainer(
        bundle,
        condition=condition,
        master_seed=master_seed,
        seeds=seeds,
        config=cfg,
        checkpoint_dir=None,
        protocol_hash=protocol_hash,
        target_mode=mode,
        algorithm_condition=algorithm,
        active_time_cost_per_step=ACTIVE_TIME_COST_PER_STEP,
    )

    if resume:
        fulls = sorted(
            ckpt_dir.glob("ckpt_step_*_full.pt"),
            key=lambda p: int(p.name.split("_")[2]),
        )
        if fulls:
            latest = fulls[-1]
            payload = load_checkpoint(latest)
            validate_resume_compatibility(
                payload,
                algorithm_condition=algorithm,
                algorithm_mode=mode.value,
                protocol_hash=protocol_hash,
                reward_condition=condition,
            )
            trainer.import_checkpoint(payload)

    start = datetime.now(timezone.utc).isoformat()
    inventory: list[dict[str, Any]] = []

    def save_step(step: int) -> dict[str, Any]:
        payload = trainer.export_checkpoint(step=step)
        payload["code_commit"] = code_commit
        payload["protocol_hash"] = protocol_hash
        payload["protocol_tag"] = PROTOCOL_TAG
        payload["base_reward_version"] = "v2_active_time"
        payload["active_time_cost_per_step"] = ACTIVE_TIME_COST_PER_STEP
        path = ckpt_dir / f"ckpt_step_{step}_full.pt"
        info = atomic_hashed_torch_save(path, payload)
        meta = {
            "seed": master_seed,
            "condition": condition,
            "algorithm_mode": mode.value,
            "reward_condition": condition,
            "base_reward_version": "v2_active_time",
            "active_time_cost_per_step": ACTIVE_TIME_COST_PER_STEP,
            "lambda_mean": LAMBDA_MEAN,
            "lambda_min": LAMBDA_MIN,
            "target_update_mode": TARGET_UPDATE_MODE,
            "learning_rate_end": LEARNING_RATE_END,
            "learning_rate_decay_environment_steps": LEARNING_RATE_DECAY_STEPS,
            "requested_step": step,
            "actual_step": trainer.env_steps,
            "path": info["path"],
            "sha256": info["sha256"],
            "size_bytes": info["size_bytes"],
            "load_test_passed": True,
            "contains_optimizer": True,
            "contains_replay": True,
            "contains_rng": True,
            "contains_schedule_state": True,
            "resumable": True,
            "protocol_hash": protocol_hash,
            "protocol_tag": PROTOCOL_TAG,
            "code_commit": code_commit,
            "storage_location": "local",
        }
        write_json_atomic(ckpt_dir / f"checkpoint_metadata_{step}.json", meta)
        inventory.append(meta)
        return meta

    if trainer.env_steps == 0:
        save_step(0)

    target = int(max_steps)
    train_steps = {s for s in CHECKPOINT_STEPS if 0 < s <= target}
    non_zero_shaping_seen = False
    while trainer.env_steps < target:
        trainer.step_once()
        if trainer.diag.non_zero_shaping_count > 0:
            non_zero_shaping_seen = True
        step = trainer.env_steps
        if step in train_steps:
            save_step(step)

    if trainer.env_steps != target:
        raise FormalEngineeringError(
            f"ended at {trainer.env_steps}, expected {target}"
        )
    if not non_zero_shaping_seen:
        raise FormalEngineeringError(
            f"condition={condition} produced zero PBRS shaping over the entire run "
            "-- reward wiring is almost certainly broken"
        )

    completion = {
        "stage": "stage9_confirmatory",
        "protocol_tag": PROTOCOL_TAG,
        "seed": master_seed,
        "condition": condition,
        "algorithm_mode": mode.value,
        "reward_condition": condition,
        "base_reward_version": "v2_active_time",
        "active_time_cost_per_step": ACTIVE_TIME_COST_PER_STEP,
        "lambda_mean": LAMBDA_MEAN,
        "lambda_min": LAMBDA_MIN,
        "target_update_mode": TARGET_UPDATE_MODE,
        "learning_rate_end": LEARNING_RATE_END,
        "learning_rate_decay_environment_steps": LEARNING_RATE_DECAY_STEPS,
        "reward_shaping_enabled": True,
        "shaping_coefficient": LAMBDA_MEAN if condition == "mean_pbrs" else LAMBDA_MIN,
        "start_time": start,
        "end_time": datetime.now(timezone.utc).isoformat(),
        "final_step": trainer.env_steps,
        "checkpoint_count": len(list(ckpt_dir.glob("ckpt_step_*_full.pt"))),
        "success": True,
        "failure_reason": "",
        "git_commit": code_commit,
        "protocol_hash": protocol_hash,
        "config_hash": cfg.sha256(),
        "checkpoint_root": str(checkpoint_root.as_posix()),
    }
    write_json_atomic(completion_path, completion)
    write_json_atomic(
        job_out / "resolved_run_config.json",
        {
            "algorithm": algorithm,
            "condition": condition,
            "target_mode": mode.value,
            "reward_condition": condition,
            "base_reward_version": "v2_active_time",
            "active_time_cost_per_step": ACTIVE_TIME_COST_PER_STEP,
            "lambda_mean": LAMBDA_MEAN,
            "lambda_min": LAMBDA_MIN,
            "master_seed": master_seed,
            "seeds": seeds,
            "max_steps": max_steps,
            "epsilon_decay_steps": EPSILON_DECAY_STEPS,
            "target_update_mode": TARGET_UPDATE_MODE,
            "learning_rate_end": LEARNING_RATE_END,
            "learning_rate_decay_environment_steps": LEARNING_RATE_DECAY_STEPS,
            "protocol_hash": protocol_hash,
            "protocol_tag": PROTOCOL_TAG,
            "timestep_definition": "one_joint_environment_transition",
        },
    )
    inv_path = output_root / "manifests" / "checkpoint_inventory_rows.jsonl"
    inv_path.parent.mkdir(parents=True, exist_ok=True)
    with inv_path.open("a", encoding="utf-8") as f:
        for row in inventory:
            f.write(json.dumps(row) + "\n")
    return completion


__all__ = ["build_config", "run_training_job"]
