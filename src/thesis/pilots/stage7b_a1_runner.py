"""Stage 7B-A1 single job runner (Vanilla vs Double DQN)."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thesis.agents.dqn_bootstrap import DQNTargetMode
from thesis.formal.formal_config import (
    FormalConfig,
    FormalDurationConfig,
    FormalExplorationConfig,
    FormalDQNConfig,
    derive_formal_job_seeds,
)
from thesis.formal.formal_trainer import FormalEngineeringError, FormalTrainer
from thesis.pilots.stage7b_a1_checkpoint import (
    atomic_hashed_torch_save,
    sha256_file,
    validate_resume_compatibility,
    write_json_atomic,
)
from thesis.pilots.stage7b_a1_config import (
    CHECKPOINT_STEPS,
    EPSILON_DECAY_STEPS,
    MAX_STEPS,
    assert_stage7b_guards,
    condition_to_target_mode,
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
        dqn=FormalDQNConfig(device="cpu"),
        allow_test_budget=True,
        formal_training_started=False,
    )
    cfg.validate()
    return cfg


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
        raise RuntimeError("Stage 7B-A1 formal device must be cpu unless non-strict")
    if condition not in {"vanilla_dqn", "double_dqn"}:
        raise RuntimeError(f"invalid condition {condition!r}")
    if not allow_smoke:
        assert_stage7b_guards(
            condition=condition,
            reward_shaping_enabled=False,
            shaping_coefficient=0.0,
            master_seed=master_seed,
            max_steps=MAX_STEPS if strict else int(max_steps),
        )
        if strict and int(max_steps) != MAX_STEPS:
            raise RuntimeError("strict mode requires max_steps=300000")
    else:
        # Smoke: temporary seed must NOT be in the frozen pilot block
        if master_seed in range(63001, 63021):
            raise RuntimeError("smoke seed must not be in 63001-63020")

    protocol_path = Path(protocol_path).resolve()
    output_root = Path(output_root).resolve()
    checkpoint_root = Path(checkpoint_root).resolve()
    if strict and not allow_smoke:
        repo_marker = protocol_path
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
                pass  # outside repo — required

    protocol_hash = sha256_file(protocol_path)
    target_mode = condition_to_target_mode(condition)
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
        condition="baseline",  # reward condition always baseline
        master_seed=master_seed,
        seeds=seeds,
        config=cfg,
        checkpoint_dir=None,
        protocol_hash=protocol_hash,
        target_mode=target_mode,
        algorithm_condition=condition,
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
                algorithm_condition=condition,
                algorithm_mode=target_mode.value,
                protocol_hash=protocol_hash,
                reward_condition="baseline",
            )
            trainer.import_checkpoint(payload)

    start = datetime.now(timezone.utc).isoformat()
    inventory: list[dict[str, Any]] = []

    def save_step(step: int) -> dict[str, Any]:
        payload = trainer.export_checkpoint(step=step)
        payload["code_commit"] = code_commit
        payload["protocol_hash"] = protocol_hash
        path = ckpt_dir / f"ckpt_step_{step}_full.pt"
        info = atomic_hashed_torch_save(path, payload)
        meta = {
            "seed": master_seed,
            "condition": condition,
            "algorithm_mode": target_mode.value,
            "reward_condition": "baseline",
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
            "code_commit": code_commit,
        }
        write_json_atomic(ckpt_dir / f"checkpoint_metadata_{step}.json", meta)
        inventory.append(meta)
        return meta

    if trainer.env_steps == 0:
        save_step(0)

    target = int(max_steps)
    train_steps = {s for s in CHECKPOINT_STEPS if 0 < s <= target}
    while trainer.env_steps < target:
        trainer.step_once()
        step = trainer.env_steps
        if step in train_steps:
            save_step(step)

    if trainer.env_steps != target:
        raise FormalEngineeringError(
            f"ended at {trainer.env_steps}, expected {target}"
        )

    completion = {
        "seed": master_seed,
        "condition": condition,
        "algorithm_mode": target_mode.value,
        "reward_condition": "baseline",
        "reward_shaping_enabled": False,
        "shaping_coefficient": 0.0,
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
            "condition": condition,
            "target_mode": target_mode.value,
            "reward_condition": "baseline",
            "master_seed": master_seed,
            "seeds": seeds,
            "max_steps": max_steps,
            "epsilon_decay_steps": EPSILON_DECAY_STEPS,
            "protocol_hash": protocol_hash,
        },
    )
    # inventory append into output manifests (small CSV/JSON only)
    inv_path = output_root / "manifests" / "checkpoint_inventory_rows.jsonl"
    inv_path.parent.mkdir(parents=True, exist_ok=True)
    with inv_path.open("a", encoding="utf-8") as f:
        for row in inventory:
            f.write(json.dumps(row) + "\n")
    return completion


__all__ = ["build_config", "run_training_job"]
