"""Stage 7A-1 single-seed Baseline budget trainer (0 → 300K)."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thesis.formal.formal_config import (
    FormalConfig,
    FormalDurationConfig,
    FormalExplorationConfig,
    FormalDQNConfig,
    derive_formal_job_seeds,
)
from thesis.formal.formal_trainer import FormalEngineeringError, FormalTrainer
from thesis.pilots.stage7a1_checkpoint import (
    atomic_hashed_torch_save,
    checkpoint_contains_flags,
    sha256_file,
    weights_payload_from_learners,
    write_checkpoint_metadata,
)
from thesis.pilots.stage7a1_config import (
    CHECKPOINT_STEPS,
    EPSILON_DECAY_STEPS,
    MAX_STEPS,
    PILOT_SEEDS,
    PRIMARY_BUDGET_CHECKPOINTS,
    TRAIN_CHECKPOINT_STEPS,
    assert_baseline_only,
)
from thesis.pilots.stage7a1_eval import evaluate_checkpoint_rich
from thesis.training.final_lock_loader import load_final_locks
from thesis.training.pilot_checkpoint import load_checkpoint


def _git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
        ).strip()
    except Exception:
        return "unknown"


def build_pilot_formal_config(*, max_steps: int = MAX_STEPS) -> FormalConfig:
    """Reuse Stage 6 FormalConfig with only the step budget extended."""
    duration = FormalDurationConfig(
        environment_steps_per_run=int(max_steps),
        checkpoint_steps=tuple(s for s in TRAIN_CHECKPOINT_STEPS if s <= max_steps),
        evaluation_steps=(),  # rich eval handled by Stage 7A-1 runner
        early_stopping=False,
        max_policy_steps=400,
    )
    exploration = FormalExplorationConfig(
        epsilon_start=1.0,
        epsilon_end=0.10,
        epsilon_decay_environment_steps=EPSILON_DECAY_STEPS,
        schedule="linear",
        epsilon_after_decay=0.10,
    )
    cfg = FormalConfig(
        duration=duration,
        exploration=exploration,
        dqn=FormalDQNConfig(device="cpu"),
        allow_test_budget=True,
        formal_training_started=False,
    )
    cfg.validate()
    if cfg.exploration.epsilon_decay_environment_steps != EPSILON_DECAY_STEPS:
        raise RuntimeError("epsilon decay must remain 50000")
    if cfg.duration.early_stopping:
        raise RuntimeError("early stopping forbidden")
    return cfg


def make_trainer(
    *,
    master_seed: int,
    protocol_hash: str,
    checkpoint_dir: Path | None,
    max_steps: int = MAX_STEPS,
) -> FormalTrainer:
    if master_seed not in PILOT_SEEDS:
        raise RuntimeError(f"seed {master_seed} not in pilot seed block 62001-62020")
    if master_seed in range(61001, 61011):
        raise RuntimeError("formal seeds 61001-61010 forbidden")
    bundle = load_final_locks()
    seeds = derive_formal_job_seeds(master_seed)
    cfg = build_pilot_formal_config(max_steps=max_steps)
    assert_baseline_only(
        condition="baseline",
        reward_shaping_enabled=False,
        shaping_coefficient=0.0,
    )
    return FormalTrainer(
        bundle,
        condition="baseline",
        master_seed=master_seed,
        seeds=seeds,
        config=cfg,
        checkpoint_dir=checkpoint_dir,
        protocol_hash=protocol_hash,
    )


def _checkpoint_index(step: int) -> int:
    return list(CHECKPOINT_STEPS).index(int(step))


def save_milestone(
    trainer: FormalTrainer,
    *,
    step: int,
    run_dir: Path,
    inventory_rows: list[dict[str, Any]],
    code_commit: str,
) -> dict[str, Any]:
    ckpt_dir = run_dir / "checkpoints"
    weights_dir = run_dir / "weights"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    full_path = ckpt_dir / f"ckpt_step_{step}_full.pt"
    weights_path = weights_dir / f"weights_step_{step}.pt"
    meta_path = ckpt_dir / f"checkpoint_metadata_{step}.json"

    payload = trainer.export_checkpoint(step=step)
    payload["code_commit"] = code_commit
    payload["requested_step"] = int(step)
    payload["actual_checkpoint_step"] = int(trainer.env_steps)
    payload["checkpoint_at_episode_boundary"] = not bool(trainer._episode_open)
    flags = checkpoint_contains_flags(payload)
    if not flags["resumable"]:
        raise RuntimeError(f"checkpoint at {step} not resumable: {flags}")

    full_info = atomic_hashed_torch_save(full_path, payload)
    w_info = atomic_hashed_torch_save(
        weights_path, weights_payload_from_learners(trainer.learners)
    )
    meta = {
        "seed": trainer.master_seed,
        "condition": "baseline",
        "requested_step": int(step),
        "actual_step": int(trainer.env_steps),
        "path": full_info["path"],
        "weights_path": w_info["path"],
        "size_bytes": full_info["size_bytes"],
        "mtime_ns": full_info["mtime_ns"],
        "sha256": full_info["sha256"],
        "full_checkpoint_sha256": full_info["sha256"],
        "weights_sha256": w_info["sha256"],
        "load_test_passed": True,
        **flags,
        "code_commit": code_commit,
        "environment_lock_hash": trainer.bundle.environment_lock_sha256_before,
        "comfort_lock_hash": trainer.bundle.comfort_lock_sha256_before,
        "formal_config_hash": trainer.config.sha256(),
        "checkpoint_at_episode_boundary": payload["checkpoint_at_episode_boundary"],
    }
    write_checkpoint_metadata(meta_path, meta)
    inventory_rows.append(meta)
    return meta


def _append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    fieldnames: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            existing = csv.DictReader(f)
            if existing.fieldnames:
                for k in existing.fieldnames:
                    if k not in fieldnames:
                        fieldnames.append(k)
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in rows:
            flat = dict(r)
            if isinstance(flat.get("evaluation_guard"), dict):
                flat["evaluation_guard"] = json.dumps(flat["evaluation_guard"])
            w.writerow(flat)


def run_seed(
    *,
    master_seed: int,
    pilot_root: Path,
    protocol_path: Path,
    max_steps: int = MAX_STEPS,
    resume: bool = True,
    force_rerun: bool = False,
    skip_eval: bool = False,
) -> dict[str, Any]:
    pilot_root = Path(pilot_root).resolve()
    protocol_path = Path(protocol_path).resolve()
    protocol_hash = sha256_file(protocol_path)
    repo_root = pilot_root
    while repo_root != repo_root.parent and not (repo_root / ".git").exists():
        repo_root = repo_root.parent
    code_commit = _git_head(repo_root)

    run_dir = pilot_root / "output" / "runs" / f"seed_{master_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    completion_path = run_dir / "run_completion.json"
    if completion_path.exists() and not force_rerun:
        prev = json.loads(completion_path.read_text(encoding="utf-8"))
        if prev.get("success") and int(prev.get("final_step", -1)) == int(max_steps):
            return {**prev, "skipped_completed": True}

    start_time = datetime.now(timezone.utc).isoformat()
    inventory_rows: list[dict[str, Any]] = []
    resume_meta = {
        "resume_count": 0,
        "resume_source_checkpoint": None,
        "resume_source_hash": None,
        "resume_timestamp": None,
    }

    trainer = make_trainer(
        master_seed=master_seed,
        protocol_hash=protocol_hash,
        checkpoint_dir=None,
        max_steps=max_steps,
    )

    # Resume from latest valid full checkpoint if present
    ckpt_dir = run_dir / "checkpoints"
    if resume and ckpt_dir.exists():
        fulls = sorted(ckpt_dir.glob("ckpt_step_*_full.pt"))
        if fulls and not force_rerun:
            latest = fulls[-1]
            payload = load_checkpoint(latest)
            trainer.import_checkpoint(payload)
            resume_meta = {
                "resume_count": 1,
                "resume_source_checkpoint": str(latest.as_posix()),
                "resume_source_hash": sha256_file(latest),
                "resume_timestamp": datetime.now(timezone.utc).isoformat(),
            }

    # Derived seed inventory
    seeds_path = pilot_root / "manifests" / "derived_seed_inventory.csv"
    seeds_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"master_seed": master_seed, **trainer.seeds}
    _append_csv(seeds_path, [row])

    eval_root = pilot_root / "output" / "evaluations"
    traj_root = pilot_root / "output" / "trajectories"
    inv_path = pilot_root / "output" / "manifests" / "checkpoint_inventory.csv"

    def _do_eval_at(step: int, meta: dict[str, Any]) -> None:
        if skip_eval:
            return
        collect_traj = step in PRIMARY_BUDGET_CHECKPOINTS
        result = evaluate_checkpoint_rich(
            trainer.bundle,
            trainer.learners,
            master_seed=master_seed,
            evaluation_seed=trainer.seeds["evaluation_seed"],
            checkpoint_step=step,
            checkpoint_index=_checkpoint_index(step),
            checkpoint_sha256=meta["sha256"],
            collect_trajectories=collect_traj,
        )
        for ep in result["episodes"]:
            ep["checkpoint_sha256"] = meta["sha256"]
        _append_csv(
            eval_root / f"seed_{master_seed}_episodes.csv",
            result["episodes"],
        )
        if collect_traj and result["trajectories"]:
            _append_csv(
                traj_root / f"seed_{master_seed}_traj_step_{step}.csv",
                result["trajectories"],
            )
        (run_dir / f"eval_guard_step_{step}.json").write_text(
            json.dumps(result["evaluation_guard"], indent=2), encoding="utf-8"
        )

    # Step-0 milestone if starting fresh
    if trainer.env_steps == 0 and 0 in CHECKPOINT_STEPS:
        meta0 = save_milestone(
            trainer,
            step=0,
            run_dir=run_dir,
            inventory_rows=inventory_rows,
            code_commit=code_commit,
        )
        _append_csv(inv_path, [meta0])
        _do_eval_at(0, meta0)

    target = int(max_steps)
    while trainer.env_steps < target:
        trainer.step_once()
        step = trainer.env_steps
        if step in TRAIN_CHECKPOINT_STEPS and step <= target:
            meta = save_milestone(
                trainer,
                step=step,
                run_dir=run_dir,
                inventory_rows=inventory_rows,
                code_commit=code_commit,
            )
            _append_csv(inv_path, [meta])
            _do_eval_at(step, meta)

    if trainer.env_steps != target:
        raise FormalEngineeringError(
            f"run ended at {trainer.env_steps}, expected {target}"
        )

    end_time = datetime.now(timezone.utc).isoformat()
    completion = {
        "seed": master_seed,
        "start_time": start_time,
        "end_time": end_time,
        "final_step": trainer.env_steps,
        "checkpoint_count": len(list((run_dir / "checkpoints").glob("ckpt_step_*_full.pt"))),
        "evaluation_count": len(list(eval_root.glob(f"seed_{master_seed}_episodes.csv"))),
        "success": True,
        "failure_reason": "",
        "git_commit": code_commit,
        "config_hash": trainer.config.sha256(),
        "protocol_hash": protocol_hash,
        **resume_meta,
    }
    completion_path.write_text(json.dumps(completion, indent=2), encoding="utf-8")
    (run_dir / "resolved_run_config.json").write_text(
        json.dumps(
            {
                "condition": "baseline",
                "master_seed": master_seed,
                "seeds": trainer.seeds,
                "max_steps": max_steps,
                "epsilon_decay_steps": EPSILON_DECAY_STEPS,
                "reward_shaping_enabled": False,
                "shaping_coefficient": 0.0,
                "early_stopping": False,
                "protocol_hash": protocol_hash,
                "formal_config_hash": trainer.config.sha256(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return completion


__all__ = [
    "build_pilot_formal_config",
    "make_trainer",
    "run_seed",
    "save_milestone",
]
