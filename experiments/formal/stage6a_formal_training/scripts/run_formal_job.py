#!/usr/bin/env python3
"""Single formal job runner (Stage 6A-0)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]


def _configure_threads() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _verify_runner_release(repo_root: Path) -> None:
    path = repo_root / "runner_release.json"
    if not path.is_file():
        raise RuntimeError("runner_release.json missing; refuse formal job start")
    data = json.loads(path.read_text(encoding="utf-8"))
    import subprocess

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
    ).strip()
    expected = str(data.get("commit", ""))
    if expected and head != expected:
        raise RuntimeError(
            f"Git commit {head} does not match runner_release.json commit {expected}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one formal 100K job")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--master-seed", type=int, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override steps for infrastructure tests only",
    )
    parser.add_argument(
        "--skip-runner-release-check",
        action="store_true",
        help="Infrastructure tests before runner_release.json exists",
    )
    args = parser.parse_args(argv)

    _configure_threads()
    sys.path.insert(0, str(REPO_ROOT / "src"))
    os.environ["PYTHONPATH"] = str(REPO_ROOT / "src")

    from thesis.formal.formal_config import (
        FormalConfig,
        FormalDurationConfig,
        FormalExplorationConfig,
        derive_formal_job_seeds,
    )
    from thesis.formal.formal_trainer import FormalEngineeringError, FormalTrainer
    from thesis.formal.status_registry import (
        STATUS_RUNNING,
        TERMINAL_COMPLETE,
        TERMINAL_FAILED,
        TERMINAL_INTERRUPTED,
    )
    from thesis.protocol.h1_r1_100k_protocol import (
        EPSILON_DECAY_STEPS,
        FORMAL_CHECKPOINT_STEPS,
        FORMAL_EVALUATION_STEPS,
        FORMAL_STEPS_PER_RUN,
    )
    from thesis.training.final_lock_loader import load_final_locks
    from thesis.training.pilot_checkpoint import atomic_torch_save, load_checkpoint

    if not args.skip_runner_release_check:
        _verify_runner_release(REPO_ROOT)

    protocol_path = Path(args.protocol_lock).resolve()
    protocol = _load_yaml(protocol_path)
    protocol_hash = _sha256(protocol_path)
    if protocol.get("protocol_version") != "5C-0-H1-R1-100K":
        raise SystemExit("protocol_version must be 5C-0-H1-R1-100K")

    env_hash = protocol["environment"]["environment_lock_sha256"]
    comfort_hash = protocol["environment"]["comfort_lock_sha256"]
    bundle = load_final_locks()
    if bundle.environment_lock_sha256_before != env_hash:
        raise SystemExit("environment lock hash mismatch vs protocol")
    if bundle.comfort_lock_sha256_before != comfort_hash:
        raise SystemExit("comfort lock hash mismatch vs protocol")

    seeds = derive_formal_job_seeds(args.master_seed)
    job_id = f"{args.condition}__{args.master_seed}"
    out_root = Path(args.output_root).resolve()
    job_dir = out_root / "jobs" / job_id
    ckpt_dir = out_root / "checkpoints" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / "job.log"

    def log(msg: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    max_steps = int(args.max_steps) if args.max_steps is not None else FORMAL_STEPS_PER_RUN
    test_budget = max_steps < FORMAL_STEPS_PER_RUN
    if test_budget:
        duration = FormalDurationConfig(
            environment_steps_per_run=max_steps,
            checkpoint_steps=(max_steps,),
            evaluation_steps=(0, max_steps),
            early_stopping=False,
        )
        exploration = FormalExplorationConfig(
            epsilon_decay_environment_steps=max(1, max_steps // 2),
        )
    else:
        duration = FormalDurationConfig(
            environment_steps_per_run=FORMAL_STEPS_PER_RUN,
            checkpoint_steps=FORMAL_CHECKPOINT_STEPS,
            evaluation_steps=FORMAL_EVALUATION_STEPS,
            early_stopping=False,
        )
        exploration = FormalExplorationConfig(
            epsilon_decay_environment_steps=EPSILON_DECAY_STEPS,
        )

    if args.device != "cpu":
        raise SystemExit("formal device must be cpu")

    cfg = FormalConfig(
        duration=duration,
        exploration=exploration,
        formal_training_started=False,
        allow_test_budget=test_budget,
    )
    cfg.validate()

    (job_dir / "resolved_run_config.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "condition": args.condition,
                "master_seed": args.master_seed,
                "seeds": seeds,
                "protocol_lock": str(protocol_path),
                "protocol_hash": protocol_hash,
                "max_steps": max_steps,
                "device": "cpu",
                "num_parallel_training_envs_per_run": 1,
                "vectorized_training": False,
                "test_budget": test_budget,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    trainer = FormalTrainer(
        bundle,
        condition=args.condition,
        master_seed=args.master_seed,
        seeds=seeds,
        config=cfg,
        checkpoint_dir=ckpt_dir,
        protocol_hash=protocol_hash,
    )
    status_path = job_dir / "status.json"
    status = STATUS_RUNNING
    reason = ""
    try:
        if args.resume:
            ckpts = sorted(ckpt_dir.glob("ckpt_step_*.pt"))
            if ckpts:
                payload = load_checkpoint(ckpts[-1])
                trainer.import_checkpoint(payload)
                log(f"resumed from {ckpts[-1]} at step {trainer.env_steps}")
        status_path.write_text(
            json.dumps({"status": status, "job_id": job_id}, indent=2), encoding="utf-8"
        )
        trainer.run(n_steps=max_steps)
        final_ckpts = sorted(ckpt_dir.glob("ckpt_step_*.pt"))
        if not final_ckpts:
            raise FormalEngineeringError("missing final checkpoint")
        load_checkpoint(final_ckpts[-1])
        status = TERMINAL_COMPLETE
        trainer.write_job_manifest(job_dir / "job_manifest.json", status=status)
        (job_dir / "episode_summaries.json").write_text(
            json.dumps(trainer.diag.episode_trace, indent=2), encoding="utf-8"
        )
        (job_dir / "evaluation_trace.json").write_text(
            json.dumps(trainer.diag.evaluation_trace, indent=2), encoding="utf-8"
        )
        atomic_torch_save(
            job_dir / "final_online_target_weights.pt",
            {
                "A_online": trainer.learners["A"].online.state_dict(),
                "A_target": trainer.learners["A"].target.state_dict(),
                "B_online": trainer.learners["B"].online.state_dict(),
                "B_target": trainer.learners["B"].target.state_dict(),
                "replay_seeds": {
                    "A": int(trainer.learners["A"].replay.seed),
                    "B": int(trainer.learners["B"].replay.seed),
                },
            },
        )
        log(f"COMPLETE steps={trainer.env_steps}")
    except KeyboardInterrupt:
        status = TERMINAL_INTERRUPTED
        reason = "keyboard_interrupt"
        emergency = ckpt_dir / f"ckpt_step_{trainer.env_steps:06d}_emergency.pt"
        atomic_torch_save(emergency, trainer.export_checkpoint(step=trainer.env_steps))
        trainer.write_job_manifest(
            job_dir / "job_manifest.json", status=status, reason=reason
        )
        status_path.write_text(
            json.dumps({"status": status, "reason": reason}, indent=2), encoding="utf-8"
        )
        log(f"INTERRUPTED_RESUMABLE: {reason}")
        return 130
    except FormalEngineeringError as exc:
        status = TERMINAL_FAILED
        reason = str(exc)
        trainer.write_job_manifest(
            job_dir / "job_manifest.json", status=status, reason=reason
        )
        status_path.write_text(
            json.dumps({"status": status, "reason": reason}, indent=2), encoding="utf-8"
        )
        log(f"FAILED_WITH_REASON: {reason}")
        return 1
    except Exception as exc:  # noqa: BLE001
        status = TERMINAL_FAILED
        reason = f"uncaught:{exc}"
        log(traceback.format_exc())
        try:
            trainer.write_job_manifest(
                job_dir / "job_manifest.json", status=status, reason=reason
            )
        except Exception:
            pass
        status_path.write_text(
            json.dumps({"status": status, "reason": reason}, indent=2), encoding="utf-8"
        )
        return 1

    status_path.write_text(
        json.dumps({"status": status, "reason": reason}, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
