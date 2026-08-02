#!/usr/bin/env python3
"""Stage 8 arm1 evaluation driver.

Loops seeds x checkpoints, loads each checkpoint's saved learner state, and
calls evaluate_checkpoint_stage8_arm0(..., protocol_tag=stage8_arm1's tag,
collect_trajectories=True) -- reused unchanged from arm0 since evaluation is
always greedy (epsilon=0) and therefore independent of arm1's single
training-time change (epsilon_decay_environment_steps). Writes a flat
evaluation_episodes.csv (same schema as arm0's/7C-Q1's, for direct
comparability) plus per-seed-per-checkpoint trajectory CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

from thesis.agents.dqn_bootstrap import DQNTargetMode  # noqa: E402
from thesis.agents.independent_dqn_v2 import DQNConfig, IndependentDQNLearner  # noqa: E402
from thesis.pilots.stage7b_a1_checkpoint import sha256_file  # noqa: E402
from thesis.pilots.stage8_arm1_config import (  # noqa: E402
    CHECKPOINT_STEPS,
    PILOT_SEEDS,
    PROTOCOL_TAG,
)
from thesis.pilots.stage8_arm0_eval import evaluate_checkpoint_stage8_arm0  # noqa: E402
from thesis.training.final_lock_loader import load_final_locks  # noqa: E402
from thesis.training.pilot_checkpoint import load_checkpoint  # noqa: E402


def _git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
        ).strip()
    except Exception:
        return "unknown"


def _append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Copied from run_stage8_arm0_evaluation.py::_append_csv (generic, unions fieldnames)."""
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
            if isinstance(flat.get("controller_role_mapping"), dict):
                flat["controller_role_mapping"] = json.dumps(flat["controller_role_mapping"])
            w.writerow(flat)


def _build_learners() -> dict[str, IndependentDQNLearner]:
    cfg = DQNConfig(
        obs_dim=27,
        n_actions=3,
        hidden_sizes=(64, 64),
        target_mode=DQNTargetMode.DOUBLE,
    )
    # Arbitrary construction seeds -- overwritten immediately by import_state().
    return {
        "A": IndependentDQNLearner("A", cfg, seed=1, replay_seed=2),
        "B": IndependentDQNLearner("B", cfg, seed=3, replay_seed=4),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "stage8_arm1_protocol.yaml",
    )
    p.add_argument("--seeds", default="65003,65004")
    p.add_argument("--checkpoints", default="0,25000,50000,75000,100000")
    args = p.parse_args(argv)

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    checkpoints = [int(x) for x in args.checkpoints.split(",") if x.strip()]
    for s in seeds:
        if s not in PILOT_SEEDS:
            raise RuntimeError(f"seed {s} not in frozen arm1 seed block {PILOT_SEEDS}")
    for c in checkpoints:
        if c not in CHECKPOINT_STEPS:
            raise RuntimeError(f"checkpoint {c} not in frozen arm1 checkpoint list {CHECKPOINT_STEPS}")

    code_commit = _git_head(REPO)
    bundle = load_final_locks()

    output_root = Path(args.output_root).resolve()
    ep_out = output_root / "raw" / "evaluation_episodes.csv"
    traj_dir = output_root / "raw" / "trajectories"
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    seed_inventory_rows: list[dict[str, Any]] = []
    total_episodes = 0
    total_step_rows = 0

    for master_seed in seeds:
        ckpt_dir = Path(args.checkpoint_root).resolve() / "baseline" / f"seed_{master_seed}"
        for step in checkpoints:
            full_path = ckpt_dir / f"ckpt_step_{step}_full.pt"
            meta_path = ckpt_dir / f"checkpoint_metadata_{step}.json"
            if not full_path.is_file():
                raise FileNotFoundError(full_path)
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
            checkpoint_sha256 = meta.get("sha256") or sha256_file(full_path)

            payload = load_checkpoint(full_path)
            learners = _build_learners()
            for aid in ("A", "B"):
                learners[aid].import_state(payload["learners"][aid])

            result = evaluate_checkpoint_stage8_arm0(
                bundle,
                learners,
                master_seed=master_seed,
                checkpoint_step=step,
                code_commit=code_commit,
                checkpoint_sha256=checkpoint_sha256,
                protocol_tag=PROTOCOL_TAG,
                collect_trajectories=True,
            )
            _append_csv(ep_out, result["episodes"])
            if result["trajectories"]:
                _append_csv(
                    traj_dir / f"seed_{master_seed}_traj_step_{step}.csv",
                    result["trajectories"],
                )
            total_episodes += len(result["episodes"])
            total_step_rows += len(result["trajectories"])
            seed_inventory_rows.append(
                {
                    "master_seed": master_seed,
                    "checkpoint_step": step,
                    "checkpoint_sha256": checkpoint_sha256,
                    "n_episodes": len(result["episodes"]),
                    "n_step_rows": len(result["trajectories"]),
                }
            )
            print(
                f"evaluated seed={master_seed} checkpoint={step}: "
                f"{len(result['episodes'])} episodes, {len(result['trajectories'])} step rows"
            )

    _append_csv(manifest_dir / "SEED_INVENTORY.csv", seed_inventory_rows)
    (manifest_dir / "PROTOCOL_PROVENANCE.json").write_text(
        json.dumps(
            {
                "stage": "stage8_arm1",
                "protocol_tag": PROTOCOL_TAG,
                "code_commit": code_commit,
                "protocol_path": str(Path(args.protocol).resolve()),
                "protocol_sha256": sha256_file(Path(args.protocol)),
                "seeds": seeds,
                "checkpoints": checkpoints,
                "total_episodes": total_episodes,
                "total_step_rows": total_step_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"TOTAL: {total_episodes} episodes, {total_step_rows} trajectory step rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
