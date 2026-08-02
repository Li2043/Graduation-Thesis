#!/usr/bin/env python3
"""Stage 8 formal gate evaluation driver.

Loops seeds x checkpoints, loads each checkpoint's saved learner state, and
evaluates it with ONE of two evaluators depending on checkpoint_step:

- checkpoint_step in RICH_LOG_CHECKPOINTS (0, 350K, 375K, 400K): rich
  per-step diagnostic evaluator (`stage8_gate_eval.evaluate_checkpoint_stage8_gate`,
  Q-values/front_gap/minimum_TTC/action sequence logged for every episode).
- all other checkpoints (25K..325K): lightweight evaluator
  (`stage7c_q1_eval.evaluate_checkpoint_stage7c`, episode-level only, reused
  UNCHANGED from Stage 7C-Q1 -- it already threads active_time_cost_per_step
  correctly and already handles both the 8-block early plan and 32-block
  late plan via its own config's n_scenario_blocks, numerically identical to
  this gate's for every checkpoint used here).

Both evaluators are called with protocol_tag=stage8-gate-protocol-v1 so the
SHA-256 evaluation-seed namespace is disjoint from Stage 7C-Q1's own.

Writes:
- raw/evaluation_episodes.csv (episode-level, all 17 checkpoints, all seeds)
- raw/trajectories/seed_{ms}_traj_step_{step}.csv (only the 4 rich checkpoints)
- raw/seed_checkpoint_summary.csv (master_seed, checkpoint_step, success_rate,
  collision_rate, truncation_rate, swap_eligibility -- exactly the schema
  `stage7c_q1_gate.evaluate_competence_gate()` requires)
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
from thesis.pilots.stage7c_q1_eval import (  # noqa: E402
    compute_swap_eligibility,
    evaluate_checkpoint_stage7c,
)
from thesis.pilots.stage8_gate_config import (  # noqa: E402
    CHECKPOINT_STEPS,
    PILOT_SEEDS,
    PROTOCOL_TAG,
    RICH_LOG_CHECKPOINTS,
)
from thesis.pilots.stage8_gate_eval import evaluate_checkpoint_stage8_gate  # noqa: E402
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
        default=Path(__file__).resolve().parents[1] / "configs" / "stage8_gate_protocol.yaml",
    )
    p.add_argument("--seeds", default=",".join(str(s) for s in PILOT_SEEDS))
    p.add_argument("--checkpoints", default=",".join(str(c) for c in CHECKPOINT_STEPS))
    args = p.parse_args(argv)

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    checkpoints = [int(x) for x in args.checkpoints.split(",") if x.strip()]
    for s in seeds:
        if s not in PILOT_SEEDS:
            raise RuntimeError(f"seed {s} not in frozen gate seed block {PILOT_SEEDS}")
    for c in checkpoints:
        if c not in CHECKPOINT_STEPS:
            raise RuntimeError(f"checkpoint {c} not in frozen gate checkpoint list {CHECKPOINT_STEPS}")

    code_commit = _git_head(REPO)
    bundle = load_final_locks()

    output_root = Path(args.output_root).resolve()
    ep_out = output_root / "raw" / "evaluation_episodes.csv"
    traj_dir = output_root / "raw" / "trajectories"
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    seed_checkpoint_rows: list[dict[str, Any]] = []
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

            if step in RICH_LOG_CHECKPOINTS:
                result = evaluate_checkpoint_stage8_gate(
                    bundle,
                    learners,
                    master_seed=master_seed,
                    checkpoint_step=step,
                    code_commit=code_commit,
                    checkpoint_sha256=checkpoint_sha256,
                    protocol_tag=PROTOCOL_TAG,
                    collect_trajectories=True,
                )
                episodes = result["episodes"]
                trajectories = result["trajectories"]
            else:
                result = evaluate_checkpoint_stage7c(
                    bundle,
                    learners,
                    master_seed=master_seed,
                    checkpoint_step=step,
                    code_commit=code_commit,
                    protocol_tag=PROTOCOL_TAG,
                )
                episodes = result["episodes"]
                for ep in episodes:
                    ep["checkpoint_sha256"] = checkpoint_sha256
                trajectories = []

            _append_csv(ep_out, episodes)
            if trajectories:
                _append_csv(
                    traj_dir / f"seed_{master_seed}_traj_step_{step}.csv",
                    trajectories,
                )
            total_episodes += len(episodes)
            total_step_rows += len(trajectories)

            n = max(len(episodes), 1)
            success_rate = sum(1.0 for e in episodes if e["success"]) / n
            collision_rate = sum(1.0 for e in episodes if e["collision"]) / n
            truncation_rate = sum(1.0 for e in episodes if e["truncation"]) / n
            swap_eligibility = compute_swap_eligibility(episodes)
            seed_checkpoint_rows.append(
                {
                    "master_seed": master_seed,
                    "checkpoint_step": step,
                    "success_rate": success_rate,
                    "collision_rate": collision_rate,
                    "truncation_rate": truncation_rate,
                    "swap_eligibility": swap_eligibility,
                    "n_episodes": len(episodes),
                    "rich_log": step in RICH_LOG_CHECKPOINTS,
                }
            )
            seed_inventory_rows.append(
                {
                    "master_seed": master_seed,
                    "checkpoint_step": step,
                    "checkpoint_sha256": checkpoint_sha256,
                    "n_episodes": len(episodes),
                    "n_step_rows": len(trajectories),
                    "rich_log": step in RICH_LOG_CHECKPOINTS,
                }
            )
            print(
                f"evaluated seed={master_seed} checkpoint={step} "
                f"({'rich' if step in RICH_LOG_CHECKPOINTS else 'lightweight'}): "
                f"{len(episodes)} episodes, {len(trajectories)} step rows, "
                f"success={success_rate:.3f} collision={collision_rate:.3f}"
            )

    _append_csv(output_root / "raw" / "seed_checkpoint_summary.csv", seed_checkpoint_rows)
    _append_csv(manifest_dir / "SEED_INVENTORY.csv", seed_inventory_rows)
    (manifest_dir / "PROTOCOL_PROVENANCE.json").write_text(
        json.dumps(
            {
                "stage": "stage8_gate",
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
