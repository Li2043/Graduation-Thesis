#!/usr/bin/env python3
"""Re-evaluation driver for the RQ3 worst-off-stakeholder extension.

Regenerates ONLY the 4 rich-log checkpoints' trajectory CSVs (0, 350K, 375K,
400K) for all three Stage 9 conditions (baseline, mean_pbrs, min_pbrs),
using the B_front/B_rear-logging-enabled `evaluate_checkpoint_stage8_gate`
(stage8_gate_eval.py, extended 2026-08-03). No retraining -- reads the
already-trained checkpoints already on disk:

  - baseline:            <checkpoint-root-baseline>/baseline/seed_{ms}/...
                          (reused verbatim from the Stage 8 gate,
                          stage8-gate-protocol-v1, seeds 65021-65040)
  - mean_pbrs/min_pbrs:  <checkpoint-root-stage9>/<condition>/seed_{ms}/...
                          (stage9-confirmatory-v1)

Writes to a NEW results root (`results/stage9_worst_off/v1/<condition>/`),
never touching the already-committed `results/stage8_gate/v1` or
`results/stage9_confirmatory/v1` trees -- this is purely additive data for
the worst-off-stakeholder analysis, not a replacement of the frozen RQ1-RQ3
gate/decision results.
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
from thesis.pilots.stage8_gate_eval import evaluate_checkpoint_stage8_gate  # noqa: E402
from thesis.pilots.stage9_config import (  # noqa: E402
    PROTOCOL_TAG,
    REUSED_BASELINE_PROTOCOL_TAG,
    RICH_LOG_CHECKPOINTS,
    SEEDS_BY_CONDITION,
)
from thesis.training.final_lock_loader import load_final_locks  # noqa: E402
from thesis.training.pilot_checkpoint import load_checkpoint  # noqa: E402

CONDITIONS: tuple[str, ...] = ("baseline", "mean_pbrs", "min_pbrs")
DEFAULT_CHECKPOINT_ROOTS = {
    "baseline": REPO.parents[0] / "final_new_experiment" / "stage8_gate_checkpoints",
    "mean_pbrs": REPO.parents[0] / "final_new_experiment" / "stage9_checkpoints",
    "min_pbrs": REPO.parents[0] / "final_new_experiment" / "stage9_checkpoints",
}
CONDITION_SUBDIR = {"baseline": "baseline", "mean_pbrs": "mean_pbrs", "min_pbrs": "min_pbrs"}
PROTOCOL_TAG_BY_CONDITION = {
    "baseline": REUSED_BASELINE_PROTOCOL_TAG,
    "mean_pbrs": PROTOCOL_TAG,
    "min_pbrs": PROTOCOL_TAG,
}


def _git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True).strip()
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
    cfg = DQNConfig(obs_dim=27, n_actions=3, hidden_sizes=(64, 64), target_mode=DQNTargetMode.DOUBLE)
    return {
        "A": IndependentDQNLearner("A", cfg, seed=1, replay_seed=2),
        "B": IndependentDQNLearner("B", cfg, seed=3, replay_seed=4),
    }


def run_condition(
    condition: str,
    *,
    checkpoint_root: Path,
    output_root: Path,
    seeds: list[int],
    checkpoints: list[int],
    bundle,
    code_commit: str,
) -> dict[str, Any]:
    protocol_tag = PROTOCOL_TAG_BY_CONDITION[condition]
    out = output_root / condition
    ep_out = out / "raw" / "evaluation_episodes.csv"
    traj_dir = out / "raw" / "trajectories"
    manifest_dir = out / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    total_episodes = 0
    total_step_rows = 0
    for master_seed in seeds:
        ckpt_dir = checkpoint_root / CONDITION_SUBDIR[condition] / f"seed_{master_seed}"
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

            result = evaluate_checkpoint_stage8_gate(
                bundle,
                learners,
                master_seed=master_seed,
                checkpoint_step=step,
                code_commit=code_commit,
                checkpoint_sha256=checkpoint_sha256,
                protocol_tag=protocol_tag,
                collect_trajectories=True,
            )
            episodes = result["episodes"]
            trajectories = result["trajectories"]

            _append_csv(ep_out, episodes)
            if trajectories:
                _append_csv(traj_dir / f"seed_{master_seed}_traj_step_{step}.csv", trajectories)
            total_episodes += len(episodes)
            total_step_rows += len(trajectories)

            n = max(len(episodes), 1)
            success_rate = sum(1.0 for e in episodes if e["success"]) / n
            collision_rate = sum(1.0 for e in episodes if e["collision"]) / n
            n_bg = sum(1 for r in trajectories if r.get("controller") in ("B_front", "B_rear"))
            print(
                f"[{condition}] seed={master_seed} checkpoint={step}: "
                f"{len(episodes)} episodes, {len(trajectories)} step rows "
                f"({n_bg} background-vehicle rows), success={success_rate:.3f} "
                f"collision={collision_rate:.3f}"
            )

    (manifest_dir / "PROTOCOL_PROVENANCE.json").write_text(
        json.dumps(
            {
                "stage": "stage9_worst_off_extension",
                "condition": condition,
                "protocol_tag": protocol_tag,
                "code_commit": code_commit,
                "checkpoint_root": str(checkpoint_root),
                "seeds": seeds,
                "checkpoints": checkpoints,
                "total_episodes": total_episodes,
                "total_step_rows": total_step_rows,
                "note": (
                    "Re-evaluation of already-trained checkpoints with "
                    "B_front/B_rear per-step logging added to "
                    "stage8_gate_eval.py, for the RQ3 worst-off-stakeholder "
                    "extension (compute_worst_off_mobility). No retraining."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"TOTAL [{condition}]: {total_episodes} episodes, {total_step_rows} trajectory step rows")
    return {"total_episodes": total_episodes, "total_step_rows": total_step_rows}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--condition", choices=list(CONDITIONS) + ["all"], default="all")
    p.add_argument(
        "--output-root",
        type=Path,
        default=REPO / "results" / "stage9_worst_off" / "v1",
    )
    p.add_argument("--checkpoints", default=",".join(str(c) for c in RICH_LOG_CHECKPOINTS))
    p.add_argument("--seeds", default=None, help="override seeds (comma-separated); for smoke-testing only")
    args = p.parse_args(argv)

    checkpoints = [int(x) for x in args.checkpoints.split(",") if x.strip()]
    for c in checkpoints:
        if c not in RICH_LOG_CHECKPOINTS:
            raise RuntimeError(f"checkpoint {c} not in rich-log set {RICH_LOG_CHECKPOINTS}")

    code_commit = _git_head(REPO)
    bundle = load_final_locks()
    output_root = Path(args.output_root).resolve()

    conditions = list(CONDITIONS) if args.condition == "all" else [args.condition]
    summary: dict[str, Any] = {}
    for condition in conditions:
        checkpoint_root = DEFAULT_CHECKPOINT_ROOTS[condition].resolve()
        allowed = SEEDS_BY_CONDITION[condition]
        if args.seeds:
            seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
            for s in seeds:
                if s not in allowed:
                    raise RuntimeError(f"seed {s} not in {condition}'s frozen seed block {allowed}")
        else:
            seeds = list(allowed)
        summary[condition] = run_condition(
            condition,
            checkpoint_root=checkpoint_root,
            output_root=output_root,
            seeds=seeds,
            checkpoints=checkpoints,
            bundle=bundle,
            code_commit=code_commit,
        )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
