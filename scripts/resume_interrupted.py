#!/usr/bin/env python3
"""Resume only runs that status.py classifies as 'interrupted'
(process died, no completion marker, no technical_failure flag --
i.e. a crash or power loss, not a scientific result). Never touches
'completed' or 'technically_failed' runs automatically -- the latter
needs a human to look at the log first."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    BUNDLE_ROOT, LOGS, PROJECT_ROOT, RUN_STATE_DIR, SB_SCRIPTS, find_latest_checkpoint,
    load_frozen_config, python_exe, read_run_manifest, run_subprocess, write_run_manifest,
)
from status import classify_run  # noqa: E402


def main() -> int:
    cfg = load_frozen_config()
    if not RUN_STATE_DIR.exists():
        print("[resume_interrupted] no run state directory -- nothing to resume.")
        return 0

    resumed = []
    for f in sorted(RUN_STATE_DIR.glob("*.json")):
        run_id = f.stem
        if run_id.startswith("curriculum_"):
            continue  # handled by resume_curriculum.py, not this script
        manifest = read_run_manifest(run_id)
        if manifest is None:
            continue
        status = classify_run(manifest)
        if status != "interrupted":
            continue

        ckpt_dir = BUNDLE_ROOT / "checkpoints" / "formal_runs" / run_id
        latest = find_latest_checkpoint(ckpt_dir)
        if latest is None:
            print(f"[resume_interrupted] {run_id}: interrupted but no checkpoint found -- "
                  "cannot safely resume, needs a human look (was it a crash before the first checkpoint?).")
            continue

        step, ckpt_path = latest
        budget_end = manifest.get("budget_end_step", 2000000)
        remaining = budget_end - step
        if remaining <= 0:
            print(f"[resume_interrupted] {run_id}: latest checkpoint ({step}) already at/past budget end "
                  f"({budget_end}) -- treating as completed, not resuming. Check status.py output.")
            continue

        seed = manifest["seed"]
        condition = manifest["condition"]
        cmd = [
            python_exe(), str(SB_SCRIPTS / "train_curriculum_stage_highwayenv.py"),
            "--scenario-bank", str(BUNDLE_ROOT / "scenario_banks" / "Q.json"),
            "--scenario-ids", *_scenario_ids(),
            "--stage-name", f"Formal_{condition}",
            "--master-seed", str(seed),
            "--output-root", str(ckpt_dir),
            "--checkpoint-root", str(ckpt_dir),
            "--resume-from", str(ckpt_path),
            "--start-step", str(step),
            "--max-additional-steps", str(remaining),
            "--episode-max-steps", str(cfg["environment"]["episode_max_steps"]),
            "--checkpoint-every", "50000",
            "--device", "cpu",
            "--replay-warmup", "512",
            "--eps-decay-steps-absolute", str(cfg["dqn"]["eps_decay_steps_absolute"]),
            "--lr-decay-steps-absolute", str(cfg["dqn"]["lr_decay_steps_absolute"]),
            "--welfare-lambda", str(cfg["welfare"]["lambda_W"]),
            "--condition", condition,
            "--action-representation", cfg["environment"]["action_representation"],
            "--local-sensing-range-m", str(cfg["observation"]["local_sensing_range_m"]),
        ]
        log_path = LOGS / f"formal_{run_id}_resumed.log"
        print(f"[resume_interrupted] resuming {run_id} from step {step} (remaining {remaining} steps)")
        write_run_manifest(run_id, started=True, resumed_from_step=step, resumed_unix=time.time())
        proc = run_subprocess(cmd, log_file=log_path, env_overrides={"OMP_NUM_THREADS": "1"})
        write_run_manifest(run_id, pid=proc.pid)
        resumed.append(run_id)

    print(f"[resume_interrupted] resumed {len(resumed)} run(s): {resumed}")
    print("[resume_interrupted] NOTE: this script only launches resumptions; it does not wait for them. "
          "Use monitor_formal.py / status.py to track progress.")
    return 0


def _scenario_ids() -> list[str]:
    data = json.loads((BUNDLE_ROOT / "scenario_banks" / "Q.json").read_text(encoding="utf-8"))
    return [s["scenario_id"] for s in data] if isinstance(data, list) else list(data.keys())


if __name__ == "__main__":
    raise SystemExit(main())
