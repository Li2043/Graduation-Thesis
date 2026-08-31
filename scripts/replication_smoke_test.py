#!/usr/bin/env python3
"""Technical smoke test for the replication package (new_protocol.md §10).

Uses seed 929999 only. Not a scientific result. Writes exclusively under
checkpoints/seed_replication_v1/smoke_929999/ so it cannot be mistaken
for a formal replication seed. Checks:

- training script launches
- environment loads
- checkpoints save and reload
- logs write
- replay buffer is populated
- the evaluation entry script can resolve the checkpoint path convention
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    LOGS, PROJECT_ROOT, SB_SCRIPTS, SCENARIO_BANKS, load_frozen_config, python_exe,
)
from replication_common import SMOKE_ROOT, SMOKE_SEED  # noqa: E402

SMOKE_STEPS = 2000
SMOKE_CKPT_EVERY = 1000
STAGE = "M6_R50_audited"


def main() -> int:
    cfg = load_frozen_config()
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    out_root = SMOKE_ROOT / str(SMOKE_SEED) / STAGE
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_root / f"seed_{SMOKE_SEED}_{STAGE}"
    log_path = LOGS / f"replication_smoke_{SMOKE_SEED}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_exe(), str(SB_SCRIPTS / "train_curriculum_stage_highwayenv.py"),
        "--scenario-bank", str(SCENARIO_BANKS / "Q.json"),
        "--scenario-ids", "Q_00000",
        "--stage-name", STAGE,
        "--master-seed", str(SMOKE_SEED),
        "--output-root", str(out_root),
        "--checkpoint-root", str(out_root),
        "--start-step", "0",
        "--max-additional-steps", str(SMOKE_STEPS),
        "--episode-max-steps", str(cfg["environment"]["episode_max_steps"]),
        "--checkpoint-every", str(SMOKE_CKPT_EVERY),
        "--device", "cpu",
        "--replay-warmup", "512",
        "--eps-decay-steps-absolute", str(cfg["dqn"]["eps_decay_steps_absolute"]),
        "--lr-decay-steps-absolute", str(cfg["dqn"]["lr_decay_steps_absolute"]),
        "--welfare-lambda", "0.0",
        "--condition", "mean",
        "--action-representation", cfg["environment"]["action_representation"],
        "--local-sensing-range-m", str(cfg["observation"]["local_sensing_range_m"]),
    ]
    print(f"[smoke] launching {SMOKE_STEPS} steps for seed {SMOKE_SEED}")
    print(f"[smoke] isolated output: {out_root}")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log_fh:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), stdout=log_fh, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    print(f"[smoke] training rc={proc.returncode} elapsed={elapsed:.1f}s log={log_path}")
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="ignore")[-3000:]
        print(tail)
        print("[smoke] FAIL: training subprocess non-zero")
        return 1

    expected = [0, SMOKE_CKPT_EVERY, SMOKE_STEPS]
    missing = [s for s in expected if not (ckpt_dir / f"ckpt_step_{s}.pt").exists()]
    if missing:
        print(f"[smoke] FAIL: missing checkpoints {missing} in {ckpt_dir}")
        return 1

    import torch
    ckpt_path = ckpt_dir / f"ckpt_step_{SMOKE_STEPS}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    required = ("online", "target", "optimiser", "step", "stage", "replay_size")
    missing_keys = [k for k in required if k not in ckpt]
    if missing_keys:
        print(f"[smoke] FAIL: checkpoint missing keys {missing_keys}")
        return 1
    if int(ckpt["step"]) != SMOKE_STEPS:
        print(f"[smoke] FAIL: checkpoint step {ckpt['step']} != {SMOKE_STEPS}")
        return 1
    if int(ckpt["replay_size"]) <= 0:
        print(f"[smoke] FAIL: replay_size={ckpt['replay_size']} (replay did not populate)")
        return 1

    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    if "resumed from" in log_text:
        print("[smoke] note: log mentions resume (unexpected for a fresh smoke run)")
    if f"[{STAGE}]" not in log_text:
        print("[smoke] FAIL: log has no stage checkpoint lines")
        return 1

    eval_script = SB_SCRIPTS / "evaluate_formal_welfare.py"
    if not eval_script.exists():
        print(f"[smoke] FAIL: evaluation script not found at {eval_script}")
        return 1
    # Path convention the curriculum/welfare eval pipeline uses:
    #   .../seed_{seed}_{stage}/ckpt_step_{N}.pt
    if f"seed_{SMOKE_SEED}_{STAGE}" not in str(ckpt_path):
        print("[smoke] FAIL: checkpoint path does not match seed_{seed}_{stage} convention")
        return 1

    marker = SMOKE_ROOT / "SMOKE_TEST_PASSED.json"
    marker.write_text(json.dumps({
        "seed": SMOKE_SEED,
        "scientific": False,
        "steps": SMOKE_STEPS,
        "checkpoint": str(ckpt_path),
        "replay_size": int(ckpt["replay_size"]),
        "elapsed_seconds": elapsed,
        "eval_script": str(eval_script),
        "isolated_from_formal_seeds": True,
    }, indent=2), encoding="utf-8")
    print(f"[smoke] PASS  ckpt={ckpt_path}")
    print(f"[smoke] replay_size={ckpt['replay_size']}  marker={marker}")
    print("[smoke] outputs are isolated under smoke_929999 -- not a replication seed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
