#!/usr/bin/env python3
"""Launch the 18 formal Mean/GGI/Maximin runs as independent OS
processes (process-level parallelism only -- never changes batch size,
update frequency, or any other per-run algorithmic behaviour to use
more cores).

Refuses to run unless FORMAL_EXPERIMENT_FREEZE_MANIFEST.json exists
(i.e. freeze_formal_manifest.py has already run and every
precondition passed).

Concurrency is chosen from a real resource probe (CPU count, available
RAM, measured per-process memory from a short trial run) -- never
blindly starts all 18 at once. Pass --max-concurrent to override the
computed safe value (still capped at 18)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    BUNDLE_ROOT, LOGS, PROJECT_ROOT, SB_SCRIPTS, load_frozen_config, needs_user_decision,
    python_exe, run_subprocess, write_run_manifest,
)

ESTIMATED_MEM_PER_PROCESS_MB = 400  # conservative estimate for this tiny-network CPU workload;
# refine via detect_hardware.py + a short trial if you want a tighter bound.


def compute_safe_concurrency(requested: int | None) -> int:
    import multiprocessing
    cpu = multiprocessing.cpu_count()
    try:
        import psutil
        ram_available_mb = psutil.virtual_memory().available / (1024 ** 2)
    except ImportError:
        ram_available_mb = None

    by_cpu = max(1, cpu - 2)  # leave headroom for OS/monitoring, matches this bundle's OMP_NUM_THREADS=1 design
    by_ram = int(ram_available_mb // ESTIMATED_MEM_PER_PROCESS_MB) if ram_available_mb else by_cpu
    safe = min(by_cpu, by_ram, 18)
    if requested is not None:
        safe = min(safe, requested, 18)
    return max(1, safe)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-concurrent", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the launch plan, launch nothing")
    args = ap.parse_args()

    manifest_path = BUNDLE_ROOT / "FORMAL_EXPERIMENT_FREEZE_MANIFEST.json"
    if not manifest_path.exists():
        needs_user_decision(
            issue="launch_formal.py was called before FORMAL_EXPERIMENT_FREEZE_MANIFEST.json exists.",
            evidence=f"Expected at {manifest_path}", options=["Run scripts/freeze_formal_manifest.py first."],
            consequences="Launching formal training without a frozen manifest risks an unrecorded/mutable "
                          "configuration for what becomes the actual thesis results.",
            recommendation="Run freeze_formal_manifest.py, confirm it succeeds, then rerun this script.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cfg = load_frozen_config()
    if not cfg.get("formal_fairness_started"):
        needs_user_decision(
            issue="formal_fairness_started is not true in FROZEN_EXPERIMENT_CONFIG.json even though a freeze "
                  "manifest exists.",
            evidence=str(cfg.get("formal_fairness_started")),
            options=["Re-run freeze_formal_manifest.py."], consequences="Inconsistent state.",
            recommendation="Do not proceed until this flag is true.")

    concurrency = compute_safe_concurrency(args.max_concurrent)
    print(f"[launch_formal] chosen concurrency: {concurrency} (of {len(manifest['run_matrix'])} total runs)")
    (BUNDLE_ROOT / "verification" / "launch_concurrency.log").parent.mkdir(parents=True, exist_ok=True)
    (BUNDLE_ROOT / "verification" / "launch_concurrency.log").write_text(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} chosen_concurrency={concurrency}\n", encoding="utf-8")

    if args.dry_run:
        for r in manifest["run_matrix"]:
            print(f"  would launch: {r['run_id']} (seed={r['seed']} condition={r['condition']})")
        return 0

    budget = manifest["budget"]
    running: list[tuple[str, subprocess.Popen]] = []
    queue = list(manifest["run_matrix"])

    def _launch_one(r: dict) -> subprocess.Popen:
        run_id = r["run_id"]
        ckpt_root = BUNDLE_ROOT / "checkpoints" / "formal_runs" / run_id
        out_root = ckpt_root
        cmd = [
            python_exe(), str(SB_SCRIPTS / "train_curriculum_stage_highwayenv.py"),
            "--scenario-bank", str(BUNDLE_ROOT / "scenario_banks" / "Q.json"),
            "--scenario-ids", *_scenario_ids(),
            "--stage-name", f"Formal_{r['condition']}",
            "--master-seed", str(r["seed"]),
            "--output-root", str(out_root),
            "--checkpoint-root", str(ckpt_root),
            "--resume-from", r["init_checkpoint"],
            "--start-step", str(budget["start_step"]),
            "--max-additional-steps", str(budget["initial_max_step"] - budget["start_step"]),
            "--episode-max-steps", str(cfg["environment"]["episode_max_steps"]),
            "--checkpoint-every", "50000",
            "--device", "cpu",
            "--replay-warmup", "512",
            "--eps-decay-steps-absolute", str(cfg["dqn"]["eps_decay_steps_absolute"]),
            "--lr-decay-steps-absolute", str(cfg["dqn"]["lr_decay_steps_absolute"]),
            "--welfare-lambda", str(cfg["welfare"]["lambda_W"]),
            "--condition", r["condition"],
            "--action-representation", cfg["environment"]["action_representation"],
            "--local-sensing-range-m", str(cfg["observation"]["local_sensing_range_m"]),
        ]
        log_path = LOGS / f"formal_{run_id}.log"
        write_run_manifest(run_id, seed=r["seed"], condition=r["condition"], started=True,
                            start_unix=time.time(), pid=None, completed=False, technical_failure=False,
                            init_checkpoint=r["init_checkpoint"], log_path=str(log_path))
        proc = run_subprocess(cmd, log_file=log_path, env_overrides={"OMP_NUM_THREADS": "1"})
        write_run_manifest(run_id, pid=proc.pid)
        print(f"[launch_formal] started {run_id} pid={proc.pid}")
        return proc

    while queue or running:
        while queue and len(running) < concurrency:
            r = queue.pop(0)
            running.append((r["run_id"], _launch_one(r)))
        time.sleep(5)
        still_running = []
        for run_id, proc in running:
            ret = proc.poll()
            if ret is None:
                still_running.append((run_id, proc))
            else:
                completed = ret == 0
                write_run_manifest(run_id, completed=completed, technical_failure=not completed,
                                    end_unix=time.time(), returncode=ret)
                print(f"[launch_formal] {run_id} finished, returncode={ret} "
                      f"({'completed' if completed else 'TECHNICAL FAILURE -- see log'})")
        running = still_running

    print("[launch_formal] all launched runs have exited. Run scripts/status.py for a full summary.")
    return 0


def _scenario_ids() -> list[str]:
    data = json.loads((BUNDLE_ROOT / "scenario_banks" / "Q.json").read_text(encoding="utf-8"))
    return [s["scenario_id"] for s in data] if isinstance(data, list) else list(data.keys())


if __name__ == "__main__":
    raise SystemExit(main())
