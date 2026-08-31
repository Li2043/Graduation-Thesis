#!/usr/bin/env python3
"""Launch / resume the 18 matched welfare fine-tunes (new_protocol.md §13).

6 seeds x Mean/GGI/Maximin, each 1,200,000 -> 2,000,000 (800k steps),
lambda_W=0.5, identical C64_R50 init checkpoint per seed. Idempotent:
re-run after pause/crash to continue from the latest verified checkpoint.
No performance-based extension (protocol §13 frozen 800k budget).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    LOGS, PROJECT_ROOT, SB_SCRIPTS, SCENARIO_BANKS, load_frozen_config, python_exe,
    write_json_atomic,
)
from replication_common import (  # noqa: E402
    CONDITIONS, REPL_RUN_STATE, SEEDS, WELFARE_END, WELFARE_ROOT, WELFARE_START,
    WELFARE_STAGE_PREFIX, _ids_from_bank, next_welfare_job, task_c64_checkpoint,
    welfare_ckpt_dir, welfare_output_root, welfare_run_id, latest_verified_checkpoint,
)

POLL_SECONDS = 15
MAX_AUTO_RETRIES = 1
CHECKPOINT_EVERY = 50_000


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        return False


def _state_path(run_id: str) -> Path:
    return REPL_RUN_STATE / f"welfare_{run_id}.json"


def _read_state(run_id: str) -> dict:
    p = _state_path(run_id)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _write_state(run_id: str, **fields) -> None:
    existing = _read_state(run_id)
    existing.update(fields)
    existing["run_id"] = run_id
    existing["last_update_unix"] = time.time()
    write_json_atomic(_state_path(run_id), existing)


def _all_jobs() -> list[tuple[str, dict | None]]:
    out = []
    for seed in SEEDS:
        for cond in CONDITIONS:
            out.append((welfare_run_id(seed, cond), next_welfare_job(seed, cond)))
    return out


def _build_cmd(job: dict, *, device: str) -> list[str]:
    cfg = load_frozen_config()
    seed, cond = job["seed"], job["condition"]
    out_root = welfare_output_root(seed, cond)
    stage = f"{WELFARE_STAGE_PREFIX}_{cond}"
    q_ids = _ids_from_bank("Q.json")
    cmd = [
        python_exe(), str(SB_SCRIPTS / "train_curriculum_stage_highwayenv.py"),
        "--scenario-bank", str(SCENARIO_BANKS / "Q.json"),
        "--scenario-ids", *q_ids,
        "--stage-name", stage,
        "--master-seed", str(seed),
        "--output-root", str(out_root),
        "--checkpoint-root", str(out_root),
        "--start-step", str(job["start_step"]),
        "--max-additional-steps", str(job["max_additional_steps"]),
        "--episode-max-steps", str(cfg["environment"]["episode_max_steps"]),
        "--checkpoint-every", str(CHECKPOINT_EVERY),
        "--device", device,
        "--replay-warmup", "512",
        "--eps-decay-steps-absolute", str(cfg["dqn"]["eps_decay_steps_absolute"]),
        "--lr-decay-steps-absolute", str(cfg["dqn"]["lr_decay_steps_absolute"]),
        "--welfare-lambda", str(cfg["welfare"]["lambda_W"]),
        "--condition", cond,
        "--action-representation", cfg["environment"]["action_representation"],
        "--local-sensing-range-m", str(cfg["observation"]["local_sensing_range_m"]),
        "--resume-from", str(job["resume_from"]),
    ]
    return cmd


def _popen(cmd: list[str], log_file: Path) -> subprocess.Popen:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    log_fh = open(log_file, "a", encoding="utf-8", buffering=1)
    kwargs: dict = dict(
        cwd=str(PROJECT_ROOT), stdout=log_fh, stderr=subprocess.STDOUT, env=env,
    )
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(cmd, **kwargs)


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-concurrent", type=int, default=18)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    REPL_RUN_STATE.mkdir(parents=True, exist_ok=True)
    WELFARE_ROOT.mkdir(parents=True, exist_ok=True)

    missing = [s for s in SEEDS if task_c64_checkpoint(s) is None]
    if missing:
        print(f"[welfare] C64_R50 step 1,200,000 missing for seeds {missing} -- finish curriculum first.")
        return 2

    print(f"[welfare] device={args.device} max_concurrent={args.max_concurrent} "
          f"budget={WELFARE_START}->{WELFARE_END} lambda_W=0.5")
    jobs_needed = []
    for run_id, job in _all_jobs():
        if job is None:
            print(f"  {run_id}: COMPLETE")
        else:
            jobs_needed.append((run_id, job))
            print(f"  {run_id}: {job['start_step']}->{WELFARE_END} "
                  f"(+{job['max_additional_steps']}) from {job['resume_from'].name}")
    if args.dry_run:
        return 0

    running: dict[str, subprocess.Popen] = {}
    retries: dict[str, int] = {welfare_run_id(s, c): 0 for s in SEEDS for c in CONDITIONS}

    def _refresh(run_id: str, seed: int, cond: str) -> None:
        latest = latest_verified_checkpoint(welfare_ckpt_dir(seed, cond))
        _write_state(
            run_id, seed=seed, condition=cond,
            current_step=latest[0] if latest else job_start_cache.get(run_id, WELFARE_START),
            latest_checkpoint=str(latest[1]) if latest else None,
            completed=latest is not None and latest[0] >= WELFARE_END,
        )

    job_start_cache: dict[str, int] = {}

    def _launch(run_id: str, job: dict) -> None:
        log_path = LOGS / f"replication_welfare_{run_id}.log"
        cmd = _build_cmd(job, device=args.device)
        print(f"[welfare] launching {run_id} {job['start_step']}->{WELFARE_END} log={log_path}")
        proc = _popen(cmd, log_path)
        running[run_id] = proc
        job_start_cache[run_id] = job["start_step"]
        _write_state(
            run_id, seed=job["seed"], condition=job["condition"], started=True,
            completed=False, technical_failure=False, pid=proc.pid,
            current_step=job["start_step"], resume_from=str(job["resume_from"]),
            log_path=str(log_path), start_unix=time.time(),
        )

    # Adopt live PIDs from a previous supervisor.
    for run_id, job in _all_jobs():
        state = _read_state(run_id)
        if job is None:
            continue
        pid = state.get("pid")
        if _pid_alive(pid):
            print(f"[welfare] adopting live pid={pid} for {run_id}")
            running[run_id] = type("P", (), {"pid": pid, "poll": lambda self=None, p=pid: None if _pid_alive(p) else 0})()

    while True:
        for run_id, job in _all_jobs():
            if run_id in running:
                continue
            state = _read_state(run_id)
            if state.get("technical_failure") and retries[run_id] >= MAX_AUTO_RETRIES:
                continue
            if job is None:
                _write_state(run_id, completed=True, pid=None, current_step=WELFARE_END)
                continue
            if len(running) >= args.max_concurrent:
                break
            _launch(run_id, job)

        if not running:
            remaining = [rid for rid, j in _all_jobs() if j is not None]
            failed = [rid for rid, _ in _all_jobs() if _read_state(rid).get("technical_failure")
                      and retries[rid] >= MAX_AUTO_RETRIES]
            if not remaining:
                print("[welfare] all 18 fine-tunes complete at step 2,000,000.")
                return 0
            if remaining and set(remaining) <= set(failed):
                print(f"[welfare] STOP: technically_failed: {failed}")
                return 2
            time.sleep(POLL_SECONDS)
            continue

        time.sleep(POLL_SECONDS)
        finished = []
        for run_id, proc in list(running.items()):
            ret = proc.poll()
            seed = int(run_id.split("_")[-1])
            cond = run_id[: run_id.rfind("_")]
            _refresh(run_id, seed, cond)
            if ret is None:
                continue
            finished.append(run_id)
            nxt = next_welfare_job(seed, cond)
            if ret != 0:
                retries[run_id] += 1
                print(f"[welfare] {run_id} rc={ret} retry {retries[run_id]}/{MAX_AUTO_RETRIES}")
                _write_state(run_id, pid=None, returncode=ret,
                             technical_failure=retries[run_id] > MAX_AUTO_RETRIES)
            else:
                print(f"[welfare] {run_id} rc=0 next={'DONE' if nxt is None else nxt['start_step']}")
                _write_state(run_id, pid=None, returncode=0, technical_failure=False,
                             completed=nxt is None)
        for run_id in finished:
            running.pop(run_id, None)


if __name__ == "__main__":
    raise SystemExit(main())
