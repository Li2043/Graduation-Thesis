#!/usr/bin/env python3
"""Launch / resume the independent-seed replication task curriculum
(new_protocol.md §11) for seeds 920101-920106.

Idempotent: inspects checkpoints/seed_replication_v1/curriculum/{seed}/
and continues each seed from its latest *verified* checkpoint. Re-run
this script after a pause, crash, or reboot -- it will not restart a
finished stage.

Does NOT implement training. Calls the frozen
train_curriculum_stage_highwayenv.py with the same flags resume_curriculum.py
uses (welfare-lambda=0, condition=mean, R=50m, meta_speed). C4 is the
protocol's frozen 300k budget (400k->700k) with no performance-based
C4_R50ext extension (new_protocol.md §11).
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
    CURRICULUM_ROOT, REPL_RUN_STATE, SEEDS, next_stage_job, seed_progress,
    stage_ckpt_dir, stage_output_root,
)

POLL_SECONDS = 15
MAX_AUTO_RETRIES = 1


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        import os as _os
        try:
            _os.kill(pid, 0)
            return True
        except OSError:
            return False
        except AttributeError:
            return True


def _run_state_path(seed: int) -> Path:
    return REPL_RUN_STATE / f"curriculum_{seed}.json"


def _read_state(seed: int) -> dict:
    p = _run_state_path(seed)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _write_state(seed: int, **fields) -> None:
    existing = _read_state(seed)
    existing.update(fields)
    existing["seed"] = seed
    existing["last_update_unix"] = time.time()
    write_json_atomic(_run_state_path(seed), existing)


def _build_cmd(job: dict, *, device: str) -> list[str]:
    cfg = load_frozen_config()
    st = job["stage"]
    seed = job["seed"]
    out_root = stage_output_root(seed, st.name)
    ckpt_root = out_root  # training script appends seed_{seed}_{stage}
    cmd = [
        python_exe(), str(SB_SCRIPTS / "train_curriculum_stage_highwayenv.py"),
        "--scenario-bank", str(SCENARIO_BANKS / st.scenario_bank),
        "--scenario-ids", *st.scenario_ids,
        "--stage-name", st.name,
        "--master-seed", str(seed),
        "--output-root", str(out_root),
        "--checkpoint-root", str(ckpt_root),
        "--start-step", str(job["start_step"]),
        "--max-additional-steps", str(job["max_additional_steps"]),
        "--episode-max-steps", str(cfg["environment"]["episode_max_steps"]),
        "--checkpoint-every", str(st.checkpoint_every),
        "--device", device,
        "--replay-warmup", "512",
        "--eps-decay-steps-absolute", str(cfg["dqn"]["eps_decay_steps_absolute"]),
        "--lr-decay-steps-absolute", str(cfg["dqn"]["lr_decay_steps_absolute"]),
        "--welfare-lambda", "0.0",
        "--condition", "mean",
        "--action-representation", cfg["environment"]["action_representation"],
        "--local-sensing-range-m", str(cfg["observation"]["local_sensing_range_m"]),
    ]
    if job["resume_from"] is not None:
        cmd += ["--resume-from", str(job["resume_from"])]
    return cmd


def _popen(cmd: list[str], log_file: Path) -> subprocess.Popen:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    log_fh = open(log_file, "a", encoding="utf-8", buffering=1)
    kwargs: dict = dict(
        cwd=str(PROJECT_ROOT),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
    )
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(cmd, **kwargs)


def _refresh_progress(seed: int) -> None:
    prog = seed_progress(seed)
    latest_path = None
    st_name = prog["current_stage"] or "C64_R50"
    ckpt = None
    if st_name:
        from replication_common import latest_verified_checkpoint
        ckpt = latest_verified_checkpoint(stage_ckpt_dir(seed, st_name if st_name != "C64_R50" or not prog["curriculum_complete"] else "C64_R50"))
        if prog["curriculum_complete"]:
            ckpt = latest_verified_checkpoint(stage_ckpt_dir(seed, "C64_R50"))
    if ckpt:
        latest_path = str(ckpt[1])
    _write_state(
        seed,
        current_stage=prog["current_stage"],
        current_step=prog["latest_step"],
        latest_checkpoint=latest_path,
        curriculum_complete=prog["curriculum_complete"],
        stage_progress=prog["stages"],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-concurrent", type=int, default=6,
                    help="max simultaneous curriculum processes (default 6 = all seeds)")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--once", action="store_true",
                    help="launch currently-needed jobs then exit without waiting (processes keep running)")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    REPL_RUN_STATE.mkdir(parents=True, exist_ok=True)
    CURRICULUM_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[replication] device={args.device} max_concurrent={args.max_concurrent}", flush=True)
    print(f"[replication] checkpoints: {CURRICULUM_ROOT}")
    print("[replication] re-run this script to resume after pause/crash; "
          "it continues from the latest verified ckpt_step_*.pt")

    jobs_preview = []
    for seed in SEEDS:
        job = next_stage_job(seed)
        prog = seed_progress(seed)
        if job is None:
            print(f"  seed {seed}: CURRICULUM COMPLETE (C64_R50 @ {prog['latest_step']})")
        else:
            st = job["stage"]
            jobs_preview.append((seed, job))
            print(f"  seed {seed}: next {st.name} steps {job['start_step']} -> {st.end_step} "
                  f"(+{job['max_additional_steps']}) resume_from={job['resume_from']}")
    if args.dry_run:
        return 0

    running: dict[int, subprocess.Popen] = {}
    retries: dict[int, int] = {s: 0 for s in SEEDS}

    def _adopt_live_pids() -> None:
        """If a previous supervisor died, do not double-launch a seed whose
        training process is still alive."""
        for seed in SEEDS:
            state = _read_state(seed)
            pid = state.get("pid")
            if state.get("curriculum_complete"):
                continue
            if _pid_alive(pid) and not state.get("completed_stage_pending"):
                # Best-effort: we cannot recover the Popen, but we can skip
                # launching until that pid exits.
                running[seed] = _PidOnly(pid)  # type: ignore[assignment]
                print(f"[replication] seed {seed}: adopting live pid={pid}")

    class _PidOnly:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def poll(self):
            return None if _pid_alive(self.pid) else 0

    _adopt_live_pids()

    def _launch(seed: int, job: dict) -> None:
        st = job["stage"]
        log_path = LOGS / f"replication_curriculum_{st.name}_{seed}.log"
        cmd = _build_cmd(job, device=args.device)
        print(f"[replication] launching seed={seed} {st.name} "
              f"{job['start_step']}->{st.end_step} log={log_path}")
        proc = _popen(cmd, log_path)
        running[seed] = proc
        _write_state(
            seed,
            started=True,
            completed=False,
            technical_failure=False,
            pid=proc.pid,
            current_stage=st.name,
            current_step=job["start_step"],
            resume_from=str(job["resume_from"]) if job["resume_from"] else None,
            log_path=str(log_path),
            start_unix=time.time(),
        )

    while True:
        for seed in SEEDS:
            if seed in running:
                continue
            state = _read_state(seed)
            if state.get("technical_failure") and retries[seed] >= MAX_AUTO_RETRIES:
                continue
            job = next_stage_job(seed)
            if job is None:
                _refresh_progress(seed)
                _write_state(seed, completed=True, pid=None, current_stage=None,
                             curriculum_complete=True)
                continue
            if len(running) >= args.max_concurrent:
                break
            _launch(seed, job)

        if args.once:
            print("[replication] --once: launched what was needed; exiting (workers keep running).")
            return 0

        if not running:
            remaining = [s for s in SEEDS if next_stage_job(s) is not None]
            failed = [s for s in SEEDS if _read_state(s).get("technical_failure")
                      and retries[s] >= MAX_AUTO_RETRIES]
            if not remaining:
                print("[replication] all six curricula complete at C64_R50 step 1,200,000.")
                return 0
            if remaining and len(failed) == len(remaining):
                print(f"[replication] STOP: remaining seeds are technically_failed: {failed}")
                print("  Inspect logs/replication_curriculum_*.log then re-run this script.")
                return 2
            time.sleep(POLL_SECONDS)
            continue

        time.sleep(POLL_SECONDS)
        finished = []
        for seed, proc in list(running.items()):
            ret = proc.poll()
            _refresh_progress(seed)
            if ret is None:
                continue
            finished.append(seed)
            job_now = next_stage_job(seed)
            if ret != 0:
                retries[seed] = retries.get(seed, 0) + 1
                print(f"[replication] seed {seed} exited rc={ret} "
                      f"(retry {retries[seed]}/{MAX_AUTO_RETRIES})")
                if retries[seed] > MAX_AUTO_RETRIES:
                    _write_state(seed, technical_failure=True, pid=None, returncode=ret)
                else:
                    _write_state(seed, pid=None, returncode=ret, technical_failure=False)
            else:
                print(f"[replication] seed {seed} subprocess finished rc=0; "
                      f"next={'DONE' if job_now is None else job_now['stage'].name}")
                _write_state(seed, pid=None, returncode=0, technical_failure=False)
        for seed in finished:
            running.pop(seed, None)


if __name__ == "__main__":
    raise SystemExit(main())
