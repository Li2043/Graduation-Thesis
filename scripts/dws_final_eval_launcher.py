"""DWS final re-evaluation -- orchestrates all 4 cells x 12 seeds = 48
independent held-out rollout jobs via dws_eval_worker.py, then merges the 48
per-shard episode CSVs into one unified dws_final_episode_level.csv.

Process-level parallelism (subprocess.Popen + poll loop), matching this
project's own established convention in scripts/launch_dense_priority.py
(OMP_NUM_THREADS=1/MKL_NUM_THREADS=1 per worker, cpu_count()-2 workers).

Read-only w.r.t. training: only invokes dws_eval_worker.py against already-
frozen checkpoints; writes new files only under OUT_ROOT.
"""
from __future__ import annotations

import csv
import multiprocessing
import os
import subprocess
import time
from pathlib import Path

DENSE_BUNDLE = Path(os.environ.get("DENSE_BUNDLE_ROOT", str(Path(__file__).resolve().parent.parent)))
ORIG_BUNDLE = Path(os.environ.get("ORIG_BUNDLE_ROOT", ""))  # checkpoints not distributed with this repo; set env var to your local bundle
PYTHON = str(DENSE_BUNDLE / ".venv" / "Scripts" / "python.exe")
WORKER = str(DENSE_BUNDLE / "scripts" / "dws_eval_worker.py")
OUT_ROOT = DENSE_BUNDLE / "outputs" / "dws_final_reevaluation_v1"
SHARD_DIR = OUT_ROOT / "episode_shards"
TRAJ_DIR = OUT_ROOT / "trajectories"
LOG_DIR = OUT_ROOT / "logs"
for d in (SHARD_DIR, TRAJ_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEEDS_ORIG = (900101, 900102, 900103, 900104, 910101, 910102)
SEEDS_REPL = (920101, 920102, 920103, 920104, 920105, 920106)
SEEDS_12 = SEEDS_ORIG + SEEDS_REPL


def cell1_ckpt_dir(seed: int) -> Path:
    if seed in SEEDS_ORIG:
        return ORIG_BUNDLE / "checkpoints" / "formal_runs" / f"maximin_{seed}" / f"seed_{seed}_Formal_maximin"
    return ORIG_BUNDLE / "checkpoints" / "seed_replication_v1" / "welfare" / str(seed) / "Maximin" / f"seed_{seed}_Formal_maximin"


def cell2_ckpt_dir(seed: int) -> Path:
    return DENSE_BUNDLE / "checkpoints" / "maximin_dense" / f"maximin_dense_{seed}" / f"seed_{seed}_Dense_maximin_dense"


def cell3_ckpt_dir(seed: int) -> Path:
    return ORIG_BUNDLE / "checkpoints" / "wsc_formal_runs_v2" / f"maximin_wsc_{seed}" / f"seed_{seed}_Formal_maximin_WSC_v2"


def cell4_ckpt_dir(seed: int) -> Path:
    return DENSE_BUNDLE / "checkpoints" / "maximin_wsc_dense" / f"maximin_wsc_dense_{seed}" / f"seed_{seed}_Dense_maximin_wsc_dense"


CELLS = {
    "cell1": dict(project_root=ORIG_BUNDLE / "project", bank=ORIG_BUNDLE / "scenario_banks" / "H1.json",
                  ckpt_dir_fn=cell1_ckpt_dir, stage_name="Formal_maximin", obs_dim=18, wsc=False, dws=False),
    "cell2": dict(project_root=DENSE_BUNDLE / "project", bank=DENSE_BUNDLE / "scenario_banks" / "H1.json",
                  ckpt_dir_fn=cell2_ckpt_dir, stage_name="Dense_maximin_dense", obs_dim=18, wsc=False, dws=True),
    "cell3": dict(project_root=ORIG_BUNDLE / "project", bank=ORIG_BUNDLE / "scenario_banks" / "H1.json",
                  ckpt_dir_fn=cell3_ckpt_dir, stage_name="Formal_maximin_WSC_v2", obs_dim=22, wsc=True, dws=False),
    "cell4": dict(project_root=DENSE_BUNDLE / "project", bank=DENSE_BUNDLE / "scenario_banks" / "H1.json",
                  ckpt_dir_fn=cell4_ckpt_dir, stage_name="Dense_maximin_wsc_dense", obs_dim=22, wsc=True, dws=True),
}


def build_jobs() -> list[dict]:
    jobs = []
    for cell_id, spec in CELLS.items():
        for seed in SEEDS_12:
            ckpt_dir = spec["ckpt_dir_fn"](seed)
            ep_csv = SHARD_DIR / f"{cell_id}_{seed}.csv"
            traj_gz = TRAJ_DIR / f"{cell_id}_{seed}.jsonl.gz"
            cmd = [
                PYTHON, WORKER,
                "--project-root", str(spec["project_root"]),
                "--scenario-bank", str(spec["bank"]),
                "--ckpt-dir", str(ckpt_dir),
                "--stage-name", spec["stage_name"],
                "--obs-dim", str(spec["obs_dim"]),
                "--seed", str(seed), "--cell", cell_id, "--condition", "maximin",
                "--out-episode-csv", str(ep_csv), "--out-trajectory-gz", str(traj_gz),
            ]
            if spec["wsc"]:
                cmd.append("--include-welfare-state")
            if spec["dws"]:
                cmd.append("--dws-on")
            jobs.append({"cell": cell_id, "seed": seed, "cmd": cmd, "ep_csv": ep_csv, "ckpt_dir": ckpt_dir})
    return jobs


def main() -> int:
    jobs = build_jobs()
    missing = [j for j in jobs if not j["ckpt_dir"].exists()]
    if missing:
        print("[dws_final_eval_launcher] MISSING checkpoint directories, aborting before any launch:")
        for j in missing:
            print(f"  {j['cell']} seed={j['seed']}: {j['ckpt_dir']}")
        return 1

    max_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"[dws_final_eval_launcher] {len(jobs)} jobs, max_workers={max_workers}")

    queue = list(jobs)
    running: list[tuple[dict, subprocess.Popen]] = []
    failed: list[dict] = []
    done: list[dict] = []
    while queue or running:
        while queue and len(running) < max_workers:
            job = queue.pop(0)
            log_path = LOG_DIR / f"{job['cell']}_{job['seed']}.log"
            log_fh = open(log_path, "w", encoding="utf-8")
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            proc = subprocess.Popen(job["cmd"], stdout=log_fh, stderr=subprocess.STDOUT, env=env)
            running.append((job, proc))
            print(f"[dws_final_eval_launcher] started {job['cell']} seed={job['seed']} pid={proc.pid}")
        time.sleep(3)
        still = []
        for job, proc in running:
            ret = proc.poll()
            if ret is None:
                still.append((job, proc))
            elif ret == 0:
                done.append(job)
                print(f"[dws_final_eval_launcher] OK {job['cell']} seed={job['seed']}")
            else:
                failed.append(job)
                print(f"[dws_final_eval_launcher] FAILED {job['cell']} seed={job['seed']} returncode={ret} -- see {LOG_DIR / (job['cell']+'_'+str(job['seed'])+'.log')}")
        running = still

    if failed:
        print(f"\n[dws_final_eval_launcher] {len(failed)}/{len(jobs)} jobs FAILED. Not merging shards.")
        return 1

    print(f"\n[dws_final_eval_launcher] all {len(jobs)} jobs completed. Merging shards...")
    all_rows = []
    fieldnames = None
    for j in jobs:
        with open(j["ep_csv"], encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise SystemExit(f"schema mismatch in {j['ep_csv']}: {reader.fieldnames} != {fieldnames}")
            all_rows.extend(reader)

    expected = len(jobs) * 256
    if len(all_rows) != expected:
        raise SystemExit(f"expected {expected} merged episode rows (48 jobs x 256), got {len(all_rows)}")

    out_csv = OUT_ROOT / "dws_final_episode_level.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"[dws_final_eval_launcher] wrote {len(all_rows)} rows -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
