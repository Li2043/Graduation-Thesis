#!/usr/bin/env python3
"""Replication Stage 2 evaluation (new_protocol.md §20-29).

Reuses the frozen evaluate_formal_welfare / evaluate_formal_behavioral /
evaluate_high_burden_diagnostic implementations unchanged except seed IDs
and checkpoint/output locations (protocol §8). Ensemble window is the
same K(2,000,000)={1.85M, 1.90M, 1.95M, 2.00M}, epsilon=0, R=50m.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import LOGS, PROJECT_ROOT, SB_SCRIPTS, python_exe  # noqa: E402
from replication_common import CONDITION_DIR, SEEDS, WELFARE_ROOT  # noqa: E402

OUT_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "seed_replication_v1"
KINDS = ("welfare", "behavioral", "high_burden")
N_SHARDS = 8


def _patch_module(mod, *, window) -> None:
    from _common import SCENARIO_BANKS

    def checkpoint_paths_for(run_id: str, seed: int):
        cond = run_id.split("_", 1)[0]
        d = WELFARE_ROOT / str(seed) / CONDITION_DIR[cond] / f"seed_{seed}_Formal_{cond}"
        return {s: d / f"ckpt_step_{s}.pt" for s in window}

    mod.SEEDS = list(SEEDS)
    mod.BANK_ROOT = SCENARIO_BANKS
    mod.CKPT_ROOT = WELFARE_ROOT
    mod.checkpoint_paths_for = checkpoint_paths_for


def _load_kind(kind: str):
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    sys.path.insert(0, str(SB_SCRIPTS))
    if kind == "welfare":
        import evaluate_formal_welfare as mod
        out = OUT_ROOT / "welfare_eval"
        shard_name = "replication_welfare_eval_shard{}.csv"
    elif kind == "behavioral":
        import evaluate_formal_behavioral as mod
        out = OUT_ROOT / "behavioral"
        shard_name = "replication_behavioral_eval_shard{}.csv"
    else:
        import evaluate_high_burden_diagnostic as mod
        out = OUT_ROOT / "behavioral"
        shard_name = "replication_high_burden_shard{}.csv"
    _patch_module(mod, window=mod.WINDOW)
    return mod, out, shard_name


def verify_ensemble_checkpoints() -> list[str]:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from thesis.study_b.q_ensemble import ensemble_window_for_stage_end
    window = ensemble_window_for_stage_end(2_000_000)
    missing = []
    for seed in SEEDS:
        for cond, dirname in CONDITION_DIR.items():
            d = WELFARE_ROOT / str(seed) / dirname / f"seed_{seed}_Formal_{cond}"
            for step in window:
                p = d / f"ckpt_step_{step}.pt"
                if not p.exists():
                    missing.append(str(p))
    return missing


def run_shard(kind: str, shard_index: int, num_shards: int) -> int:
    mod, out_dir, shard_name = _load_kind(kind)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = mod.all_jobs()
    my_jobs = [j for i, j in enumerate(jobs) if i % num_shards == shard_index]
    print(f"[evaluate_replication {kind} shard {shard_index}/{num_shards}] {len(my_jobs)} jobs", flush=True)
    all_rows = []
    for run_id, condition, seed, bank in my_jobs:
        print(f"  {run_id} {bank} ...", flush=True)
        rows = mod.run_one(run_id, condition, seed, bank)
        all_rows.extend(rows)
        n = len(rows)
        if kind == "welfare":
            comp = sum(r["completion"] for r in rows) / n
            print(f"    n={n} completion={comp:.3f}", flush=True)
        else:
            print(f"    n={n}", flush=True)
    out_csv = out_dir / shard_name.format(shard_index)
    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=mod.CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {len(all_rows)} rows -> {out_csv}", flush=True)
    return 0


def launch_kind(kind: str, num_shards: int, max_concurrent: int) -> int:
    missing = verify_ensemble_checkpoints()
    if missing:
        print(f"[evaluate_replication] missing ensemble checkpoints ({len(missing)}):")
        for p in missing[:12]:
            print(f"  {p}")
        return 2
    LOGS.mkdir(parents=True, exist_ok=True)
    running: list[tuple[int, subprocess.Popen]] = []
    rc_worst = 0
    queue = list(range(num_shards))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    while queue or running:
        while queue and len(running) < max_concurrent:
            i = queue.pop(0)
            log = LOGS / f"replication_eval_{kind}_shard{i}.log"
            cmd = [
                python_exe(), str(Path(__file__).resolve()),
                "--kind", kind, "--shard-index", str(i), "--num-shards", str(num_shards),
            ]
            log_fh = open(log, "w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd, cwd=str(PROJECT_ROOT.parent), stdout=log_fh, stderr=subprocess.STDOUT, env=env,
            )
            print(f"[evaluate_replication] started {kind} shard {i} pid={proc.pid} log={log}", flush=True)
            running.append((i, proc))
        still = []
        for i, proc in running:
            ret = proc.poll()
            if ret is None:
                still.append((i, proc))
            else:
                print(f"[evaluate_replication] {kind} shard {i} rc={ret}", flush=True)
                if ret != 0:
                    rc_worst = ret
        running = still
        if running:
            import time
            time.sleep(5)
    return rc_worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=(*KINDS, "all"), default="all")
    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=N_SHARDS)
    ap.add_argument("--max-concurrent", type=int, default=8)
    args = ap.parse_args()

    if args.shard_index is not None:
        return run_shard(args.kind if args.kind != "all" else "welfare", args.shard_index, args.num_shards)

    kinds = KINDS if args.kind == "all" else (args.kind,)
    for kind in kinds:
        print(f"\n===== {kind} =====", flush=True)
        rc = launch_kind(kind, args.num_shards, args.max_concurrent)
        if rc != 0:
            print(f"[evaluate_replication] {kind} failed rc={rc}")
            return rc
    print("[evaluate_replication] all requested kinds finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
