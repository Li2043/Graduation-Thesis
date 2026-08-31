#!/usr/bin/env python3
"""Minimal single-condition launcher for the Dense Reward Study's Priority 1-4
(Maximin and GGI -- README.md Priority 5-6 (Mean) still need the same extension
applied to _PRIORITY_CONDITION/_PRIORITY_RUN_TAG in build_plan() below).

Does NOT reuse/modify launch_formal.py or launch_wsc_formal_batch_v2.py (both
run their own full internal matrix with no per-condition filter -- see
README.md Section 10's gap note). This is a separate, minimal script.

  Priority 1: Maximin + WSC + Dense   -> train_curriculum_stage_highwayenv_wsc.py
  Priority 2: Maximin + Dense-only    -> train_curriculum_stage_highwayenv.py
  Priority 3: GGI + WSC + Dense       -> train_curriculum_stage_highwayenv_wsc.py
  Priority 4: GGI + Dense-only        -> train_curriculum_stage_highwayenv.py

Per README.md Section 6's sequencing rule, Priority 3/4 must NOT be launched for
real (non-dry-run) until Priority 1/2 (Maximin) are formally complete -- this
script does not enforce that itself (no programmatic gate), it is a human
discipline rule, same as every other stopping-rule in this project.

All four use the SAME 12 formal seeds, the SAME real 1.2M-step C64 source
checkpoint per seed (checkpoints/formal_init/<seed>/C64_R50/.../ckpt_step_1200000.pt),
welfare-lambda=0.5 (FROZEN_EXPERIMENT_CONFIG.json's single condition-agnostic
welfare.lambda_W, confirmed shared across Mean/GGI/Maximin by launch_formal.py),
PLUS --dense-welfare-shaping with the frozen configs/dense_reward_protocol_v1.json
magnitude/epsilon. Priority differs by welfare condition (maximin/ggi) and by
which training script is used (WSC observation on/off).

--dry-run prints the exact launch plan (run count, seeds, checkpoints,
commands) and starts nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import BUNDLE_ROOT, LOGS, SB_SCRIPTS, find_latest_checkpoint, python_exe, run_subprocess, write_run_manifest  # noqa: E402

FORMAL_SEEDS: tuple[int, ...] = (
    900101, 900102, 900103, 900104, 910101, 910102,
    920101, 920102, 920103, 920104, 920105, 920106,
)
C64_SOURCE_STEP = 1_200_000
FORMAL_BUDGET_END_STEP = 2_000_000
EPISODE_MAX_STEPS = 200
# Despite the name, this is FROZEN_EXPERIMENT_CONFIG.json's single, condition-agnostic
# welfare.lambda_W (0.5) -- launch_formal.py applies the same lambda_W to Mean/GGI/Maximin
# alike (verified against that config directly), so it is reused unchanged for GGI/Mean here.
WELFARE_LAMBDA_MAXIMIN = 0.5


def _source_checkpoint(seed: int) -> Path:
    return (
        BUNDLE_ROOT / "checkpoints" / "formal_init" / str(seed) / "C64_R50"
        / f"seed_{seed}_C64_R50" / f"ckpt_step_{C64_SOURCE_STEP}.pt"
    )


def _scenario_ids() -> list[str]:
    data = json.loads((BUNDLE_ROOT / "scenario_banks" / "Q.json").read_text(encoding="utf-8"))
    return [s["scenario_id"] for s in data] if isinstance(data, list) else list(data.keys())


def _load_dense_protocol() -> dict:
    path = BUNDLE_ROOT / "configs" / "dense_reward_protocol_v1.json"
    if not path.exists():
        raise SystemExit(
            f"Refusing to build a launch plan: {path} does not exist. "
            "Freeze the Dense Reward Study protocol (magnitude/epsilon) before running this launcher."
        )
    return json.loads(path.read_text(encoding="utf-8"))


# Priority -> (welfare condition, run_tag). WSC-on priorities are odd, Dense-only
# (WSC-off) priorities are even, matching README.md Section 6's numbering:
#   1/2 = Maximin (WSC+Dense / Dense-only), 3/4 = GGI, 5/6 = Mean.
_PRIORITY_CONDITION = {1: "maximin", 2: "maximin", 3: "ggi", 4: "ggi", 5: "mean", 6: "mean"}
_PRIORITY_RUN_TAG = {
    1: "maximin_wsc_dense", 2: "maximin_dense",
    3: "ggi_wsc_dense", 4: "ggi_dense",
    5: "mean_wsc_dense", 6: "mean_dense",
}


def build_plan(priority: int, *, live_smoke_test: bool = False, smoke_steps: int = 20, smoke_seed_count: int = 2, device: str = "cpu", seeds: tuple[int, ...] | None = None, continue_from_latest: bool = False) -> list[dict]:
    if priority not in (1, 2, 3, 4):
        raise SystemExit(
            f"only Priority 1/2 (Maximin) and 3/4 (GGI) are implemented by this launcher, got --priority {priority}. "
            "Priority 5/6 (Mean) still need the same extension applied to _PRIORITY_CONDITION/_PRIORITY_RUN_TAG."
        )
    protocol = _load_dense_protocol()
    if protocol["dense_shaping_mode"] != "discrete":
        raise SystemExit(f"unsupported dense_shaping_mode {protocol['dense_shaping_mode']!r}")

    condition = _PRIORITY_CONDITION[priority]
    use_wsc = priority % 2 == 1
    run_tag = _PRIORITY_RUN_TAG[priority]
    script = SB_SCRIPTS / ("train_curriculum_stage_highwayenv_wsc.py" if use_wsc else "train_curriculum_stage_highwayenv.py")

    # --live-smoke-test: a REAL (non-dry-run) launch of this exact code path (subprocess spawn,
    # log/manifest writing, Windows multiprocessing), but with a tiny step budget and only the
    # first `smoke_seed_count` seeds, writing to a run_tag+"_livesmoke" root that is never read by
    # the real formal launch below -- so this can never collide with or be mistaken for a formal run.
    if seeds is not None:
        unknown = [s for s in seeds if s not in FORMAL_SEEDS]
        if unknown:
            raise SystemExit(f"seeds not in the formal 12-seed set: {unknown}")
        selected_seeds = seeds
    elif live_smoke_test:
        selected_seeds = FORMAL_SEEDS[:smoke_seed_count]
    else:
        selected_seeds = FORMAL_SEEDS
    effective_run_tag = f"{run_tag}_livesmoke" if live_smoke_test else run_tag
    checkpoint_every = smoke_steps if live_smoke_test else 50_000
    replay_warmup = min(5, smoke_steps) if live_smoke_test else 512

    plan = []
    for seed in selected_seeds:
        run_id = f"{effective_run_tag}_{seed}"
        ckpt_root = BUNDLE_ROOT / "checkpoints" / effective_run_tag / run_id
        out_root = ckpt_root
        src_ckpt = _source_checkpoint(seed)
        start_step = C64_SOURCE_STEP
        if continue_from_latest:
            if live_smoke_test:
                raise SystemExit("--continue-from-latest is not used with --live-smoke-test")
            seed_dir = ckpt_root / f"seed_{seed}_Dense_{effective_run_tag}"
            latest = find_latest_checkpoint(seed_dir)
            if latest is None:
                raise SystemExit(f"no checkpoint to continue for seed {seed} under {seed_dir}")
            start_step, src_ckpt = latest
            if start_step >= FORMAL_BUDGET_END_STEP:
                print(f"[launch_dense_priority] skip seed {seed}: already at step {start_step}")
                continue
        max_additional_steps = smoke_steps if live_smoke_test else (FORMAL_BUDGET_END_STEP - start_step)
        cmd = [
            python_exe(), str(script),
            "--scenario-bank", str(BUNDLE_ROOT / "scenario_banks" / "Q.json"),
            "--scenario-ids", *_scenario_ids(),
            "--stage-name", f"Dense_{effective_run_tag}",
            "--master-seed", str(seed),
            "--output-root", str(out_root),
            "--checkpoint-root", str(ckpt_root),
            "--resume-from", str(src_ckpt),
            "--start-step", str(start_step),
            "--max-additional-steps", str(max_additional_steps),
            "--episode-max-steps", str(EPISODE_MAX_STEPS),
            "--checkpoint-every", str(checkpoint_every),
            "--device", device,
            "--replay-warmup", str(replay_warmup),
            "--welfare-lambda", str(WELFARE_LAMBDA_MAXIMIN),
            "--condition", condition,
            "--dense-welfare-shaping",
            "--dense-shaping-mode", protocol["dense_shaping_mode"],
            "--dense-shaping-magnitude", str(protocol["dense_shaping_magnitude"]),
            "--dense-shaping-epsilon", str(protocol["dense_shaping_epsilon"]),
        ]
        plan.append({
            "run_id": run_id, "seed": seed, "use_wsc": use_wsc,
            "source_checkpoint": str(src_ckpt), "source_checkpoint_exists": src_ckpt.exists(),
            "checkpoint_root": str(ckpt_root), "cmd": cmd,
        })
    return plan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--priority", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--max-workers", type=int, default=None,
                    help="Default: cpu_count() - 2 (matches launch_formal.py's existing formula).")
    ap.add_argument("--device", type=str, default="cpu",
                    help="Hardware execution detail only (cpu or cuda). Does not change scientific parameters.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--live-smoke-test", action="store_true",
                    help="Real (non-dry-run) launch of this exact code path -- subprocess spawn, log/manifest "
                         "writing, Windows multiprocessing -- but with only --smoke-seed-count seeds and "
                         "--smoke-steps additional steps, writing to a clearly separate '..._livesmoke' "
                         "checkpoint root that the real formal launch never reads. Use this once on a new "
                         "machine before committing to the real 800,000-step/12-seed launch.")
    ap.add_argument("--smoke-steps", type=int, default=20)
    ap.add_argument("--smoke-seed-count", type=int, default=2)
    ap.add_argument("--seeds", type=str, default=None,
                    help="Optional comma-separated subset of the formal 12 seeds. Default: all 12.")
    ap.add_argument("--continue-from-latest", action="store_true",
                    help="Resume each seed from the latest ckpt_step_*.pt in its dense run directory "
                         "instead of restarting from the C64 1.2M init checkpoint.")
    args = ap.parse_args()

    if args.live_smoke_test and args.dry_run:
        raise SystemExit("--live-smoke-test and --dry-run are mutually exclusive")

    seed_tuple = None
    if args.seeds:
        seed_tuple = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    plan = build_plan(
        args.priority, live_smoke_test=args.live_smoke_test,
        smoke_steps=args.smoke_steps, smoke_seed_count=args.smoke_seed_count,
        device=args.device, seeds=seed_tuple,
        continue_from_latest=args.continue_from_latest,
    )
    _LABELS = {
        1: "Maximin + WSC + Dense", 2: "Maximin + Dense-only",
        3: "GGI + WSC + Dense", 4: "GGI + Dense-only",
    }
    label = _LABELS[args.priority]

    if args.dry_run:
        print(f"[launch_dense_priority] Priority {args.priority}: {label}")
        print(f"[launch_dense_priority] {len(plan)} runs, seeds: {[r['seed'] for r in plan]}")
        bad = [r for r in plan if not r["source_checkpoint_exists"]]
        if bad:
            print(f"[launch_dense_priority] MISSING SOURCE CHECKPOINTS for seeds: {[r['seed'] for r in bad]}")
        for r in plan:
            print(f"  would launch: {r['run_id']} (seed={r['seed']}) "
                  f"source_checkpoint_exists={r['source_checkpoint_exists']}")
            print(f"    {' '.join(r['cmd'])}")
        return 1 if bad else 0

    import multiprocessing
    max_workers = args.max_workers if args.max_workers is not None else max(1, multiprocessing.cpu_count() - 2)
    mode = f"LIVE SMOKE TEST ({args.smoke_seed_count} seeds x {args.smoke_steps} steps, separate checkpoint root)" \
        if args.live_smoke_test else "LIVE FORMAL LAUNCH (real 800,000-step budget, all 12 seeds)"
    print(f"[launch_dense_priority] {mode}: Priority {args.priority} ({label}), "
          f"{len(plan)} runs, max_workers={max_workers}, device={args.device}. This script does not itself gate this further -- "
          f"the coordinator/user must confirm this is an intentional formal launch before invoking without --dry-run.")

    running: list[tuple[str, object]] = []
    queue = list(plan)
    while queue or running:
        while queue and len(running) < max_workers:
            r = queue.pop(0)
            Path(r["checkpoint_root"]).mkdir(parents=True, exist_ok=True)
            log_path = LOGS / f"{r['run_id']}.log"
            write_run_manifest(r["run_id"], seed=r["seed"], condition="maximin", started=True,
                                start_unix=time.time(), pid=None, completed=False, technical_failure=False,
                                init_checkpoint=r["source_checkpoint"], log_path=str(log_path))
            proc = run_subprocess(r["cmd"], log_file=log_path, env_overrides={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
            write_run_manifest(r["run_id"], pid=proc.pid)
            print(f"[launch_dense_priority] started {r['run_id']} pid={proc.pid}")
            running.append((r["run_id"], proc))
        time.sleep(5)
        still_running = []
        for run_id, proc in running:
            ret = proc.poll()
            if ret is None:
                still_running.append((run_id, proc))
            else:
                write_run_manifest(run_id, completed=(ret == 0), technical_failure=(ret != 0),
                                    end_unix=time.time(), returncode=ret)
                print(f"[launch_dense_priority] {run_id} finished, returncode={ret}")
        running = still_running

    print("[launch_dense_priority] all launched runs have exited.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
