#!/usr/bin/env python3
"""Held-out evaluation (H0 homogeneous, H1 heterogeneous, RUNBOOK sec
50-52) of each completed formal run's final checkpoint-Q-ensemble,
using the SAME authoritative methodology as every curriculum/
qualification gate this project has used (stage_q_ensemble_gate.py,
epsilon=0, K(S)={S-150K,S-100K,S-50K,S}, --local-sensing-range-m 50.0).

Does not run formal training. Only evaluates runs status.py reports as
'completed'. Writes outputs/held_out_eval/{run_id}/."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    BUNDLE_ROOT, OUTPUTS, PROJECT_ROOT, RUN_STATE_DIR, SB_SCRIPTS, load_frozen_config,
    python_exe, read_run_manifest,
)
from status import classify_run  # noqa: E402


def _scenario_ids(bank_path: Path) -> list[str]:
    data = json.loads(bank_path.read_text(encoding="utf-8"))
    return [s["scenario_id"] for s in data] if isinstance(data, list) else list(data.keys())


def evaluate_one(run_id: str, seed: int, final_step: int, bank_name: str) -> None:
    cfg = load_frozen_config()
    bank_path = BUNDLE_ROOT / "scenario_banks" / f"{bank_name}.json"
    condition = run_id.split("_", 1)[0]
    # stage_q_ensemble_gate expects the dir that directly contains ckpt_step_*.pt
    # and whose name includes seed_{seed}_ (same layout as training output).
    ckpt_root = (BUNDLE_ROOT / "checkpoints" / "formal_runs" / run_id
                 / f"seed_{seed}_Formal_{condition}")
    out_dir = OUTPUTS / "held_out_eval" / run_id / bank_name
    cmd = [
        python_exe(), str(SB_SCRIPTS / "stage_q_ensemble_gate.py"),
        "--stage-label", f"{run_id}_{bank_name}",
        "--stage-end-step", str(final_step),
        "--checkpoint-stage-name", f"Formal_{condition}",
        "--scenario-bank", str(bank_path),
        "--scenario-ids", *_scenario_ids(bank_path),
        "--seed-checkpoint-roots", f"{seed}:{ckpt_root}",
        "--local-sensing-range-m", str(cfg["observation"]["local_sensing_range_m"]),
        "--output-dir", str(out_dir),
    ]
    print(f"[evaluate_formal] {run_id} on {bank_name}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    print(proc.stdout[-1500:])
    if proc.returncode != 0:
        print(f"[evaluate_formal] WARNING: {run_id}/{bank_name} eval exited {proc.returncode}\n{proc.stderr[-1500:]}")


def main() -> int:
    if not RUN_STATE_DIR.exists():
        print("[evaluate_formal] no run state found -- nothing to evaluate yet.")
        return 0
    completed = []
    for f in sorted(RUN_STATE_DIR.glob("*.json")):
        if f.stem.startswith("curriculum_"):
            continue
        manifest = read_run_manifest(f.stem)
        if manifest and classify_run(manifest) == "completed":
            completed.append((f.stem, manifest))

    if not completed:
        print("[evaluate_formal] no completed formal runs yet. Nothing to do.")
        return 0

    for run_id, manifest in completed:
        seed = manifest["seed"]
        final_step = manifest.get("current_step") or manifest.get("budget_end_step") or 2000000
        for bank in ("H0", "H1"):
            evaluate_one(run_id, seed, final_step, bank)

    print(f"[evaluate_formal] evaluated {len(completed)} completed run(s) on H0 and H1. "
          f"See outputs/held_out_eval/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
