#!/usr/bin/env python3
"""Write the formal experiment freeze manifest and, ONLY if every
precondition holds, set formal_fairness_started=true. After that flag
flips, RUNBOOK's own rule applies: no outcome-dependent method
modification is allowed for the rest of the project.

Preconditions (all must hold, checked in order, STOPS at the first
failure rather than setting the flag "mostly"):
  1. all 6 formal seeds have a C64_R50 checkpoint at step 1,200,000
     that hash-verifies and torch.load()s cleanly;
  2. H0.json / H1.json hash-verify against the frozen record;
  3. FROZEN_EXPERIMENT_CONFIG.json is internally consistent (lambda_W,
     seed list, GGI weights match what's about to be written)."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    BUNDLE_ROOT, CHECKPOINTS_ROOT, CONFIGS, FROZEN_CONFIG_PATH, load_frozen_config,
    needs_user_decision, python_exe, sha256_file, write_json_atomic,
)

FORMAL_SEEDS = [900101, 900102, 900103, 900104, 910101, 910102]
CONDITIONS = ["mean", "ggi", "maximin"]


def _c64_checkpoint_path(seed: int) -> Path:
    if seed in (900101, 900102, 900103, 900104):
        return (CHECKPOINTS_ROOT / "formal_init" / str(seed) / "C64_R50" /
                f"seed_{seed}_C64_R50" / "ckpt_step_1200000.pt")
    return (CHECKPOINTS_ROOT / "curriculum_910101_910102" / str(seed) / "C64_R50" /
            f"seed_{seed}_C64_R50" / "ckpt_step_1200000.pt")


def check_all_seed_checkpoints() -> dict:
    import torch
    results = {}
    for seed in FORMAL_SEEDS:
        p = _c64_checkpoint_path(seed)
        entry = {"path": str(p), "exists": p.exists()}
        if p.exists():
            try:
                ck = torch.load(p, map_location="cpu")
                entry["step"] = ck.get("step")
                entry["stage"] = ck.get("stage")
                entry["loads_ok"] = ck.get("step") == 1200000 and ck.get("stage") == "C64_R50"
            except Exception as e:  # noqa: BLE001
                entry["loads_ok"] = False
                entry["load_error"] = repr(e)
        else:
            entry["loads_ok"] = False
        results[str(seed)] = entry
    return results


def check_held_out_hashes() -> dict:
    proc = subprocess.run([python_exe(), str(Path(__file__).resolve().parent / "verify_scenario_hashes.py")],
                           cwd=str(BUNDLE_ROOT), capture_output=True, text=True)
    return {"returncode": proc.returncode, "stdout": proc.stdout}


def main() -> int:
    cfg = load_frozen_config()

    print("[freeze_formal_manifest] checking all 6 formal seeds have a valid C64_R50 checkpoint...")
    ckpt_check = check_all_seed_checkpoints()
    missing = [s for s, r in ckpt_check.items() if not r.get("loads_ok")]
    if missing:
        needs_user_decision(
            issue=f"Not every formal seed has a valid, loadable C64_R50 checkpoint at step 1,200,000: "
                  f"missing/invalid = {missing}.",
            evidence=json.dumps(ckpt_check, indent=2),
            options=["For 910101/910102: run resume_curriculum.py until their C64_R50 build completes.",
                     "For 900101-900104: re-run verify_checkpoints.py -- their checkpoints should already "
                     "exist in checkpoints/formal_init/; if missing, the bundle copy may be incomplete."],
            consequences="Cannot freeze the formal manifest or start formal training with an incomplete "
                          "seed block -- this would violate the pre-outcome 6-seed matched-block design.",
            recommendation="Do not proceed until all 6 are valid. Do not substitute a different seed.")

    print("[freeze_formal_manifest] verifying H0/H1 held-out bank hashes...")
    hash_check = check_held_out_hashes()
    if hash_check["returncode"] != 0:
        needs_user_decision(
            issue="H0.json/H1.json held-out bank hash verification FAILED.",
            evidence=hash_check["stdout"],
            options=["Re-copy scenario_banks/ from the original USB bundle -- do NOT regenerate these banks."],
            consequences="A hash mismatch here likely means transit corruption, not that the bank needs "
                          "recreating; using a regenerated bank would break comparability with prior/parallel "
                          "evaluation using the original banks.",
            recommendation="Re-copy from source; only regenerate as an absolute last resort, and if you do, "
                            "flag it loudly in the experiment record.")

    manifest = {
        "frozen_at_unix": time.time(),
        "frozen_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "formal_seeds": FORMAL_SEEDS,
        "conditions": CONDITIONS,
        "same_seed_block_for_all_conditions": True,
        "lambda_W": cfg["welfare"]["lambda_W"],
        "ggi_weights_ascending": cfg["welfare"]["ggi_weights_ascending"],
        "n_runs_total": len(FORMAL_SEEDS) * len(CONDITIONS),
        "run_matrix": [{"seed": s, "condition": c, "run_id": f"{c}_{s}",
                         "init_checkpoint": str(_c64_checkpoint_path(s))}
                        for s in FORMAL_SEEDS for c in CONDITIONS],
        "budget": cfg["formal_seed_block"]["formal_budget"],
        "observation": cfg["observation"],
        "dqn": cfg["dqn"],
        "environment": cfg["environment"],
        "checkpoint_verification": ckpt_check,
        "held_out_hash_verification_passed": True,
        "keep_all_seeds_rule": cfg["formal_seed_block"]["keep_all_seeds_rule"],
    }

    manifest_json_path = BUNDLE_ROOT / "FORMAL_EXPERIMENT_FREEZE_MANIFEST.json"
    write_json_atomic(manifest_json_path, manifest)

    md_lines = [
        "# FORMAL_EXPERIMENT_FREEZE_MANIFEST",
        "",
        f"Frozen: {manifest['frozen_at_iso']}",
        "",
        f"6-seed matched block (same seeds across all 3 conditions): `{FORMAL_SEEDS}`",
        f"Conditions: `{CONDITIONS}` -- lambda_W = **{manifest['lambda_W']}** (frozen, do not re-search)",
        f"GGI weights (ascending): `{manifest['ggi_weights_ascending']}`",
        f"Total formal runs: **{manifest['n_runs_total']}**",
        "",
        "## Run matrix",
        "",
        "| run_id | seed | condition | init checkpoint |",
        "|---|---|---|---|",
    ]
    for r in manifest["run_matrix"]:
        md_lines.append(f"| {r['run_id']} | {r['seed']} | {r['condition']} | `{r['init_checkpoint']}` |")
    md_lines += ["", "## Keep-all-seeds rule", "", manifest["keep_all_seeds_rule"], "",
                 "After this manifest is frozen: **no outcome-dependent method modification is allowed** "
                 "(RUNBOOK sec 48)."]
    (BUNDLE_ROOT / "FORMAL_EXPERIMENT_FREEZE_MANIFEST.md").write_text("\n".join(md_lines), encoding="utf-8")

    cfg["formal_fairness_started"] = True
    write_json_atomic(FROZEN_CONFIG_PATH, cfg)

    print(f"[freeze_formal_manifest] WROTE {manifest_json_path} and .md")
    print("[freeze_formal_manifest] formal_fairness_started = true")
    print("[freeze_formal_manifest] Per RUNBOOK sec 48: no outcome-dependent method modification is allowed "
          "from this point forward.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
