#!/usr/bin/env python3
"""Hash-verify every checkpoint under checkpoints/ against
CHECKSUMS.sha256, and confirm each one torch.load()s cleanly with the
expected step/stage fields. Run this after copying the bundle to a new
machine/drive, before trusting any checkpoint as a training starting
point."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import BUNDLE_ROOT, CHECKPOINTS_ROOT, CHECKSUMS_PATH, VERIFICATION, sha256_file, write_json_atomic  # noqa: E402


def load_checksums() -> dict[str, str]:
    out = {}
    if not CHECKSUMS_PATH.exists():
        return out
    for line in CHECKSUMS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        h, _, relpath = line.partition("  ")
        out[relpath.replace("\\", "/")] = h
    return out


def main() -> int:
    import torch
    checksums = load_checksums()
    results = []
    n_ok, n_hash_mismatch, n_missing_hash, n_load_error = 0, 0, 0, 0

    for ckpt in sorted(CHECKPOINTS_ROOT.rglob("*.pt")):
        rel = str(ckpt.relative_to(BUNDLE_ROOT)).replace("\\", "/")
        entry = {"path": rel}
        actual_hash = sha256_file(ckpt)
        entry["sha256"] = actual_hash
        expected_hash = checksums.get(rel)
        if expected_hash is None:
            entry["hash_status"] = "NO_RECORDED_HASH"
            n_missing_hash += 1
        elif expected_hash == actual_hash:
            entry["hash_status"] = "MATCH"
        else:
            entry["hash_status"] = "MISMATCH"
            n_hash_mismatch += 1

        try:
            ck = torch.load(ckpt, map_location="cpu")
            entry["load_ok"] = True
            entry["step"] = ck.get("step")
            entry["stage"] = ck.get("stage")
            entry["has_online"] = "online" in ck
            entry["has_optimiser"] = "optimiser" in ck
        except Exception as e:  # noqa: BLE001
            entry["load_ok"] = False
            entry["load_error"] = repr(e)
            n_load_error += 1

        if entry.get("hash_status") == "MATCH" and entry.get("load_ok"):
            n_ok += 1
        results.append(entry)

    summary = {
        "total_checkpoints": len(results), "ok": n_ok,
        "hash_mismatch": n_hash_mismatch, "no_recorded_hash": n_missing_hash,
        "load_error": n_load_error,
        "overall_ok": n_hash_mismatch == 0 and n_load_error == 0,
        "checkpoints": results,
    }
    write_json_atomic(VERIFICATION / "checkpoint_verification_report.json", summary)
    print(f"checkpoints: {len(results)} total, {n_ok} OK, {n_hash_mismatch} hash mismatch, "
          f"{n_missing_hash} no recorded hash, {n_load_error} load errors")
    if n_hash_mismatch or n_load_error:
        print("[FAIL] Do not trust any MISMATCH/load_error checkpoint as a training starting point.")
        return 1
    print("[OK] All checkpoints hash-verified and load cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
