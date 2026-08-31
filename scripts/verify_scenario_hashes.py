#!/usr/bin/env python3
"""Hash-verify Q/M/H0/H1/OOD_speed/C1/C4/C16 scenario banks. These are
FROZEN -- if any hash mismatches, STOP and do not regenerate; a
mismatch means the bundle was corrupted in transit, not that the bank
needs recreating."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CHECKSUMS_PATH, SCENARIO_BANKS, VERIFICATION, sha256_file, write_json_atomic  # noqa: E402


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
    checksums = load_checksums()
    results = []
    n_mismatch = 0
    for bank in sorted(SCENARIO_BANKS.glob("*.json")):
        rel = f"scenario_banks/{bank.name}"
        actual = sha256_file(bank)
        expected = checksums.get(rel)
        status = "NO_RECORDED_HASH" if expected is None else ("MATCH" if expected == actual else "MISMATCH")
        if status == "MISMATCH":
            n_mismatch += 1
        results.append({"file": bank.name, "sha256": actual, "expected": expected, "status": status})
        print(f"{bank.name}: {status}")

    write_json_atomic(VERIFICATION / "scenario_hash_report.json", {"results": results, "mismatches": n_mismatch})
    if n_mismatch:
        print(f"[FAIL] {n_mismatch} scenario bank(s) do not match the frozen hash. "
              "Do NOT regenerate -- re-copy the bundle from the original USB/source instead.")
        return 1
    print("[OK] All scenario banks hash-verified against the frozen record.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
