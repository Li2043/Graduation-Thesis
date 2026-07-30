#!/usr/bin/env python3
"""Seed collision audit + storage preflight for Stage 7A-1 (pre-training)."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[4]
EXP = Path(__file__).resolve().parents[1]
PROTOCOL = EXP / "configs" / "stage7a1_baseline_budget_protocol.yaml"
PILOT_SEEDS = list(range(62001, 62021))
FORBIDDEN = list(range(61001, 61011))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_seeds() -> dict:
    hits: list[dict] = []
    # Scan for master_seed JSON/YAML fields equal to pilot seeds in final_new
    patterns = [
        REPO / "experiments",
        REPO / "src",
        Path(r"C:\Users\HP\Desktop\毕业项目\thesis\final_new_results_100k\formal_results"),
        Path(r"C:\Users\HP\Desktop\毕业项目\thesis\final_new_analysis_100k\experiments"),
    ]
    needles = {str(s) for s in PILOT_SEEDS}
    for root in patterns:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".json", ".yaml", ".yml", ".csv", ".md", ".py"}:
                continue
            # skip this pilot's own outputs/protocol once created
            if "stage7a1_baseline_budget" in path.as_posix() and path.name != "stage7a1_baseline_budget_protocol.yaml":
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            # Look for master_seed assignments / lists — not bare integers in unrelated columns
            for s in PILOT_SEEDS:
                markers = (
                    f"master_seed\": {s}",
                    f"master_seed': {s}",
                    f"master_seed: {s}",
                    f",{s}," if path.suffix.lower() == ".csv" and "master_seed" in text[:200].lower() else None,
                    f"range(62001",
                    f"62001–62020",
                    f"62001-62020",
                )
                for m in markers:
                    if m and m in text:
                        # ignore chapter prose mentioning numbers accidentally unless master_seed nearby
                        if m.startswith("range") or m.startswith("62001"):
                            if "master_seed" in text or "master_seeds" in text or "seeds" in path.name:
                                hits.append({"seed": s, "path": path.as_posix(), "marker": m})
                        else:
                            hits.append({"seed": s, "path": str(path.relative_to(REPO)).replace("\\", "/") if path.is_relative_to(REPO) else path.as_posix(), "marker": m})
                        break
    # Formal seeds must not appear as pilot seeds
    collision = sorted({h["seed"] for h in hits})
    # Deduplicate hits
    uniq = []
    seen = set()
    for h in hits:
        key = (h["seed"], h["path"], h["marker"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(h)
    return {
        "pilot_seeds": PILOT_SEEDS,
        "forbidden_formal_seeds": FORBIDDEN,
        "collision_seeds": collision,
        "hits": uniq,
        "status": "ABORT" if collision else "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def storage_preflight(output_root: Path) -> dict:
    # Empirical Stage 6A full ckpt sizes ~19–46 MB; use 45 MB upper estimate
    per_ckpt_mb = 45.0
    n_seeds = 20
    n_ckpts = 10
    full_ckpts_gb = per_ckpt_mb * n_seeds * n_ckpts / 1024.0
    weights_gb = 0.12 * n_seeds * n_ckpts / 1024.0
    evals_gb = 2.0
    traj_gb = 8.0
    logs_gb = 1.0
    temp_gb = full_ckpts_gb * 0.15
    total_gb = full_ckpts_gb + weights_gb + evals_gb + traj_gb + logs_gb + temp_gb
    with_margin_gb = total_gb * 1.20
    usage = shutil.disk_usage(str(output_root if output_root.exists() else output_root.parent))
    free_gb = usage.free / (1024**3)
    abort = with_margin_gb > 0.80 * free_gb
    return {
        "per_full_checkpoint_mb_estimate": per_ckpt_mb,
        "n_seeds": n_seeds,
        "n_checkpoints": n_ckpts,
        "estimated_full_checkpoints_gb": round(full_ckpts_gb, 3),
        "estimated_total_with_20pct_margin_gb": round(with_margin_gb, 3),
        "free_disk_gb": round(free_gb, 3),
        "output_root": str(output_root).replace("\\", "/"),
        "status": "ABORT" if abort else "PASS",
        "abort_reason": "estimated demand exceeds 80% of free disk" if abort else None,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    EXP.mkdir(parents=True, exist_ok=True)
    (EXP / "manifests").mkdir(parents=True, exist_ok=True)
    (EXP / "output" / "manifests").mkdir(parents=True, exist_ok=True)
    proto = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    assert proto["condition"] == "baseline"
    assert proto["master_seeds"] == PILOT_SEEDS
    assert proto["maximum_training_steps"] == 300000
    audit = audit_seeds()
    (EXP / "manifests" / "seed_collision_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    out_root = EXP / "output"
    storage = storage_preflight(out_root)
    (EXP / "manifests" / "storage_preflight.json").write_text(
        json.dumps(storage, indent=2) + "\n", encoding="utf-8"
    )
    (EXP / "configs" / "protocol_hash.txt").write_text(
        sha256_file(PROTOCOL) + "\n", encoding="utf-8"
    )
    print(json.dumps({"seed_audit": audit["status"], "storage": storage["status"], "collisions": audit["collision_seeds"]}, indent=2))
    if audit["status"] == "ABORT":
        print("BLOCKED — preregistered seed collision")
        return 2
    if storage["status"] == "ABORT":
        print("ABORT — insufficient disk")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
