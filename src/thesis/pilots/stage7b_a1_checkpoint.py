"""Stage 7B-A1 checkpoint write/load with algorithm-mode guards."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from thesis.training.pilot_checkpoint import _to_cpu_state, load_checkpoint


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_resumable_payload(payload: dict[str, Any]) -> None:
    for key in (
        "algorithm_mode",
        "learners",
        "ic_schedule",
        "global_rng",
        "env_steps",
        "protocol_hash",
    ):
        if key not in payload:
            raise ValueError(f"checkpoint missing required field {key}")
    for aid in ("A", "B"):
        la = payload["learners"].get(aid, {})
        for k in ("online", "target", "optimiser", "replay", "learner_rng"):
            if k not in la:
                raise ValueError(f"checkpoint missing learners[{aid}].{k}")


def atomic_hashed_torch_save(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_resumable_payload(payload)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    torch.save(_to_cpu_state(payload), tmp)
    with open(tmp, "rb") as f:
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    load_checkpoint(tmp)
    digest = sha256_file(tmp)
    if path.exists():
        existing = sha256_file(path)
        if existing == digest:
            tmp.unlink(missing_ok=True)
            st = path.stat()
            return {
                "path": str(path.as_posix()),
                "sha256": existing,
                "size_bytes": int(st.st_size),
                "reused": True,
                "load_test_passed": True,
            }
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"refusing overwrite of different checkpoint: {path}")
    os.replace(tmp, path)
    st = path.stat()
    return {
        "path": str(path.as_posix()),
        "sha256": digest,
        "size_bytes": int(st.st_size),
        "reused": False,
        "load_test_passed": True,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with open(tmp, "rb") as f:
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def validate_resume_compatibility(
    payload: dict[str, Any],
    *,
    algorithm_condition: str,
    algorithm_mode: str,
    protocol_hash: str,
    reward_condition: str = "baseline",
) -> None:
    assert_resumable_payload(payload)
    if str(payload.get("condition")) != reward_condition:
        raise ValueError(
            f"reward condition mismatch: ckpt={payload.get('condition')!r} "
            f"expected={reward_condition!r}"
        )
    mode = payload.get("algorithm_mode") or payload.get("target_mode")
    if str(mode) != algorithm_mode:
        raise ValueError(
            f"cross-condition resume rejected: ckpt_mode={mode!r} "
            f"requested={algorithm_mode!r}"
        )
    algo = payload.get("algorithm_condition")
    if algo is not None and str(algo) != algorithm_condition:
        raise ValueError(
            f"cross-condition resume rejected: ckpt_condition={algo!r} "
            f"requested={algorithm_condition!r}"
        )
    if protocol_hash and str(payload.get("protocol_hash", "")) not in {"", protocol_hash}:
        raise ValueError("protocol hash mismatch on resume")


__all__ = [
    "assert_resumable_payload",
    "atomic_hashed_torch_save",
    "sha256_file",
    "validate_resume_compatibility",
    "write_json_atomic",
]
