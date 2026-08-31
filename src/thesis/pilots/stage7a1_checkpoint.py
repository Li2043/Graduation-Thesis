"""Stage 7A-1 atomic checkpoint helpers with hash inventory."""

from __future__ import annotations

import hashlib
import json
import os
import time
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


def atomic_hashed_torch_save(
    path: Path,
    payload: dict[str, Any],
    *,
    allow_identical_reuse: bool = True,
) -> dict[str, Any]:
    """Write via temp+fsync+rename; refuse silent overwrite of different content."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    clean = _to_cpu_state(payload)
    torch.save(clean, tmp)
    with open(tmp, "rb") as f:
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    # load-test from temp
    probe = torch.load(tmp, map_location="cpu", weights_only=False)
    if not isinstance(probe, dict):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checkpoint load test failed: {path}")
    digest = sha256_file(tmp)
    size = tmp.stat().st_size

    if path.exists():
        existing = sha256_file(path)
        if existing == digest and allow_identical_reuse:
            tmp.unlink(missing_ok=True)
            st = path.stat()
            return {
                "path": str(path.as_posix()),
                "sha256": existing,
                "size_bytes": int(st.st_size),
                "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
                "load_test_passed": True,
                "reused": True,
            }
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"refusing overwrite of existing checkpoint with different hash: {path}"
        )

    os.replace(tmp, path)
    st = path.stat()
    # verify final
    load_checkpoint(path)
    return {
        "path": str(path.as_posix()),
        "sha256": digest,
        "size_bytes": int(size),
        "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
        "load_test_passed": True,
        "reused": False,
    }


def write_checkpoint_metadata(path: Path, meta: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    with open(tmp, "rb") as f:
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    if path.exists():
        # metadata may be rewritten with same scientific content; compare sha of body
        new_sha = sha256_file(tmp)
        old_sha = sha256_file(path)
        if new_sha != old_sha:
            # allow metadata refresh only if scientific keys match
            old = json.loads(path.read_text(encoding="utf-8"))
            keys = ("seed", "requested_step", "actual_step", "full_checkpoint_sha256")
            if any(old.get(k) != meta.get(k) for k in keys if k in old and k in meta):
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"refusing overwrite of different metadata: {path}")
    os.replace(tmp, path)


def weights_payload_from_learners(learners: dict[str, Any]) -> dict[str, Any]:
    return {
        "A_online": {k: v.detach().cpu() for k, v in learners["A"].online.state_dict().items()},
        "A_target": {k: v.detach().cpu() for k, v in learners["A"].target.state_dict().items()},
        "B_online": {k: v.detach().cpu() for k, v in learners["B"].online.state_dict().items()},
        "B_target": {k: v.detach().cpu() for k, v in learners["B"].target.state_dict().items()},
        "replay_seeds": {
            "A": int(learners["A"].replay.seed),
            "B": int(learners["B"].replay.seed),
        },
        "update_counts": {
            "A": int(learners["A"]._update_count),
            "B": int(learners["B"]._update_count),
        },
    }


def checkpoint_contains_flags(payload: dict[str, Any]) -> dict[str, bool]:
    la = payload.get("learners", {}).get("A", {})
    return {
        "contains_optimizer": "optimiser" in la or "optimizer" in la,
        "contains_replay": "replay" in la,
        "contains_rng": "learner_rng" in la and "global_rng" in payload,
        "contains_schedule_state": "ic_schedule" in payload,
        "resumable": all(
            [
                "optimiser" in la or "optimizer" in la,
                "replay" in la,
                "learner_rng" in la,
                "ic_schedule" in payload,
                "env_steps" in payload,
            ]
        ),
    }


__all__ = [
    "atomic_hashed_torch_save",
    "checkpoint_contains_flags",
    "sha256_file",
    "weights_payload_from_learners",
    "write_checkpoint_metadata",
]
