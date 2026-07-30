"""Atomic checkpoint I/O for Stage 5B-0 pilot."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _to_cpu_state(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu_state(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_state(v) for v in obj]
    return obj


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    """Write temporary file, fsync, rename to final path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    clean = _to_cpu_state(payload)
    torch.save(clean, tmp)
    # fsync where supported
    with open(tmp, "rb") as f:
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def capture_global_rng_states() -> dict[str, Any]:
    return {
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_global_rng_states(payload: dict[str, Any]) -> None:
    torch.set_rng_state(payload["torch_cpu"])
    np.random.set_state(payload["numpy"])
    random.setstate(payload["python"])


def learner_fingerprint(learner) -> dict[str, str]:
    import hashlib

    online = learner.parameter_vector(network="online").tobytes()
    target = learner.parameter_vector(network="target").tobytes()
    return {
        "online_sha256": hashlib.sha256(online).hexdigest(),
        "target_sha256": hashlib.sha256(target).hexdigest(),
        "update_count": str(int(learner._update_count)),
        "replay_size": str(len(learner.replay)),
    }


__all__ = [
    "atomic_torch_save",
    "capture_global_rng_states",
    "learner_fingerprint",
    "load_checkpoint",
    "restore_global_rng_states",
]
