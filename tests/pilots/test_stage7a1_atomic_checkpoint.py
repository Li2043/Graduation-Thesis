"""Atomic checkpoint overwrite protection."""

from __future__ import annotations

import pytest
import torch

from thesis.pilots.stage7a1_checkpoint import atomic_hashed_torch_save, sha256_file


def test_atomic_refuse_different_overwrite(tmp_path):
    path = tmp_path / "ckpt.pt"
    info1 = atomic_hashed_torch_save(path, {"v": 1, "t": torch.tensor([1.0])})
    assert path.is_file()
    # identical reuse ok
    info2 = atomic_hashed_torch_save(path, {"v": 1, "t": torch.tensor([1.0])})
    assert info2["reused"] is True
    assert info2["sha256"] == info1["sha256"]
    # different content must fail
    with pytest.raises(RuntimeError, match="refusing overwrite"):
        atomic_hashed_torch_save(path, {"v": 2, "t": torch.tensor([2.0])})
    assert sha256_file(path) == info1["sha256"]
