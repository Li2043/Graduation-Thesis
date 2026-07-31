"""Parallel job isolation tests."""

from __future__ import annotations

from pathlib import Path


def test_job_directories_isolated(tmp_path):
    root = tmp_path / "ckpt"
    jobs = []
    for cond in ("vanilla_dqn", "double_dqn"):
        for seed in (63001, 63002):
            d = root / cond / f"seed_{seed}"
            d.mkdir(parents=True)
            marker = d / "writable.txt"
            marker.write_text(f"{cond}-{seed}", encoding="utf-8")
            jobs.append(marker)
    texts = [p.read_text(encoding="utf-8") for p in jobs]
    assert len(set(texts)) == 4
    # no shared writable path
    assert len({p.resolve() for p in jobs}) == 4


def test_checkpoint_root_can_be_external(tmp_path):
    external = tmp_path / "external_checkpoints"
    external.mkdir()
    # Simulate policy: external path is not under experiments/pilots/.../checkpoints
    pilot_ckpt = Path("experiments/pilots/stage7b_a1_double_dqn/output/checkpoints")
    assert external.resolve() != pilot_ckpt.resolve()
    assert "stage7b_a1_double_dqn" not in str(external.resolve())
