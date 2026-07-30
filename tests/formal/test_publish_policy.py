"""Publish allowlist, large-file rejection, local-only replay checkpoints."""

from pathlib import Path

from thesis.formal.publish import (
    MAX_ORDINARY_GIT_FILE_MIB,
    build_publish_manifest,
    is_local_only_checkpoint,
    is_publish_allowed,
    reject_oversized_git_files,
)


def test_publish_policy(tmp_path):
    assert MAX_ORDINARY_GIT_FILE_MIB == 90
    ckpt = tmp_path / "checkpoints" / "job" / "ckpt_step_000100.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"x")
    assert is_local_only_checkpoint(ckpt) is True
    assert is_publish_allowed(ckpt) is False

    report = tmp_path / "report.md"
    report.write_text("ok", encoding="utf-8")
    assert is_publish_allowed(report) is True

    big = tmp_path / "big.bin"
    big.write_bytes(b"0" * (91 * 1024 * 1024))
    violations = reject_oversized_git_files([big])
    assert violations

    small = tmp_path / "small.csv"
    small.write_text("a,b\n1,2\n", encoding="utf-8")
    manifest = build_publish_manifest(
        [small],
        root=tmp_path,
        source_job="baseline__61001",
        protocol_hash="abc",
        runner_commit="deadbeef",
    )
    assert manifest["n_files"] == 1
    assert manifest["files"][0]["sha256"]
    assert manifest["protocol_hash"] == "abc"
