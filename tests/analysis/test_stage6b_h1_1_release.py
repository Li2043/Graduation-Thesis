"""Stage 6B-H1.1 release metadata tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from thesis.analysis.h1_manifest import (
    is_absolute_path_string,
    sha256_file,
    verify_manifest_hashes,
)

H1 = Path(__file__).resolve().parents[2] / "experiments/formal/stage6b_h1"
OUT = H1 / "output"


@pytest.mark.skipif(not (OUT / "diagnostics/nonutility_mismatches.csv").is_file(), reason="H1 output missing")
def test_nonutility_mismatch_csv_has_header_and_zero_rows() -> None:
    path = OUT / "diagnostics/nonutility_mismatches.csv"
    df = pd.read_csv(path)
    assert len(df) == 0
    assert list(df.columns) == [
        "condition",
        "master_seed",
        "block_id",
        "assignment",
        "field",
        "old",
        "new",
    ]
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 1


@pytest.mark.skipif(not (OUT / "manifests/analysis_manifest.json").is_file(), reason="manifest missing")
def test_nonutility_mismatch_csv_manifest_hash_matches() -> None:
    man = json.loads((OUT / "manifests/analysis_manifest.json").read_text(encoding="utf-8"))
    rel = "output/diagnostics/nonutility_mismatches.csv"
    assert rel in man["output_hashes"]
    assert man["output_hashes"][rel] == sha256_file(OUT / "diagnostics/nonutility_mismatches.csv")


@pytest.mark.skipif(not (OUT / "manifests/analysis_manifest.json").is_file(), reason="manifest missing")
def test_manifest_contains_paper_integrity_before_and_after() -> None:
    man = json.loads((OUT / "manifests/analysis_manifest.json").read_text(encoding="utf-8"))
    hashes = man["output_hashes"]
    assert "output/diagnostics/paper_file_integrity_before.csv" in hashes
    assert "output/diagnostics/paper_file_integrity_after.csv" in hashes


@pytest.mark.skipif(not (OUT / "manifests/analysis_manifest.json").is_file(), reason="manifest missing")
def test_manifest_paths_are_relative() -> None:
    man = json.loads((OUT / "manifests/analysis_manifest.json").read_text(encoding="utf-8"))
    for rel in man["output_hashes"]:
        assert not is_absolute_path_string(rel)
        assert ".." not in Path(rel).parts
    for p in man.get("figure_paths", []):
        assert not is_absolute_path_string(p)


@pytest.mark.skipif(not (OUT / "manifests/analysis_manifest.json").is_file(), reason="manifest missing")
def test_manifest_paths_do_not_escape_root() -> None:
    man = json.loads((OUT / "manifests/analysis_manifest.json").read_text(encoding="utf-8"))
    for rel in man["output_hashes"]:
        resolved = (H1 / rel).resolve()
        assert str(resolved).startswith(str(H1.resolve()))


@pytest.mark.skipif(not (OUT / "manifests/analysis_manifest.json").is_file(), reason="manifest missing")
def test_manifest_contains_no_windows_absolute_paths() -> None:
    text = (OUT / "manifests/analysis_manifest.json").read_text(encoding="utf-8")
    assert "C:\\\\Users" not in text
    assert "C:\\Users" not in text


@pytest.mark.skipif(not (OUT / "manifests/acceptance_checks.json").is_file(), reason="acceptance missing")
def test_reference_tolerance_is_not_looser_than_one_e_minus_six() -> None:
    acc = json.loads((OUT / "manifests/acceptance_checks.json").read_text(encoding="utf-8"))
    assert float(acc["reference_tolerance"]) <= 1e-6


@pytest.mark.skipif(not (OUT / "manifests/analysis_manifest.json").is_file(), reason="manifest missing")
def test_manifest_hashes_verify() -> None:
    verify_manifest_hashes(
        artifact_root=H1,
        manifest_path=OUT / "manifests/analysis_manifest.json",
    )


@pytest.mark.skipif(not (OUT / "manifests/analysis_manifest.json").is_file(), reason="manifest missing")
def test_execution_and_release_commits_are_recorded() -> None:
    man = json.loads((OUT / "manifests/analysis_manifest.json").read_text(encoding="utf-8"))
    assert man.get("execution_commit")
    assert man.get("release_commit")


@pytest.mark.skipif(not (OUT / "manifests/analysis_manifest.json").is_file(), reason="manifest missing")
def test_paper_files_unchanged() -> None:
    man = json.loads((OUT / "manifests/analysis_manifest.json").read_text(encoding="utf-8"))
    assert man["paper_integrity"]["changed_file_count"] == 0
    assert man["paper_integrity"]["verified_unchanged"] is True
