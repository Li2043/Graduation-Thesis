#!/usr/bin/env python3
"""Package Stage 6B-H1 tree into a reproducible release ZIP (includes gitignored output/)."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from thesis.analysis.h1_manifest import sha256_file, verify_manifest_hashes

SCRIPT = Path(__file__).resolve()
H1_ROOT = SCRIPT.parents[1]
REPO_ROOT = SCRIPT.parents[4]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h1-root", type=Path, default=H1_ROOT)
    parser.add_argument("--releases-dir", type=Path, default=REPO_ROOT / "releases")
    parser.add_argument("--commit", default="")
    args = parser.parse_args()

    h1 = Path(args.h1_root).resolve()
    releases = Path(args.releases_dir).resolve()
    releases.mkdir(parents=True, exist_ok=True)
    short = (args.commit or "local")[:8]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = releases / f"stage6b_h1_1_release_{short}_{stamp}.zip"

    exclude_names = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
    exclude_suffix = {".pyc", ".tmp"}

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(h1.rglob("*")):
            if not path.is_file():
                continue
            if any(part in exclude_names for part in path.parts):
                continue
            if path.suffix in exclude_suffix:
                continue
            # skip nested large accidental zips inside h1
            if path.suffix == ".zip" and path.parent == h1:
                continue
            arcname = Path("stage6b_h1") / path.relative_to(h1)
            zf.write(path, arcname.as_posix())

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(tmp_path)
        extracted_h1 = tmp_path / "stage6b_h1"
        man = extracted_h1 / "output" / "manifests" / "analysis_manifest.json"
        missing = []
        required = [
            "output/manifests/analysis_manifest.json",
            "output/data/evaluation_episodes_h1.csv",
            "output/statistics/primary_endpoint_contrasts_h1.csv",
            "output/diagnostics/paper_file_integrity_after.csv",
            "analysis_requirements_h1.txt",
            "environment_snapshot.json",
            "pip_freeze.txt",
            "logs/stage6b_h1_runner.log",
        ]
        for rel in required:
            if not (extracted_h1 / rel).is_file():
                missing.append(rel)
        errors: list[str] = []
        valid = False
        if man.is_file() and not missing:
            try:
                verify_manifest_hashes(artifact_root=extracted_h1, manifest_path=man)
                valid = True
            except Exception as exc:
                errors.append(str(exc))
        else:
            errors.append("missing required files or manifest")

        validation = {
            "archive_path": str(archive.relative_to(REPO_ROOT)).replace("\\", "/"),
            "archive_sha256": sha256_file(archive),
            "archive_size_bytes": int(archive.stat().st_size),
            "extracted_manifest_valid": valid,
            "missing_required_files": missing,
            "unexpected_validation_errors": errors,
        }
        out_val = h1 / "output" / "manifests" / "release_archive_validation.json"
        # Also write under releases for discoverability
        (releases / "release_archive_validation.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(validation, indent=2))
        return 0 if valid and not missing and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
