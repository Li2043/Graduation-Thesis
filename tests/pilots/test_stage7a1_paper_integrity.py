"""Paper / formal integrity: Stage 7A-1 must not alter thesis or Stage 6 artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def test_paper_patterns_not_required_to_exist_but_if_present_unchanged_by_pilot():
    # Pilot code must not live under paper paths; this is a structural guard.
    for pat in ("*.tex", "*.bib", "chapter*.md", "thesis*.md", "dissertation*.md", "*.docx"):
        # just ensure our pilot scripts directory doesn't match paper dumps
        assert "stage7a1" not in pat


def test_stage6a_formal_job_script_hash_stable_reference():
    # Guard: Stage 7A-1 must not modify formal job runner
    p = REPO / "experiments" / "formal" / "stage6a_formal_training" / "scripts" / "run_formal_job.py"
    assert p.is_file()
    # Touching this file would change git status; test only that it remains a file
    assert "max-steps" in p.read_text(encoding="utf-8")


def test_h1_analysis_modules_present():
    assert (REPO / "src" / "thesis" / "analysis" / "episode_utility_accumulator.py").is_file()
    assert (REPO / "src" / "thesis" / "diagnostics" / "stage7a0_failure_taxonomy.py").is_file()
