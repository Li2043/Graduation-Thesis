"""Regression expectations for Stage 6B-H1 (post-run checks when outputs exist)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

OUT = Path(__file__).resolve().parents[2] / "experiments/formal/stage6b_h1/output"


@pytest.mark.skipif(not (OUT / "diagnostics/nonutility_mismatches.csv").is_file(), reason="H1 not run")
def test_nonutility_mismatches_empty() -> None:
    df = pd.read_csv(OUT / "diagnostics/nonutility_mismatches.csv")
    assert len(df) == 0


@pytest.mark.skipif(not (OUT / "data/evaluation_episodes_h1.csv").is_file(), reason="H1 not run")
def test_episode_count_480() -> None:
    df = pd.read_csv(OUT / "data/evaluation_episodes_h1.csv")
    assert len(df) == 480


@pytest.mark.skipif(not (OUT / "manifests/acceptance_checks.json").is_file(), reason="H1 not run")
def test_acceptance_success_collision_match() -> None:
    import json

    payload = json.loads((OUT / "manifests/acceptance_checks.json").read_text(encoding="utf-8"))
    assert payload["success_rate"]["baseline"] == pytest.approx(0.35)
    assert payload["collision_rate"]["baseline"] == pytest.approx(0.04375)
