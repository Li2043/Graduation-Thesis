"""Tests for the Stage 11 Study A held-out evaluation driver
(stage11_confirmatory_eval_runner.py). The pure aggregation/gate functions
are tested with synthetic data (fast, no checkpoint files needed); the
end-to-end loop is tested against real confirmatory checkpoints when
present on this machine, skipped otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thesis.pilots.stage11_confirmatory_config import (
    ALL_CONFIRMATORY_SEEDS,
    BASELINE_SEEDS,
    GATE_CHECKPOINTS,
    MEAN_PBRS_SEEDS,
    MIN_PBRS_SEEDS,
)
from thesis.pilots.stage11_confirmatory_eval_runner import (
    checkpoint_path_for,
    compute_adjacent_drop_stability,
    compute_competence_gate,
    compute_gate_checkpoint_stability,
    condition_for_seed,
    run_confirmatory_evaluation,
    summarize_episodes,
)

CONFIRMATORY_CHECKPOINT_ROOT = Path("external_checkpoints/stage11_confirmatory/checkpoints")
CONFIRMATORY_OUTPUT_ROOT = Path("external_checkpoints/stage11_confirmatory/output")


def test_gate_checkpoints_are_all_multiples_of_10k():
    for step in GATE_CHECKPOINTS:
        assert step % 10_000 == 0, step


def test_condition_for_seed_covers_every_formal_seed():
    for seed in ALL_CONFIRMATORY_SEEDS:
        condition_for_seed(seed)  # must not raise


def test_condition_for_seed_rejects_pilot_seed():
    with pytest.raises(ValueError):
        condition_for_seed(69113)


def test_checkpoint_path_for_shape(tmp_path):
    path = checkpoint_path_for(tmp_path, 69121, 350000)
    assert path == tmp_path / "seed_69121" / "ckpt_step_350000.pt"


def _episode(*, success, collision, truncated, u_ramp, u_mainline):
    return {
        "success": success,
        "collision": collision,
        "truncated": truncated,
        "U_ramp": u_ramp,
        "U_mainline": u_mainline,
    }


def test_summarize_episodes_basic_rates():
    episodes = [
        _episode(success=True, collision=False, truncated=False, u_ramp=0.9, u_mainline=1.0),
        _episode(success=False, collision=True, truncated=False, u_ramp=0.0, u_mainline=0.5),
    ]
    s = summarize_episodes(episodes)
    assert s["n_episodes"] == 2
    assert s["completion_rate"] == 0.5
    assert s["collision_free_rate"] == 0.5
    assert s["truncation_rate"] == 0.0
    assert s["min_U_mean"] == pytest.approx((0.9 + 0.0) / 2)
    assert s["gap_mean"] == pytest.approx((0.1 + 0.5) / 2)


def test_compute_adjacent_drop_stability_flags_late_collapse(tmp_path):
    steps = list(range(300_000, 400_001, 10_000))
    completion_by_step = {s: 0.95 for s in steps}
    completion_by_step[400_000] = 0.20  # simulated late collapse
    manifest = {"checkpoints": [{"step": s, "window": {"completion_rate": c}} for s, c in completion_by_step.items()]}
    seed = ALL_CONFIRMATORY_SEEDS[0]
    (tmp_path / f"seed_{seed}_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = compute_adjacent_drop_stability(output_root=tmp_path, seeds=(seed,))
    assert seed in result["flagged_seeds"]
    assert result["max_drop_per_seed"][seed] == pytest.approx(0.75)


def test_compute_adjacent_drop_stability_clean_run_not_flagged(tmp_path):
    steps = list(range(300_000, 400_001, 10_000))
    manifest = {"checkpoints": [{"step": s, "window": {"completion_rate": 0.95}} for s in steps]}
    seed = ALL_CONFIRMATORY_SEEDS[0]
    (tmp_path / f"seed_{seed}_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = compute_adjacent_drop_stability(output_root=tmp_path, seeds=(seed,))
    assert result["flagged_seeds"] == {}


def test_compute_gate_checkpoint_stability_flags_only_the_collapsing_seed():
    seeds = ALL_CONFIRMATORY_SEEDS
    steps = GATE_CHECKPOINTS
    eval_results = {}
    for seed in seeds:
        comps = [1.0, 1.0, 1.0]
        if seed == 69132:
            comps = [1.0, 0.5625, 0.0]
        eval_results[seed] = {
            step: {"episodes": [], "summary": {"completion_rate": c}} for step, c in zip(steps, comps)
        }
    result = compute_gate_checkpoint_stability(eval_results)
    assert set(result["flagged_seeds"].keys()) == {69132}
    assert result["max_drop_per_seed"][69121] == 0.0


def _fake_eval_results(*, passing_seeds, failing_seed=None):
    results = {}
    for seed in ALL_CONFIRMATORY_SEEDS:
        summary_ok = {
            "n_episodes": 16,
            "completion_rate": 0.95,
            "collision_free_rate": 0.98,
            "truncation_rate": 0.0,
            "mean_U_mean": 0.9,
            "min_U_mean": 0.85,
            "gap_mean": 0.05,
        }
        summary_bad = dict(summary_ok, completion_rate=0.5)
        by_step = {}
        for step in GATE_CHECKPOINTS:
            summary = summary_bad if seed == failing_seed else summary_ok
            by_step[step] = {"episodes": [], "summary": summary}
        results[seed] = by_step
    return results


def test_compute_competence_gate_passes_when_all_seeds_qualify():
    eval_results = _fake_eval_results(passing_seeds=ALL_CONFIRMATORY_SEEDS)
    stability = {"flagged_seeds": {}}
    gate = compute_competence_gate(eval_results, stability)
    assert gate["verdict"] == "PASS"
    for condition in ("baseline", "mean_pbrs", "min_pbrs"):
        assert gate["per_condition_qualifying_count"][condition] == 8


def test_compute_competence_gate_fails_below_seed_intersection_min():
    # Fail 3 of baseline's 8 seeds -> only 5 qualify, below GATE_SEED_INTERSECTION_MIN=6.
    eval_results = _fake_eval_results(passing_seeds=ALL_CONFIRMATORY_SEEDS)
    for seed in BASELINE_SEEDS[:3]:
        for step in GATE_CHECKPOINTS:
            eval_results[seed][step]["summary"] = dict(eval_results[seed][step]["summary"], completion_rate=0.5)
    stability = {"flagged_seeds": {}}
    gate = compute_competence_gate(eval_results, stability)
    assert gate["verdict"] == "FAIL"
    assert gate["intersection_ok"]["baseline"] is False
    assert gate["intersection_ok"]["mean_pbrs"] is True
    assert gate["intersection_ok"]["min_pbrs"] is True


def test_compute_competence_gate_tolerates_one_flagged_seed_within_intersection_budget():
    # A single stability-flagged seed excludes only that seed from the
    # qualifying set (7/8 remains >= GATE_SEED_INTERSECTION_MIN=6) -- folded
    # into per-seed qualification, not a separate whole-run veto.
    eval_results = _fake_eval_results(passing_seeds=ALL_CONFIRMATORY_SEEDS)
    stability = {"flagged_seeds": {MEAN_PBRS_SEEDS[3]: [{"from_step": 380000, "to_step": 400000, "drop": 0.7}]}}
    gate = compute_competence_gate(eval_results, stability)
    assert gate["verdict"] == "PASS"
    assert MEAN_PBRS_SEEDS[3] not in gate["per_condition_qualifying_seeds"]["mean_pbrs"]
    assert gate["per_condition_qualifying_count"]["mean_pbrs"] == 7


def test_compute_competence_gate_fails_when_flagged_seeds_exceed_intersection_budget():
    eval_results = _fake_eval_results(passing_seeds=ALL_CONFIRMATORY_SEEDS)
    stability = {
        "flagged_seeds": {
            seed: [{"from_step": 380000, "to_step": 400000, "drop": 0.7}] for seed in MEAN_PBRS_SEEDS[:3]
        }
    }
    gate = compute_competence_gate(eval_results, stability)
    assert gate["verdict"] == "FAIL"
    assert gate["intersection_ok"]["mean_pbrs"] is False
    assert gate["per_condition_qualifying_count"]["mean_pbrs"] == 5


pytestmark_integration = pytest.mark.skipif(
    not (CONFIRMATORY_CHECKPOINT_ROOT / "seed_69121" / "ckpt_step_350000.pt").exists(),
    reason="Study A confirmatory checkpoints not present on this machine",
)


@pytestmark_integration
def test_run_confirmatory_evaluation_end_to_end_smoke():
    result = run_confirmatory_evaluation(
        checkpoint_root=CONFIRMATORY_CHECKPOINT_ROOT,
        seeds=(69121,),
        checkpoints=(350_000,),
    )
    assert set(result.keys()) == {69121}
    assert set(result[69121].keys()) == {350_000}
    summary = result[69121][350_000]["summary"]
    assert summary["n_episodes"] == 16
    assert 0.0 <= summary["completion_rate"] <= 1.0
