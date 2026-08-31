from __future__ import annotations

import csv

import numpy as np
import pytest

from thesis.study_b.analysis.behaviour import mean_hard_brake_rate, worse_off_frequency_by_class
from thesis.study_b.analysis.bootstrap import holm_correction, paired_bootstrap_contrast
from thesis.study_b.analysis.competence import check_no_collapse, check_qualification_gate
from thesis.study_b.analysis.plots import plot_bootstrap_forest, plot_learning_curves
from thesis.study_b.analysis.welfare import seed_level_summary


EVAL_FIELDS = [
    "scenario_id", "traffic_type", "term_reason", "completion", "collision", "timeout",
    "mean_U", "min_U", "min_U_vehicle", "min_U_role", "min_U_speed_class", "ggi", "gini",
    "C_max", "C_mean", "hard_brake_total",
] + [f"{prefix}_{vid}" for vid in ("V0", "V1", "V2", "V3") for prefix in ("role", "speed_class", "U", "C", "hard_brake")]


def _write_eval_csv(path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _base_row(**overrides) -> dict:
    row = {
        "scenario_id": "s0", "traffic_type": "heterogeneous", "term_reason": "success",
        "completion": 1, "collision": 0, "timeout": 0,
        "mean_U": 0.9, "min_U": 0.8, "min_U_vehicle": "V0", "min_U_role": "ramp", "min_U_speed_class": "slow",
        "ggi": 0.85, "gini": 0.05, "C_max": 1.0, "C_mean": 0.5, "hard_brake_total": 0,
    }
    for vid in ("V0", "V1", "V2", "V3"):
        row[f"role_{vid}"] = "ramp" if vid in ("V0", "V1") else "mainline"
        row[f"speed_class_{vid}"] = "slow" if vid in ("V0", "V2") else "fast"
        row[f"U_{vid}"] = 0.9
        row[f"C_{vid}"] = 0.5
        row[f"hard_brake_{vid}"] = 0
    row.update(overrides)
    return row


def test_seed_level_summary_basic(tmp_path):
    rows = [_base_row(scenario_id=f"s{i}") for i in range(10)]
    path = tmp_path / "eval.csv"
    _write_eval_csv(path, rows)
    summary = seed_level_summary(path)
    assert summary["n_scenarios"] == 10
    assert summary["completion_rate"] == pytest.approx(1.0)
    assert summary["mean_U"] == pytest.approx(0.9)
    assert summary["gini"] == pytest.approx(0.05)
    assert summary["all_zero_utility_rate"] == pytest.approx(0.0)


def test_seed_level_summary_excludes_na_gini(tmp_path):
    rows = [_base_row(scenario_id="s0", gini=""), _base_row(scenario_id="s1", gini=0.2)]
    path = tmp_path / "eval.csv"
    _write_eval_csv(path, rows)
    summary = seed_level_summary(path)
    assert summary["gini"] == pytest.approx(0.2)  # only the non-NA row counted
    assert summary["all_zero_utility_rate"] == pytest.approx(0.5)


def test_paired_bootstrap_contrast_detects_clear_positive_effect():
    baseline = [0.5, 0.52, 0.48, 0.51, 0.49, 0.50]
    condition = [0.7, 0.72, 0.68, 0.71, 0.69, 0.70]  # consistently +0.2
    result = paired_bootstrap_contrast(condition, baseline, n_replicates=2000, seed=0)
    assert result.point_estimate == pytest.approx(0.2, abs=0.01)
    assert result.ci_lower > 0
    assert result.p_value < 0.05


def test_paired_bootstrap_contrast_null_effect_wide_ci():
    rng = np.random.default_rng(1)
    baseline = list(rng.normal(0.5, 0.05, size=6))
    condition = list(baseline)  # identical -> zero effect
    result = paired_bootstrap_contrast(condition, baseline, n_replicates=2000, seed=0)
    assert result.point_estimate == pytest.approx(0.0, abs=1e-9)
    assert result.ci_lower <= 0.0 <= result.ci_upper


def test_paired_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap_contrast([0.1, 0.2], [0.1], n_replicates=100)


def test_holm_correction_stops_at_first_failure():
    # p1 very small (should reject), p2 moderate (should reject at its own
    # threshold), p3 large (should NOT reject) -- classic step-down case.
    p_values = {"h1": 0.001, "h2": 0.03, "h3": 0.2}
    results = holm_correction(p_values, alpha=0.05)
    assert results["h1"]["reject_null"] is True
    # m=3: h1 threshold=0.05/3=0.0167 (0.001 passes); h2 threshold=0.05/2=0.025 (0.03 FAILS)
    assert results["h2"]["reject_null"] is False
    assert results["h3"]["reject_null"] is False  # stopped once h2 failed


def test_holm_correction_all_reject_when_all_tiny():
    p_values = {"h1": 0.0001, "h2": 0.0002}
    results = holm_correction(p_values, alpha=0.05)
    assert all(r["reject_null"] for r in results.values())


def test_check_qualification_gate_pass():
    seeds = [
        {"completion_rate": 0.92, "collision_rate": 0.02, "timeout_rate": 0.01},
        {"completion_rate": 0.91, "collision_rate": 0.03, "timeout_rate": 0.02},
        {"completion_rate": 0.93, "collision_rate": 0.01, "timeout_rate": 0.01},
        {"completion_rate": 0.60, "collision_rate": 0.10, "timeout_rate": 0.05},  # the 4th, allowed to miss
    ]
    result = check_qualification_gate(seeds)
    assert result["overall_pass"] is True
    assert result["n_seeds_at_target"] == 3


def test_check_qualification_gate_fails_on_collision():
    seeds = [{"completion_rate": 0.95, "collision_rate": 0.20, "timeout_rate": 0.0}] * 4
    result = check_qualification_gate(seeds)
    assert result["pass_collision"] is False
    assert result["overall_pass"] is False


def test_check_no_collapse_detects_drop_after_reaching_target():
    records = [
        {"step": 0, "window": {"completion_rate": 0.1}},
        {"step": 50000, "window": {"completion_rate": 0.92}},  # reaches target
        {"step": 100000, "window": {"completion_rate": 0.70}},  # drop of 0.22 -> violation
    ]
    result = check_no_collapse(records)
    assert result["pass"] is False
    assert len(result["violations"]) == 1


def test_check_no_collapse_ignores_pre_target_noise():
    records = [
        {"step": 0, "window": {"completion_rate": 0.05}},
        {"step": 50000, "window": {"completion_rate": 0.40}},  # big jump, but before reaching 0.90
        {"step": 100000, "window": {"completion_rate": 0.10}},  # big drop, but STILL before 0.90
    ]
    result = check_no_collapse(records)
    assert result["reached_target"] is False
    assert result["pass"] is True


def test_worse_off_frequency_by_class_sums_to_one():
    rows = [_base_row(min_U_role="ramp", min_U_speed_class="slow") for _ in range(7)]
    rows += [_base_row(min_U_role="mainline", min_U_speed_class="fast") for _ in range(3)]
    freq = worse_off_frequency_by_class(rows)
    assert freq["ramp_slow"] == pytest.approx(0.7)
    assert freq["mainline_fast"] == pytest.approx(0.3)
    assert sum(freq.values()) == pytest.approx(1.0)


def test_mean_hard_brake_rate():
    rows = [_base_row(hard_brake_total=4) for _ in range(5)]
    result = mean_hard_brake_rate(rows)
    assert result["overall"] == pytest.approx(4.0)


def test_plot_learning_curves_smoke(tmp_path):
    manifests = {
        "baseline": [
            {"checkpoints": [{"step": 0, "window": {"completion_rate": 0.1}}, {"step": 100, "window": {"completion_rate": 0.5}}]},
            {"checkpoints": [{"step": 0, "window": {"completion_rate": 0.2}}, {"step": 100, "window": {"completion_rate": 0.6}}]},
        ],
    }
    out = tmp_path / "curve.png"
    plot_learning_curves(manifests, metric="completion_rate", output_path=out)
    assert out.exists()


def test_plot_bootstrap_forest_smoke(tmp_path):
    from thesis.study_b.analysis.bootstrap import BootstrapResult

    results = {
        "mean_pbrs - baseline": BootstrapResult(0.05, -0.01, 0.11, 0.08, 0.9, 6),
        "min_pbrs - baseline": BootstrapResult(-0.02, -0.1, 0.06, 0.6, 0.4, 6),
    }
    out = tmp_path / "forest.png"
    plot_bootstrap_forest(results, output_path=out)
    assert out.exists()
