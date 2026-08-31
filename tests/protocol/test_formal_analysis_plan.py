"""Stage 5C-0 — formal analysis plan tests."""

from __future__ import annotations

from thesis.protocol.final_training_protocol import build_formal_analysis_plan


def test_analysis_plan_frozen():
    plan = build_formal_analysis_plan()
    assert plan["statistical_unit"] == "formal_training_seed"
    assert plan["pairwise_contrasts"] == [
        "mean_pbrs - baseline",
        "min_pbrs - baseline",
        "min_pbrs - mean_pbrs",
    ]
    assert len(plan["primary_endpoints_at_step_20000"]) == 5
    assert plan["bootstrap"]["replicates"] == 10_000
    assert plan["bootstrap"]["rng_seed"] == 91001
    assert plan["multiple_comparisons"]["method"] == "Holm"
    assert plan["hypothesis_tests"]["alpha"] == 0.05
    assert plan["hypothesis_tests"]["sidedness"] == "two_sided"
    assert plan["missing_convention_policy"]["zero_fill"] is False
    assert plan["best_checkpoint_selection"] is False
    assert plan["primary_formal_endpoint_checkpoint"] == 20_000
