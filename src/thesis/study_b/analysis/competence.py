"""Phase 1 qualification gate + Phase 2/3 no-collapse check --
new_research_plan.md's "Competence acceptance" section."""

from __future__ import annotations

from typing import Any, Sequence

__all__ = ["check_qualification_gate", "check_no_collapse"]


def check_qualification_gate(
    seed_summaries: Sequence[dict[str, Any]],
    *,
    completion_target: float = 0.90,
    min_seeds_at_target: int = 3,
    collision_max: float = 0.05,
    timeout_max: float = 0.05,
) -> dict[str, Any]:
    """``seed_summaries``: one ``welfare.seed_level_summary`` dict per
    qualification seed. Pooled collision/timeout rate here is the plain
    mean of per-seed rates (a reasonable approximation of a scenario-count-
    weighted pooled rate when every seed evaluates on the same bank size,
    which Phase 1's fixed qualification bank guarantees)."""
    n = len(seed_summaries)
    n_at_target = sum(1 for s in seed_summaries if s["completion_rate"] >= completion_target)
    pooled_collision = sum(s["collision_rate"] for s in seed_summaries) / n
    pooled_timeout = sum(s["timeout_rate"] for s in seed_summaries) / n

    pass_completion = n_at_target >= min_seeds_at_target
    pass_collision = pooled_collision <= collision_max
    pass_timeout = pooled_timeout <= timeout_max

    return {
        "n_seeds": n,
        "n_seeds_at_target": n_at_target,
        "min_seeds_required": min_seeds_at_target,
        "pass_completion": pass_completion,
        "pooled_collision_rate": pooled_collision,
        "pass_collision": pass_collision,
        "pooled_timeout_rate": pooled_timeout,
        "pass_timeout": pass_timeout,
        "overall_pass": pass_completion and pass_collision and pass_timeout,
    }


def check_no_collapse(
    checkpoint_records: Sequence[dict[str, Any]], *, drop_threshold: float = 0.10, reach_threshold: float = 0.90
) -> dict[str, Any]:
    """``checkpoint_records``: a training manifest's ``checkpoints`` list
    (each ``{"step":..., "window": {"completion_rate":...}}``). Only
    counts a drop as a violation AFTER first reaching ``reach_threshold``
    -- early-training noise before competence is established is expected
    and not itself a collapse."""
    reached = False
    max_drop = 0.0
    violations: list[dict[str, Any]] = []
    prev_rate: float | None = None
    for record in checkpoint_records:
        rate = record["window"]["completion_rate"]
        if rate >= reach_threshold:
            reached = True
        if reached and prev_rate is not None:
            drop = prev_rate - rate
            if drop > drop_threshold:
                violations.append({"step": record["step"], "drop": drop})
            max_drop = max(max_drop, drop)
        prev_rate = rate
    return {"reached_target": reached, "max_drop": max_drop, "violations": violations, "pass": len(violations) == 0}
