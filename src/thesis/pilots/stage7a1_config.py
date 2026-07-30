"""Stage 7A-1 Baseline budget pilot constants and decision helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

PILOT_SEEDS: tuple[int, ...] = tuple(range(62001, 62021))
FORBIDDEN_FORMAL_SEEDS: tuple[int, ...] = tuple(range(61001, 61011))
MAX_STEPS = 300_000
CHECKPOINT_STEPS: tuple[int, ...] = (
    0,
    10_000,
    25_000,
    50_000,
    75_000,
    100_000,
    150_000,
    200_000,
    250_000,
    300_000,
)
TRAIN_CHECKPOINT_STEPS: tuple[int, ...] = tuple(s for s in CHECKPOINT_STEPS if s > 0)
EVALUATION_STEPS: tuple[int, ...] = CHECKPOINT_STEPS
PRIMARY_BUDGET_CHECKPOINTS: tuple[int, ...] = (100_000, 150_000, 200_000, 250_000, 300_000)
EPSILON_DECAY_STEPS = 50_000


def assert_baseline_only(
    *,
    condition: str,
    reward_shaping_enabled: bool,
    shaping_coefficient: float,
) -> None:
    if condition != "baseline":
        raise RuntimeError(f"Stage 7A-1 allows only baseline, got {condition!r}")
    if condition in {"mean_pbrs", "min_pbrs"}:
        raise RuntimeError("PBRS conditions forbidden")
    if reward_shaping_enabled:
        raise RuntimeError("reward shaping forbidden")
    if float(shaping_coefficient) != 0.0:
        raise RuntimeError("shaping coefficient must be 0")


def competence_gate_pass(row: Mapping[str, Any]) -> dict[str, Any]:
    """Frozen provisional gate. ``row`` is condition-level checkpoint summary."""
    checks = {
        "seeds_ge_075": int(row.get("seeds_success_ge_0_75", 0)) >= 16,
        "mean_success": float(row.get("mean_success", 0.0)) >= 0.75,
        "collision": float(row.get("mean_collision", 1.0)) <= 0.05,
        "truncation": float(row.get("mean_truncation", 1.0)) <= 0.15,
        "swap_eligible": float(row.get("swap_eligible_pair_proportion", 0.0)) >= 0.75,
    }
    return {"passed": all(checks.values()), "checks": checks}


def non_degradation(earlier: Mapping[str, Any], later: Mapping[str, Any]) -> bool:
    success_drop = float(earlier["mean_success"]) - float(later["mean_success"])
    coll_inc = float(later["mean_collision"]) - float(earlier["mean_collision"])
    trunc_inc = float(later["mean_truncation"]) - float(earlier["mean_truncation"])
    seeds_dec = int(earlier["seeds_success_ge_0_75"]) - int(later["seeds_success_ge_0_75"])
    return (
        success_drop <= 0.05
        and coll_inc <= 0.02
        and trunc_inc <= 0.05
        and seeds_dec <= 2
    )


def select_stable_budget(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Earliest B such that gate(B) and gate(B') and non-degradation."""
    by_step = {int(r["checkpoint_step"]): r for r in summaries}
    pairs = (
        (100_000, 150_000),
        (150_000, 200_000),
        (200_000, 250_000),
        (250_000, 300_000),
    )
    for b, bp in pairs:
        if b not in by_step or bp not in by_step:
            continue
        g_b = competence_gate_pass(by_step[b])
        g_bp = competence_gate_pass(by_step[bp])
        if g_b["passed"] and g_bp["passed"] and non_degradation(by_step[b], by_step[bp]):
            return {
                "stable_sufficient_budget": b,
                "confirmation_checkpoint": bp,
                "status": "budget-responsive and competence-qualified",
                "gate_b": g_b,
                "gate_bp": g_bp,
            }
    # only 300K passes?
    if 300_000 in by_step and competence_gate_pass(by_step[300_000])["passed"]:
        return {
            "stable_sufficient_budget": None,
            "confirmation_checkpoint": None,
            "status": "promising but stability not established",
            "note": "only 300K passed gate; extend beyond 300K before freezing",
            "gate_300k": competence_gate_pass(by_step[300_000]),
        }
    return {
        "stable_sufficient_budget": None,
        "confirmation_checkpoint": None,
        "status": "budget extension alone did not establish competence",
    }


__all__ = [
    "CHECKPOINT_STEPS",
    "EPSILON_DECAY_STEPS",
    "EVALUATION_STEPS",
    "FORBIDDEN_FORMAL_SEEDS",
    "MAX_STEPS",
    "PILOT_SEEDS",
    "PRIMARY_BUDGET_CHECKPOINTS",
    "TRAIN_CHECKPOINT_STEPS",
    "assert_baseline_only",
    "competence_gate_pass",
    "non_degradation",
    "select_stable_budget",
]
