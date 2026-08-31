"""Stage 7C-Q1 competence gate: PASS / FAIL / INVALID only."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from thesis.pilots.stage7c_q1_config import (
    GATE_ADJACENT_SUCCESS_DROP_MAX,
    GATE_CHECKPOINTS,
    GATE_COLLISION_MAX,
    GATE_MATERIAL_REGRESSION,
    GATE_MAX_LATE_COLLAPSE_SEEDS,
    GATE_MAX_MATERIAL_REGRESSION_SEEDS,
    GATE_MEAN_SUCCESS_MIN,
    GATE_MIN_QUALIFIED_SEEDS,
    GATE_SEED_SUCCESS_MIN,
    GATE_SWAP_ELIGIBILITY_MIN,
    GATE_TRUNCATION_MAX,
    LEARNING_CURVE_CHECKPOINTS,
    PILOT_SEEDS,
    late_collapse_7c,
)


def evaluate_competence_gate(
    seed_checkpoint_df: pd.DataFrame,
    *,
    expected_seeds: tuple[int, ...] = PILOT_SEEDS,
    integrity_ok: bool = True,
    integrity_errors: list[str] | None = None,
) -> dict[str, Any]:
    """``seed_checkpoint_df`` columns required:
    master_seed, checkpoint_step, success_rate, collision_rate, truncation_rate, swap_eligibility
    """
    if not integrity_ok:
        return {
            "status": "INVALID",
            "reason": "; ".join(integrity_errors or ["integrity_failed"]),
            "components": {},
        }

    required = {
        "master_seed",
        "checkpoint_step",
        "success_rate",
        "collision_rate",
        "truncation_rate",
        "swap_eligibility",
    }
    missing_cols = required - set(seed_checkpoint_df.columns)
    if missing_cols:
        return {
            "status": "INVALID",
            "reason": f"missing columns: {sorted(missing_cols)}",
            "components": {},
        }

    df = seed_checkpoint_df.copy()
    seeds_present = sorted(int(s) for s in df["master_seed"].unique())
    if seeds_present != sorted(expected_seeds):
        return {
            "status": "INVALID",
            "reason": f"seed set mismatch: present={seeds_present}",
            "components": {},
        }

    for ckpt in GATE_CHECKPOINTS:
        if df[df["checkpoint_step"] == ckpt].empty:
            return {
                "status": "INVALID",
                "reason": f"missing gate checkpoint {ckpt}",
                "components": {},
            }

    components: dict[str, Any] = {}
    qualified_sets: list[set[int]] = []
    for ckpt in GATE_CHECKPOINTS:
        g = df[df["checkpoint_step"] == ckpt]
        mean_s = float(g["success_rate"].mean())
        mean_c = float(g["collision_rate"].mean())
        mean_t = float(g["truncation_rate"].mean())
        mean_swap = float(g["swap_eligibility"].mean())
        seed_ok = set(
            int(r.master_seed)
            for r in g.itertuples(index=False)
            if float(r.success_rate) >= GATE_SEED_SUCCESS_MIN
        )
        qualified_sets.append(seed_ok)
        comps = {
            "mean_success": mean_s >= GATE_MEAN_SUCCESS_MIN,
            "collision": mean_c <= GATE_COLLISION_MAX,
            "truncation": mean_t <= GATE_TRUNCATION_MAX,
            "swap_eligibility": mean_swap >= GATE_SWAP_ELIGIBILITY_MIN,
            "mean_success_value": mean_s,
            "mean_collision_value": mean_c,
            "mean_truncation_value": mean_t,
            "mean_swap_value": mean_swap,
            "seeds_ge_threshold": len(seed_ok),
        }
        components[str(ckpt)] = comps

    intersection = set.intersection(*qualified_sets) if qualified_sets else set()
    components["qualified_seed_intersection"] = sorted(intersection)
    components["qualified_seed_intersection_count"] = len(intersection)
    intersection_ok = len(intersection) >= GATE_MIN_QUALIFIED_SEEDS

    # Learning-curve adjacent drops on late 64-episode checkpoints
    curve_ok = True
    curve_violations: list[str] = []
    means = {}
    for ckpt in LEARNING_CURVE_CHECKPOINTS:
        g = df[df["checkpoint_step"] == ckpt]
        if g.empty:
            return {
                "status": "INVALID",
                "reason": f"missing learning-curve checkpoint {ckpt}",
                "components": components,
            }
        means[ckpt] = float(g["success_rate"].mean())
    for a, b in zip(LEARNING_CURVE_CHECKPOINTS, LEARNING_CURVE_CHECKPOINTS[1:]):
        drop = means[a] - means[b]
        if drop > GATE_ADJACENT_SUCCESS_DROP_MAX:
            curve_ok = False
            curve_violations.append(f"{a}->{b} drop={drop:.4f}")

    # Material regressions 350→400
    material_seeds = []
    late_collapse_seeds = []
    for seed in expected_seeds:
        g = df[df["master_seed"] == seed]
        by_ckpt = {
            int(r.checkpoint_step): float(r.success_rate) for r in g.itertuples(index=False)
        }
        # material regression any adjacent in 350..400
        late_ckpts = [c for c in LEARNING_CURVE_CHECKPOINTS if c >= 350_000]
        for a, b in zip(late_ckpts, late_ckpts[1:]):
            if by_ckpt.get(a, 0) - by_ckpt.get(b, 0) > GATE_MATERIAL_REGRESSION:
                material_seeds.append(int(seed))
                break
        if late_collapse_7c(by_ckpt):
            late_collapse_seeds.append(int(seed))

    material_ok = len(set(material_seeds)) <= GATE_MAX_MATERIAL_REGRESSION_SEEDS
    collapse_ok = len(set(late_collapse_seeds)) <= GATE_MAX_LATE_COLLAPSE_SEEDS

    gate_ckpt_ok = all(
        components[str(c)]["mean_success"]
        and components[str(c)]["collision"]
        and components[str(c)]["truncation"]
        and components[str(c)]["swap_eligibility"]
        for c in GATE_CHECKPOINTS
    )

    components["learning_curve_ok"] = curve_ok
    components["learning_curve_violations"] = curve_violations
    components["material_regression_seeds"] = sorted(set(material_seeds))
    components["late_collapse_seeds"] = sorted(set(late_collapse_seeds))
    components["intersection_ok"] = intersection_ok
    components["gate_checkpoint_means_ok"] = gate_ckpt_ok

    passed = (
        gate_ckpt_ok
        and intersection_ok
        and curve_ok
        and material_ok
        and collapse_ok
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "reason": "" if passed else "one_or_more_qualification_criteria_failed",
        "components": components,
    }


def evaluate_gate_from_csv(path: str, *, integrity_ok: bool = True) -> dict[str, Any]:
    df = pd.read_csv(path)
    return evaluate_competence_gate(df, integrity_ok=integrity_ok)


__all__ = ["evaluate_competence_gate", "evaluate_gate_from_csv"]
