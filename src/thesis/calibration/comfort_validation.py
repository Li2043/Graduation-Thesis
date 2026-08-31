"""Holdout validation and confirmatory online/offline checks (Stage 3B-R1)."""

from __future__ import annotations

import math
from typing import Any

from thesis.calibration.joint_comfort_calibration import (
    TIE,
    TraceBundle,
    _is_active_reward_transition,
    _run_cell_with_substeps,
    evaluate_tuple,
)
from thesis.certification.choice_state_scenarios import (
    EnvironmentCandidate,
    InitialConditionBlock,
    expand_label_assignments,
)
from thesis.rewards.base_reward_v2 import compute_hard_braking_cost


def validate_selected_tuple(
    bundle: TraceBundle,
    *,
    a_comfort: float,
    a_hard: float,
    eta_h: float,
) -> dict[str, Any]:
    """Evaluate selected tuple once on validation blocks; never reselect."""
    raw = evaluate_tuple(
        bundle, a_comfort=a_comfort, a_hard=a_hard, eta_h=eta_h, block_set="validation"
    )
    reasons: list[str] = []
    if int(raw.get("n_usable_safe_blocks") or 0) < 7:
        reasons.append(f"usable_safe_blocks={raw.get('n_usable_safe_blocks')}<7")
    med = float(raw["median_nominal_share"])
    mx = float(raw["max_nominal_share"])
    if not (0.02 - TIE <= med <= 0.06 + TIE):
        reasons.append(f"median_nominal_share={med}")
    if not (mx <= 0.10 + TIE):
        reasons.append(f"max_nominal_share={mx}")
    if not (float(raw["hard_nonzero_rate"]) >= 0.80 - TIE):
        reasons.append(f"hard_nonzero_rate={raw['hard_nonzero_rate']}")
    if not (float(raw["mean_H_hard"]) >= 0.20 - TIE):
        reasons.append(f"hard_mean_H={raw['mean_H_hard']}")
    if not (float(raw["median_paired_share_diff"]) >= 0.02 - TIE):
        reasons.append(f"median_paired_diff={raw['median_paired_share_diff']}")
    # paired positivity already encoded in evaluate_tuple reasons for cal; re-check val
    for r in raw.get("rejection_reasons") or []:
        if "hard_share_not_gt_nominal" in r or "paired_hard_share" in r:
            reasons.append(r)
    if int(raw.get("ordering_violations") or 0) > 0:
        reasons.append(f"ordering_violations={raw['ordering_violations']}")
    if not (float(raw["median_order_gap"]) <= 0.05 + TIE):
        reasons.append(f"median_order_gap={raw['median_order_gap']}")
    if not (float(raw["max_order_gap"]) <= 0.10 + TIE):
        reasons.append(f"max_order_gap={raw['max_order_gap']}")

    # Integrity from bundle (shared)
    for k, v in bundle.integrity.items():
        if int(v) != 0:
            reasons.append(f"integrity_{k}={v}")

    seen: set[str] = set()
    uniq = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return {
        **raw,
        "pass": len(uniq) == 0,
        "rejection_reasons": uniq,
        "selection_used_validation": False,
        "reselection_triggered": False,
    }


def confirmatory_online_rerun(
    *,
    candidate: EnvironmentCandidate,
    calibration_blocks: list[InitialConditionBlock],
    validation_blocks: list[InitialConditionBlock],
    offline_transitions: list[dict[str, Any]],
    a_comfort: float,
    a_hard: float,
    eta_h: float,
    lock_hash: str,
    run_id: str,
) -> dict[str, Any]:
    """Online reconstruction with comfort in the reward path; compare to offline."""
    comfort = {"a_comfort": a_comfort, "a_hard": a_hard, "eta_H": eta_h}
    online_rows: list[dict[str, Any]] = []
    max_rew_err = 0.0
    max_ret_err = 0.0

    off_index: dict[tuple, dict[str, Any]] = {}
    for t in offline_transitions:
        key = (
            t["block_id"],
            t["label_assignment"],
            t["matrix_cell"],
            t["controller_id"],
            int(t["policy_step"]),
        )
        off_index[key] = t

    for block in list(calibration_blocks) + list(validation_blocks):
        for assignment in expand_label_assignments(block):
            for cell in ("GO_GO", "GO_YIELD", "YIELD_GO", "YIELD_YIELD"):
                _meta, trans, _subs = _run_cell_with_substeps(
                    candidate=candidate,
                    block=assignment,
                    cell=cell,  # type: ignore[arg-type]
                    lock_hash=lock_hash,
                    run_id=run_id,
                    comfort=comfort,
                )
                off_ret = 0.0
                on_ret = 0.0
                for t in trans:
                    if not _is_active_reward_transition(t):
                        continue
                    online_total = float(t["total_base_reward"])
                    a_pol = float(t["policy_level_acceleration"])
                    H = float(t.get("hard_braking_cost") or 0.0)
                    hard_c = float(t.get("hard_braking_component") or 0.0)
                    key = (
                        t["block_id"],
                        t["label_assignment"],
                        t["matrix_cell"],
                        t["controller_id"],
                        int(t["policy_step"]),
                    )
                    off = off_index.get(key)
                    if off is not None and _is_active_reward_transition(off):
                        H_off = compute_hard_braking_cost(
                            float(off["policy_level_acceleration"]), a_comfort, a_hard
                        )
                        off_total = float(off["core_reward"]) - eta_h * H_off
                        max_rew_err = max(
                            max_rew_err,
                            abs(online_total - off_total),
                            abs(a_pol - float(off["policy_level_acceleration"])),
                            abs(H - H_off),
                            abs(hard_c - (-eta_h * H_off)),
                        )
                        off_ret += float(off["gamma"]) * off_total
                    on_ret += float(t["gamma"]) * online_total
                    online_rows.append(
                        {
                            **{k: t[k] for k in (
                                "block_set", "block_id", "label_assignment", "matrix_cell",
                                "controller_id", "policy_step", "policy_level_acceleration",
                                "physics_substep_accelerations", "commanded_action",
                                "active_on_road", "progress_component", "exit_component",
                                "collision_component", "gamma", "terminated", "truncated",
                                "fixture_flag", "environment_lock_hash",
                            )},
                            "run_id": run_id,
                            "H": H,
                            "hard_braking_component": hard_c,
                            "total_base_reward": online_total,
                        }
                    )
                max_ret_err = max(max_ret_err, abs(on_ret - off_ret))

    passed = max_rew_err <= 1e-10 + 1e-15 and max_ret_err <= 1e-10 + 1e-15
    return {
        "pass": passed,
        "max_per_transition_reward_error": max_rew_err,
        "max_discounted_return_error": max_ret_err,
        "online_transitions": online_rows,
        "n_online_transitions": len(online_rows),
    }


__all__ = ["confirmatory_online_rerun", "validate_selected_tuple"]
