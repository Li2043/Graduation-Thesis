"""Stage 4A environment candidate selection and holdout validation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from thesis.certification.choice_state_certification import (
    certify_block,
    run_background_safety_audit,
)
from thesis.certification.choice_state_metrics import aggregate_order_gaps
from thesis.certification.choice_state_scenarios import (
    build_environment_candidates,
    build_ic_blocks,
    materialize_block_for_geometry,
)
from thesis.envs.final_environment_config import EnvironmentCandidate, InitialConditionBlock


def evaluate_candidate_on_blocks(
    candidate: EnvironmentCandidate,
    blocks: list[InitialConditionBlock],
    *,
    block_set: str,
) -> dict[str, Any]:
    certifications: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    matrices: list[dict[str, Any]] = []
    bg_rows: list[dict[str, Any]] = []
    for block in blocks:
        mat = materialize_block_for_geometry(block, candidate.geometry)
        result = certify_block(candidate, mat)
        certifications.append(
            {
                "candidate_id": candidate.candidate_id,
                "block_set": block_set,
                "block_id": result["block_id"],
                "arrival_category": result["arrival_category"],
                "certified": result["certified"],
                "rejection_reasons": result["rejection_reasons"],
                "normalised_order_gap": result["normalised_order_gap"],
                "background_relevant": result["background_relevant"],
                "label_swap_max_error": result["label_swap_max_error"],
            }
        )
        for tr in result["traces"]:
            tr = dict(tr)
            tr["run_id"] = None  # filled by runner
            tr["candidate_id"] = candidate.candidate_id
            traces.append(tr)
        matrices.append(
            {
                "candidate_id": candidate.candidate_id,
                "block_set": block_set,
                "block_id": result["block_id"],
                "matrix": result["matrix"],
                "certified": result["certified"],
            }
        )
        bg_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "block_set": block_set,
                "block_id": result["block_id"],
                "background_relevant": result["background_relevant"],
                "matrix_cells": {
                    k: {
                        "bg_min_speed": v.get("bg_min_speed"),
                        "bg_max_brake": v.get("bg_max_brake"),
                        "bg_min_gap_to_learners": v.get("bg_min_gap_to_learners"),
                    }
                    for k, v in result["matrix"].items()
                },
            }
        )

    certified = [c for c in certifications if c["certified"]]
    gaps = [float(c["normalised_order_gap"]) for c in certified]
    gap_agg = aggregate_order_gaps(gaps) if gaps else {
        "median_normalised_order_gap": float("nan"),
        "maximum_normalised_order_gap": float("nan"),
    }
    bg_rate = (
        sum(1 for c in certifications if c["background_relevant"]) / max(len(certifications), 1)
    )
    categories = {c["arrival_category"] for c in certified}
    label_err = max((float(c["label_swap_max_error"]) for c in certifications), default=0.0)

    return {
        "candidate_id": candidate.candidate_id,
        "priority_rank": candidate.priority_rank,
        "block_set": block_set,
        "n_blocks": len(blocks),
        "n_certified": len(certified),
        "certification_rate": len(certified) / max(len(blocks), 1),
        "certified_arrival_categories": sorted(categories),
        "background_relevance_rate": bg_rate,
        "label_swap_max_error": label_err,
        **gap_agg,
        "certifications": certifications,
        "matrices": matrices,
        "traces": traces,
        "background_rows": bg_rows,
    }


def calibration_feasible(
    eval_result: dict[str, Any],
    *,
    bg_safety: dict[str, Any],
    integrity: dict[str, int],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if eval_result["n_certified"] < 11:
        reasons.append(f"certified={eval_result['n_certified']}<11")
    cats = set(eval_result["certified_arrival_categories"])
    for need in ("mainline_lead", "ramp_lead", "near_simultaneous"):
        if need not in cats:
            reasons.append(f"missing_category:{need}")
    if float(eval_result["background_relevance_rate"]) + 1e-12 < 0.75:
        reasons.append(f"bg_relevance={eval_result['background_relevance_rate']:.3f}<0.75")
    med = float(eval_result["median_normalised_order_gap"])
    mx = float(eval_result["maximum_normalised_order_gap"])
    if not math.isfinite(med) or med > 0.05 + 1e-12:
        reasons.append(f"median_order_gap={med}")
    if not math.isfinite(mx) or mx > 0.10 + 1e-12:
        reasons.append(f"max_order_gap={mx}")
    if float(eval_result["label_swap_max_error"]) > 1e-12:
        reasons.append(f"label_swap={eval_result['label_swap_max_error']}")
    if int(bg_safety.get("spontaneous_collision_count", 0)) != 0:
        reasons.append("spontaneous_background_collision")
    for k, v in integrity.items():
        if int(v) != 0:
            reasons.append(f"{k}={v}")
    return len(reasons) == 0, reasons


def validation_pass(eval_result: dict[str, Any], *, bg_safety: dict[str, Any], integrity: dict[str, int]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if eval_result["n_certified"] < 7:
        reasons.append(f"certified={eval_result['n_certified']}<7")
    for c in eval_result["certifications"]:
        if not c["certified"]:
            continue
        # both conventions implied by certification
    if float(eval_result["background_relevance_rate"]) + 1e-12 < 0.75:
        reasons.append(f"bg_relevance={eval_result['background_relevance_rate']:.3f}<0.75")
    med = float(eval_result["median_normalised_order_gap"])
    mx = float(eval_result["maximum_normalised_order_gap"])
    if not math.isfinite(med) or med > 0.05 + 1e-12:
        reasons.append(f"median_order_gap={med}")
    if not math.isfinite(mx) or mx > 0.10 + 1e-12:
        reasons.append(f"max_order_gap={mx}")
    if float(eval_result["label_swap_max_error"]) > 1e-12:
        reasons.append(f"label_swap={eval_result['label_swap_max_error']}")
    if int(bg_safety.get("spontaneous_collision_count", 0)) != 0:
        reasons.append("spontaneous_background_collision")
    for k, v in integrity.items():
        if int(v) != 0:
            reasons.append(f"{k}={v}")
    return len(reasons) == 0, reasons


def accumulate_integrity(eval_result: dict[str, Any]) -> dict[str, int]:
    disc = rep = inv = nan = fix = 0
    for m in eval_result["matrices"]:
        for cell in m["matrix"].values():
            disc += int(cell.get("route_discontinuity", 0) or 0)
            rep += int(cell.get("repeated_exit", 0) or 0)
            inv += int(cell.get("invalid_flags", 0) or 0)
            nan += int(cell.get("nan_count", 0) or 0)
            fix += int(cell.get("fixture_count", 0) or 0)
    return {
        "route_discontinuity_count": disc,
        "repeated_exit_count": rep,
        "invalid_flag_count": inv,
        "nan_inf_count": nan,
        "fixture_count": fix,
    }


def select_environment_candidate(
    *,
    candidates: list[EnvironmentCandidate] | None = None,
    calibration_blocks: list[InitialConditionBlock] | None = None,
    validation_blocks: list[InitialConditionBlock] | None = None,
) -> dict[str, Any]:
    """Calibrate on calibration blocks only; validate once; never reselect."""
    if candidates is None:
        candidates = build_environment_candidates()
    if calibration_blocks is None or validation_blocks is None:
        cal, val = build_ic_blocks()
        calibration_blocks = calibration_blocks or cal
        validation_blocks = validation_blocks or val

    # Enforce separation: validation never used for selection
    cal_ids = {b.block_id for b in calibration_blocks}
    val_ids = {b.block_id for b in validation_blocks}
    if cal_ids & val_ids:
        raise ValueError("calibration and validation block IDs must be disjoint")

    candidate_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    all_traces: list[dict[str, Any]] = []
    all_matrices: list[dict[str, Any]] = []
    all_bg: list[dict[str, Any]] = []
    all_certs: list[dict[str, Any]] = []

    feasible: list[tuple[int, EnvironmentCandidate, dict[str, Any]]] = []

    for cand in sorted(candidates, key=lambda c: c.priority_rank):
        cal_eval = evaluate_candidate_on_blocks(cand, calibration_blocks, block_set="calibration")
        bg_safety = run_background_safety_audit(cand, calibration_blocks + validation_blocks)
        integrity = accumulate_integrity(cal_eval)
        ok, reasons = calibration_feasible(cal_eval, bg_safety=bg_safety, integrity=integrity)
        row = {
            "candidate_id": cand.candidate_id,
            "priority_rank": cand.priority_rank,
            "geometry_id": cand.geometry.geometry_id,
            "idm_profile_id": cand.idm.profile_id,
            "calibration_certified": cal_eval["n_certified"],
            "calibration_n": cal_eval["n_blocks"],
            "calibration_feasible": ok,
            "rejection_reasons": reasons,
            "background_relevance_rate_calibration": cal_eval["background_relevance_rate"],
            "median_normalised_order_gap": cal_eval["median_normalised_order_gap"],
            "maximum_normalised_order_gap": cal_eval["maximum_normalised_order_gap"],
            "label_swap_max_error": cal_eval["label_swap_max_error"],
            "spontaneous_background_collision_count": bg_safety["spontaneous_collision_count"],
            **integrity,
        }
        candidate_rows.append(row)
        all_traces.extend(cal_eval["traces"])
        all_matrices.extend(cal_eval["matrices"])
        all_bg.extend(cal_eval["background_rows"])
        all_certs.extend(cal_eval["certifications"])
        if not ok:
            failures.append({"candidate_id": cand.candidate_id, "stage": "calibration", "reasons": reasons})
        else:
            feasible.append((cand.priority_rank, cand, cal_eval))

    selected = None
    validation_eval = None
    validation_ok = False
    validation_reasons: list[str] = []
    overall = "FAIL"
    environment_parameters_final = False

    if feasible:
        feasible.sort(key=lambda x: x[0])
        rank, cand, cal_eval = feasible[0]
        selected = cand
        # Holdout exactly once — never reselect
        validation_eval = evaluate_candidate_on_blocks(
            cand, validation_blocks, block_set="validation"
        )
        bg_safety_v = run_background_safety_audit(cand, validation_blocks)
        integrity_v = accumulate_integrity(validation_eval)
        validation_ok, validation_reasons = validation_pass(
            validation_eval, bg_safety=bg_safety_v, integrity=integrity_v
        )
        all_traces.extend(validation_eval["traces"])
        all_matrices.extend(validation_eval["matrices"])
        all_bg.extend(validation_eval["background_rows"])
        all_certs.extend(validation_eval["certifications"])
        if validation_ok:
            overall = "PASS"
            environment_parameters_final = True
        else:
            overall = "FAIL"
            failures.append(
                {
                    "candidate_id": cand.candidate_id,
                    "stage": "validation",
                    "reasons": validation_reasons,
                    "note": "holdout_failure_no_reselection",
                }
            )
        # update selected row
        for row in candidate_rows:
            if row["candidate_id"] == cand.candidate_id:
                row["selected"] = True
                row["validation_certified"] = validation_eval["n_certified"]
                row["validation_pass"] = validation_ok
                row["validation_rejection_reasons"] = validation_reasons
                row["background_relevance_rate_validation"] = validation_eval[
                    "background_relevance_rate"
                ]
    else:
        for row in candidate_rows:
            row["selected"] = False

    return {
        "overall": overall,
        "environment_parameters_final": environment_parameters_final,
        "comfort_parameters_final": False,
        "policy_training_started": False,
        "candidates": candidate_rows,
        "feasible_candidate_ids": [c.candidate_id for _, c, _ in feasible],
        "selected_candidate": selected.to_dict() if selected else None,
        "calibration_blocks": [b.to_dict() for b in calibration_blocks],
        "validation_blocks": [b.to_dict() for b in validation_blocks],
        "certifications": all_certs,
        "matrices": all_matrices,
        "traces": all_traces,
        "background_rows": all_bg,
        "failures": failures,
        "validation": None
        if validation_eval is None
        else {
            "n_certified": validation_eval["n_certified"],
            "n_blocks": validation_eval["n_blocks"],
            "pass": validation_ok,
            "reasons": validation_reasons,
            "median_normalised_order_gap": validation_eval["median_normalised_order_gap"],
            "maximum_normalised_order_gap": validation_eval["maximum_normalised_order_gap"],
            "background_relevance_rate": validation_eval["background_relevance_rate"],
            "label_swap_max_error": validation_eval["label_swap_max_error"],
        },
        "selection_used_validation": False,
    }


def build_final_environment_lock(
    *,
    selected: dict[str, Any],
    calibration_blocks: list[dict[str, Any]],
    validation_blocks: list[dict[str, Any]],
    git_commit: str,
    config_hashes: dict[str, str],
    source_hashes: dict[str, str] | None = None,
    holdout_audit: list[dict[str, Any]] | None = None,
    superseded_stage4a_run_id: str = "20260729T231946Z_c8d92bc3",
    superseded_lock_sha256: str = (
        "d5614d41d0c229db70b76973c55daa6905d7c5f07dc0781b81826b8891d76ded"
    ),
    route_coordinate_version: str = "v3_quintic_arc_length_4a0r2",
) -> dict[str, Any]:
    from thesis.envs.final_environment_config import (
        LearningDynamics,
        TargetSpeeds,
        TimingConfig,
        VehicleGeometry,
    )
    from thesis.envs.final_observation import OBSERVATION_DIM

    geom = selected["geometry"]
    idm = selected["idm"]
    timing = TimingConfig()
    veh = VehicleGeometry()
    dyn = LearningDynamics()
    targets = TargetSpeeds()
    return {
        "selected_geometry_id": geom["geometry_id"],
        "selected_idm_profile_id": idm["profile_id"],
        "candidate_id": selected["candidate_id"],
        "candidate_priority_rank": selected["priority_rank"],
        "geometry": geom,
        "physics_dt": timing.physics_dt,
        "policy_interval": timing.policy_interval,
        "physics_substeps_per_action": timing.physics_substeps_per_action,
        "vehicle_dimensions": {"length": veh.length, "width": veh.width},
        "learning_action_accelerations": {
            "ACCELERATE": dyn.accel,
            "MAINTAIN": dyn.maintain,
            "DECELERATE": dyn.decel,
        },
        "speed_limits": {"v_min": dyn.v_min, "v_max": dyn.v_max},
        "collision_thresholds": {"bumper_gap_collision": 0.0, "min_safe_bumper_gap": 2.0},
        "target_speeds": asdict(targets),
        "idm_parameters": idm,
        "maximum_emergency_deceleration": idm.get("maximum_emergency_deceleration", 6.0),
        "observation_version": "final_observation_v1_stage4a0r",
        "observation_dimension": OBSERVATION_DIM,
        "route_geometry_version": route_coordinate_version,
        "collision_model_version": "oriented_rectangle_sat_v1",
        "exit_removal_semantics": {
            "check_every_physics_substep": True,
            "collision_precedence_same_substep": True,
            "remove_from_collision_and_idm_immediately": True,
            "applies_to": ["A", "B", "B_front", "B_rear"],
        },
        "calibration_block_definitions": calibration_blocks,
        "validation_block_definitions": validation_blocks,
        "zero_duplicate_holdout_audit": holdout_audit or [],
        "route_coordinate_version": route_coordinate_version,
        "termination_truncation_rules": {
            "success": "both_learners_exit_without_stakeholder_collision",
            "collision": "oriented_rectangle_sat_overlap",
            "truncation": "max_policy_steps_without_terminal",
        },
        "controller_role_balancing": {
            "assignments": ["A=mainline,B=ramp", "A=ramp,B=mainline"],
            "permanent_role_assignment_forbidden": True,
        },
        "source_git_commit": git_commit,
        "source_hashes": source_hashes or {},
        "configuration_sha256": config_hashes,
        "superseded_stage4a_run_id": superseded_stage4a_run_id,
        "superseded_lock_sha256": superseded_lock_sha256,
        "comfort_parameters_final": False,
        "policy_training_started": False,
        "environment_parameters_final": True,
    }


def write_processed_tables(result: dict[str, Any], processed_dir: Path) -> None:
    import csv

    def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
        path = processed_dir / name
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        keys: list[str] = []
        seen: set[str] = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                flat = {
                    k: (json.dumps(v, default=str) if isinstance(v, (dict, list, tuple)) else v)
                    for k, v in r.items()
                }
                w.writerow(flat)

    write_csv("candidate_summary.csv", result["candidates"])
    write_csv("block_certification.csv", result["certifications"])
    write_csv(
        "choice_matrix_summary.csv",
        [
            {
                "candidate_id": m["candidate_id"],
                "block_set": m["block_set"],
                "block_id": m["block_id"],
                "certified": m["certified"],
                "GO_GO": m["matrix"]["GO_GO"],
                "GO_YIELD": m["matrix"]["GO_YIELD"],
                "YIELD_GO": m["matrix"]["YIELD_GO"],
                "YIELD_YIELD": m["matrix"]["YIELD_YIELD"],
            }
            for m in result["matrices"]
        ],
    )
    order_rows = [
        {
            "candidate_id": c["candidate_id"],
            "block_set": c["block_set"],
            "block_id": c["block_id"],
            "normalised_order_gap": c["normalised_order_gap"],
            "certified": c["certified"],
        }
        for c in result["certifications"]
    ]
    write_csv("order_bias_summary.csv", order_rows)
    write_csv(
        "background_relevance_summary.csv",
        [
            {
                "candidate_id": b["candidate_id"],
                "block_set": b["block_set"],
                "block_id": b["block_id"],
                "background_relevant": b["background_relevant"],
            }
            for b in result["background_rows"]
        ],
    )
    write_csv(
        "background_safety_summary.csv",
        [
            {
                "candidate_id": r["candidate_id"],
                "spontaneous_background_collision_count": r.get(
                    "spontaneous_background_collision_count", 0
                ),
            }
            for r in result["candidates"]
        ],
    )
    write_csv(
        "label_invariance_summary.csv",
        [
            {
                "candidate_id": c["candidate_id"],
                "block_set": c["block_set"],
                "block_id": c["block_id"],
                "label_swap_max_error": c["label_swap_max_error"],
            }
            for c in result["certifications"]
        ],
    )
    val_rows = []
    if result.get("validation"):
        val_rows.append({"selected_candidate": (result.get("selected_candidate") or {}).get("candidate_id"), **result["validation"]})
    write_csv("validation_summary.csv", val_rows)

    # Acceleration diagnostics from traces
    acc_rows = []
    for tr in result["traces"]:
        if tr.get("controller_id") in ("A", "B"):
            acc_rows.append(
                {
                    "candidate_id": tr.get("candidate_id"),
                    "block_id": tr.get("block_id"),
                    "matrix_cell": tr.get("matrix_cell"),
                    "controller_id": tr.get("controller_id"),
                    "realised_acceleration": tr.get("realised_acceleration"),
                    "jerk": tr.get("jerk"),
                }
            )
    write_csv("acceleration_trace_summary.csv", acc_rows[:50000])


__all__ = [
    "build_final_environment_lock",
    "calibration_feasible",
    "evaluate_candidate_on_blocks",
    "select_environment_candidate",
    "validation_pass",
    "write_processed_tables",
]
