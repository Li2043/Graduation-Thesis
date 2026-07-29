"""Execute choice matrices and certify physical blocks (Stage 4A)."""

from __future__ import annotations

import math
from typing import Any

from thesis.certification.choice_state_metrics import (
    GAMMA,
    CellOutcome,
    background_meaningful,
    cell_physically_safe,
    classify_exit_order,
    core_ordering_ok,
    discounted_return,
    go_go_problematic,
    no_unilateral_guarantee,
    normalised_order_gap,
    yield_yield_inefficient,
)
from thesis.certification.choice_state_scenarios import (
    GO_PROFILES,
    YIELD_PROFILES,
    EnvironmentCandidate,
    InitialConditionBlock,
    MatrixCell,
    cell_kinds,
    expand_label_assignments,
    least_intervention_profile,
    macro_action_sequence,
    materialize_block_for_geometry,
)
from thesis.envs.final_environment_config import TimingConfig
from thesis.envs.merge_env_candidate_v3 import MergeEnvCandidateV3, MergeEnvCandidateV3Config


def _profile_pairs(cell: MatrixCell) -> list[tuple[Any, Any]]:
    """Preregistered profile combinations; least-intervention order first."""
    gos = sorted(GO_PROFILES, key=lambda p: (p.n_steps, p.absolute_acceleration, p.profile_id))
    ylds = sorted(YIELD_PROFILES, key=lambda p: (p.n_steps, p.absolute_acceleration, p.profile_id))
    if cell == "GO_GO":
        return [(g1, g2) for g1 in gos for g2 in gos]
    if cell == "YIELD_YIELD":
        return [(y1, y2) for y1 in ylds for y2 in ylds]
    if cell == "GO_YIELD":
        return [(g, y) for g in gos for y in ylds]
    if cell == "YIELD_GO":
        return [(g, y) for g in gos for y in ylds]
    return [(gos[0], ylds[0])]


def _run_cell(
    *,
    candidate: EnvironmentCandidate,
    block: InitialConditionBlock,
    cell: MatrixCell,
    total_steps: int = 240,
) -> tuple[CellOutcome, list[dict[str, Any]]]:
    ml_kind, rp_kind = cell_kinds(cell)
    best: CellOutcome | None = None
    best_trace: list[dict[str, Any]] = []

    for p_ml, p_rp in _profile_pairs(cell):
        # For GO/GO and YIELD/YIELD both slots are same-kind profiles.
        if cell == "GO_GO":
            go_ml, go_rp = p_ml, p_rp
            acts = macro_action_sequence(
                "GO",
                "GO",
                go=go_ml,
                yield_p=YIELD_PROFILES[0],
                total_steps=total_steps,
                role_A=block.role_A,
                go_ramp=go_rp,
            )
            g_prof, y_prof = go_ml, go_rp
        elif cell == "YIELD_YIELD":
            y_ml, y_rp = p_ml, p_rp
            acts = macro_action_sequence(
                "YIELD",
                "YIELD",
                go=GO_PROFILES[0],
                yield_p=y_ml,
                total_steps=total_steps,
                role_A=block.role_A,
                yield_ramp=y_rp,
            )
            g_prof, y_prof = y_ml, y_rp
        else:
            g_prof, y_prof = p_ml, p_rp
            acts = macro_action_sequence(
                ml_kind,
                rp_kind,
                go=g_prof,
                yield_p=y_prof,
                total_steps=total_steps,
                role_A=block.role_A,
            )
        cfg = MergeEnvCandidateV3Config(candidate=candidate, block=block, timing=TimingConfig())
        env = MergeEnvCandidateV3(cfg)
        env.reset(seed=block.seed)
        rewards_A: list[float] = []
        rewards_B: list[float] = []
        min_gap = math.inf
        min_ttc = math.inf
        min_acc = math.inf
        max_acc = -math.inf
        disc_count = 0
        nan_count = 0
        invalid = 0
        fixture = 0
        bg_min_speed = {"B_front": math.inf, "B_rear": math.inf}
        bg_max_speed = {"B_front": -math.inf, "B_rear": -math.inf}
        bg_max_brake = {"B_front": 0.0, "B_rear": 0.0}
        bg_min_gap = math.inf
        trace: list[dict[str, Any]] = []
        term = trunc = False
        term_reason = "ongoing"
        info: dict[str, Any] = {}
        prev_acc: dict[str, float] = {"A": 0.0, "B": 0.0}

        for a in acts:
            _obs, rew, term, trunc, info = env.step(a)
            if term and trunc:
                invalid += 1
            if info.get("fixture_only"):
                fixture += 1
            if info.get("route_discontinuity"):
                disc_count += 1
            rewards_A.append(float(rew["A"]))
            rewards_B.append(float(rew["B"]))
            for aid in ("A", "B"):
                acc = float(info["vehicles_t1"][aid]["acceleration"])
                min_acc = min(min_acc, acc)
                max_acc = max(max_acc, acc)
                for v in info["components"][aid].values():
                    if isinstance(v, (int, float)) and not math.isfinite(float(v)):
                        nan_count += 1
                jerk = acc - prev_acc[aid]
                prev_acc[aid] = acc
            g = info.get("min_bumper_gap")
            if g is not None:
                min_gap = min(min_gap, float(g))
            ttc = info.get("ttc")
            if ttc is not None:
                min_ttc = min(min_ttc, float(ttc))
            for bid in ("B_front", "B_rear"):
                sp = float(info["vehicles_t1"][bid]["speed"])
                accb = float(info["vehicles_t1"][bid]["acceleration"])
                bg_min_speed[bid] = min(bg_min_speed[bid], sp)
                bg_max_speed[bid] = max(bg_max_speed[bid], sp)
                bg_max_brake[bid] = max(bg_max_brake[bid], max(0.0, -accb))
            for bid in ("B_front", "B_rear"):
                for aid in ("A", "B"):
                    xa = info["vehicles_t1"][aid]["world_x"]
                    xb = info["vehicles_t1"][bid]["world_x"]
                    bg_min_gap = min(bg_min_gap, abs(float(xa) - float(xb)))

            for aid in ("A", "B"):
                acc = float(info["vehicles_t1"][aid]["acceleration"])
                jerk_val = acc - (prev_acc[aid] if False else acc)  # set below
                jerk_val = None
                if info["policy_step"] > 1:
                    # approximate from stored prev before overwrite — recompute
                    pass
                trace.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "block_id": block.block_id,
                        "block_set": block.block_set,
                        "label_assignment": f"A={block.role_A},B={block.role_B}",
                        "matrix_cell": cell,
                        "selected_macro_profiles": {
                            "mainline": g_prof.profile_id if ml_kind == "GO" else y_prof.profile_id,
                            "ramp": g_prof.profile_id if rp_kind == "GO" else y_prof.profile_id,
                        },
                        "policy_step": info["policy_step"],
                        "physics_substep": info["physics_substeps"] - 1,
                        "controller_id": aid,
                        "traffic_role": info["vehicles_t1"][aid]["role"],
                        "action": int(a[aid]),
                        "world_position": info["vehicles_t1"][aid]["world_x"],
                        "route_position": info["vehicles_t1"][aid]["route_position"],
                        "speed": info["vehicles_t1"][aid]["speed"],
                        "realised_acceleration": info["vehicles_t1"][aid]["acceleration"],
                        "jerk": None,
                        "bumper_gap": info.get("min_bumper_gap"),
                        "ttc": info.get("ttc"),
                        "completed": info["completion"][aid],
                        "exit_event": info["events"]["exit_event"][aid],
                        "collision_registry": info["events"]["stakeholder_collided"],
                        "collision_pairs": info["events"]["collision_pairs"],
                        "terminated": term,
                        "truncated": trunc,
                        "progress_component": info["components"][aid]["progress_component"],
                        "exit_component": info["components"][aid]["exit_component"],
                        "collision_component": info["components"][aid]["collision_component"],
                        "core_reward": info["components"][aid]["core_reward"],
                        "discounted_core_reward": info["discount_factor"]
                        * info["components"][aid]["core_reward"],
                        "route_integrity_flag": bool(info.get("route_discontinuity")),
                        "fixture_flag": False,
                    }
                )
            term_reason = info.get("term_reason", term_reason)
            if term or trunc:
                break

        et = info.get("exit_time", {"A": None, "B": None})
        role_of = {"A": block.role_A, "B": block.role_B}
        t_ml = t_rp = None
        for aid in ("A", "B"):
            if role_of[aid] == "mainline":
                t_ml = et.get(aid)
            else:
                t_rp = et.get(aid)
        order = classify_exit_order(t_ml, t_rp)
        g_a = discounted_return(rewards_A, GAMMA) if rewards_A else 0.0
        g_b = discounted_return(rewards_B, GAMMA) if rewards_B else 0.0
        outcome = CellOutcome(
            cell=cell,
            success=term_reason == "success",
            collision=term_reason == "collision",
            truncated=bool(trunc),
            exit_order=order,
            G_team_core=g_a + g_b,
            G_A_core=g_a,
            G_B_core=g_b,
            episode_length=int(info.get("policy_step", 0)),
            exit_time_mainline=t_ml,
            exit_time_ramp=t_rp,
            min_bumper_gap=None if min_gap is math.inf else float(min_gap),
            min_ttc=None if min_ttc is math.inf else float(min_ttc),
            min_accel=float(min_acc if min_acc < math.inf else 0.0),
            max_accel=float(max_acc if max_acc > -math.inf else 0.0),
            route_discontinuity=disc_count,
            nan_count=nan_count,
            invalid_flags=invalid,
            repeated_exit=max(0, env._exit_count["A"] - 1) + max(0, env._exit_count["B"] - 1),
            fixture_count=fixture,
            bg_min_speed={k: (0.0 if v is math.inf else float(v)) for k, v in bg_min_speed.items()},
            bg_max_brake=bg_max_brake,
            bg_min_gap_to_learners=None if bg_min_gap is math.inf else float(bg_min_gap),
            selected_macros={"GO": g_prof.profile_id, "YIELD": y_prof.profile_id},
        )
        # attach max speed for relevance diagnostics
        outcome.bg_min_speed = {
            **outcome.bg_min_speed,
            **{f"{k}_max": (0.0 if v is -math.inf else float(v)) for k, v in bg_max_speed.items()},
        }
        safe, reasons = cell_physically_safe(outcome)
        outcome.physically_safe = safe
        outcome.rejection_reasons = reasons

        if cell == "GO_YIELD":
            ok = safe and order == "mainline_first"
        elif cell == "YIELD_GO":
            ok = safe and order == "ramp_first"
        elif cell == "YIELD_YIELD":
            # Prefer least-intervention non-colliding YY realisation
            if not outcome.collision:
                best = outcome
                best_trace = trace
                break
            if best is None:
                best = outcome
                best_trace = trace
            continue
        elif cell == "GO_GO":
            # Least-intervention GO/GO only (first pair)
            best = outcome
            best_trace = trace
            break
        else:
            ok = False
        if ok:
            best = outcome
            best_trace = trace
            break
        if best is None:
            best = outcome
            best_trace = trace

    assert best is not None
    return best, best_trace


def certify_block(
    candidate: EnvironmentCandidate,
    block: InitialConditionBlock,
) -> dict[str, Any]:
    """Certify one physical block under both label assignments."""
    matrices: list[dict[str, CellOutcome]] = []
    traces: list[dict[str, Any]] = []
    label_errors: list[float] = []

    physical_results: dict[str, CellOutcome] | None = None
    for assignment in expand_label_assignments(block):
        matrix: dict[str, CellOutcome] = {}
        for cell in ("GO_GO", "GO_YIELD", "YIELD_GO", "YIELD_YIELD"):
            out, tr = _run_cell(candidate=candidate, block=assignment, cell=cell)  # type: ignore[arg-type]
            matrix[cell] = out
            traces.extend(tr)
        matrices.append(matrix)
        if physical_results is None:
            physical_results = matrix
        else:
            for cell, o1 in physical_results.items():
                o2 = matrix[cell]
                label_errors.append(abs(o1.G_team_core - o2.G_team_core))
                label_errors.append(abs((o1.exit_time_mainline or -1) - (o2.exit_time_mainline or -1)))
                label_errors.append(abs((o1.exit_time_ramp or -1) - (o2.exit_time_ramp or -1)))

    assert physical_results is not None
    ml = physical_results["GO_YIELD"]
    rp = physical_results["YIELD_GO"]
    gg = physical_results["GO_GO"]
    yy = physical_results["YIELD_YIELD"]

    reasons: list[str] = []
    if not (ml.physically_safe and ml.exit_order == "mainline_first"):
        reasons.append("mainline_first_convention_failed:" + ",".join(ml.rejection_reasons))
    if not (rp.physically_safe and rp.exit_order == "ramp_first"):
        reasons.append("ramp_first_convention_failed:" + ",".join(rp.rejection_reasons))
    if not go_go_problematic(gg, ml, rp):
        reasons.append("go_go_not_problematic")
    if not yield_yield_inefficient(yy, ml, rp):
        reasons.append("yield_yield_not_inefficient")
    if not no_unilateral_guarantee(physical_results):
        reasons.append("unilateral_guarantee")
    both_conventions = (
        ml.physically_safe
        and ml.exit_order == "mainline_first"
        and rp.physically_safe
        and rp.exit_order == "ramp_first"
    )
    og = normalised_order_gap(ml.G_team_core, rp.G_team_core)
    if both_conventions and og["normalised_order_gap"] > 0.10 + 1e-12:
        reasons.append(f"order_gap={og['normalised_order_gap']:.4f}>0.10")
    if both_conventions and not core_ordering_ok(ml, rp, yy, gg):
        reasons.append("core_ordering_failed")
    max_label_err = max(label_errors) if label_errors else 0.0
    if max_label_err > 1e-12:
        reasons.append(f"label_swap_err={max_label_err}")
    bg_rel = background_meaningful(ml, rp)
    integrity = (
        ml.route_discontinuity
        + rp.route_discontinuity
        + ml.nan_count
        + rp.nan_count
        + ml.invalid_flags
        + rp.invalid_flags
        + ml.fixture_count
        + rp.fixture_count
        + ml.repeated_exit
        + rp.repeated_exit
    )
    if integrity:
        reasons.append(f"integrity={integrity}")

    # Safe conventions must not cause background collision (already in collision flag)
    if ml.collision or rp.collision:
        # already covered by convention failed
        pass

    certified = len(reasons) == 0
    return {
        "block_id": block.block_id,
        "block_set": block.block_set,
        "arrival_category": block.arrival_category,
        "certified": certified,
        "rejection_reasons": reasons,
        "matrix": {k: v.to_dict() for k, v in physical_results.items()},
        "normalised_order_gap": og["normalised_order_gap"],
        "background_relevant": bg_rel,
        "label_swap_max_error": max_label_err,
        "traces": traces,
    }


def run_background_safety_audit(
    candidate: EnvironmentCandidate,
    blocks: list[InitialConditionBlock],
) -> dict[str, Any]:
    """Background-only / conservative / safe-convention scripts; count spontaneous collisions."""
    spontaneous = 0
    details: list[dict[str, Any]] = []
    for block0 in blocks:
        block = materialize_block_for_geometry(block0, candidate.geometry)
        for assignment in expand_label_assignments(block):
            # Passiveive scripts: only pure background–background collisions are spontaneous
            for script_name, acts in (
                ("background_only_maintain", [{"A": 0, "B": 0}] * 200),
                (
                    "conservative_yield_both",
                    [{"A": 2, "B": 2}] * 8 + [{"A": 0, "B": 0}] * 192,
                ),
            ):
                cfg = MergeEnvCandidateV3Config(
                    candidate=candidate, block=assignment, timing=TimingConfig()
                )
                env = MergeEnvCandidateV3(cfg)
                env.reset(seed=assignment.seed)
                for a in acts:
                    _o, _r, term, trunc, info = env.step(a)
                    pairs = [set(p) for p in info["events"]["collision_pairs"]]
                    if pairs and all(p.issubset({"B_front", "B_rear"}) for p in pairs):
                        spontaneous += 1
                        details.append(
                            {
                                "block_id": assignment.block_id,
                                "label": f"A={assignment.role_A}",
                                "script": script_name,
                                "pairs": [sorted(p) for p in pairs],
                            }
                        )
                        break
                    if term or trunc:
                        break
            # Safe conventions: use the same profile search as certification
            for cell, script_name in (
                ("GO_YIELD", "safe_mainline_first"),
                ("YIELD_GO", "safe_ramp_first"),
            ):
                out, _ = _run_cell(candidate=candidate, block=assignment, cell=cell)  # type: ignore[arg-type]
                if out.collision:
                    spontaneous += 1
                    details.append(
                        {
                            "block_id": assignment.block_id,
                            "label": f"A={assignment.role_A}",
                            "script": script_name,
                            "pairs": "learner_collision",
                        }
                    )
    return {
        "spontaneous_collision_count": spontaneous,
        "details": details,
    }


__all__ = ["certify_block", "run_background_safety_audit"]
