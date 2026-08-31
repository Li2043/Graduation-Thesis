"""Joint comfort calibration on the locked Stage 4A-R1 environment (Stage 3B-R1)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from thesis.audits.audit_metrics import median, normalised_order_gap
from thesis.calibration.policy_acceleration import braking_magnitude, policy_braking_acceleration
from thesis.certification.choice_state_certification import _profile_pairs
from thesis.certification.choice_state_scenarios import (
    GO_PROFILES,
    YIELD_PROFILES,
    EnvironmentCandidate,
    InitialConditionBlock,
    MatrixCell,
    cell_kinds,
    expand_label_assignments,
    macro_action_sequence,
)
from thesis.envs.final_environment_config import TimingConfig
from thesis.envs.merge_env_candidate_v3 import HighLevelAction, MergeEnvCandidateV3, MergeEnvCandidateV3Config
from thesis.rewards.base_reward_v2 import compute_hard_braking_cost

GAMMA = 0.995
TIE = 1e-12
HARD_BRAKE_MAG_MIN = 2.5

A_COMFORT_GRID = (1.5, 2.0, 2.5)
A_HARD_GRID = (3.0, 3.5, 4.0, 5.0, 6.0)
ETA_GRID = tuple(round(0.0050 + i * 0.0025, 4) for i in range(19))


def _is_active_reward_transition(t: dict[str, Any]) -> bool:
    return (
        bool(t.get("active_on_road"))
        and not bool(t.get("fixture_flag"))
        and bool(t.get("finite", True))
        and not bool(t.get("invalid_term_trunc"))
    )


def valid_threshold_pair_r1(a_comfort: float, a_hard: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not (a_hard > a_comfort):
        reasons.append("a_hard_not_greater_than_a_comfort")
    if (a_hard - a_comfort) < 1.0 - TIE:
        reasons.append("a_hard_minus_a_comfort_lt_1.0")
    return len(reasons) == 0, reasons


def build_threshold_pairs() -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for ac in A_COMFORT_GRID:
        for ah in A_HARD_GRID:
            ok, _ = valid_threshold_pair_r1(ac, ah)
            if ok:
                pairs.append((ac, ah))
    return pairs


def build_complete_tuples() -> list[tuple[float, float, float]]:
    return [(ac, ah, eta) for ac, ah in build_threshold_pairs() for eta in ETA_GRID]


@dataclass
class TraceBundle:
    transitions: list[dict[str, Any]] = field(default_factory=list)
    substep_rows: list[dict[str, Any]] = field(default_factory=list)
    episode_rows: list[dict[str, Any]] = field(default_factory=list)
    hard_windows: list[dict[str, Any]] = field(default_factory=list)
    integrity: dict[str, int] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)


def _physical_safety_flags(outcome_like: dict[str, Any]) -> dict[str, Any]:
    """Lightweight GO_GO / cell physical classification from episode aggregates."""
    reasons: list[str] = []
    collision = bool(outcome_like.get("collision"))
    truncated = bool(outcome_like.get("truncated"))
    success = bool(outcome_like.get("success"))
    gap = outcome_like.get("min_bumper_gap")
    ttc = outcome_like.get("min_ttc")
    if collision:
        reasons.append("collision")
    if gap is not None and float(gap) < 2.0 - TIE:
        reasons.append("min_gap")
    if ttc is not None and float(ttc) < 1.0 - TIE:
        reasons.append("ttc")
    if truncated:
        reasons.append("truncated")
    if not success:
        reasons.append("not_success")
    unsafe = len(reasons) > 0
    return {
        "physically_unsafe": unsafe,
        "physical_safety_reasons": reasons,
        "collision": collision,
        "truncated": truncated,
        "success": success,
    }


def _run_cell_with_substeps(
    *,
    candidate: EnvironmentCandidate,
    block: InitialConditionBlock,
    cell: MatrixCell,
    lock_hash: str,
    run_id: str,
    total_steps: int = 240,
    comfort: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Same macro selection as Stage 4A-R1 certification, with substep accel capture."""
    from thesis.rewards.base_reward_v2 import BaseRewardConfig

    comfort_cfg: BaseRewardConfig | None = None
    if comfort is not None:
        if isinstance(comfort, BaseRewardConfig):
            comfort_cfg = comfort
        else:
            comfort_cfg = BaseRewardConfig(
                a_comfort=float(comfort["a_comfort"]),
                a_hard=float(comfort["a_hard"]),
                eta_hard_brake=float(comfort["eta_H"]),
            )
    ml_kind, rp_kind = cell_kinds(cell)
    best_meta: dict[str, Any] | None = None
    best_trans: list[dict[str, Any]] = []
    best_subs: list[dict[str, Any]] = []

    def gogo_score(meta: dict[str, Any]) -> int:
        if meta["collision"]:
            return 0
        if meta.get("min_bumper_gap") is not None and meta["min_bumper_gap"] < 2.0:
            return 1
        if meta.get("min_ttc") is not None and meta["min_ttc"] < 1.0:
            return 2
        if meta["truncated"] or not meta["success"]:
            return 3
        return 4

    for p_ml, p_rp in _profile_pairs(cell):
        if cell == "GO_GO":
            acts = macro_action_sequence(
                "GO",
                "GO",
                go=p_ml,
                yield_p=YIELD_PROFILES[0],
                total_steps=total_steps,
                role_A=block.role_A,
                go_ramp=p_rp,
            )
            g_prof, y_prof = p_ml, p_rp
        elif cell == "YIELD_YIELD":
            acts = macro_action_sequence(
                "YIELD",
                "YIELD",
                go=GO_PROFILES[0],
                yield_p=p_ml,
                total_steps=total_steps,
                role_A=block.role_A,
                yield_ramp=p_rp,
            )
            g_prof, y_prof = p_ml, p_rp
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

        cfg = MergeEnvCandidateV3Config(
            candidate=candidate, block=block, timing=TimingConfig(), comfort=comfort_cfg
        )
        env = MergeEnvCandidateV3(cfg)
        env.reset(seed=block.seed)
        transitions: list[dict[str, Any]] = []
        sub_rows: list[dict[str, Any]] = []
        min_gap = math.inf
        min_ttc = math.inf
        term = trunc = False
        term_reason = "ongoing"
        info: dict[str, Any] = {}
        label = f"A={block.role_A},B={block.role_B}"

        for a in acts:
            snap_active = {
                aid: bool(env._vehicles[aid].active_on_road) for aid in ("A", "B")
            }
            _obs, rew, term, trunc, info = env.step(a)
            g = info.get("min_bumper_gap")
            if g is not None:
                min_gap = min(min_gap, float(g))
            ttc = info.get("ttc")
            if ttc is not None:
                min_ttc = min(min_ttc, float(ttc))

            sub_recs = info.get("substep_records") or []
            for aid in ("A", "B"):
                sub_accels: list[float | None] = []
                for rec in sub_recs:
                    veh = rec["vehicles"][aid]
                    # Active during substep if still on road at that substep snapshot
                    if not bool(veh.get("active_on_road", True)) and bool(veh.get("completed", False)):
                        # Exited earlier: placeholder
                        if not snap_active[aid]:
                            sub_accels.append(None)
                        else:
                            # Was active at transition start; include realised until exit
                            sub_accels.append(float(veh["realised_acceleration"]))
                    elif not snap_active[aid]:
                        sub_accels.append(None)
                    else:
                        sub_accels.append(float(veh["realised_acceleration"]))
                    sub_rows.append(
                        {
                            "run_id": run_id,
                            "block_set": block.block_set,
                            "block_id": block.block_id,
                            "label_assignment": label,
                            "matrix_cell": cell,
                            "controller_id": aid,
                            "traffic_role": info["vehicles_t1"][aid]["role"],
                            "policy_step": info["policy_step"],
                            "physics_substep": rec["physics_substep"],
                            "realised_acceleration": float(veh["realised_acceleration"]),
                            "commanded_acceleration": veh.get("commanded_acceleration"),
                            "active_on_road": bool(veh.get("active_on_road")),
                            "completed": bool(veh.get("completed")),
                            "environment_lock_hash": lock_hash,
                        }
                    )
                a_policy = policy_braking_acceleration(sub_accels)
                active = bool(snap_active[aid]) and not bool(info["vehicles_t"][aid].get("completed"))
                # Prefer transition-start active flag
                active = bool(snap_active[aid])
                completed_now = bool(info["completion"][aid])
                fixture = bool(info.get("fixture_only"))
                invalid = bool(term and trunc)
                finite_ok = all(
                    math.isfinite(float(v))
                    for v in (
                        rew[aid],
                        info["components"][aid]["progress_component"],
                        a_policy,
                    )
                )
                transitions.append(
                    {
                        "run_id": run_id,
                        "block_set": block.block_set,
                        "block_id": block.block_id,
                        "label_assignment": label,
                        "matrix_cell": cell,
                        "controller_id": aid,
                        "traffic_role": info["vehicles_t1"][aid]["role"],
                        "policy_step": info["policy_step"],
                        "physics_substep_accelerations": [
                            None if x is None else float(x) for x in sub_accels
                        ],
                        "policy_level_acceleration": a_policy,
                        "commanded_action": int(a[aid]),
                        "commanded_acceleration": info["vehicles_t1"][aid].get(
                            "commanded_acceleration"
                        ),
                        "active_on_road": active,
                        "completed": completed_now,
                        "progress_component": float(info["components"][aid]["progress_component"]),
                        "exit_component": float(info["components"][aid]["exit_component"]),
                        "collision_component": float(info["components"][aid]["collision_component"]),
                        "core_reward": float(info["components"][aid]["core_reward"]),
                        "hard_braking_component": float(
                            info["components"][aid].get("hard_braking_component", 0.0)
                        ),
                        "hard_braking_cost": float(
                            info["components"][aid].get("hard_braking_cost", 0.0)
                        ),
                        "total_base_reward": float(
                            info["components"][aid].get(
                                "total_base_reward", info["components"][aid]["core_reward"]
                            )
                        ),
                        "delta_rho": float(info["components"][aid]["delta_rho"]),
                        "gamma": float(info["discount_factor"]),
                        "terminated": bool(term),
                        "truncated": bool(trunc),
                        "term_reason": info.get("term_reason"),
                        "fixture_flag": fixture,
                        "invalid_term_trunc": invalid,
                        "finite": finite_ok,
                        "route_discontinuity": bool(info.get("route_discontinuity")),
                        "nan_count": int(info.get("nan_count") or 0),
                        "exit_order_hint": None,
                        "environment_lock_hash": lock_hash,
                        "selected_macros": {
                            "mainline": g_prof.profile_id if ml_kind == "GO" else y_prof.profile_id,
                            "ramp": g_prof.profile_id if rp_kind == "GO" else y_prof.profile_id,
                        },
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
        if t_ml is None or t_rp is None:
            order = "partial"
        elif t_ml == t_rp:
            order = "simultaneous"
        else:
            order = "mainline_first" if t_ml < t_rp else "ramp_first"

        meta = {
            "success": term_reason == "success",
            "collision": term_reason == "collision",
            "truncated": bool(trunc),
            "exit_order": order,
            "min_bumper_gap": None if min_gap is math.inf else float(min_gap),
            "min_ttc": None if min_ttc is math.inf else float(min_ttc),
            "selected_macros": {
                "GO": g_prof.profile_id,
                "YIELD": y_prof.profile_id,
            },
            "G_team_core": float(
                sum(
                    float(t["gamma"]) * float(t["core_reward"])
                    for t in transitions
                    if t["controller_id"] in ("A", "B") and _is_active_reward_transition(t)
                )
            ),
            "episode_length": int(info.get("policy_step", 0)),
        }
        meta.update(_physical_safety_flags(meta))

        accept = False
        if cell == "GO_YIELD":
            accept = meta["success"] and not meta["collision"] and order == "mainline_first"
        elif cell == "YIELD_GO":
            accept = meta["success"] and not meta["collision"] and order == "ramp_first"
        elif cell == "YIELD_YIELD":
            accept = not meta["collision"]
        elif cell == "GO_GO":
            if best_meta is None or gogo_score(meta) < gogo_score(best_meta):
                best_meta, best_trans, best_subs = meta, transitions, sub_rows
            if gogo_score(meta) == 0 and g_prof.profile_id == "GO_1" and y_prof.profile_id == "GO_1":
                break
            continue
        if accept:
            best_meta, best_trans, best_subs = meta, transitions, sub_rows
            break
        if best_meta is None:
            best_meta, best_trans, best_subs = meta, transitions, sub_rows

    assert best_meta is not None
    for t in best_trans:
        t["physical_safety_classification"] = best_meta
        t["exit_order"] = best_meta["exit_order"]
    return best_meta, best_trans, best_subs


def generate_immutable_traces(
    *,
    candidate: EnvironmentCandidate,
    calibration_blocks: Sequence[InitialConditionBlock],
    validation_blocks: Sequence[InitialConditionBlock],
    lock_hash: str,
    run_id: str,
) -> TraceBundle:
    """Generate all physical traces once before offline candidate evaluation."""
    bundle = TraceBundle()
    integrity = {
        "route_discontinuity_count": 0,
        "repeated_exit_count": 0,
        "invalid_flag_count": 0,
        "fixture_count": 0,
        "nan_inf_count": 0,
        "missing_substep_acceleration_count": 0,
    }
    included = excluded = 0
    hard_total = 0
    hard_blocks_cal: set[str] = set()

    for block in list(calibration_blocks) + list(validation_blocks):
        for assignment in expand_label_assignments(block):
            for cell in ("GO_GO", "GO_YIELD", "YIELD_GO", "YIELD_YIELD"):
                meta, trans, subs = _run_cell_with_substeps(
                    candidate=candidate,
                    block=assignment,
                    cell=cell,  # type: ignore[arg-type]
                    lock_hash=lock_hash,
                    run_id=run_id,
                )
                label = f"A={assignment.role_A},B={assignment.role_B}"
                bundle.episode_rows.append(
                    {
                        "run_id": run_id,
                        "block_set": block.block_set,
                        "block_id": block.block_id,
                        "label_assignment": label,
                        "matrix_cell": cell,
                        **meta,
                        "environment_lock_hash": lock_hash,
                    }
                )
                for t in trans:
                    if t["route_discontinuity"]:
                        integrity["route_discontinuity_count"] += 1
                    if t["invalid_term_trunc"]:
                        integrity["invalid_flag_count"] += 1
                    if t["fixture_flag"]:
                        integrity["fixture_count"] += 1
                    if not t["finite"] or t["nan_count"]:
                        integrity["nan_inf_count"] += 1
                    n_sub = len(t["physics_substep_accelerations"])
                    if t["active_on_road"] and n_sub != 4:
                        # Allow early collision break with fewer substeps
                        if n_sub == 0:
                            integrity["missing_substep_acceleration_count"] += 1
                    # Inclusion rule for nominal-safe later; count all active here
                    if t["active_on_road"] and not t["fixture_flag"] and t["finite"]:
                        included += 1
                    else:
                        excluded += 1
                    # Hard windows: safe conventions only, designated later after filter
                    bundle.transitions.append(t)
                bundle.substep_rows.extend(subs)

    # Designate hard windows on safe-convention episodes
    safe_eps = {
        (e["block_id"], e["label_assignment"], e["matrix_cell"])
        for e in bundle.episode_rows
        if e["matrix_cell"] in ("GO_YIELD", "YIELD_GO")
        and e["success"]
        and not e["collision"]
        and e["exit_order"]
        in (
            "mainline_first" if e["matrix_cell"] == "GO_YIELD" else "ramp_first",
        )
    }
    # Fix exit order check properly
    safe_eps = set()
    for e in bundle.episode_rows:
        if e["matrix_cell"] == "GO_YIELD" and e["success"] and not e["collision"] and e["exit_order"] == "mainline_first":
            safe_eps.add((e["block_id"], e["label_assignment"], e["matrix_cell"]))
        if e["matrix_cell"] == "YIELD_GO" and e["success"] and not e["collision"] and e["exit_order"] == "ramp_first":
            safe_eps.add((e["block_id"], e["label_assignment"], e["matrix_cell"]))

    for t in bundle.transitions:
        key = (t["block_id"], t["label_assignment"], t["matrix_cell"])
        if key not in safe_eps:
            continue
        if not t["active_on_road"] or t["fixture_flag"] or not t["finite"]:
            continue
        if int(t["commanded_action"]) != int(HighLevelAction.DECELERATE):
            continue
        mag = braking_magnitude(float(t["policy_level_acceleration"]))
        if mag + TIE < HARD_BRAKE_MAG_MIN:
            continue
        hw = {
            **{k: t[k] for k in (
                "run_id", "block_set", "block_id", "label_assignment", "matrix_cell",
                "controller_id", "policy_step", "policy_level_acceleration", "commanded_action",
                "environment_lock_hash",
            )},
            "braking_magnitude": mag,
            "designated_hard_braking_window": True,
        }
        bundle.hard_windows.append(hw)
        t["designated_hard_braking_window"] = True
        hard_total += 1
        if t["block_set"] == "calibration":
            hard_blocks_cal.add(t["block_id"])

    for t in bundle.transitions:
        t.setdefault("designated_hard_braking_window", False)

    bundle.integrity = integrity
    bundle.counts = {
        "included_transitions": included,
        "excluded_transitions": excluded,
        "hard_window_transitions": hard_total,
        "hard_window_calibration_blocks": len(hard_blocks_cal),
        "n_episodes": len(bundle.episode_rows),
        "n_transitions": len(bundle.transitions),
    }
    return bundle


def assert_hard_window_coverage(bundle: TraceBundle) -> None:
    """BLOCKED if hard-window coverage is insufficient."""
    n_blocks = int(bundle.counts.get("hard_window_calibration_blocks", 0))
    n_hw = int(bundle.counts.get("hard_window_transitions", 0))
    if n_blocks < 10 or n_hw < 20:
        raise RuntimeError(
            f"BLOCKED: hard-window coverage failed "
            f"(calibration_blocks_with_hw={n_blocks}<10 or total_hw={n_hw}<20)"
        )


def episode_braking_shares(
    transitions: Sequence[dict[str, Any]],
    *,
    a_comfort: float,
    a_hard: float,
    eta_h: float,
) -> dict[str, float]:
    """Per-learner braking shares for one episode; team = mean of available."""
    by: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
    for t in transitions:
        if not _is_active_reward_transition(t):
            continue
        by[t["controller_id"]].append(t)
    shares: list[float] = []
    out: dict[str, float] = {}
    for aid, rows in by.items():
        if not rows:
            continue
        B = 0.0
        D = 0.0
        for t in rows:
            g = float(t["gamma"])
            H = compute_hard_braking_cost(
                float(t["policy_level_acceleration"]), a_comfort, a_hard
            )
            b = g * eta_h * H
            d = g * (
                abs(float(t["progress_component"]))
                + abs(float(t["exit_component"]))
                + abs(float(t["collision_component"]))
                + eta_h * H
            )
            B += b
            D += d
        s = B / max(D, 1e-12)
        out[aid] = s
        shares.append(s)
    out["team"] = float(sum(shares) / len(shares)) if shares else float("nan")
    return out


def comfort_adjusted_team_return(
    transitions: Sequence[dict[str, Any]],
    *,
    a_comfort: float,
    a_hard: float,
    eta_h: float,
) -> float:
    total = 0.0
    for t in transitions:
        if t["controller_id"] not in ("A", "B"):
            continue
        if not _is_active_reward_transition(t):
            continue
        H = compute_hard_braking_cost(
            float(t["policy_level_acceleration"]), a_comfort, a_hard
        )
        total += float(t["gamma"]) * (float(t["core_reward"]) - eta_h * H)
    return float(total)


def evaluate_tuple(
    bundle: TraceBundle,
    *,
    a_comfort: float,
    a_hard: float,
    eta_h: float,
    block_set: str = "calibration",
) -> dict[str, Any]:
    """Offline evaluation of one complete (a_comfort, a_hard, eta) tuple."""
    reasons: list[str] = []
    eps = [
        e
        for e in bundle.episode_rows
        if e["block_set"] == block_set
    ]
    trans = [t for t in bundle.transitions if t["block_set"] == block_set]

    # Safe-convention episodes (primary label A=mainline for metrics aggregation)
    safe_cells = ("GO_YIELD", "YIELD_GO")
    safe_eps = [
        e
        for e in eps
        if e["matrix_cell"] in safe_cells
        and e["success"]
        and not e["collision"]
        and (
            (e["matrix_cell"] == "GO_YIELD" and e["exit_order"] == "mainline_first")
            or (e["matrix_cell"] == "YIELD_GO" and e["exit_order"] == "ramp_first")
        )
        and e["label_assignment"].startswith("A=mainline")
    ]
    safe_keys = {(e["block_id"], e["label_assignment"], e["matrix_cell"]) for e in safe_eps}

    nominal_H: list[float] = []
    hard_H: list[float] = []
    nominal_shares: list[float] = []
    paired_diffs: list[float] = []
    block_h_ok = 0
    cal_block_ids = sorted({e["block_id"] for e in eps if e["block_set"] == block_set})

    per_block: list[dict[str, Any]] = []
    order_gaps: list[float] = []
    ordering_violations = 0

    for bid in cal_block_ids:
        block_nom_H: list[float] = []
        block_hard_H: list[float] = []
        # Use A=mainline assignment for block-level metrics
        label = "A=mainline,B=ramp"
        gy = next(
            (
                e
                for e in eps
                if e["block_id"] == bid
                and e["label_assignment"] == label
                and e["matrix_cell"] == "GO_YIELD"
            ),
            None,
        )
        yg = next(
            (
                e
                for e in eps
                if e["block_id"] == bid
                and e["label_assignment"] == label
                and e["matrix_cell"] == "YIELD_GO"
            ),
            None,
        )
        yy = next(
            (
                e
                for e in eps
                if e["block_id"] == bid
                and e["label_assignment"] == label
                and e["matrix_cell"] == "YIELD_YIELD"
            ),
            None,
        )
        gg = next(
            (
                e
                for e in eps
                if e["block_id"] == bid
                and e["label_assignment"] == label
                and e["matrix_cell"] == "GO_GO"
            ),
            None,
        )

        usable = (
            gy is not None
            and yg is not None
            and gy["success"]
            and yg["success"]
            and not gy["collision"]
            and not yg["collision"]
            and gy["exit_order"] == "mainline_first"
            and yg["exit_order"] == "ramp_first"
        )

        def ep_trans(cell: str) -> list[dict[str, Any]]:
            return [
                t
                for t in trans
                if t["block_id"] == bid
                and t["label_assignment"] == label
                and t["matrix_cell"] == cell
            ]

        if usable:
            for cell in ("GO_YIELD", "YIELD_GO"):
                rows = [
                    t
                    for t in ep_trans(cell)
                    if _is_active_reward_transition(t)
                ]
                for t in rows:
                    H = compute_hard_braking_cost(
                        float(t["policy_level_acceleration"]), a_comfort, a_hard
                    )
                    nominal_H.append(H)
                    block_nom_H.append(H)
                    if t.get("designated_hard_braking_window"):
                        hard_H.append(H)
                        block_hard_H.append(H)
                sh = episode_braking_shares(
                    ep_trans(cell), a_comfort=a_comfort, a_hard=a_hard, eta_h=eta_h
                )
                if math.isfinite(sh["team"]):
                    nominal_shares.append(sh["team"])
                # hard-window share vs episode share
                hw_rows = [t for t in rows if t.get("designated_hard_braking_window")]
                if hw_rows:
                    # share restricted to hard windows only
                    B = D = 0.0
                    for t in hw_rows:
                        g = float(t["gamma"])
                        H = compute_hard_braking_cost(
                            float(t["policy_level_acceleration"]), a_comfort, a_hard
                        )
                        B += g * eta_h * H
                        dmag = (
                            abs(float(t["progress_component"]))
                            + abs(float(t["exit_component"]))
                            + abs(float(t["collision_component"]))
                            + eta_h * H
                        )
                        D += g * dmag
                    hw_share = B / max(D, 1e-12)
                    if math.isfinite(sh["team"]):
                        diff = hw_share - sh["team"]
                        paired_diffs.append(diff)
                        if not (hw_share > sh["team"] + TIE):
                            reasons.append(f"hard_share_not_gt_nominal:{bid}:{cell}")

            if block_hard_H and block_nom_H:
                if (sum(block_hard_H) / len(block_hard_H)) > (
                    sum(block_nom_H) / len(block_nom_H)
                ) + TIE:
                    block_h_ok += 1

            # Ordering with comfort-adjusted returns
            g_gy = comfort_adjusted_team_return(
                ep_trans("GO_YIELD"), a_comfort=a_comfort, a_hard=a_hard, eta_h=eta_h
            )
            g_yg = comfort_adjusted_team_return(
                ep_trans("YIELD_GO"), a_comfort=a_comfort, a_hard=a_hard, eta_h=eta_h
            )
            g_yy = comfort_adjusted_team_return(
                ep_trans("YIELD_YIELD"), a_comfort=a_comfort, a_hard=a_hard, eta_h=eta_h
            )
            g_gg = comfort_adjusted_team_return(
                ep_trans("GO_GO"), a_comfort=a_comfort, a_hard=a_hard, eta_h=eta_h
            )
            # Core-only returns (eta=0) to detect pre-existing safe-GG dominance
            g_gy0 = comfort_adjusted_team_return(
                ep_trans("GO_YIELD"), a_comfort=a_comfort, a_hard=a_hard, eta_h=0.0
            )
            g_yg0 = comfort_adjusted_team_return(
                ep_trans("YIELD_GO"), a_comfort=a_comfort, a_hard=a_hard, eta_h=0.0
            )
            g_gg0 = comfort_adjusted_team_return(
                ep_trans("GO_GO"), a_comfort=a_comfort, a_hard=a_hard, eta_h=0.0
            )

            yy_safe_or_unresolved = yy is not None and (
                (not yy["collision"]) or (not yy["success"])
            )
            if yy_safe_or_unresolved and yy is not None and not yy["collision"]:
                if not (g_gy > g_yy + TIE and g_yg > g_yy + TIE):
                    ordering_violations += 1
                    reasons.append(f"asymmetric_not_above_yy:{bid}")

            if gg is not None and gg["collision"]:
                if not (g_gy > g_gg + TIE and g_yg > g_gg + TIE):
                    ordering_violations += 1
                    reasons.append(f"asymmetric_not_above_collision_gg:{bid}")
            if gg is not None and gg["truncated"] and not gg["collision"]:
                if not (g_gy > g_gg + TIE and g_yg > g_gg + TIE):
                    ordering_violations += 1
                    reasons.append(f"asymmetric_not_above_trunc_gg:{bid}")

            # Safe GO_GO must not dominate both conventions after comfort — unless
            # core returns already exhibit that dominance (Stage 4A-R1 leaves such
            # blocks uncertified; comfort cannot create braking on pure GO macros).
            if gg is not None and not gg.get("physically_unsafe", True) and gg["success"]:
                core_dom = (
                    g_gg0 > g_gy0 + TIE
                    and g_gg0 > g_yg0 + TIE
                    and int(gg["episode_length"])
                    <= min(int(gy["episode_length"]), int(yg["episode_length"]))
                )
                comfort_dom = (
                    g_gg > g_gy + TIE
                    and g_gg > g_yg + TIE
                    and int(gg["episode_length"])
                    <= min(int(gy["episode_length"]), int(yg["episode_length"]))
                )
                if comfort_dom and not core_dom:
                    ordering_violations += 1
                    reasons.append(f"safe_gogo_dominates:{bid}")

            og = normalised_order_gap(g_gy, g_yg)["normalised_order_gap"]
            order_gaps.append(og)
            if og > 0.10 + TIE:
                reasons.append(f"order_gap>{0.10}:{bid}")

            per_block.append(
                {
                    "block_id": bid,
                    "usable": True,
                    "G_GO_YIELD": g_gy,
                    "G_YIELD_GO": g_yg,
                    "G_YIELD_YIELD": g_yy,
                    "G_GO_GO": g_gg,
                    "order_gap": og,
                    "mean_H_hard": (sum(block_hard_H) / len(block_hard_H)) if block_hard_H else None,
                    "mean_H_nominal": (sum(block_nom_H) / len(block_nom_H)) if block_nom_H else None,
                }
            )
        else:
            per_block.append({"block_id": bid, "usable": False})

    def _rate(xs: list[float], pred) -> float:
        return (sum(1 for x in xs if pred(x)) / len(xs)) if xs else float("nan")

    nom_nonzero = _rate(nominal_H, lambda h: h > 0)
    nom_sat = _rate(nominal_H, lambda h: abs(h - 1.0) <= TIE)
    hard_nonzero = _rate(hard_H, lambda h: h > 0)
    mean_nom = float(sum(nominal_H) / len(nominal_H)) if nominal_H else float("nan")
    mean_hard = float(sum(hard_H) / len(hard_H)) if hard_H else float("nan")
    sep = mean_hard - mean_nom if math.isfinite(mean_hard) and math.isfinite(mean_nom) else float("nan")

    med_share = median(nominal_shares) if nominal_shares else float("nan")
    max_share = max(nominal_shares) if nominal_shares else float("nan")
    med_pair = median(paired_diffs) if paired_diffs else float("nan")
    med_gap = median(order_gaps) if order_gaps else float("nan")
    max_gap = max(order_gaps) if order_gaps else float("nan")

    # Feasibility checks (calibration only uses full criteria when block_set=calibration)
    if block_set == "calibration":
        if nom_nonzero > 0.10 + TIE:
            reasons.append(f"nominal_nonzero_rate={nom_nonzero}")
        if nom_sat > 0.01 + TIE:
            reasons.append(f"nominal_saturation_rate={nom_sat}")
        if not (hard_nonzero >= 0.80 - TIE):
            reasons.append(f"hard_nonzero_rate={hard_nonzero}")
        if not (mean_hard >= 0.20 - TIE):
            reasons.append(f"hard_mean_H={mean_hard}")
        if not (sep >= 0.15 - TIE):
            reasons.append(f"H_separation={sep}")
        if block_h_ok < 10:
            reasons.append(f"blocks_hard_gt_nominal={block_h_ok}<10")
        if not (0.02 - TIE <= med_share <= 0.06 + TIE):
            reasons.append(f"median_nominal_share={med_share}")
        if not (max_share <= 0.10 + TIE):
            reasons.append(f"max_nominal_share={max_share}")
        if any(d <= TIE for d in paired_diffs):
            reasons.append("paired_hard_share_not_strictly_greater")
        if not (med_pair >= 0.02 - TIE):
            reasons.append(f"median_paired_diff={med_pair}")
        if ordering_violations:
            reasons.append(f"ordering_violations={ordering_violations}")
        if not (med_gap <= 0.05 + TIE):
            reasons.append(f"median_order_gap={med_gap}")
        if not (max_gap <= 0.10 + TIE):
            reasons.append(f"max_order_gap={max_gap}")
        for k, v in bundle.integrity.items():
            if int(v) != 0:
                reasons.append(f"integrity_{k}={v}")

    # Deduplicate reasons while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            uniq.append(r)

    feasible = len(uniq) == 0 if block_set == "calibration" else None
    return {
        "a_comfort": a_comfort,
        "a_hard": a_hard,
        "eta_H": eta_h,
        "block_set": block_set,
        "feasible": feasible,
        "rejection_reasons": uniq,
        "nominal_nonzero_rate": nom_nonzero,
        "nominal_saturation_rate": nom_sat,
        "hard_nonzero_rate": hard_nonzero,
        "mean_H_nominal": mean_nom,
        "mean_H_hard": mean_hard,
        "H_separation": sep,
        "blocks_hard_gt_nominal": block_h_ok,
        "median_nominal_share": med_share,
        "max_nominal_share": max_share,
        "median_paired_share_diff": med_pair,
        "n_paired_diffs": len(paired_diffs),
        "ordering_violations": ordering_violations,
        "median_order_gap": med_gap,
        "max_order_gap": max_gap,
        "n_usable_safe_blocks": sum(1 for p in per_block if p.get("usable")),
        "per_block": per_block,
        "selection_key": None,
    }


def select_feasible_tuple(results: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Lexicographic joint selection among feasible complete tuples."""
    feasible = [r for r in results if r.get("feasible") is True]
    if not feasible:
        return None

    def key(r: dict[str, Any]) -> tuple:
        # 1 smallest eta; 2 largest H separation; 3 lowest median share;
        # 4 highest a_comfort; 5 highest a_hard
        return (
            float(r["eta_H"]),
            -float(r["H_separation"]),
            float(r["median_nominal_share"]),
            -float(r["a_comfort"]),
            -float(r["a_hard"]),
        )

    feasible_sorted = sorted(feasible, key=key)
    for i, r in enumerate(feasible_sorted, start=1):
        r["selection_rank"] = i
        r["selection_key"] = list(key(r))
    return feasible_sorted[0]


def run_joint_calibration(bundle: TraceBundle) -> dict[str, Any]:
    """Evaluate all complete tuples offline; select without using validation."""
    pairs = build_threshold_pairs()
    tuples = build_complete_tuples()
    rows: list[dict[str, Any]] = []
    for ac, ah, eta in tuples:
        rows.append(
            evaluate_tuple(bundle, a_comfort=ac, a_hard=ah, eta_h=eta, block_set="calibration")
        )
    selected = select_feasible_tuple(rows)
    return {
        "threshold_pairs": [{"a_comfort": a, "a_hard": b} for a, b in pairs],
        "n_threshold_pairs": len(pairs),
        "n_complete_tuples": len(tuples),
        "tuple_results": rows,
        "n_feasible": sum(1 for r in rows if r.get("feasible")),
        "selected": selected,
        "selection_used_validation": False,
    }


__all__ = [
    "A_COMFORT_GRID",
    "A_HARD_GRID",
    "ETA_GRID",
    "GAMMA",
    "TraceBundle",
    "assert_hard_window_coverage",
    "build_complete_tuples",
    "build_threshold_pairs",
    "comfort_adjusted_team_return",
    "episode_braking_shares",
    "evaluate_tuple",
    "generate_immutable_traces",
    "run_joint_calibration",
    "select_feasible_tuple",
    "valid_threshold_pair_r1",
]
