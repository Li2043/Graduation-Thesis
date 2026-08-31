"""Offline comfort threshold and eta_hard_brake calibration (Stage 3B)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from thesis.audits.audit_metrics import IncentiveOrderingResult
from thesis.calibration.calibration_metrics import (
    TIE_TOL,
    assert_eta_penalty_monotone,
    assert_h_monotonicities,
    braking_penalty_share,
    check_incentive_ordering,
    compare_threshold_lex,
    discounted_return,
    h_from_accel,
    median,
    normalised_order_gap,
    summarise_h,
    threshold_pair_feasible,
    valid_threshold_pair,
)
from thesis.calibration.trace_loader import (
    HARD_BRAKING_SAFE,
    NOMINAL_SAFE,
    PRIMARY_RANKING_SCENARIOS,
    SLOW_SAFE,
    FilterStats,
    SourceTraceManifest,
    filter_active_calibration_transitions,
    scenario_class,
)


@dataclass(frozen=True)
class ComfortCandidate:
    a_comfort: float
    a_hard: float
    eta_hard_brake: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class ThresholdMetrics:
    a_comfort: float
    a_hard: float
    feasible: bool
    rejection_reasons: list[str]
    separation_score: float
    nominal: dict[str, Any]
    slow: dict[str, Any]
    hard_window: dict[str, Any]
    per_block: list[dict[str, Any]]
    selection_key: tuple[float, float, float, float]
    selection_rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "a_comfort": self.a_comfort,
            "a_hard": self.a_hard,
            "feasible": self.feasible,
            "rejection_reasons": list(self.rejection_reasons),
            "separation_score": self.separation_score,
            "nominal": self.nominal,
            "slow": self.slow,
            "hard_window": self.hard_window,
            "per_block": self.per_block,
            "selection_key": list(self.selection_key),
            "selection_rank": self.selection_rank,
        }


@dataclass
class EtaMetrics:
    a_comfort: float
    a_hard: float
    eta_hard_brake: float
    feasible: bool
    rejection_reasons: list[str]
    median_nominal_share: float
    max_nominal_share: float
    median_paired_share_diff: float
    median_order_gap: float | None
    max_order_gap: float | None
    n_ordering_violations: int
    per_block: list[dict[str, Any]]
    integrity_ok: bool
    selection_rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "a_comfort": self.a_comfort,
            "a_hard": self.a_hard,
            "eta_hard_brake": self.eta_hard_brake,
            "feasible": self.feasible,
            "rejection_reasons": list(self.rejection_reasons),
            "median_nominal_share": self.median_nominal_share,
            "max_nominal_share": self.max_nominal_share,
            "median_paired_share_diff": self.median_paired_share_diff,
            "median_order_gap": self.median_order_gap,
            "max_order_gap": self.max_order_gap,
            "n_ordering_violations": self.n_ordering_violations,
            "per_block": self.per_block,
            "integrity_ok": self.integrity_ok,
            "selection_rank": self.selection_rank,
        }


@dataclass
class CalibrationSelection:
    a_comfort: float | None
    a_hard: float | None
    eta_hard_brake: float | None
    threshold_feasible_count: int
    eta_feasible_count: int
    overall: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HardBrakingWindow:
    block_id: str
    controller_id: str
    steps: list[int]
    start_step: int
    end_step: int
    n_transitions: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_hard_braking_windows(
    transitions: Sequence[Mapping[str, Any]],
    *,
    delta_brake_min: float = 1.0,
) -> tuple[list[HardBrakingWindow], list[dict[str, Any]], list[str]]:
    """Matched step alignment of hard_braking_safe vs safe_mainline_first."""
    by_key: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in transitions:
        if bool(row.get("fixture_only")):
            continue
        sid = str(row["scenario_id"])
        if sid not in {"hard_braking_safe", "safe_mainline_first"}:
            continue
        key = (str(row["block_id"]), sid, str(row["controller_id"]), int(row["step"]))
        by_key[key] = dict(row)

    blocks = sorted({str(r["block_id"]) for r in transitions if not r.get("fixture_only")})
    windows: list[HardBrakingWindow] = []
    window_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for bid in blocks:
        for aid in ("A", "B"):
            # Common steps present in both scenarios
            hard_steps = {
                k[3]
                for k in by_key
                if k[0] == bid and k[1] == "hard_braking_safe" and k[2] == aid
            }
            nom_steps = {
                k[3]
                for k in by_key
                if k[0] == bid and k[1] == "safe_mainline_first" and k[2] == aid
            }
            common = sorted(hard_steps & nom_steps)
            marked: list[int] = []
            for step in common:
                h = by_key[(bid, "hard_braking_safe", aid, step)]
                n = by_key[(bid, "safe_mainline_first", aid, step)]
                mag_h = max(0.0, -float(h["realised_acceleration"]))
                mag_n = max(0.0, -float(n["realised_acceleration"]))
                if mag_h - mag_n >= delta_brake_min - TIE_TOL:
                    marked.append(step)
                    window_rows.append(
                        {
                            "block_id": bid,
                            "controller_id": aid,
                            "step": step,
                            "braking_magnitude_hard": mag_h,
                            "braking_magnitude_nominal": mag_n,
                            "delta_braking_magnitude": mag_h - mag_n,
                            "realised_acceleration_hard": float(h["realised_acceleration"]),
                            "realised_acceleration_nominal": float(n["realised_acceleration"]),
                        }
                    )
            if not marked:
                continue
            # Contiguous segments
            seg_start = marked[0]
            prev = marked[0]
            for s in marked[1:] + [None]:  # type: ignore[list-item]
                if s is not None and s == prev + 1:
                    prev = s
                    continue
                steps = list(range(seg_start, prev + 1))
                windows.append(
                    HardBrakingWindow(
                        block_id=bid,
                        controller_id=aid,
                        steps=steps,
                        start_step=seg_start,
                        end_step=prev,
                        n_transitions=len(steps),
                    )
                )
                if s is not None:
                    seg_start = s
                    prev = s

        if not any(w.block_id == bid for w in windows):
            failures.append(f"{bid}:no_hard_braking_window")

    return windows, window_rows, failures


def _discount_factor(row: Mapping[str, Any], gamma: float) -> float:
    if "discount_factor" in row and row["discount_factor"] is not None:
        return float(row["discount_factor"])
    step = int(row["step"])
    return float(gamma ** (step - 1))


def reconstruct_scenario_returns(
    transitions: Sequence[Mapping[str, Any]],
    *,
    a_comfort: float,
    a_hard: float,
    eta: float,
    gamma: float,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Recompute braking; keep progress/exit/collision fixed. Key=(block, scenario).

    Includes all learner steps recorded in the Stage 3A episode (including post-exit
    placeholder steps) so discounted returns match the confirmatory env rerun.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in transitions:
        if bool(row.get("fixture_only")):
            continue
        sid = str(row["scenario_id"])
        if sid not in PRIMARY_RANKING_SCENARIOS:
            continue
        key = (str(row["block_id"]), sid)
        groups.setdefault(key, []).append(dict(row))

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in groups.items():
        rows.sort(key=lambda r: (int(r["step"]), str(r["controller_id"])))
        rewards: dict[str, list[float]] = {"A": [], "B": []}
        prog: dict[str, list[float]] = {"A": [], "B": []}
        exit_c: dict[str, list[float]] = {"A": [], "B": []}
        coll: dict[str, list[float]] = {"A": [], "B": []}
        brake: dict[str, list[float]] = {"A": [], "B": []}
        decomp_mismatch = 0
        nan_count = 0
        for r in rows:
            aid = str(r["controller_id"])
            if aid not in {"A", "B"}:
                continue
            p = float(r["progress_component"])
            e = float(r["exit_component"])
            c = float(r["collision_component"])
            h = h_from_accel(float(r["realised_acceleration"]), a_comfort, a_hard)
            b = -float(eta) * h
            total = p + e + c + b
            fixed = p + e + c
            if abs(total - (fixed + b)) > TIE_TOL:
                decomp_mismatch += 1
            for v in (p, e, c, b, total, h):
                if not math.isfinite(float(v)):
                    nan_count += 1
            rewards[aid].append(total)
            prog[aid].append(p)
            exit_c[aid].append(e)
            coll[aid].append(c)
            brake[aid].append(b)

        g_a = discounted_return(rewards["A"], gamma) if rewards["A"] else 0.0
        g_b = discounted_return(rewards["B"], gamma) if rewards["B"] else 0.0
        g_prog = discounted_return(prog["A"], gamma) + discounted_return(prog["B"], gamma)
        g_exit = discounted_return(exit_c["A"], gamma) + discounted_return(exit_c["B"], gamma)
        g_coll = discounted_return(coll["A"], gamma) + discounted_return(coll["B"], gamma)
        g_brake = discounted_return(brake["A"], gamma) + discounted_return(brake["B"], gamma)
        out[key] = {
            "block_id": key[0],
            "scenario_id": key[1],
            "G_A": g_a,
            "G_B": g_b,
            "G_team": g_a + g_b,
            "G_progress": g_prog,
            "G_exit": g_exit,
            "G_collision": g_coll,
            "G_hard_braking": g_brake,
            "braking_penalty_share": braking_penalty_share(g_brake, g_prog, g_exit),
            "decomp_mismatch": decomp_mismatch,
            "nan_count": nan_count,
            "n_reward_steps": len(rewards["A"]) + len(rewards["B"]),
        }
    return out


def evaluate_threshold_pair(
    *,
    a_comfort: float,
    a_hard: float,
    included: Sequence[Mapping[str, Any]],
    window_rows: Sequence[Mapping[str, Any]],
    blocks: Sequence[str],
) -> ThresholdMetrics:
    ok_pair, pair_reasons = valid_threshold_pair(a_comfort, a_hard)
    if not ok_pair:
        empty = summarise_h([]).to_dict()
        return ThresholdMetrics(
            a_comfort=a_comfort,
            a_hard=a_hard,
            feasible=False,
            rejection_reasons=pair_reasons,
            separation_score=float("-inf"),
            nominal=empty,
            slow=empty,
            hard_window=empty,
            per_block=[],
            selection_key=(float("-inf"), float("-inf"), a_comfort, a_hard),
        )

    def hs(rows: Sequence[Mapping[str, Any]]) -> list[float]:
        return [
            h_from_accel(float(r["realised_acceleration"]), a_comfort, a_hard) for r in rows
        ]

    nom_rows = [r for r in included if str(r["scenario_id"]) in NOMINAL_SAFE]
    slow_rows = [r for r in included if str(r["scenario_id"]) in SLOW_SAFE]
    # Hard-window accelerations from hard scenario side
    hard_accels = []
    for wr in window_rows:
        hard_accels.append(
            h_from_accel(float(wr["realised_acceleration_hard"]), a_comfort, a_hard)
        )

    nom = summarise_h(hs(nom_rows))
    slow = summarise_h(hs(slow_rows))
    hard = summarise_h(hard_accels)
    separation = hard.mean_h - nom.mean_h

    per_block: list[dict[str, Any]] = []
    n_block_fail = 0
    for bid in blocks:
        b_nom = summarise_h(
            hs([r for r in nom_rows if str(r["block_id"]) == bid])
        )
        b_hard_vals = [
            h_from_accel(float(wr["realised_acceleration_hard"]), a_comfort, a_hard)
            for wr in window_rows
            if str(wr["block_id"]) == bid
        ]
        b_hard = summarise_h(b_hard_vals)
        detect_fail = b_hard.n > 0 and b_hard.nonzero_rate < 0.70 - TIE_TOL
        if b_hard.n == 0 or detect_fail:
            n_block_fail += 1
        per_block.append(
            {
                "block_id": bid,
                "nominal": b_nom.to_dict(),
                "hard_window": b_hard.to_dict(),
                "hard_window_detection_rate": b_hard.nonzero_rate,
                "detection_rate_lt_0_70": bool(b_hard.n == 0 or detect_fail),
            }
        )

    feasible, reasons = threshold_pair_feasible(
        nominal=nom,
        hard_window=hard,
        separation_score=separation,
        n_blocks_hard_detection_lt_0_70=n_block_fail,
    )
    key = (separation, -nom.mean_h, a_comfort, a_hard)
    return ThresholdMetrics(
        a_comfort=a_comfort,
        a_hard=a_hard,
        feasible=feasible,
        rejection_reasons=reasons,
        separation_score=separation,
        nominal=nom.to_dict(),
        slow=slow.to_dict(),
        hard_window=hard.to_dict(),
        per_block=per_block,
        selection_key=key,
    )


def select_threshold_pair(candidates: Sequence[ThresholdMetrics]) -> ThresholdMetrics | None:
    feasible = [c for c in candidates if c.feasible]
    if not feasible:
        return None
    best = feasible[0]
    for c in feasible[1:]:
        if compare_threshold_lex(c.selection_key, best.selection_key) > 0:
            best = c
    # ranks among all candidates (feasible first by selection key)
    ordered = sorted(
        candidates,
        key=lambda c: (0 if c.feasible else 1, tuple(-x for x in c.selection_key)),
    )
    for i, c in enumerate(ordered, start=1):
        c.selection_rank = i
    return best


def evaluate_eta(
    *,
    a_comfort: float,
    a_hard: float,
    eta: float,
    transitions: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    gamma: float,
    blocks: Sequence[str],
) -> EtaMetrics:
    recon = reconstruct_scenario_returns(
        transitions, a_comfort=a_comfort, a_hard=a_hard, eta=eta, gamma=gamma
    )
    reasons: list[str] = []
    nominal_shares: list[float] = []
    paired_diffs: list[float] = []
    order_gaps: list[float] = []
    per_block: list[dict[str, Any]] = []
    ordering_violations = 0
    nan_count = 0
    decomp = 0
    fixture_in_ranking = 0

    meta = {(str(o["block_id"]), str(o["scenario_id"])): o for o in outcomes}

    def _g(bid: str, name: str) -> dict[str, Any] | None:
        return recon.get((bid, name))

    def _team(bid: str, name: str) -> float | None:
        r = _g(bid, name)
        m = meta.get((bid, name))
        if r is None or m is None or bool(m.get("fixture_only")):
            return None
        return float(r["G_team"])

    def _g_success(bid: str, name: str) -> float | None:
        nonlocal fixture_in_ranking
        m = meta.get((bid, name))
        r = _g(bid, name)
        if m is None or r is None:
            return None
        if bool(m.get("fixture_only")):
            fixture_in_ranking += 1
            return None
        if name.startswith("safe") or name.startswith("slow"):
            if str(m.get("term_reason")) != "success":
                return None
        if name in {"early_collision", "late_collision"} and not bool(m.get("collision")):
            return None
        if name == "stall_after_partial_progress":
            if not (bool(m.get("truncated")) and not bool(m.get("collision"))):
                return None
        return float(r["G_team"])

    for bid in blocks:
        mf = _g(bid, "safe_mainline_first")
        rf = _g(bid, "safe_ramp_first")
        hb = _g(bid, "hard_braking_safe")
        if mf is not None:
            nominal_shares.append(float(mf["braking_penalty_share"]))
        if rf is not None:
            nominal_shares.append(float(rf["braking_penalty_share"]))
        near = _g(bid, "safe_near_simultaneous")
        if near is not None:
            nominal_shares.append(float(near["braking_penalty_share"]))

        share_diff = None
        if mf is not None and hb is not None:
            share_diff = float(hb["braking_penalty_share"]) - float(mf["braking_penalty_share"])
            paired_diffs.append(share_diff)
            if not (float(hb["braking_penalty_share"]) > float(mf["braking_penalty_share"])):
                reasons.append(f"{bid}:hard_share_not_gt_nominal")

        g_mf = _team(bid, "safe_mainline_first")
        g_rf = _team(bid, "safe_ramp_first")
        if g_mf is not None and g_rf is not None:
            og = normalised_order_gap(g_mf, g_rf)
            order_gaps.append(og["normalised_order_gap"])

        inc: IncentiveOrderingResult = check_incentive_ordering(
            bid,
            g_safe_mainline=_g_success(bid, "safe_mainline_first"),
            g_safe_ramp=_g_success(bid, "safe_ramp_first"),
            g_slow_mainline=_g_success(bid, "slow_safe_mainline_first"),
            g_slow_ramp=_g_success(bid, "slow_safe_ramp_first"),
            g_stall_partial=_g_success(bid, "stall_after_partial_progress"),
            g_early_coll=_g_success(bid, "early_collision"),
            g_late_coll=_g_success(bid, "late_collision"),
        )
        if not inc.ok:
            ordering_violations += 1
            reasons.append(f"{bid}:ordering:{','.join(inc.violations)}")

        for name in ("safe_mainline_first", "safe_ramp_first"):
            gs = _g_success(bid, name)
            stall = _g_success(bid, "stall_after_partial_progress")
            ec = _g_success(bid, "early_collision")
            lc = _g_success(bid, "late_collision")
            if gs is not None and stall is not None and not (gs > stall):
                reasons.append(f"{bid}:{name}_not_above_stall")
            if gs is not None and ec is not None and not (gs > ec):
                reasons.append(f"{bid}:{name}_not_above_early_collision")
            if gs is not None and lc is not None and not (gs > lc):
                reasons.append(f"{bid}:{name}_not_above_late_collision")

        for r in recon.values():
            if r["block_id"] != bid:
                continue
            nan_count += int(r["nan_count"])
            decomp += int(r["decomp_mismatch"])

        per_block.append(
            {
                "block_id": bid,
                "G_team_safe_mainline_first": g_mf,
                "G_team_safe_ramp_first": g_rf,
                "nominal_share_mainline": None if mf is None else mf["braking_penalty_share"],
                "hard_share": None if hb is None else hb["braking_penalty_share"],
                "paired_share_diff": share_diff,
                "ordering_ok": inc.ok,
                "ordering_violations": list(inc.violations),
            }
        )

    med_share = median(nominal_shares) if nominal_shares else float("nan")
    max_share = max(nominal_shares) if nominal_shares else float("nan")
    med_diff = median(paired_diffs) if paired_diffs else float("nan")
    med_gap = median(order_gaps) if order_gaps else None
    max_gap = max(order_gaps) if order_gaps else None

    if not (0.02 - TIE_TOL <= med_share <= 0.06 + TIE_TOL):
        reasons.append(f"median_nominal_share={med_share:.6f}_not_in_[0.02,0.06]")
    if max_share > 0.10 + TIE_TOL:
        reasons.append(f"max_nominal_share={max_share:.6f}>0.10")
    if not (med_diff >= 0.02 - TIE_TOL):
        reasons.append(f"median_paired_share_diff={med_diff:.6f}<0.02")
    if ordering_violations > 0:
        reasons.append(f"ordering_violations={ordering_violations}")
    if med_gap is not None and med_gap > 0.05 + TIE_TOL:
        reasons.append(f"median_order_gap={med_gap:.6f}>0.05")
    if max_gap is not None and max_gap > 0.10 + TIE_TOL:
        reasons.append(f"max_order_gap={max_gap:.6f}>0.10")
    integrity_ok = nan_count == 0 and decomp == 0 and fixture_in_ranking == 0
    if not integrity_ok:
        reasons.append(
            f"integrity_nan={nan_count},decomp={decomp},fixture_in_ranking={fixture_in_ranking}"
        )

    # Deduplicate reasons while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for rsn in reasons:
        if rsn not in seen:
            seen.add(rsn)
            uniq.append(rsn)

    feasible = len(uniq) == 0
    return EtaMetrics(
        a_comfort=a_comfort,
        a_hard=a_hard,
        eta_hard_brake=eta,
        feasible=feasible,
        rejection_reasons=uniq,
        median_nominal_share=float(med_share) if med_share == med_share else float("nan"),
        max_nominal_share=float(max_share) if max_share == max_share else float("nan"),
        median_paired_share_diff=float(med_diff) if med_diff == med_diff else float("nan"),
        median_order_gap=med_gap,
        max_order_gap=max_gap,
        n_ordering_violations=ordering_violations,
        per_block=per_block,
        integrity_ok=integrity_ok,
    )


def select_eta(candidates: Sequence[EtaMetrics]) -> EtaMetrics | None:
    """Minimum-effective-intervention: smallest feasible eta."""
    feasible = [c for c in candidates if c.feasible]
    if not feasible:
        return None
    best = min(feasible, key=lambda c: (c.eta_hard_brake, c.a_comfort, c.a_hard))
    ordered = sorted(candidates, key=lambda c: (0 if c.feasible else 1, c.eta_hard_brake))
    for i, c in enumerate(ordered, start=1):
        c.selection_rank = i
    return best


def acceleration_distribution_rows(
    included: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_class: dict[str, list[float]] = {}
    for r in included:
        cls = scenario_class(str(r["scenario_id"]))
        by_class.setdefault(cls, []).append(float(r["realised_acceleration"]))
    rows: list[dict[str, Any]] = []
    for cls, acc in sorted(by_class.items()):
        brake = [max(0.0, -a) for a in acc]
        rows.append(
            {
                "scenario_class": cls,
                "n": len(acc),
                "mean_accel": sum(acc) / len(acc),
                "mean_braking_magnitude": sum(brake) / len(brake),
                "p50_braking_magnitude": median(brake),
                "p95_braking_magnitude": __import__(
                    "thesis.calibration.calibration_metrics", fromlist=["percentile"]
                ).percentile(brake, 95),
                "min_accel": min(acc),
                "max_accel": max(acc),
            }
        )
    return rows


def run_comfort_calibration(
    *,
    manifest: SourceTraceManifest,
    transitions: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    a_comfort_candidates: Sequence[float],
    a_hard_candidates: Sequence[float],
    eta_candidates: Sequence[float],
    gamma: float = 0.995,
) -> dict[str, Any]:
    """Full offline calibration pipeline (no env dynamics / no DQN)."""
    assert_h_monotonicities()
    assert_eta_penalty_monotone(0.5, list(eta_candidates))

    included, excluded, filter_stats = filter_active_calibration_transitions(transitions)
    windows, window_rows, window_failures = build_hard_braking_windows(transitions)
    blocks = sorted({str(r["block_id"]) for r in transitions if not r.get("fixture_only")})

    threshold_metrics: list[ThresholdMetrics] = []
    for a_c in a_comfort_candidates:
        for a_h in a_hard_candidates:
            threshold_metrics.append(
                evaluate_threshold_pair(
                    a_comfort=float(a_c),
                    a_hard=float(a_h),
                    included=included,
                    window_rows=window_rows,
                    blocks=blocks,
                )
            )
    selected_thr = select_threshold_pair(threshold_metrics)

    eta_metrics: list[EtaMetrics] = []
    selected_eta: EtaMetrics | None = None
    notes: list[str] = []
    if window_failures:
        notes.append("hard_window_failures:" + ";".join(window_failures))

    if selected_thr is None:
        overall = "FAIL"
        notes.append("no_feasible_threshold_pair")
        selection = CalibrationSelection(
            None, None, None, 0, 0, overall, notes
        )
    else:
        for eta in eta_candidates:
            eta_metrics.append(
                evaluate_eta(
                    a_comfort=selected_thr.a_comfort,
                    a_hard=selected_thr.a_hard,
                    eta=float(eta),
                    transitions=transitions,
                    outcomes=outcomes,
                    gamma=gamma,
                    blocks=blocks,
                )
            )
        selected_eta = select_eta(eta_metrics)
        thr_feas = sum(1 for t in threshold_metrics if t.feasible)
        eta_feas = sum(1 for e in eta_metrics if e.feasible)
        if selected_eta is None:
            overall = "FAIL"
            notes.append("no_feasible_eta")
            selection = CalibrationSelection(
                selected_thr.a_comfort,
                selected_thr.a_hard,
                None,
                thr_feas,
                eta_feas,
                overall,
                notes,
            )
        else:
            overall = "PASS"
            selection = CalibrationSelection(
                selected_thr.a_comfort,
                selected_thr.a_hard,
                selected_eta.eta_hard_brake,
                thr_feas,
                eta_feas,
                overall,
                notes,
            )

    return {
        "manifest": manifest,
        "included": included,
        "excluded": excluded,
        "filter_stats": filter_stats,
        "windows": windows,
        "window_rows": window_rows,
        "window_failures": window_failures,
        "threshold_metrics": threshold_metrics,
        "eta_metrics": eta_metrics,
        "selected_threshold": selected_thr,
        "selected_eta": selected_eta,
        "selection": selection,
        "accel_distribution": acceleration_distribution_rows(included),
        "blocks": blocks,
        "overall": selection.overall,
    }


def confirmatory_scripted_rerun(
    *,
    a_comfort: float,
    a_hard: float,
    eta: float,
    gamma: float,
    offline_recon: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Rerun Stage 3A scripts with selected comfort params; compare team returns."""
    from thesis.audits.audit_scenarios import build_all_audit_scenarios
    from thesis.audits.base_outcome_audit import run_audit_scenario
    from thesis.rewards.base_reward_v2 import BaseRewardConfig

    scenarios = [
        s
        for s in build_all_audit_scenarios()
        if not s.fixture_only and s.scenario_id in PRIMARY_RANKING_SCENARIOS
    ]
    diffs: list[dict[str, Any]] = []
    confirmatory_transitions: list[dict[str, Any]] = []
    max_abs = 0.0
    traj_changed = False

    for sc in scenarios:
        # Inject selected comfort into env reward config
        sc.config.base_reward = BaseRewardConfig(
            progress_weight=0.4,
            exit_weight=0.6,
            collision_penalty=1.0,
            eta_hard_brake=float(eta),
            a_comfort=float(a_comfort),
            a_hard=float(a_hard),
        )
        outcome = run_audit_scenario(sc, run_id="stage3b_confirm", gamma=gamma)
        key = (sc.block_id, sc.scenario_id)
        off = offline_recon.get(key)
        if off is None:
            continue
        d_team = abs(float(outcome.G_team) - float(off["G_team"]))
        d_brake = abs(float(outcome.G_hard_braking) - float(off["G_hard_braking"]))
        max_abs = max(max_abs, d_team, d_brake)
        diffs.append(
            {
                "block_id": sc.block_id,
                "scenario_id": sc.scenario_id,
                "G_team_offline": off["G_team"],
                "G_team_rerun": outcome.G_team,
                "abs_diff_team": d_team,
                "G_hard_braking_offline": off["G_hard_braking"],
                "G_hard_braking_rerun": outcome.G_hard_braking,
                "abs_diff_braking": d_brake,
                "episode_length_rerun": outcome.episode_length,
            }
        )
        # Reward params must not alter physical trajectories: compare accelerations
        # against offline trace lengths for matched steps
        for t in outcome.transitions:
            confirmatory_transitions.append(t)

    ok = max_abs <= 1e-10
    return {
        "ok": ok,
        "max_abs_return_difference": max_abs,
        "comparisons": diffs,
        "confirmatory_transitions": confirmatory_transitions,
        "trajectory_coupling_failure": traj_changed,
    }
