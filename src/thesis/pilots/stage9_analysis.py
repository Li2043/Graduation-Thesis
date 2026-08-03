"""Stage 9 RQ1-RQ3 analysis: certified-choice-state convention metrics
(q, p_MF, D_swap), learner mobility, and non-inferiority testing.

Certified choice states (Chapter 2 SS2.8): computed directly from
`thesis.certification.choice_state_certification.certify_block` against the
frozen environment candidate (G1-I1) and the 8 physical validation blocks --
not re-derived per evaluation run, since certification is a policy-
independent, scripted-macro property of the physical block itself. Computed
once in this session (2026-08-03): 7/8 blocks certify --
validation_001 fails ("go_go_not_problematic", "core_ordering_failed").
Recorded here as a frozen constant (`CERTIFIED_TEMPLATE_INDICES`) rather
than recomputed on every analysis run, both for speed and so a future
environment-candidate change cannot silently change this stage's filter
without the constant being explicitly revisited.

All functions here are read-only analysis over already-written
`evaluation_episodes.csv` / trajectory CSVs -- no training, no simulation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Frozen 2026-08-03: certify_block(candidate="G1-I1", block) for all 8
# validation blocks. template_validation_index = scenario_block % 8.
CERTIFIED_TEMPLATE_INDICES: frozenset[int] = frozenset({1, 2, 3, 4, 5, 6, 7})
UNCERTIFIED_TEMPLATE_INDICES: frozenset[int] = frozenset({0})  # validation_001

TARGET_SPEED = 20.0  # single fixed constant shared by all stakeholders (merge_env_v2.MergeEnvConfig)


def recompute_certified_template_indices() -> dict[str, Any]:
    """Recompute certification from scratch, for drift-checking against the
    frozen CERTIFIED_TEMPLATE_INDICES constant above. Not used in the hot
    analysis path -- call explicitly if the environment lock ever changes."""
    from thesis.certification.choice_state_certification import certify_block
    from thesis.certification.choice_state_scenarios import (
        build_environment_candidates,
        build_ic_blocks,
    )

    candidates = build_environment_candidates()
    cand = next(c for c in candidates if c.candidate_id == "G1-I1")
    _, val_blocks = build_ic_blocks()
    results = {}
    for i, b in enumerate(val_blocks):
        r = certify_block(cand, b)
        results[i] = {"block_id": b.block_id, "certified": r["certified"], "reasons": r["rejection_reasons"]}
    return results


def _template_index(ep: pd.DataFrame) -> pd.Series:
    return ep["scenario_block"].astype(int) % 8


def filter_certified(ep: pd.DataFrame) -> pd.DataFrame:
    """Restrict episodes to certified choice states only."""
    idx = _template_index(ep)
    return ep[idx.isin(CERTIFIED_TEMPLATE_INDICES)].copy()


def compute_rq1_metrics(ep: pd.DataFrame, *, checkpoint_steps: tuple[int, ...] | None = None) -> dict[str, Any]:
    """q, p_MF, D_swap within certified choice states, per Chapter 2 SS2.8:

        q      = P(safe resolution)                         -- success rate
        p_MF   = P(mainline_first | safe resolution)
        D_swap = P(order changes after swapping A and B)     -- among scenario
                 blocks where BOTH assignments (0/1) safely resolved, the
                 fraction with a complementary (not identical) passing_order.

    Restricted to `checkpoint_steps` if given (else all checkpoints present
    in `ep`). Returns per-checkpoint and pooled values, plus the raw
    resolution-proportion denominator -- reported *before* any of q/p_MF/
    D_swap so a low denominator is visible, not hidden behind a ratio
    (per protocol doc SS6 / Chapter 2 SS2.9's own interpretive constraint).
    """
    cert = filter_certified(ep)
    if checkpoint_steps is not None:
        cert = cert[cert["checkpoint_step"].isin(checkpoint_steps)]
    cert["assignment"] = cert["assignment"].astype(int)
    cert["scenario_block"] = cert["scenario_block"].astype(int)

    out: dict[str, Any] = {"n_certified_episodes": int(len(cert))}
    if cert.empty:
        out["status"] = "NO_CERTIFIED_EPISODES"
        return out

    q = float(cert["success"].mean())
    resolved = cert[cert["success"]]
    p_mf = float((resolved["passing_order"] == "mainline_first").mean()) if len(resolved) else float("nan")

    swap_rows = []
    for (seed, ckpt, block), g in cert.groupby(["master_seed", "checkpoint_step", "scenario_block"]):
        a = g[g["assignment"] == 0]
        b = g[g["assignment"] == 1]
        if len(a) != 1 or len(b) != 1:
            continue
        a, b = a.iloc[0], b.iloc[0]
        if not (bool(a["success"]) and bool(b["success"])):
            swap_rows.append("one_or_both_unresolved")
            continue
        orders = {str(a["passing_order"]), str(b["passing_order"])}
        swap_rows.append("complementary" if orders == {"mainline_first", "ramp_first"} else "same_order")
    swap_series = pd.Series(swap_rows)
    both_resolved = swap_series[swap_series != "one_or_both_unresolved"]
    d_swap = float((both_resolved == "complementary").mean()) if len(both_resolved) else float("nan")

    out.update(
        {
            "status": "OK",
            "n_certified_pairs_total": int(len(swap_series)),
            "n_certified_pairs_both_resolved": int(len(both_resolved)),
            "q_safe_resolution_rate": q,
            "p_MF_mainline_first_given_resolved": p_mf,
            "D_swap_complementary_order_rate": d_swap,
            "swap_pair_breakdown": swap_series.value_counts().to_dict(),
        }
    )
    return out


def compute_rq1_metrics_per_seed(ep: pd.DataFrame, *, checkpoint_steps: tuple[int, ...]) -> pd.DataFrame:
    """Per-seed q / D_swap (and collision_rate, for RQ3 convenience), within
    certified choice states, restricted to `checkpoint_steps`. One row per
    master_seed -- the natural independent-replication unit for the
    two-sample tests in SS7.1/SS7.2 of the protocol (n=20/condition), as
    opposed to `compute_rq1_metrics`'s pooled-episode view (used for the
    single-condition RQ1 descriptive report only).
    """
    cert = filter_certified(ep)
    cert = cert[cert["checkpoint_step"].isin(checkpoint_steps)].copy()
    cert["assignment"] = cert["assignment"].astype(int)
    cert["scenario_block"] = cert["scenario_block"].astype(int)

    rows = []
    for seed, g in cert.groupby("master_seed"):
        q = float(g["success"].mean())
        collision_rate = float(g["collision"].mean())
        swap_cat = []
        for (ckpt, block), gb in g.groupby(["checkpoint_step", "scenario_block"]):
            a = gb[gb["assignment"] == 0]
            b = gb[gb["assignment"] == 1]
            if len(a) != 1 or len(b) != 1:
                continue
            a, b = a.iloc[0], b.iloc[0]
            if not (bool(a["success"]) and bool(b["success"])):
                swap_cat.append("unresolved")
                continue
            orders = {str(a["passing_order"]), str(b["passing_order"])}
            swap_cat.append("complementary" if orders == {"mainline_first", "ramp_first"} else "same_order")
        resolved_pairs = [c for c in swap_cat if c != "unresolved"]
        d_swap = float(np.mean([c == "complementary" for c in resolved_pairs])) if resolved_pairs else float("nan")
        rows.append(
            {
                "master_seed": int(seed),
                "q": q,
                "collision_rate": collision_rate,
                "D_swap": d_swap,
                "n_episodes": int(len(g)),
                "n_resolved_pairs": len(resolved_pairs),
            }
        )
    return pd.DataFrame(rows)


def compute_learner_mobility(
    ep: pd.DataFrame,
    traj_dir: Path,
    *,
    checkpoint_steps: tuple[int, ...],
    target_speed: float = TARGET_SPEED,
) -> dict[str, Any]:
    """Stage-6B-H1-style corrected mean mobility (U_i) for controllers A, B
    only -- see protocol doc SS7.2 note on scope (B_front/B_rear telemetry is
    not present in current per-step trajectory logs).

    U_i(episode) = 0                                              if collision
                 = mean over active on-road steps of clip01(speed/target_speed)   otherwise
    """
    ep2 = ep[ep["checkpoint_step"].isin(checkpoint_steps)].copy()
    ep2["episode_index"] = ep2["episode_index"].astype(int)
    ep2["assignment"] = ep2["assignment"].astype(int)
    ep2["scenario_block"] = ep2["scenario_block"].astype(int)

    traj_cache: dict[tuple[int, int], pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for _, erow in ep2.iterrows():
        seed = int(erow["master_seed"])
        ckpt = int(erow["checkpoint_step"])
        key = (seed, ckpt)
        if key not in traj_cache:
            p = traj_dir / f"seed_{seed}_traj_step_{ckpt}.csv"
            traj_cache[key] = pd.read_csv(p) if p.is_file() else pd.DataFrame()
        traj = traj_cache[key]
        if traj.empty:
            continue
        mask = (
            (traj["validation_block_id"] == erow["validation_block_id"])
            & (traj["assignment"] == erow["assignment"])
            & (traj["scenario_block"] == erow["scenario_block"])
            & (traj["episode_index"] == erow["episode_index"])
        )
        et = traj.loc[mask]
        if et.empty:
            continue
        collided = bool(erow["collision"])
        for controller in ("A", "B"):
            sub = et[et["controller"] == controller].sort_values("policy_step")
            if sub.empty:
                continue
            active = sub[sub["active"] == True] if "active" in sub.columns else sub  # noqa: E712
            if collided:
                u = 0.0
            elif active.empty:
                u = float("nan")
            else:
                e = (active["speed"] / target_speed).clip(0.0, 1.0)
                u = float(e.mean())
            rows.append({"master_seed": seed, "checkpoint_step": ckpt, "controller": controller, "U_i": u})

    df = pd.DataFrame(rows).dropna(subset=["U_i"])
    if df.empty:
        return {"status": "NO_DATA", "n_rows": 0}
    per_seed = df.groupby("master_seed")["U_i"].mean()
    return {
        "status": "OK",
        "n_episode_controller_rows": int(len(df)),
        "mean_U": float(df["U_i"].mean()),
        "sd_U": float(df["U_i"].std(ddof=1)),
        "by_checkpoint": {
            int(k): {"mean": float(v["mean"]), "sd": float(v["std"]), "n": int(v["count"])}
            for k, v in df.groupby("checkpoint_step")["U_i"].agg(["mean", "std", "count"]).iterrows()
        },
        "per_seed_mean_U": {int(k): float(v) for k, v in per_seed.items()},
    }


ALL_STAKEHOLDER_CONTROLLERS: tuple[str, ...] = ("A", "B", "B_front", "B_rear")


def compute_stakeholder_mobility(
    ep: pd.DataFrame,
    traj_dir: Path,
    *,
    checkpoint_steps: tuple[int, ...],
    target_speed: float = TARGET_SPEED,
    controllers: tuple[str, ...] = ALL_STAKEHOLDER_CONTROLLERS,
) -> pd.DataFrame:
    """Per-episode, per-controller U_i for all four stakeholders (A, B,
    B_front, B_rear) -- same U_i definition as compute_learner_mobility (0
    if collision, else mean clip01(speed/target_speed) over logged steps),
    but returned as one row per (episode, controller) rather than collapsed
    into a per-seed mean, so callers can take a per-episode MIN across
    controllers (the worst-off stakeholder) before aggregating.

    B_front/B_rear are scripted IDM background vehicles with no
    active-on-road gating (they never "complete"/exit -- see
    merge_env_v2._registry.completed, which only tracks "A"/"B"): every
    logged step counts, unlike A/B's `active` filter.

    Requires trajectory CSVs written by the B_front/B_rear-logging-enabled
    evaluator (stage8_gate_eval.py, extended 2026-08-03); older trajectory
    CSVs will simply have no B_front/B_rear rows to find.
    """
    ep2 = ep[ep["checkpoint_step"].isin(checkpoint_steps)].copy()
    ep2["episode_index"] = ep2["episode_index"].astype(int)
    ep2["assignment"] = ep2["assignment"].astype(int)
    ep2["scenario_block"] = ep2["scenario_block"].astype(int)

    traj_cache: dict[tuple[int, int], pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for _, erow in ep2.iterrows():
        seed = int(erow["master_seed"])
        ckpt = int(erow["checkpoint_step"])
        key = (seed, ckpt)
        if key not in traj_cache:
            p = traj_dir / f"seed_{seed}_traj_step_{ckpt}.csv"
            traj_cache[key] = pd.read_csv(p) if p.is_file() else pd.DataFrame()
        traj = traj_cache[key]
        if traj.empty:
            continue
        mask = (
            (traj["validation_block_id"] == erow["validation_block_id"])
            & (traj["assignment"] == erow["assignment"])
            & (traj["scenario_block"] == erow["scenario_block"])
            & (traj["episode_index"] == erow["episode_index"])
        )
        et = traj.loc[mask]
        if et.empty:
            continue
        collided = bool(erow["collision"])
        for controller in controllers:
            sub = et[et["controller"] == controller].sort_values("policy_step")
            if sub.empty:
                continue
            if controller in ("A", "B") and "active" in sub.columns:
                sub = sub[sub["active"] == True]  # noqa: E712
            if collided:
                u = 0.0
            elif sub.empty:
                u = float("nan")
            else:
                e = (sub["speed"] / target_speed).clip(0.0, 1.0)
                u = float(e.mean())
            rows.append(
                {
                    "master_seed": seed,
                    "checkpoint_step": ckpt,
                    "validation_block_id": erow["validation_block_id"],
                    "assignment": int(erow["assignment"]),
                    "scenario_block": int(erow["scenario_block"]),
                    "episode_index": int(erow["episode_index"]),
                    "controller": controller,
                    "U_i": u,
                }
            )
    return pd.DataFrame(rows)


def compute_worst_off_mobility(
    ep: pd.DataFrame,
    traj_dir: Path,
    *,
    checkpoint_steps: tuple[int, ...],
    target_speed: float = TARGET_SPEED,
) -> dict[str, Any]:
    """U_worst(episode) = min_i U_i(episode) across all four stakeholders --
    the direct operationalisation of the Rawlsian worst-off-stakeholder
    quantity, as opposed to compute_learner_mobility's mean-of-two (A/B
    only) proxy. Only episodes with all 4 controllers logged are used for
    the min (an episode missing a controller would silently understate how
    many parties were checked); `n_incomplete_episodes` reports how many
    were dropped for this reason so a large count is visible, not hidden.
    """
    detail = compute_stakeholder_mobility(
        ep, traj_dir, checkpoint_steps=checkpoint_steps, target_speed=target_speed
    )
    if detail.empty:
        return {"status": "NO_DATA", "n_rows": 0}
    controllers_present = sorted(detail["controller"].unique().tolist())
    episode_keys = [
        "master_seed",
        "checkpoint_step",
        "validation_block_id",
        "assignment",
        "scenario_block",
        "episode_index",
    ]
    valid = detail.dropna(subset=["U_i"])
    counts = valid.groupby(episode_keys)["controller"].nunique()
    n_expected = len(ALL_STAKEHOLDER_CONTROLLERS)
    complete_keys = counts[counts == n_expected].index
    n_incomplete = int((counts != n_expected).sum())
    per_episode_all = valid.groupby(episode_keys)["U_i"].min().rename("U_worst").reset_index()
    per_episode = (
        per_episode_all.set_index(episode_keys).loc[complete_keys].reset_index()
        if len(complete_keys)
        else per_episode_all.iloc[0:0]
    )
    if per_episode.empty:
        return {
            "status": "NO_DATA",
            "n_rows": 0,
            "controllers_present": controllers_present,
            "n_incomplete_episodes": n_incomplete,
        }
    per_seed = per_episode.groupby("master_seed")["U_worst"].mean()
    return {
        "status": "OK",
        "controllers_present": controllers_present,
        "n_episodes": int(len(per_episode)),
        "n_incomplete_episodes": n_incomplete,
        "mean_U_worst": float(per_episode["U_worst"].mean()),
        "sd_U_worst": float(per_episode["U_worst"].std(ddof=1)),
        "per_seed_mean_U_worst": {int(k): float(v) for k, v in per_seed.items()},
    }


def one_sided_superiority_test(
    treatment: np.ndarray, control: np.ndarray, *, alpha: float = 0.05
) -> dict[str, Any]:
    """One-sided Welch's t-test, H1: mean(treatment) > mean(control).

    Neither `two_sample_diff_test` (two-sided, difference-detection framing)
    nor `non_inferiority_test` (not-worse framing) directly tests RQ3's
    "improvement" half -- does PBRS make the worst-off stakeholder better
    off than baseline, not merely no worse. This is the untested half
    flagged in the protocol doc SS7.2 scope note; added to close it once
    `compute_worst_off_mobility` supplies the U_worst quantity to test.
    """
    from scipy import stats

    t = np.asarray(treatment, dtype=float)
    c = np.asarray(control, dtype=float)
    diff = float(t.mean() - c.mean())
    t_stat, p_two_sided = stats.ttest_ind(t, c, equal_var=False)
    p_one_sided = float(p_two_sided / 2) if t_stat > 0 else float(1.0 - p_two_sided / 2)
    return {
        "mean_treatment": float(t.mean()),
        "mean_control": float(c.mean()),
        "diff_treatment_minus_control": diff,
        "n_treatment": int(len(t)),
        "n_control": int(len(c)),
        "t_stat": float(t_stat),
        "p_value_one_sided": p_one_sided,
        "significant_at_0.05": bool(p_one_sided < alpha),
        "superior": bool(p_one_sided < alpha and diff > 0),
    }


def two_sample_diff_test(
    a: np.ndarray, b: np.ndarray, *, mes: float
) -> dict[str, Any]:
    """RQ1/RQ2 difference-detection framing (protocol doc SS7.1): Welch's
    t-test plus an explicit flag for whether the observed |difference|
    clears the pre-registered detection-floor MES for this design. A
    non-significant OR sub-MES result must be reported as "not detected at
    this design's resolution," never as "no effect" (PAPER_CHANGES_REQUIRED_
    LATER.md item 9 -- carried into this stage's own governance)."""
    from scipy import stats

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = float(a.mean() - b.mean())
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
    return {
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "diff_a_minus_b": diff,
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant_at_0.05": bool(p_value < 0.05),
        "mes": mes,
        "abs_diff_ge_mes": bool(abs(diff) >= mes),
        "interpretable_as_detected_difference": bool(p_value < 0.05 and abs(diff) >= mes),
    }


def non_inferiority_test(
    treatment: np.ndarray,
    control: np.ndarray,
    *,
    margin: float,
    higher_is_better: bool,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """One-sided (1-alpha) CI on (treatment - control), classified into
    THREE outcomes, not two:

      "non_inferior" -- the CI excludes a loss larger than `margin` (the
                         classic non-inferiority criterion: even the
                         unfavourable bound is within the tolerable loss).
      "inferior"      -- the POINT ESTIMATE itself already shows a loss
                         larger than `margin` (positive evidence of harm,
                         not just insufficient precision to rule it out).
      "inconclusive"  -- neither: the point estimate does not show a loss
                         beyond the margin, but the CI is wide enough that a
                         loss beyond the margin cannot be excluded either.
                         This is the honest outcome when a fixed sample size
                         cannot resolve the pre-registered margin -- it must
                         be reported as such, not collapsed into "non_inferior:
                         false" (which reads as evidence of harm when it may
                         only be evidence of imprecision) nor covered up by
                         loosening the margin after the fact.

    `higher_is_better=True` for metrics like mobility/q (loss = decrease);
    `False` for collision rate (loss = increase).
    """
    from scipy import stats

    t = np.asarray(treatment, dtype=float)
    c = np.asarray(control, dtype=float)
    diff = float(t.mean() - c.mean())
    n_t, n_c = len(t), len(c)
    se = float(np.sqrt(t.var(ddof=1) / n_t + c.var(ddof=1) / n_c))
    df = (t.var(ddof=1) / n_t + c.var(ddof=1) / n_c) ** 2 / (
        (t.var(ddof=1) / n_t) ** 2 / (n_t - 1) + (c.var(ddof=1) / n_c) ** 2 / (n_c - 1)
    )
    t_crit = float(stats.t.ppf(1 - alpha, df))  # one-sided

    if higher_is_better:
        bound = diff - t_crit * se  # unfavourable (lower) bound
        point_shows_harm = diff < -margin
        bound_within_margin = bound >= -margin
    else:
        bound = diff + t_crit * se  # unfavourable (upper) bound
        point_shows_harm = diff > margin
        bound_within_margin = bound <= margin

    if bound_within_margin:
        outcome = "non_inferior"
    elif point_shows_harm:
        outcome = "inferior"
    else:
        outcome = "inconclusive"

    return {
        "mean_treatment": float(t.mean()),
        "mean_control": float(c.mean()),
        "diff_treatment_minus_control": diff,
        "n_treatment": n_t,
        "n_control": n_c,
        "margin": margin,
        "higher_is_better": higher_is_better,
        "one_sided_bound": float(bound),
        "outcome": outcome,
        "non_inferior": outcome == "non_inferior",  # kept for backward-compat callers
    }


__all__ = [
    "ALL_STAKEHOLDER_CONTROLLERS",
    "CERTIFIED_TEMPLATE_INDICES",
    "TARGET_SPEED",
    "UNCERTIFIED_TEMPLATE_INDICES",
    "compute_learner_mobility",
    "compute_rq1_metrics",
    "compute_rq1_metrics_per_seed",
    "compute_stakeholder_mobility",
    "compute_worst_off_mobility",
    "filter_certified",
    "non_inferiority_test",
    "one_sided_superiority_test",
    "recompute_certified_template_indices",
    "two_sample_diff_test",
]
