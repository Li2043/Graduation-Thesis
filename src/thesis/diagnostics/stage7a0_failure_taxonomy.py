"""Failure taxonomy for truncated Baseline episodes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS = {
    "near_stop_speed": 1.0,
    "low_speed": 3.0,
    "minimal_progress_25": 0.5,  # meters along route over 25 steps
    "minimal_progress_50": 1.0,  # meters along route over 50 steps
    "high_action_switch_rate": 0.50,
    "dominant_action_fraction": 0.80,
}

PRECEDENCE = [
    "environment_or_exit_anomaly",
    "post_exit_survivor_stall",
    "mutual_yielding",
    "merge_zone_deadlock",
    "unilateral_stall",
    "downstream_completion_failure",
    "oscillatory_control",
    "persistent_accelerate_conflict",
    "stable_but_maladaptive",
    "low_confidence_policy",
    "other_unresolved",
]


def _window(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df.empty:
        return df
    last = int(df["policy_step"].max())
    return df[df["policy_step"] >= last - n + 1]


def _switch_rate(actions: list[int]) -> float:
    if len(actions) < 2:
        return 0.0
    switches = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i - 1])
    return switches / (len(actions) - 1)


def _alt_rate(actions: list[int]) -> float:
    if len(actions) < 2:
        return 0.0
    alt = 0
    for i in range(1, len(actions)):
        a0, a1 = actions[i - 1], actions[i]
        if {a0, a1} == {1, 2}:
            alt += 1
    return alt / (len(actions) - 1)


def classify_truncated_episode(
    episode: dict[str, Any],
    steps: pd.DataFrame,
    *,
    thresholds: dict[str, float] | None = None,
    q_margin_ref: pd.Series | None = None,
) -> dict[str, Any]:
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    flags: dict[str, bool] = {k: False for k in PRECEDENCE}

    if not bool(episode.get("truncated")):
        return {
            "primary_failure_label": "not_truncated",
            "secondary_flags_json": "[]",
            **{f"flag_{k}": False for k in PRECEDENCE},
            "reason": "episode not truncated",
        }

    a = steps[steps["controller"] == "A"].sort_values("policy_step")
    b = steps[steps["controller"] == "B"].sort_values("policy_step")
    w25a, w25b = _window(a, 25), _window(b, 25)
    w50a, w50b = _window(a, 50), _window(b, 50)

    def prog_gain(w: pd.DataFrame) -> float:
        if len(w) < 2:
            return 0.0
        return float(w["route_progress"].iloc[-1] - w["route_progress"].iloc[0])

    def mean_speed(w: pd.DataFrame) -> float:
        return float(w["speed"].mean()) if len(w) else 0.0

    both_active_end = bool(w25a["active"].iloc[-1]) and bool(w25b["active"].iloc[-1]) if len(w25a) and len(w25b) else False
    one_exited = bool(episode.get("first_exit_controller")) and (
        (len(w25a) and not bool(w25a["active"].iloc[-1]))
        or (len(w25b) and not bool(w25b["active"].iloc[-1]))
    )

    # Environment anomaly heuristics
    for w in (a, b):
        if len(w) >= 2 and (w["route_progress"].diff().dropna() < -5.0).any():
            flags["environment_or_exit_anomaly"] = True
        if len(w) and bool(w["exited"].iloc[-1]) and bool(w["active"].iloc[-1]):
            flags["environment_or_exit_anomaly"] = True

    # Post-exit survivor stall
    if episode.get("first_exit_controller") and episode.get("second_exit_step") is None:
        survivor = "B" if episode["first_exit_controller"] == "A" else "A"
        ws = w50a if survivor == "A" else w50b
        if len(ws) and bool(ws["active"].iloc[-1]) and prog_gain(ws) < thr["minimal_progress_50"]:
            flags["post_exit_survivor_stall"] = True

    # Mutual yielding
    if both_active_end:
        if (
            mean_speed(w25a) < thr["low_speed"]
            and mean_speed(w25b) < thr["low_speed"]
            and prog_gain(w25a) < thr["minimal_progress_25"]
            and prog_gain(w25b) < thr["minimal_progress_25"]
        ):
            joints = w25a["joint_action_category"].tolist() if "joint_action_category" in w25a else []
            yieldish = sum(
                1
                for j in joints
                if j
                in {
                    "maintain-maintain",
                    "maintain-decelerate",
                    "decelerate-maintain",
                    "decelerate-decelerate",
                }
            )
            if joints and yieldish / len(joints) >= thr["dominant_action_fraction"]:
                flags["mutual_yielding"] = True

    # Merge-zone deadlock (both active, little progress)
    if both_active_end and prog_gain(w50a) < thr["minimal_progress_50"] and prog_gain(w50b) < thr["minimal_progress_50"]:
        flags["merge_zone_deadlock"] = True

    # Unilateral stall
    for label, ws, peer in (("A", w50a, w50b), ("B", w50b, w50a)):
        if not len(ws):
            continue
        if (
            bool(ws["active"].iloc[-1])
            and mean_speed(ws) < thr["near_stop_speed"]
            and prog_gain(ws) < thr["minimal_progress_50"]
            and (prog_gain(peer) >= thr["minimal_progress_50"] or (len(peer) and not bool(peer["active"].iloc[-1])))
        ):
            flags["unilateral_stall"] = True

    # Downstream completion failure
    if one_exited and not flags["post_exit_survivor_stall"]:
        flags["downstream_completion_failure"] = True

    # Oscillation
    for w in (w50a, w50b):
        acts = [int(x) for x in w["commanded_action"].tolist()] if len(w) else []
        if _switch_rate(acts) > thr["high_action_switch_rate"] and prog_gain(w) < thr["minimal_progress_50"]:
            flags["oscillatory_control"] = True
        if _alt_rate(acts) > 0.35 and prog_gain(w) < thr["minimal_progress_50"]:
            flags["oscillatory_control"] = True

    # Persistent accelerate conflict
    if both_active_end and len(w25a):
        frac_acc = (
            ((w25a["commanded_action"] == 1).mean() + (w25b["commanded_action"] == 1).mean()) / 2.0
        )
        if frac_acc >= 0.5 and prog_gain(w50a) < thr["minimal_progress_50"]:
            flags["persistent_accelerate_conflict"] = True

    # Q margin relative classification
    qm = pd.concat([w50a["Q_margin"], w50b["Q_margin"]], ignore_index=True).dropna()
    median_qm = float(qm.median()) if len(qm) else float("nan")
    q_band = "unavailable"
    if q_margin_ref is not None and len(q_margin_ref) and np.isfinite(median_qm):
        q10 = float(q_margin_ref.quantile(0.10))
        q25 = float(q_margin_ref.quantile(0.25))
        if median_qm < q10:
            q_band = "below_10th_percentile"
            flags["low_confidence_policy"] = True
        elif median_qm < q25:
            q_band = "10th_25th_percentile"
        else:
            q_band = "above_25th_percentile"

    # Stable but maladaptive
    if np.isfinite(median_qm) and median_qm > 0.05:
        for w in (w50a, w50b):
            acts = [int(x) for x in w["commanded_action"].tolist()] if len(w) else []
            if acts and _switch_rate(acts) < 0.2 and prog_gain(w) < thr["minimal_progress_50"]:
                flags["stable_but_maladaptive"] = True

    primary = "other_unresolved"
    for key in PRECEDENCE:
        if flags[key]:
            primary = key
            break
    secondary = [k for k, v in flags.items() if v and k != primary]
    reason = f"primary={primary}; secondary={secondary}; q_band={q_band}"
    out = {
        "primary_failure_label": primary,
        "secondary_flags_json": str(secondary),
        "median_Q_margin_final50": median_qm,
        "q_margin_band": q_band,
        "reason": reason,
    }
    for k in PRECEDENCE:
        out[f"flag_{k}"] = bool(flags[k])
    return out


def build_failure_taxonomy(
    episodes: pd.DataFrame,
    steps: pd.DataFrame,
    *,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    trunc = episodes[episodes["truncated"]].copy()
    ref = steps["Q_margin"].dropna()
    rows = []
    for _, ep in trunc.iterrows():
        mask = (
            (steps["master_seed"] == ep["master_seed"])
            & (steps["validation_block_id"] == ep["validation_block_id"])
            & (steps["assignment"] == ep["assignment"])
        )
        ep_steps = steps.loc[mask]
        cls = classify_truncated_episode(ep.to_dict(), ep_steps, thresholds=thresholds, q_margin_ref=ref)
        row = {**ep.to_dict(), **cls}
        rows.append(row)
    return pd.DataFrame(rows)
