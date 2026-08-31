"""Pure H1-R1 endpoint and convention functions (no training side effects)."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from thesis.certification.choice_state_metrics import classify_exit_order


STAKEHOLDERS = ("A", "B", "B_front", "B_rear")


def _finite(name: str, value: float) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"non-finite {name}: {v}")
    return v


def role_exit_times(
    *,
    exit_time: Mapping[str, Any],
    roles: Mapping[str, str],
) -> tuple[int | None, int | None]:
    """Map learner identities to mainline/ramp exit times via roles (not IDs)."""
    t_ml: int | None = None
    t_rp: int | None = None
    for sid in ("A", "B"):
        role = str(roles[sid])
        raw = exit_time.get(sid)
        if raw is None:
            continue
        t = int(raw)
        if role == "mainline":
            t_ml = t if t_ml is None else min(t_ml, t)
        elif role == "ramp":
            t_rp = t if t_rp is None else min(t_rp, t)
        else:
            raise ValueError(f"unexpected role {role!r} for {sid}")
    return t_ml, t_rp


def classify_convention(
    *,
    success: bool,
    exit_time: Mapping[str, Any],
    roles: Mapping[str, str],
) -> str | None:
    """Convention for a successful episode; None if not classifiable."""
    if not success:
        return None
    t_ml, t_rp = role_exit_times(exit_time=exit_time, roles=roles)
    order = classify_exit_order(t_ml, t_rp)
    if order in {"mainline_first", "ramp_first", "simultaneous"}:
        return order
    return None


def convention_consistency(conventions: Sequence[str | None]) -> float | None:
    """Seed-level convention consistency.

    Proportion of successful episodes following the modal non-simultaneous
    convention. Simultaneous successes remain in the denominator when a unique
    non-simultaneous mode exists. Missing (not zero) when:
    - no successful episodes;
    - no non-simultaneous successes;
    - tied mainline/ramp non-simultaneous modes.
    """
    successful = [c for c in conventions if c is not None]
    if not successful:
        return None
    non_sim = [c for c in successful if c in {"mainline_first", "ramp_first"}]
    if not non_sim:
        return None
    counts = Counter(non_sim)
    top = counts.most_common()
    if len(top) >= 2 and top[0][1] == top[1][1]:
        return None  # tie
    mode = top[0][0]
    return float(sum(1 for c in successful if c == mode) / len(successful))


def episode_stakeholder_utilities(experiences: Mapping[str, float]) -> dict[str, float]:
    out = {}
    for sid in STAKEHOLDERS:
        out[sid] = _finite(f"U_{sid}", float(experiences[sid]))
    return out


def mean_stakeholder_utility(utilities: Mapping[str, float]) -> float:
    vals = [utilities[s] for s in STAKEHOLDERS]
    return float(sum(vals) / len(vals))


def minimum_stakeholder_utility(utilities: Mapping[str, float]) -> float:
    return float(min(utilities[s] for s in STAKEHOLDERS))


def worst_off_stakeholder(utilities: Mapping[str, float]) -> str:
    return min(STAKEHOLDERS, key=lambda s: (utilities[s], s))


def aggregate_seed_checkpoint_primary(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate validated evaluation episodes for one condition×seed×checkpoint."""
    if len(episodes) != 16:
        raise ValueError(f"expected 16 evaluation episodes, got {len(episodes)}")
    n = len(episodes)
    success_flags = [bool(e["success"]) for e in episodes]
    collision_flags = [bool(e["collision"]) for e in episodes]
    utilities = [e["stakeholder_utilities"] for e in episodes]
    mean_u = [mean_stakeholder_utility(u) for u in utilities]
    min_u = [minimum_stakeholder_utility(u) for u in utilities]
    conventions = [e.get("convention") for e in episodes]
    # Only successful episodes contribute convention labels; unsuccessful -> None
    conv_for_consistency = []
    for e in episodes:
        if e["success"]:
            conv_for_consistency.append(e.get("convention"))
        else:
            conv_for_consistency.append(None)
    return {
        "n_episodes": n,
        "evaluation_success_rate": float(sum(success_flags) / n),
        "stakeholder_collision_rate": float(sum(collision_flags) / n),
        "mean_stakeholder_episode_utility": float(sum(mean_u) / n),
        "minimum_stakeholder_episode_utility": float(sum(min_u) / n),
        "convention_consistency": convention_consistency(conv_for_consistency),
        "mainline_first_frequency": float(
            sum(1 for c in conventions if c == "mainline_first") / n
        ),
        "ramp_first_frequency": float(
            sum(1 for c in conventions if c == "ramp_first") / n
        ),
        "n_success": int(sum(success_flags)),
        "n_collision": int(sum(collision_flags)),
    }


def trapezoidal_auc(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Trapezoidal AUC over fixed checkpoints; no interpolation of gaps."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    if any(y is None or (isinstance(y, float) and not math.isfinite(y)) for y in ys):
        return None
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    total = 0.0
    for a, b in zip(order, order[1:]):
        total += 0.5 * (float(ys[a]) + float(ys[b])) * (float(xs[b]) - float(xs[a]))
    return float(total)


__all__ = [
    "STAKEHOLDERS",
    "aggregate_seed_checkpoint_primary",
    "classify_convention",
    "convention_consistency",
    "episode_stakeholder_utilities",
    "mean_stakeholder_utility",
    "minimum_stakeholder_utility",
    "role_exit_times",
    "trapezoidal_auc",
    "worst_off_stakeholder",
]
