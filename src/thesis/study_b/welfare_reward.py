"""Study B DIRECT terminal welfare reward -- revised_new_research_plan.md /
change.md's final formal reward design.

This REPLACES, for formal Mean/GGI/Maximin training, the earlier PBRS
shaping mechanism in ``pbrs_reward.py`` -- that module is left completely
unmodified (per change.md: "不要修改正在运行实验所依赖的文件") and
remains valid for interpreting the already-running PBRS/baseline
qualification diagnostic run's results; it is a solver/engineering
diagnostic now, not the formal reward mechanism.

Reward structure (revised_new_research_plan.md sec 13):

    non-terminal:  r_t^(c) = r_t^task                (unchanged)
    terminal:      r_T^(c) = r_T^task + R_c^W
    R_c^W = lambda_W * (W_c(U) - 1),  lambda_W = 1.0

This is explicitly NOT PBRS: there is no potential function and no
``gamma*Phi(s')-Phi(s)`` term anywhere here. ``W_c(U)`` uses the FULL
episode's failure-aware utility (``utility.episode_utilities``), only
computable once an episode has actually ended (terminated OR truncated).

Critical correctness point (change.md #3): ``W_c(U)`` must be computed
EXACTLY ONCE per finished episode, using all N agents' terminal
utilities -- never once per agent. The single resulting scalar
``R_c^W`` is then added identically to every agent's own terminal task
reward (this is not "adding the bonus 4 times" -- each agent still only
receives it once, in its own single terminal transition; it is the same
NUMBER because there is only one team-level welfare state being
rewarded, not because of any double-counting bug). See
``terminal_welfare_bonus``'s docstring and
``tests/study_b/test_welfare_reward.py``'s
``test_bonus_added_once_per_agent_not_multiplied`` for the regression
test this exact point is checked by.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from thesis.study_b.utility import generalized_gini_welfare, mean_welfare, min_welfare

__all__ = [
    "WELFARE_LAMBDA",
    "WelfareCondition",
    "MEAN",
    "GGI",
    "MAXIMIN",
    "condition_by_name",
    "terminal_welfare_bonus",
]

# revised_new_research_plan.md sec 13: lambda_W = 1.0, frozen.
WELFARE_LAMBDA: float = 1.0


@dataclass(frozen=True)
class WelfareCondition:
    name: str
    welfare_fn: Callable[[Sequence[float]], float]


MEAN = WelfareCondition("mean", mean_welfare)
GGI = WelfareCondition("ggi", generalized_gini_welfare)
MAXIMIN = WelfareCondition("maximin", min_welfare)

_BY_NAME = {c.name: c for c in (MEAN, GGI, MAXIMIN)}


def condition_by_name(name: str) -> WelfareCondition:
    if name not in _BY_NAME:
        raise ValueError(f"unknown welfare condition {name!r}, expected one of {sorted(_BY_NAME)}")
    return _BY_NAME[name]


def terminal_welfare_bonus(
    condition: WelfareCondition, episode_utilities: Sequence[float], *, lam: float = WELFARE_LAMBDA
) -> float:
    """R_c^W for one finished episode. Call this EXACTLY ONCE per episode
    (typically right after the environment reports ``terminated or
    truncated``, using ``utility.episode_utilities(traces)`` for ALL
    active agents that episode), then add the single returned float to
    every agent's own terminal-transition task reward -- do not call this
    again per agent."""
    if not episode_utilities:
        raise ValueError("episode_utilities must be non-empty")
    w_c = condition.welfare_fn(list(episode_utilities))
    return float(lam * (w_c - 1.0))
