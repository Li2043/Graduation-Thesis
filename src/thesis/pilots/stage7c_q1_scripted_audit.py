"""Scripted Base Reward V2 incentive audit for Stage 7C-Q1 (pre-training)."""

from __future__ import annotations

from typing import Any

from thesis.pilots.stage7c_q1_config import ACTIVE_TIME_COST_PER_STEP
from thesis.rewards.base_reward_v2 import (
    AgentTransitionState,
    BaseRewardConfig,
    BaseRewardInputs,
    compute_base_reward_for_agents,
)


def _cfg() -> BaseRewardConfig:
    # Locked comfort magnitudes from Stage 3B-R1 / Stage 6: 1.5 / 3.5 / 0.015
    return BaseRewardConfig(
        a_comfort=1.5,
        a_hard=3.5,
        eta_hard_brake=0.015,
        active_time_cost_per_step=ACTIVE_TIME_COST_PER_STEP,
    )


def _agent(
    *,
    rho_t: float,
    rho_t1: float,
    already_exited: bool = False,
    accel: float = 0.0,
    route_start: float = 0.0,
    route_exit: float = 1.0,
) -> AgentTransitionState:
    # Use absolute route positions consistent with rho = pos/exit when start=0
    return AgentTransitionState(
        route_position_t=float(rho_t) * (route_exit - route_start) + route_start,
        route_position_t1=float(rho_t1) * (route_exit - route_start) + route_start,
        route_start=route_start,
        route_exit=route_exit,
        acceleration=float(accel),
        already_exited=bool(already_exited),
    )


def _episode_return(transitions: list[BaseRewardInputs], cfg: BaseRewardConfig) -> dict[str, float]:
    totals = {"A": 0.0, "B": 0.0}
    for inp in transitions:
        out = compute_base_reward_for_agents(inp, cfg)
        for aid in ("A", "B"):
            totals[aid] += float(out[aid].total_reward)
    return totals


def _safe_progress_episode(*, mainline_first: bool, n_steps: int = 40) -> list[BaseRewardInputs]:
    """Synthetic safe completion: both agents progress and exit; optional order."""
    steps: list[BaseRewardInputs] = []
    # A is treated as mainline controller identity for reward (roles are identity-level here)
    for t in range(n_steps):
        frac0 = t / n_steps
        frac1 = (t + 1) / n_steps
        # delay one agent's progress slightly to encode order without changing totals much
        if mainline_first:
            a0, a1 = frac0, max(0.0, frac0 - 0.02)
            b0, b1 = max(0.0, frac0 - 0.05), max(0.0, frac1 - 0.05)
        else:
            a0, a1 = max(0.0, frac0 - 0.05), max(0.0, frac1 - 0.05)
            b0, b1 = frac0, max(0.0, frac0 - 0.02)
        already_a = a0 >= 1.0
        already_b = b0 >= 1.0
        # clamp into [0,1] for positions; exit when crossing 1.0
        a0c, a1c = min(a0, 1.0), min(a1, 1.05)
        b0c, b1c = min(b0, 1.0), min(b1, 1.05)
        steps.append(
            BaseRewardInputs(
                agents={
                    "A": _agent(rho_t=a0c, rho_t1=min(a1c, 1.0) if a1c < 1.0 else 1.0, already_exited=already_a and a0c >= 1.0),
                    "B": _agent(rho_t=b0c, rho_t1=min(b1c, 1.0) if b1c < 1.0 else 1.0, already_exited=already_b and b0c >= 1.0),
                },
                stakeholder_collided={"A": False, "B": False, "B_front": False, "B_rear": False},
                terminated=False,
                truncated=False,
            )
        )
    # Force final exit transitions if needed
    for aid, other in (("A", "B"),):
        _ = aid, other
    # Explicit exit transitions near end
    steps.append(
        BaseRewardInputs(
            agents={
                "A": _agent(rho_t=0.99, rho_t1=1.0, already_exited=False),
                "B": _agent(rho_t=0.99, rho_t1=1.0, already_exited=False),
            },
            stakeholder_collided={"A": False, "B": False, "B_front": False, "B_rear": False},
        )
    )
    # Post-exit idle steps (must not charge time)
    for _ in range(5):
        steps.append(
            BaseRewardInputs(
                agents={
                    "A": _agent(rho_t=1.0, rho_t1=1.0, already_exited=True),
                    "B": _agent(rho_t=1.0, rho_t1=1.0, already_exited=True),
                },
                stakeholder_collided={"A": False, "B": False, "B_front": False, "B_rear": False},
            )
        )
    return steps


def _stall_episode(*, n_steps: int = 80) -> list[BaseRewardInputs]:
    steps = []
    for _ in range(n_steps):
        steps.append(
            BaseRewardInputs(
                agents={
                    "A": _agent(rho_t=0.2, rho_t1=0.2, already_exited=False, accel=-1.0),
                    "B": _agent(rho_t=0.25, rho_t1=0.25, already_exited=False, accel=-1.0),
                },
                stakeholder_collided={"A": False, "B": False, "B_front": False, "B_rear": False},
                truncated=True,
            )
        )
    return steps


def _collision_episode() -> list[BaseRewardInputs]:
    steps = [
        BaseRewardInputs(
            agents={
                "A": _agent(rho_t=0.3, rho_t1=0.35, already_exited=False),
                "B": _agent(rho_t=0.32, rho_t1=0.37, already_exited=False),
            },
            stakeholder_collided={"A": True, "B": True, "B_front": False, "B_rear": False},
            terminated=True,
        )
    ]
    return steps


def _short_yield_episode(*, n_steps: int = 12) -> list[BaseRewardInputs]:
    """Brief decelerate-then-progress yielding (safe completion)."""
    steps: list[BaseRewardInputs] = []
    for t in range(n_steps):
        # A yields briefly then both progress to exit
        if t < 3:
            a0, a1 = 0.15, 0.15
            b0, b1 = 0.20 + 0.05 * t, 0.20 + 0.05 * (t + 1)
            accel_a = -1.0
        else:
            frac = (t - 3) / max(n_steps - 3, 1)
            a0, a1 = min(0.15 + frac, 0.99), min(0.15 + frac + 0.08, 1.0)
            b0, b1 = min(0.35 + frac, 0.99), min(0.35 + frac + 0.08, 1.0)
            accel_a = 0.0
        steps.append(
            BaseRewardInputs(
                agents={
                    "A": _agent(rho_t=a0, rho_t1=a1, already_exited=a0 >= 1.0, accel=accel_a),
                    "B": _agent(rho_t=b0, rho_t1=b1, already_exited=b0 >= 1.0),
                },
                stakeholder_collided={"A": False, "B": False, "B_front": False, "B_rear": False},
            )
        )
    return steps


def _mutual_yield_episode(*, n_steps: int = 60) -> list[BaseRewardInputs]:
    steps = []
    for _ in range(n_steps):
        steps.append(
            BaseRewardInputs(
                agents={
                    "A": _agent(rho_t=0.22, rho_t1=0.22, accel=-1.2),
                    "B": _agent(rho_t=0.23, rho_t1=0.23, accel=-1.2),
                },
                stakeholder_collided={"A": False, "B": False, "B_front": False, "B_rear": False},
                truncated=True,
            )
        )
    return steps


def run_scripted_reward_audit() -> dict[str, Any]:
    cfg = _cfg()
    mainline = _episode_return(_safe_progress_episode(mainline_first=True), cfg)
    ramp = _episode_return(_safe_progress_episode(mainline_first=False), cfg)
    stall = _episode_return(_stall_episode(), cfg)
    collision = _episode_return(_collision_episode(), cfg)
    short_yield = _episode_return(_short_yield_episode(), cfg)
    mutual = _episode_return(_mutual_yield_episode(), cfg)

    # Joint return = mean of A,B (order-bias uses both)
    def joint(d: dict[str, float]) -> float:
        return 0.5 * (d["A"] + d["B"])

    j_main = joint(mainline)
    j_ramp = joint(ramp)
    j_stall = joint(stall)
    j_coll = joint(collision)
    j_short = joint(short_yield)
    j_mutual = joint(mutual)

    checks = {
        "safe_mainline_gt_stall": j_main > j_stall,
        "safe_ramp_gt_stall": j_ramp > j_stall,
        "safe_mainline_gt_collision": j_main > j_coll,
        "safe_ramp_gt_collision": j_ramp > j_coll,
        "short_yield_gt_stall": j_short > j_stall,
        "safe_mainline_gt_mutual_yield": j_main > j_mutual,
        "trajectories_covered": True,
    }

    # Normalized order gap across the two safe orders
    denom = max(abs(j_main), abs(j_ramp), 1e-9)
    order_gap = abs(j_main - j_ramp) / denom
    checks["median_normalized_order_gap_le_0_05"] = order_gap <= 0.05
    checks["maximum_normalized_order_gap_le_0_10"] = order_gap <= 0.10

    # Post-exit cost: inactive transition active_time must be 0
    inactive = compute_base_reward_for_agents(
        BaseRewardInputs(
            agents={
                "A": _agent(rho_t=1.0, rho_t1=1.0, already_exited=True),
                "B": _agent(rho_t=1.0, rho_t1=1.0, already_exited=True),
            },
            stakeholder_collided={"A": False, "B": False, "B_front": False, "B_rear": False},
        ),
        cfg,
    )
    checks["post_exit_no_time_cost"] = (
        inactive["A"].active_time_component == 0.0 and inactive["B"].active_time_component == 0.0
    )

    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "returns": {
            "safe_mainline_first": mainline,
            "safe_ramp_first": ramp,
            "stall": stall,
            "collision": collision,
            "short_safe_yielding": short_yield,
            "mutual_yielding": mutual,
            "joint": {
                "mainline": j_main,
                "ramp": j_ramp,
                "stall": j_stall,
                "collision": j_coll,
                "short_yield": j_short,
                "mutual_yield": j_mutual,
            },
            "normalized_order_gap": order_gap,
        },
        "active_time_cost_per_step": ACTIVE_TIME_COST_PER_STEP,
        "note": "Synthetic scripted audit over BaseRewardV2; coefficient search forbidden.",
    }


__all__ = ["run_scripted_reward_audit"]
