#!/usr/bin/env python3
"""M5 -- oracle feasibility gate on the HighwayEnv backend (runbook sec
29). Ports the unmodified rule-based ``oracle_controller.oracle_actions``
(pure functions over positions/roles) onto
``StudyBHeterogeneousHighwayEnv``: the only backend-specific detail is
where "position" comes from (real HighwayEnv world x-coordinate instead
of the legacy scalar ``route_position``) and what merge_start/merge_end
mean in the new road geometry (``before_merge_length`` /
``before_merge_length + converge_merge_length`` -- see
``scenario_adapter.py``'s module docstring for why these are the correct
analogues).

Reads global vehicle state deliberately (this is the oracle, not a
learned policy -- see the legacy script's identical docstring note).

**2026-08-16 CONTROL_AUTHORITY amendment, oracle-side adjustments**
(command-timing only, per that amendment's explicit authorization --
does not touch the physical control envelope):

1. **meta_speed target_speed debounce.** The oracle's own decision
   function (``oracle_actions``) is memoryless -- it re-evaluates from
   scratch every step and was designed against direct_accel's memoryless
   per-step acceleration commands. Under ``meta_speed``, naively
   forwarding its raw ACCELERATE/BRAKE decision every step causes
   unbounded ``target_speed`` accumulation (confirmed empirically:
   target_speed observed running to -108 m/s during one sustained yield
   decision), causing 0% oracle completion (100% timeout) before this
   fix. Fixed by debouncing: only forward a fresh ACCELERATE/BRAKE nudge
   if the vehicle's ``target_speed`` hasn't already moved past its
   current ``speed`` in that direction; otherwise forward HOLD. This
   recovered completion from 0% to 96.9% on the Q bank.
2. **Real-time same-lane check for meta_speed only.**
   ``oracle_actions``'s new optional ``lateral_positions`` parameter
   (real world-y) is passed ONLY when ``action_representation ==
   "meta_speed"`` here -- NOT for ``direct_accel``, whose oracle result
   was already independently validated at 100%/0%/0% under the original
   role-based same-lane check (M5, sec 29); there is no reason to touch
   an already-passing, already-frozen case. Under meta_speed, this
   correctly recognizes a ramp vehicle that has already physically
   merged as same-lane with a trailing mainline vehicle (the scenario's
   fixed ``role`` field alone can't express this) and closed the
   remaining 2/64 Q-bank collisions left after the debounce fix.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(REPO_SRC))

from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig  # noqa: E402
from thesis.study_b.envs.highwayenv_wrapper import (  # noqa: E402
    StudyBHeterogeneousHighwayEnv,
    StudyBHighwayWrapperConfig,
)
from thesis.study_b.oracle_controller import ACCELERATE, DECELERATE, MAINTAIN, oracle_actions  # noqa: E402
from thesis.study_b.training_common import load_scenario_bank  # noqa: E402
from thesis.study_b.utility import episode_utilities  # noqa: E402

__all__ = ["run_oracle_controller_highwayenv", "ENVIRONMENT_FEASIBILITY_COMPLETION_TARGET"]

ENVIRONMENT_FEASIBILITY_COMPLETION_TARGET = 0.90


def _debounce_meta_speed_action(raw_action: int, vehicle) -> int:
    """Prevents target_speed windup (2026-08-16 amendment, see module
    docstring item 1): only forwards a fresh ACCELERATE/BRAKE nudge if
    the vehicle isn't already leaning that direction relative to its
    current speed; otherwise HOLD."""
    if raw_action == ACCELERATE:
        return MAINTAIN if vehicle.target_speed > vehicle.speed else ACCELERATE
    if raw_action == DECELERATE:
        return MAINTAIN if vehicle.target_speed < vehicle.speed else DECELERATE
    return MAINTAIN


def run_oracle_controller_highwayenv(*, scenario_bank: Path, action_representation: str = "direct_accel") -> list[dict]:
    scenarios = load_scenario_bank(scenario_bank)
    env_config = ThesisHighwayMergeEnvConfig(action_representation=action_representation)
    merge_start = env_config.before_merge_length
    merge_end = env_config.before_merge_length + env_config.converge_merge_length
    env = StudyBHeterogeneousHighwayEnv(StudyBHighwayWrapperConfig(env_config=env_config))
    is_meta_speed = action_representation == "meta_speed"

    rows = []
    for scenario in scenarios:
        env.reset(seed=0, scenario=scenario)
        term_reason = "truncation"
        for _t in range(env_config.episode_max_steps):
            positions = {
                vid: env._env.world_xy(env._env._vehicle_by_id[vid])[0]  # noqa: SLF001 -- oracle reads global state
                for vid in env.active_vehicle_ids
            }
            lateral_positions = (
                {vid: env._env.world_xy(env._env._vehicle_by_id[vid])[1] for vid in env.active_vehicle_ids}  # noqa: SLF001
                if is_meta_speed else None
            )
            raw_actions = oracle_actions(
                scenario=scenario, positions=positions, merge_start=merge_start, merge_end=merge_end,
                active_vehicle_ids={vid: True for vid in env.active_vehicle_ids},
                lateral_positions=lateral_positions,
            )
            actions = (
                {vid: _debounce_meta_speed_action(a, env._env._vehicle_by_id[vid]) for vid, a in raw_actions.items()}  # noqa: SLF001
                if is_meta_speed else raw_actions
            )
            _obs, _reward, terminated, truncated, info = env.step(actions)
            if terminated:
                term_reason = "collision" if info["collision_event"] else "success"
                break
            if truncated:
                term_reason = "truncation"
                break

        traces = env.episode_traces()
        utilities = episode_utilities(traces)
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "term_reason": term_reason,
                "completion": int(term_reason == "success"),
                "collision": int(term_reason == "collision"),
                "timeout": int(term_reason == "truncation"),
                "mean_U": sum(utilities.values()) / len(utilities),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario-bank", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--action-representation", type=str, default="direct_accel", choices=["direct_accel", "meta_speed"])
    args = p.parse_args(argv)

    rows = run_oracle_controller_highwayenv(scenario_bank=args.scenario_bank, action_representation=args.action_representation)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario_id", "term_reason", "completion", "collision", "timeout", "mean_U"])
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    completion_rate = sum(r["completion"] for r in rows) / n
    collision_rate = sum(r["collision"] for r in rows) / n
    timeout_rate = sum(r["timeout"] for r in rows) / n

    if completion_rate >= 0.98 and collision_rate <= 0.01 and timeout_rate <= 0.01:
        verdict = "STRONG_PASS"
    elif completion_rate >= 0.95 and (collision_rate + timeout_rate) <= 0.05:
        verdict = "ACCEPTABLE_PASS"
    elif completion_rate >= 0.80:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "FAIL"

    summary = {
        "n_scenarios": n, "completion_rate": completion_rate, "collision_rate": collision_rate,
        "timeout_rate": timeout_rate, "environment_feasibility_verdict": verdict,
    }
    (args.output.parent / "oracle_controller_highwayenv_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"evaluated {n} scenarios -> {args.output}")
    print(f"completion_rate={completion_rate:.4f} collision_rate={collision_rate:.4f} timeout_rate={timeout_rate:.4f}")
    print(f"M5 verdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
