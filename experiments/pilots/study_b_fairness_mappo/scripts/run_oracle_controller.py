#!/usr/bin/env python3
"""VDN_Conditional_Amendment_Protocol.md sec 8 (Diagnostic 3): run
``oracle_controller.py``'s rule-based, global-state controller against a
frozen scenario bank and report the environment-feasibility verdict.

**Not a formal learned policy** -- this script reads global vehicle state
(``env._env._vehicles``) that no learned agent in this project is ever
allowed to see, exactly because its only purpose is answering whether the
N=4 matched-TTC environment is physically/procedurally solvable at all,
independent of any local-observation or credit-assignment limitation a
learned policy might have.

Needs NO trained checkpoint and NO new RL training -- can run as soon as
it's built, independent of any training job's outcome.

Target (document sec 8): completion > 0.90 with low collision =
environment-feasibility PASS."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(REPO_SRC))

from thesis.study_b.heterogeneous_env import StudyBEnvConfig, StudyBHeterogeneousEnv  # noqa: E402
from thesis.study_b.oracle_controller import oracle_actions  # noqa: E402
from thesis.study_b.training_common import load_scenario_bank  # noqa: E402
from thesis.study_b.utility import episode_utilities  # noqa: E402

__all__ = ["run_oracle_controller", "ENVIRONMENT_FEASIBILITY_COMPLETION_TARGET"]

ENVIRONMENT_FEASIBILITY_COMPLETION_TARGET = 0.90


def run_oracle_controller(
    *, scenario_bank: Path, episode_max_steps: int = 200, merge_start: float = 200.0, merge_end: float = 300.0,
) -> list[dict]:
    scenarios = load_scenario_bank(scenario_bank)
    env = StudyBHeterogeneousEnv(StudyBEnvConfig(episode_max_steps=episode_max_steps))

    rows = []
    for scenario in scenarios:
        obs, _info = env.reset(seed=0, scenario=scenario)
        term_reason = "truncation"
        for _t in range(episode_max_steps):
            positions = {vid: env._env._vehicles[vid].route_position for vid in env.active_vehicle_ids}  # noqa: SLF001 -- oracle explicitly reads global state
            actions = oracle_actions(
                scenario=scenario, positions=positions, merge_start=merge_start, merge_end=merge_end,
                active_vehicle_ids={vid: True for vid in env.active_vehicle_ids},
            )
            obs, _base_reward, terminated, truncated, step_info = env.step(actions)
            if terminated or truncated:
                term_reason = step_info["term_reason"]
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
    p.add_argument("--episode-max-steps", type=int, default=200)
    p.add_argument("--merge-start", type=float, default=200.0)
    p.add_argument("--merge-end", type=float, default=300.0)
    args = p.parse_args(argv)

    rows = run_oracle_controller(
        scenario_bank=args.scenario_bank, episode_max_steps=args.episode_max_steps,
        merge_start=args.merge_start, merge_end=args.merge_end,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario_id", "term_reason", "completion", "collision", "timeout", "mean_U"])
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    completion_rate = sum(r["completion"] for r in rows) / n
    collision_rate = sum(r["collision"] for r in rows) / n
    timeout_rate = sum(r["timeout"] for r in rows) / n
    verdict = "PASS" if completion_rate > ENVIRONMENT_FEASIBILITY_COMPLETION_TARGET else "FAIL"

    summary = {
        "n_scenarios": n, "completion_rate": completion_rate, "collision_rate": collision_rate,
        "timeout_rate": timeout_rate, "environment_feasibility_verdict": verdict,
    }
    (args.output.parent / "oracle_controller_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"evaluated {n} scenarios -> {args.output}")
    print(f"completion_rate={completion_rate:.4f} collision_rate={collision_rate:.4f} timeout_rate={timeout_rate:.4f}")
    print(f"environment-feasibility verdict: {verdict} (target: completion > {ENVIRONMENT_FEASIBILITY_COMPLETION_TARGET})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
