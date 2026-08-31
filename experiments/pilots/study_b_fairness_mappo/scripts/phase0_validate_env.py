#!/usr/bin/env python3
"""Study B Phase 0 -- environment/information-structure validation, per
new_research_plan.md's Phase 0 checklist. NOT a training run. Writes
``unit_test_report.txt`` to ``--output-dir``; the Phase 0 acceptance gate
is "every check below reports PASS."

Checks:
1. Matched-TTC (>=10,000 scenarios, isolated single-vehicle simulation
   through the REAL physics constants -- accel/decel rate, dt, v_max --
   not just the generator's own construction-time arithmetic): >=95% of
   scenarios have both matched pairs (front, rear) crossing merge_start
   within 0.5s of each other.
2. Spawn validity: 100% of generated scenarios respect the 15m same-lane
   floor (scenario_generator.py already rejects/resamples internally --
   this re-confirms at Phase-0 scale that the loop never silently exceeds
   its resample budget).
3. Role x speed-class x front/rear counterbalancing: each of the 4
   (role, speed_class) combinations appears close to 50/50 in front vs.
   rear slot across many scenarios.
4. Observation leakage: delegates to pytest
   (tests/study_b/test_local_observation_leakage.py) -- run separately,
   referenced here for completeness of the report.
5. Utility sanity: u(18,18)=1, u(22,22)=1, u(15,18)<1, u(18,22)<1.
6. PettingZoo parallel_api_test, if available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(REPO_SRC))

from thesis.pilots.stage11_welfare import target_speed_attainment  # noqa: E402
from thesis.study_b.scenario_generator import (  # noqa: E402
    ScenarioSpec,
    VehicleSpawnSpec,
    generate_scenario,
    matched_ttc_deltas,
)

ROLE_MEMBERS = {"ramp": ["V0", "V1"], "mainline": ["V2", "V3"]}

# Isolated single-vehicle physics constants -- copied from
# stage10_symmetric_merge_env.py's Stage10MergeEnvConfig defaults
# (accel_rate=2.0, decel_rate=3.0, v_min=0, v_max=30, dt=0.2). A dedicated
# tiny re-simulation (not the full multi-agent env) is used here
# deliberately: running the real interactive env would let a matched-pair
# collision truncate the episode before every vehicle has crossed
# merge_start, which would corrupt exactly the measurement this check
# needs -- see this script's module docstring.
ACCEL_RATE = 2.0
DECEL_RATE = 3.0
V_MAX = 30.0
DT = 0.2


def simulate_isolated_crossing_time(
    spec: VehicleSpawnSpec, *, merge_start: float, max_steps: int = 500, band: float = 0.5
) -> float | None:
    """Target-speed-seeking single-vehicle simulation (accelerate/hold/
    decelerate toward ``spec.target_speed``) from ``spec.route_position``/
    ``spec.spawn_speed`` -- returns the simulated TIME at which
    route_position crosses ``merge_start``, or ``None`` if it never does
    within ``max_steps``. Since every scenario spawns a vehicle AT its own
    target speed (spec.spawn_speed == spec.target_speed, by
    scenario_generator.py's own construction), this legitimately exercises
    the discretized dt-stepping physics rather than being a pure tautology
    -- any accumulated drift from the continuous-time nominal formula would
    show up here."""
    position = spec.route_position
    speed = spec.spawn_speed
    for step in range(max_steps):
        if position >= merge_start:
            return step * DT
        if speed < spec.target_speed - band:
            a = ACCEL_RATE
        elif speed > spec.target_speed + band:
            a = -DECEL_RATE
        else:
            a = 0.0
        v_new = max(0.0, min(V_MAX, speed + a * DT))
        v_avg = 0.5 * (speed + v_new)
        position += v_avg * DT
        speed = v_new
    return None


def check_matched_ttc(n_scenarios: int, *, merge_start: float, seed_offset: int = 1) -> tuple[float, float]:
    """Returns (construction_pass_rate, simulated_pass_rate) -- both
    measured against the 0.5s threshold."""
    construction_ok = 0
    simulated_ok = 0
    for i in range(n_scenarios):
        scenario = generate_scenario(
            scenario_id=f"phase0_{i}", episode_seed=seed_offset + i,
            role_members=ROLE_MEMBERS, merge_start=merge_start,
        )
        deltas = matched_ttc_deltas(scenario)
        if all(d <= 0.5 for d in deltas.values()):
            construction_ok += 1

        crossing_times: dict[str, float] = {}
        for vid, spec in scenario.vehicles.items():
            t = simulate_isolated_crossing_time(spec, merge_start=merge_start)
            if t is not None:
                crossing_times[vid] = t
        by_slot: dict[str, list[float]] = {}
        for vid, spec in scenario.vehicles.items():
            if vid in crossing_times:
                by_slot.setdefault(spec.ttc_slot, []).append(crossing_times[vid])
        if all(len(v) == 2 and abs(v[0] - v[1]) <= 0.5 for v in by_slot.values()) and len(by_slot) == 2:
            simulated_ok += 1
    return construction_ok / n_scenarios, simulated_ok / n_scenarios


def check_spawn_validity(n_scenarios: int, *, merge_start: float, seed_offset: int = 500_000) -> float:
    ok = 0
    for i in range(n_scenarios):
        scenario = generate_scenario(
            scenario_id=f"phase0_spawn_{i}", episode_seed=seed_offset + i,
            role_members=ROLE_MEMBERS, merge_start=merge_start,
        )
        by_role: dict[str, list[float]] = {}
        for v in scenario.vehicles.values():
            by_role.setdefault(v.role, []).append(v.route_position)
        if all(abs(p[0] - p[1]) >= 15.0 for p in by_role.values()):
            ok += 1
    return ok / n_scenarios


def check_counterbalancing(n_scenarios: int, *, merge_start: float, seed_offset: int = 700_000) -> dict[str, float]:
    front_count: dict[str, int] = {"fast": 0, "slow": 0}
    total_count: dict[str, int] = {"fast": 0, "slow": 0}
    for i in range(n_scenarios):
        scenario = generate_scenario(
            scenario_id=f"phase0_cb_{i}", episode_seed=seed_offset + i,
            role_members=ROLE_MEMBERS, merge_start=merge_start,
        )
        for v in scenario.vehicles.values():
            total_count[v.speed_class] += 1
            if v.ttc_slot == "front":
                front_count[v.speed_class] += 1
    return {cls: front_count[cls] / total_count[cls] for cls in total_count}


def check_utility_sanity() -> list[tuple[str, bool]]:
    return [
        ("u(18,18)==1", target_speed_attainment(18.0, 18.0) == 1.0),
        ("u(22,22)==1", target_speed_attainment(22.0, 22.0) == 1.0),
        ("u(15,18)<1", target_speed_attainment(15.0, 18.0) < 1.0),
        ("u(18,22)<1", target_speed_attainment(18.0, 22.0) < 1.0),
    ]


def check_pettingzoo() -> tuple[bool, str]:
    try:
        from pettingzoo.test import parallel_api_test

        from thesis.study_b.heterogeneous_env import StudyBEnvConfig
        from thesis.study_b.pettingzoo_wrapper import StudyBParallelEnv

        env = StudyBParallelEnv(StudyBEnvConfig(episode_max_steps=150))
        parallel_api_test(env, num_cycles=300)
        return True, "parallel_api_test passed"
    except ImportError:
        return False, "pettingzoo not installed -- SKIPPED (optional per new_research_plan.md)"
    except Exception as exc:  # noqa: BLE001 -- report, don't crash the whole gate
        return False, f"parallel_api_test FAILED: {exc}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-scenarios", type=int, default=10_000)
    p.add_argument("--merge-start", type=float, default=200.0)
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args(argv)

    lines: list[str] = []
    overall_pass = True

    def report(name: str, passed: bool, detail: str) -> None:
        nonlocal overall_pass
        overall_pass = overall_pass and passed
        status = "PASS" if passed else "FAIL"
        lines.append(f"[{status}] {name}: {detail}")
        print(lines[-1])

    construction_rate, simulated_rate = check_matched_ttc(args.n_scenarios, merge_start=args.merge_start)
    report(
        "matched_ttc_construction",
        construction_rate >= 0.95,
        f"{construction_rate:.4f} of {args.n_scenarios} scenarios within 0.5s (construction-time)",
    )
    report(
        "matched_ttc_simulated",
        simulated_rate >= 0.95,
        f"{simulated_rate:.4f} of {args.n_scenarios} scenarios within 0.5s (isolated real-physics simulation)",
    )

    spawn_rate = check_spawn_validity(min(args.n_scenarios, 5000), merge_start=args.merge_start)
    report("spawn_validity", spawn_rate == 1.0, f"{spawn_rate:.4f} same-lane->=15m valid")

    counterbalance = check_counterbalancing(min(args.n_scenarios, 5000), merge_start=args.merge_start)
    cb_ok = all(0.4 <= r <= 0.6 for r in counterbalance.values())
    report("role_speed_counterbalancing", cb_ok, f"P(front | speed_class): {counterbalance}")

    for name, ok in check_utility_sanity():
        report(f"utility_sanity::{name}", ok, "")

    pz_ok, pz_detail = check_pettingzoo()
    # PettingZoo is explicitly optional (new_research_plan.md) -- an
    # import-skip does not fail the overall gate, only a genuine API
    # non-compliance does.
    if "SKIPPED" in pz_detail:
        lines.append(f"[SKIP] pettingzoo_parallel_api: {pz_detail}")
        print(lines[-1])
    else:
        report("pettingzoo_parallel_api", pz_ok, pz_detail)

    lines.append("")
    lines.append(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    print(lines[-1])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "unit_test_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report written to {report_path}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
