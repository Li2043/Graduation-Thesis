"""Deterministic scripted scenarios for Stage 2B-1 integration fixtures.

These are integration fixtures, not final choice-state certification scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Iterator

from thesis.envs.merge_env_v2 import HighLevelAction, MergeEnvConfig, MergeEnvV2


@dataclass
class ScenarioSpec:
    scenario_id: str
    config: MergeEnvConfig
    actions: list[dict[str, int]]
    description: str = ""


def _cfg(**kwargs: Any) -> MergeEnvConfig:
    return MergeEnvConfig(**kwargs)


def _maintain(n: int) -> list[dict[str, int]]:
    return [{"A": int(HighLevelAction.MAINTAIN), "B": int(HighLevelAction.MAINTAIN)} for _ in range(n)]


def _accel(n: int) -> list[dict[str, int]]:
    return [
        {"A": int(HighLevelAction.ACCELERATE), "B": int(HighLevelAction.ACCELERATE)}
        for _ in range(n)
    ]


def build_scenarios() -> dict[str, ScenarioSpec]:
    """Return named deterministic scenario fixtures."""
    scenarios: dict[str, ScenarioSpec] = {}

    scenarios["nominal_forward"] = ScenarioSpec(
        scenario_id="nominal_forward",
        config=_cfg(seed=0, max_steps=80, spawn_speed_A=18.0, spawn_speed_B=16.0),
        actions=_accel(40),
        description="Both accelerate forward on distinct roles",
    )

    scenarios["mainline_first"] = ScenarioSpec(
        scenario_id="mainline_first",
        config=_cfg(
            seed=1,
            role_A="mainline",
            role_B="ramp",
            spawn_route_A=150.0,
            spawn_speed_A=22.0,
            spawn_route_B=10.0,
            spawn_speed_B=12.0,
            max_steps=80,
        ),
        actions=_accel(50),
        description="Mainline controller starts closer to exit",
    )

    scenarios["ramp_first"] = ScenarioSpec(
        scenario_id="ramp_first",
        config=_cfg(
            seed=2,
            role_A="mainline",
            role_B="ramp",
            spawn_route_A=10.0,
            spawn_speed_A=12.0,
            spawn_route_B=160.0,
            spawn_speed_B=22.0,
            max_steps=80,
        ),
        actions=_accel(50),
        description="Ramp controller starts closer to its exit",
    )

    # Force near-exit spawns for clean exit events
    scenarios["A_exits_first"] = ScenarioSpec(
        scenario_id="A_exits_first",
        config=_cfg(
            seed=3,
            role_A="mainline",
            role_B="ramp",
            spawn_route_A=195.0,
            spawn_speed_A=20.0,
            spawn_route_B=20.0,
            spawn_speed_B=10.0,
            max_steps=40,
            # Keep background far away from post-exit learners
            spawn_route_B_front=2000.0,
            spawn_route_B_rear=-200.0,
            spawn_speed_B_front=0.0,
            spawn_speed_B_rear=0.0,
        ),
        actions=_accel(15),
        description="A crosses exit before B",
    )

    scenarios["B_exits_first"] = ScenarioSpec(
        scenario_id="B_exits_first",
        config=_cfg(
            seed=4,
            role_A="mainline",
            role_B="ramp",
            spawn_route_A=20.0,
            spawn_speed_A=10.0,
            spawn_route_B=200.0,
            spawn_speed_B=20.0,
            max_steps=40,
            spawn_route_B_front=2000.0,
            spawn_route_B_rear=-200.0,
            spawn_speed_B_front=0.0,
            spawn_speed_B_rear=0.0,
        ),
        actions=_accel(15),
        description="B crosses exit before A",
    )

    scenarios["simultaneous_exit"] = ScenarioSpec(
        scenario_id="simultaneous_exit",
        config=_cfg(
            seed=5,
            role_A="mainline",
            role_B="ramp",
            spawn_route_A=190.0,
            spawn_speed_A=20.0,
            spawn_route_B=206.0,
            spawn_speed_B=20.0,
            max_steps=20,
            spawn_route_B_front=2000.0,
            spawn_route_B_rear=-200.0,
            spawn_speed_B_front=0.0,
            spawn_speed_B_rear=0.0,
        ),
        actions=_accel(10),
        description="A and B cross exits nearly together without colliding",
    )

    def collision_scenario(
        name: str,
        targets: list[str],
        seed: int,
    ) -> ScenarioSpec:
        return ScenarioSpec(
            scenario_id=name,
            config=_cfg(
                seed=seed,
                max_steps=5,
                fixture_mode="controlled_collision",
                fixture_payload={
                    "collide_at_step": 1,
                    "target_ids": targets,
                    "collision_world_x": 60.0,
                    "collision_speed": 12.0,
                },
                spawn_route_B_front=90.0,
                spawn_route_B_rear=20.0,
            ),
            actions=_maintain(3),
            description=f"Forced collision involving {targets}",
        )

    scenarios["controlled_collision_A"] = collision_scenario(
        "controlled_collision_A", ["A", "B_front"], 10
    )
    scenarios["controlled_collision_B"] = collision_scenario(
        "controlled_collision_B", ["B", "B_rear"], 11
    )
    scenarios["background_front_collision"] = collision_scenario(
        "background_front_collision", ["B_front", "A"], 12
    )
    scenarios["background_rear_collision"] = collision_scenario(
        "background_rear_collision", ["B_rear", "B"], 13
    )

    scenarios["external_truncation"] = ScenarioSpec(
        scenario_id="external_truncation",
        config=_cfg(
            seed=20,
            max_steps=5,
            spawn_route_A=10.0,
            spawn_route_B=10.0,
            spawn_speed_A=5.0,
            spawn_speed_B=5.0,
            spawn_route_B_front=200.0,
            spawn_route_B_rear=-40.0,
        ),
        actions=_maintain(8),
        description="Hit max_steps without success/collision",
    )

    scenarios["hard_braking_trace"] = ScenarioSpec(
        scenario_id="hard_braking_trace",
        config=_cfg(
            seed=21,
            max_steps=20,
            decel_rate=6.0,
            spawn_speed_A=20.0,
            spawn_speed_B=18.0,
            spawn_route_B_front=250.0,
            spawn_route_B_rear=-40.0,
        ),
        actions=[
            {"A": int(HighLevelAction.DECELERATE), "B": int(HighLevelAction.MAINTAIN)}
            for _ in range(15)
        ],
        description="A hard-brakes for braking-cost diagnostics",
    )

    return scenarios


def run_scenario(
    spec: ScenarioSpec,
) -> tuple[MergeEnvV2, list[dict[str, Any]]]:
    """Execute a scenario; return env and per-step transition records (partial)."""
    env = MergeEnvV2(spec.config)
    obs, info = env.reset(seed=spec.config.seed)
    records: list[dict[str, Any]] = []
    for action in spec.actions:
        obs, reward, terminated, truncated, step_info = env.step(action)
        records.append(
            {
                "scenario_id": spec.scenario_id,
                "obs": obs,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": step_info,
            }
        )
        if terminated or truncated:
            break
    return env, records


def iter_scripted_transitions(
    scenario_ids: list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    specs = build_scenarios()
    ids = scenario_ids or list(specs.keys())
    for sid in ids:
        _, records = run_scenario(specs[sid])
        for rec in records:
            yield rec
