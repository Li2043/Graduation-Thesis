"""Matched initial-condition blocks and deterministic scripted audit scenarios.

Stage 3A uses scripted actions only. Primary ranking scenarios evolve through
environment dynamics without teleportation. Fixture-injected collisions are
marked ``fixture_only=True`` and excluded from behavioural ranking.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from thesis.envs.merge_env_v2 import HighLevelAction, MergeEnvConfig


ACC = int(HighLevelAction.ACCELERATE)
DEC = int(HighLevelAction.DECELERATE)
MNT = int(HighLevelAction.MAINTAIN)


@dataclass(frozen=True)
class MatchedBlock:
    """Fixed initial condition shared by all scripts in the block."""

    block_id: str
    seed: int
    role_A: str
    role_B: str
    spawn_route_A: float
    spawn_route_B: float
    spawn_speed_A: float
    spawn_speed_B: float
    spawn_route_B_front: float
    spawn_route_B_rear: float
    spawn_speed_B_front: float
    spawn_speed_B_rear: float
    target_speed: float
    max_steps: int
    dt: float = 0.2
    collision_distance: float = 4.0
    # Allow reverse for oscillation audit only when applied in scenario config
    v_min: float = 0.0
    description: str = ""

    def base_config(self, **overrides: Any) -> MergeEnvConfig:
        kwargs = dict(
            seed=self.seed,
            role_A=self.role_A,
            role_B=self.role_B,
            spawn_route_A=self.spawn_route_A,
            spawn_route_B=self.spawn_route_B,
            spawn_speed_A=self.spawn_speed_A,
            spawn_speed_B=self.spawn_speed_B,
            spawn_route_B_front=self.spawn_route_B_front,
            spawn_route_B_rear=self.spawn_route_B_rear,
            spawn_speed_B_front=self.spawn_speed_B_front,
            spawn_speed_B_rear=self.spawn_speed_B_rear,
            target_speed=self.target_speed,
            max_steps=self.max_steps,
            dt=self.dt,
            collision_distance=self.collision_distance,
            v_min=self.v_min,
            # Keep background far from learners for safe scripts by default
            fixture_mode=None,
            fixture_payload={},
        )
        kwargs.update(overrides)
        return MergeEnvConfig(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AuditScenario:
    block_id: str
    scenario_id: str
    config: MergeEnvConfig
    actions: list[dict[str, int]]
    fixture_only: bool = False
    primary_ranking: bool = True
    description: str = ""
    # Exact post-reset route overrides (removes spawn jitter for matched ICs)
    fix_route_A: float | None = None
    fix_route_B: float | None = None
    fix_speed_A: float | None = None
    fix_speed_B: float | None = None
    # Oscillation / reverse needs negative v_min
    force_v_min: float | None = None


def _seq(actions_A: list[int], actions_B: list[int]) -> list[dict[str, int]]:
    if len(actions_A) != len(actions_B):
        raise ValueError("action sequences must have equal length")
    return [{"A": a, "B": b} for a, b in zip(actions_A, actions_B)]


def _repeat(a: int, b: int, n: int) -> list[dict[str, int]]:
    return [{"A": a, "B": b} for _ in range(n)]


def build_matched_blocks() -> list[MatchedBlock]:
    """Eight fixed matched blocks. Initial conditions are not tuned after ranking."""
    # Background parked far away for most blocks
    far_front, far_rear = 2000.0, -200.0
    blocks = [
        MatchedBlock(
            block_id="block_001",
            seed=101,
            role_A="mainline",
            role_B="ramp",
            spawn_route_A=40.0,
            spawn_route_B=30.0,
            spawn_speed_A=16.0,
            spawn_speed_B=15.0,
            spawn_route_B_front=far_front,
            spawn_route_B_rear=far_rear,
            spawn_speed_B_front=0.0,
            spawn_speed_B_rear=0.0,
            target_speed=20.0,
            max_steps=220,
            description="Moderate gap, moderate speeds",
        ),
        MatchedBlock(
            block_id="block_002",
            seed=102,
            role_A="mainline",
            role_B="ramp",
            spawn_route_A=35.0,
            spawn_route_B=45.0,
            spawn_speed_A=14.0,
            spawn_speed_B=17.0,
            spawn_route_B_front=far_front,
            spawn_route_B_rear=far_rear,
            spawn_speed_B_front=0.0,
            spawn_speed_B_rear=0.0,
            target_speed=20.0,
            max_steps=220,
            description="Ramp slightly ahead in route progress toward merge",
        ),
        MatchedBlock(
            block_id="block_003",
            seed=103,
            role_A="mainline",
            role_B="ramp",
            spawn_route_A=55.0,
            spawn_route_B=25.0,
            spawn_speed_A=18.0,
            spawn_speed_B=13.0,
            spawn_route_B_front=far_front,
            spawn_route_B_rear=far_rear,
            spawn_speed_B_front=0.0,
            spawn_speed_B_rear=0.0,
            target_speed=20.0,
            max_steps=220,
            description="Mainline closer to merge, higher speed",
        ),
        MatchedBlock(
            block_id="block_004",
            seed=104,
            role_A="mainline",
            role_B="ramp",
            spawn_route_A=30.0,
            spawn_route_B=20.0,
            spawn_speed_A=12.0,
            spawn_speed_B=12.0,
            spawn_route_B_front=far_front,
            spawn_route_B_rear=far_rear,
            spawn_speed_B_front=0.0,
            spawn_speed_B_rear=0.0,
            target_speed=18.0,
            max_steps=260,
            description="Lower speeds, smaller merge gap",
        ),
        MatchedBlock(
            block_id="block_005",
            seed=105,
            role_A="mainline",
            role_B="ramp",
            spawn_route_A=45.0,
            spawn_route_B=50.0,
            spawn_speed_A=20.0,
            spawn_speed_B=14.0,
            spawn_route_B_front=far_front,
            spawn_route_B_rear=far_rear,
            spawn_speed_B_front=0.0,
            spawn_speed_B_rear=0.0,
            target_speed=22.0,
            max_steps=220,
            description="Higher mainline speed, ramp nearer join",
        ),
        MatchedBlock(
            block_id="block_006",
            seed=106,
            role_A="mainline",
            role_B="ramp",
            spawn_route_A=25.0,
            spawn_route_B=35.0,
            spawn_speed_A=15.0,
            spawn_speed_B=18.0,
            spawn_route_B_front=far_front,
            spawn_route_B_rear=far_rear,
            spawn_speed_B_front=0.0,
            spawn_speed_B_rear=0.0,
            target_speed=20.0,
            max_steps=240,
            description="Ramp faster; both safe orders achievable",
        ),
        MatchedBlock(
            block_id="block_007",
            seed=107,
            role_A="mainline",
            role_B="ramp",
            spawn_route_A=50.0,
            spawn_route_B=40.0,
            spawn_speed_A=17.0,
            spawn_speed_B=16.0,
            spawn_route_B_front=far_front,
            spawn_route_B_rear=far_rear,
            spawn_speed_B_front=0.0,
            spawn_speed_B_rear=0.0,
            target_speed=20.0,
            max_steps=220,
            description="Near-merge starts, medium speeds",
        ),
        MatchedBlock(
            block_id="block_008",
            seed=108,
            role_A="mainline",
            role_B="ramp",
            spawn_route_A=38.0,
            spawn_route_B=28.0,
            spawn_speed_A=13.0,
            spawn_speed_B=19.0,
            spawn_route_B_front=far_front,
            spawn_route_B_rear=far_rear,
            spawn_speed_B_front=0.0,
            spawn_speed_B_rear=0.0,
            target_speed=20.0,
            max_steps=240,
            description="Asymmetric speeds; larger arrival-time difference",
        ),
    ]
    return blocks


def _safe_actions_mainline_first(n_fast: int = 90, n_yield: int = 15) -> list[dict[str, int]]:
    # Mainline (A) accelerates; ramp (B) yields briefly then accelerates
    acts = _repeat(ACC, DEC, n_yield) + _repeat(ACC, ACC, n_fast)
    return acts


def _safe_actions_ramp_first(n_yield: int = 15, n_fast: int = 90) -> list[dict[str, int]]:
    # Mainline yields; ramp accelerates
    return _repeat(DEC, ACC, n_yield) + _repeat(ACC, ACC, n_fast)


def _safe_actions_simultaneous(n: int = 100) -> list[dict[str, int]]:
    return _repeat(ACC, ACC, n)


def _slow_safe(n_slow: int = 160) -> list[dict[str, int]]:
    # Mostly maintain with sparse accelerate → late completion
    acts: list[dict[str, int]] = []
    for i in range(n_slow):
        if i % 4 == 0:
            acts.append({"A": ACC, "B": ACC})
        else:
            acts.append({"A": MNT, "B": MNT})
    return acts


def build_block_scenarios(block: MatchedBlock) -> list[AuditScenario]:
    """All primary + fixture scenarios for one matched block."""
    out: list[AuditScenario] = []
    bid = block.block_id
    fix = dict(
        fix_route_A=block.spawn_route_A,
        fix_route_B=block.spawn_route_B,
        fix_speed_A=block.spawn_speed_A,
        fix_speed_B=block.spawn_speed_B,
    )

    out.append(
        AuditScenario(
            block_id=bid,
            scenario_id="safe_mainline_first",
            config=block.base_config(max_steps=block.max_steps),
            actions=_safe_actions_mainline_first(),
            description="Both complete; mainline role exits first",
            **fix,
        )
    )
    out.append(
        AuditScenario(
            block_id=bid,
            scenario_id="safe_ramp_first",
            config=block.base_config(max_steps=block.max_steps),
            actions=_safe_actions_ramp_first(),
            description="Both complete; ramp role exits first",
            **fix,
        )
    )
    out.append(
        AuditScenario(
            block_id=bid,
            scenario_id="safe_near_simultaneous",
            config=block.base_config(max_steps=block.max_steps),
            actions=_safe_actions_simultaneous(),
            description="Both complete with small exit-time difference",
            **fix,
        )
    )
    out.append(
        AuditScenario(
            block_id=bid,
            scenario_id="slow_safe_mainline_first",
            config=block.base_config(max_steps=max(block.max_steps, 280)),
            actions=_repeat(ACC, DEC, 10) + _slow_safe(200),
            description="Late safe completion, mainline-leaning yield",
            **fix,
        )
    )
    out.append(
        AuditScenario(
            block_id=bid,
            scenario_id="slow_safe_ramp_first",
            config=block.base_config(max_steps=max(block.max_steps, 280)),
            actions=_repeat(DEC, ACC, 10) + _slow_safe(200),
            description="Late safe completion, ramp-leaning yield",
            **fix,
        )
    )
    out.append(
        AuditScenario(
            block_id=bid,
            scenario_id="stall_at_start",
            config=block.base_config(max_steps=40),
            actions=_repeat(MNT, MNT, 50),
            description="No meaningful progress until truncation",
            **fix,
        )
    )
    out.append(
        AuditScenario(
            block_id=bid,
            scenario_id="stall_after_partial_progress",
            config=block.base_config(max_steps=50),
            actions=_repeat(ACC, ACC, 12) + _repeat(MNT, MNT, 50),
            description="Partial progress then unresolved truncation",
            **fix,
        )
    )

    # Physical early collision: place both on shared mainline with closing speeds
    # A mainline at 70, B ramp past join (route 85 → world ≈ 75), B faster closing?
    # B world = 50 + (route-60) = route - 10. For B world ~ 66 and A at 70:
    # B route = 76, world=66; A=70. B speed high, A low → catch from behind.
    out.append(
        AuditScenario(
            block_id=bid,
            scenario_id="early_collision",
            config=block.base_config(
                max_steps=40,
                spawn_route_A=70.0,
                spawn_route_B=76.0,
                spawn_speed_A=8.0,
                spawn_speed_B=22.0,
            ),
            actions=_repeat(MNT, ACC, 30),
            description="Physically evolved early collision on shared lane",
            fix_route_A=70.0,
            fix_route_B=76.0,
            fix_speed_A=8.0,
            fix_speed_B=22.0,
        )
    )
    # Late collision: progress far then close the gap
    out.append(
        AuditScenario(
            block_id=bid,
            scenario_id="late_collision",
            config=block.base_config(
                max_steps=80,
                spawn_route_A=120.0,
                spawn_route_B=125.0,
                spawn_speed_A=10.0,
                spawn_speed_B=24.0,
            ),
            actions=_repeat(ACC, ACC, 8) + _repeat(MNT, ACC, 40),
            description="Substantial progress then physical collision",
            fix_route_A=120.0,
            fix_route_B=125.0,
            fix_speed_A=10.0,
            fix_speed_B=24.0,
        )
    )
    out.append(
        AuditScenario(
            block_id=bid,
            scenario_id="hard_braking_safe",
            config=block.base_config(max_steps=block.max_steps, decel_rate=6.0),
            actions=_repeat(DEC, ACC, 8) + _repeat(ACC, ACC, 100),
            description="Hard braking then safe completion",
            **fix,
        )
    )
    # Oscillation: enable reverse via v_min < 0
    osc_actions: list[dict[str, int]] = []
    for _ in range(6):
        osc_actions.extend(_repeat(ACC, ACC, 4))
        osc_actions.extend(_repeat(DEC, DEC, 8))  # reverse when v_min < 0
        osc_actions.extend(_repeat(ACC, ACC, 4))
    out.append(
        AuditScenario(
            block_id=bid,
            scenario_id="oscillation_closed_cycle",
            config=block.base_config(max_steps=80, v_min=-12.0, decel_rate=4.0),
            actions=osc_actions,
            description="Closed route-progress cycles (reverse enabled)",
            force_v_min=-12.0,
            **fix,
        )
    )
    out.append(
        AuditScenario(
            block_id=bid,
            scenario_id="reverse_then_recover",
            config=block.base_config(max_steps=block.max_steps, v_min=-10.0, decel_rate=4.0),
            actions=_repeat(DEC, DEC, 10) + _repeat(ACC, ACC, 120),
            description="Brief reverse then recover to safe completion if possible",
            force_v_min=-10.0,
            **fix,
        )
    )

    # Fixture-only stakeholder collision diagnostics (excluded from ranking)
    for target, name in (
        (["A", "B_front"], "fixture_collision_A"),
        (["B", "B_rear"], "fixture_collision_B"),
        (["B_front", "A"], "fixture_collision_B_front"),
        (["B_rear", "B"], "fixture_collision_B_rear"),
    ):
        out.append(
            AuditScenario(
                block_id=bid,
                scenario_id=name,
                config=block.base_config(
                    max_steps=5,
                    fixture_mode="controlled_collision",
                    fixture_payload={
                        "collide_at_step": 1,
                        "target_ids": target,
                        "collision_world_x": 60.0,
                        "collision_speed": 12.0,
                    },
                ),
                actions=_repeat(MNT, MNT, 3),
                fixture_only=True,
                primary_ranking=False,
                description=f"Fixture-only collision involving {target}",
                **fix,
            )
        )
    return out


def build_all_audit_scenarios() -> list[AuditScenario]:
    scenarios: list[AuditScenario] = []
    for block in build_matched_blocks():
        scenarios.extend(build_block_scenarios(block))
    return scenarios
