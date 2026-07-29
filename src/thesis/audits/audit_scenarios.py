"""Matched initial-condition blocks and deterministic scripted audit scenarios.

Stage 3A uses scripted actions only. Primary ranking scenarios evolve through
environment dynamics without teleportation. Fixture-injected collisions are
marked ``fixture_only=True`` and excluded from behavioural ranking.

Matched blocks share one initial state per block across all scripts. Blocks are
chosen so both safe orderings are physically achievable; ICs are not tuned
after observing reward rankings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
            fixture_mode=None,
            fixture_payload={},
        )
        kwargs.update(overrides)
        return MergeEnvConfig(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def ic_fix(self) -> dict[str, float]:
        return {
            "fix_route_A": self.spawn_route_A,
            "fix_route_B": self.spawn_route_B,
            "fix_speed_A": self.spawn_speed_A,
            "fix_speed_B": self.spawn_speed_B,
        }


@dataclass
class AuditScenario:
    block_id: str
    scenario_id: str
    config: MergeEnvConfig
    actions: list[dict[str, int]]
    fixture_only: bool = False
    primary_ranking: bool = True
    description: str = ""
    fix_route_A: float | None = None
    fix_route_B: float | None = None
    fix_speed_A: float | None = None
    fix_speed_B: float | None = None
    force_v_min: float | None = None


def _seq(actions_A: list[int], actions_B: list[int]) -> list[dict[str, int]]:
    if len(actions_A) != len(actions_B):
        raise ValueError("action sequences must have equal length")
    return [{"A": a, "B": b} for a, b in zip(actions_A, actions_B)]


def _repeat(a: int, b: int, n: int) -> list[dict[str, int]]:
    return [{"A": a, "B": b} for _ in range(n)]


def build_matched_blocks() -> list[MatchedBlock]:
    """Eight fixed matched blocks covering gap/speed/arrival variation.

    Geometry places mainline A behind the join and ramp B on the approach so
    either vehicle can clear first under different yield scripts.
    """
    far_front, far_rear = 2000.0, -200.0
    # Speeds kept in a narrow band so physical early/late collisions and both
    # safe orders remain achievable from the shared IC without teleportation.
    specs = [
        ("block_001", 101, 20.0, 40.0, 9.0, 9.0, "Baseline equal moderate-low speeds"),
        ("block_002", 102, 18.0, 39.0, 9.0, 9.0, "Slightly smaller gap"),
        ("block_003", 103, 22.0, 42.0, 9.0, 9.0, "Slightly larger gap"),
        ("block_004", 104, 21.0, 41.0, 9.5, 9.0, "Mainline mild arrival lead"),
        ("block_005", 105, 20.0, 42.0, 9.0, 10.0, "Ramp faster arrival tendency"),
        ("block_006", 106, 16.0, 40.0, 9.0, 9.0, "Larger initial merge separation"),
        ("block_007", 107, 24.0, 40.0, 9.0, 9.0, "Smaller separation; A closer to join"),
        ("block_008", 108, 19.0, 41.0, 9.5, 9.0, "Mild asymmetric speeds and gap"),
    ]
    return [
        MatchedBlock(
            block_id=bid,
            seed=seed,
            role_A="mainline",
            role_B="ramp",
            spawn_route_A=ra,
            spawn_route_B=rb,
            spawn_speed_A=sa,
            spawn_speed_B=sb,
            spawn_route_B_front=far_front,
            spawn_route_B_rear=far_rear,
            spawn_speed_B_front=0.0,
            spawn_speed_B_rear=0.0,
            target_speed=20.0,
            max_steps=260,
            description=desc,
        )
        for bid, seed, ra, rb, sa, sb, desc in specs
    ]


def _safe_actions_mainline_first() -> list[dict[str, int]]:
    # Brief yield on ramp (DEC), then coast, then both accelerate
    return _repeat(ACC, DEC, 12) + _repeat(ACC, MNT, 50) + _repeat(ACC, ACC, 160)


def _safe_actions_ramp_first() -> list[dict[str, int]]:
    return _repeat(DEC, ACC, 12) + _repeat(MNT, ACC, 50) + _repeat(ACC, ACC, 160)


def _safe_actions_simultaneous() -> list[dict[str, int]]:
    # Mild mutual coast then accelerate — keeps exit times close without forcing contact
    return _repeat(MNT, MNT, 4) + _repeat(ACC, ACC, 160)


def _slow_cruise(n: int) -> list[dict[str, int]]:
    acts: list[dict[str, int]] = []
    for i in range(n):
        if i % 5 == 0:
            acts.append({"A": ACC, "B": ACC})
        else:
            acts.append({"A": MNT, "B": MNT})
    return acts


def _slow_safe_mainline() -> list[dict[str, int]]:
    return _repeat(ACC, DEC, 12) + _slow_cruise(220)


def _slow_safe_ramp() -> list[dict[str, int]]:
    return _repeat(DEC, ACC, 12) + _slow_cruise(220)


def _early_collision_actions() -> list[dict[str, int]]:
    # B joins then brakes; A closes from behind (short horizon)
    return _repeat(ACC, ACC, 4) + _repeat(ACC, DEC, 12) + _repeat(ACC, MNT, 40)


def _late_collision_actions() -> list[dict[str, int]]:
    return (
        _repeat(ACC, ACC, 5)
        + _repeat(MNT, ACC, 20)
        + _repeat(ACC, DEC, 20)
        + _repeat(ACC, MNT, 40)
    )


def _hard_braking_safe_actions() -> list[dict[str, int]]:
    # Extra hard-braking burst beyond the matched smooth mainline-first script
    return (
        _repeat(ACC, DEC, 12)
        + _repeat(DEC, MNT, 4)
        + _repeat(ACC, MNT, 50)
        + _repeat(ACC, ACC, 160)
    )


def _oscillation_actions() -> list[dict[str, int]]:
    """Brake toward rest, then closed forward/reverse micro-cycles."""
    acts = _repeat(DEC, DEC, 30)
    for _ in range(6):
        acts.extend(_repeat(ACC, ACC, 2))
        acts.extend(_repeat(DEC, DEC, 4))
    return acts


def build_block_scenarios(block: MatchedBlock) -> list[AuditScenario]:
    """All primary + fixture scenarios for one matched block (shared IC)."""
    out: list[AuditScenario] = []
    bid = block.block_id
    fix = block.ic_fix()

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
            config=block.base_config(max_steps=max(block.max_steps, 320)),
            actions=_slow_safe_mainline(),
            description="Late safe completion, mainline-first leaning",
            **fix,
        )
    )
    out.append(
        AuditScenario(
            block_id=bid,
            scenario_id="slow_safe_ramp_first",
            config=block.base_config(max_steps=max(block.max_steps, 320)),
            actions=_slow_safe_ramp(),
            description="Late safe completion, ramp-first leaning",
            **fix,
        )
    )
    out.append(
        AuditScenario(
            block_id=bid,
            scenario_id="stall_at_start",
            config=block.base_config(max_steps=40),
            actions=_repeat(DEC, DEC, 25) + _repeat(MNT, MNT, 40),
            description="Brake to stop then no progress until truncation",
            **fix,
        )
    )
    out.append(
        AuditScenario(
            block_id=bid,
            scenario_id="stall_after_partial_progress",
            config=block.base_config(max_steps=50),
            actions=_repeat(ACC, ACC, 5) + _repeat(DEC, DEC, 28) + _repeat(MNT, MNT, 40),
            description="Partial progress, stop, then unresolved truncation",
            **fix,
        )
    )
    out.append(
        AuditScenario(
            block_id=bid,
            scenario_id="early_collision",
            config=block.base_config(max_steps=60, decel_rate=4.0),
            actions=_early_collision_actions(),
            description="Physically evolved early collision on shared lane",
            **fix,
        )
    )
    out.append(
        AuditScenario(
            block_id=bid,
            scenario_id="late_collision",
            config=block.base_config(max_steps=120, decel_rate=6.0),
            actions=_late_collision_actions(),
            description="Substantial progress then physical collision",
            **fix,
        )
    )
    out.append(
        AuditScenario(
            block_id=bid,
            scenario_id="hard_braking_safe",
            config=block.base_config(max_steps=block.max_steps),
            actions=_hard_braking_safe_actions(),
            description="Extra braking burst then safe completion",
            **fix,
        )
    )
    out.append(
        AuditScenario(
            block_id=bid,
            scenario_id="oscillation_closed_cycle",
            config=block.base_config(
                max_steps=80,
                v_min=-2.5,
                accel_rate=2.5,
                decel_rate=2.5,
            ),
            actions=_oscillation_actions(),
            description="Closed route-progress cycles (reverse enabled)",
            force_v_min=-2.5,
            **fix,
        )
    )
    out.append(
        AuditScenario(
            block_id=bid,
            scenario_id="reverse_then_recover",
            config=block.base_config(max_steps=block.max_steps, v_min=-4.0, decel_rate=3.0),
            actions=_repeat(DEC, DEC, 10) + _safe_actions_mainline_first(),
            description="Brief reverse then recover to safe completion",
            force_v_min=-4.0,
            **fix,
        )
    )

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
