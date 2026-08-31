from __future__ import annotations

import pytest

from thesis.study_b.scenario_generator import (
    MIN_SAME_LANE_GAP_M,
    ScenarioSpec,
    generate_scenario,
    matched_ttc_deltas,
)

ROLE_MEMBERS = {"ramp": ["V0", "V1"], "mainline": ["V2", "V3"]}


def test_generate_scenario_produces_four_vehicles_two_per_role():
    spec = generate_scenario(scenario_id="t1", episode_seed=1, role_members=ROLE_MEMBERS)
    assert set(spec.vehicles.keys()) == {"V0", "V1", "V2", "V3"}
    members = spec.role_members()
    assert len(members["ramp"]) == 2
    assert len(members["mainline"]) == 2


def test_generate_scenario_heterogeneous_has_one_fast_one_slow_per_role():
    spec = generate_scenario(scenario_id="t2", episode_seed=2, role_members=ROLE_MEMBERS)
    for role, members in spec.role_members().items():
        classes = sorted(spec.vehicles[m].speed_class for m in members)
        assert classes == ["fast", "slow"], f"role {role} classes: {classes}"


def test_generate_scenario_homogeneous_all_at_v_ref():
    spec = generate_scenario(
        scenario_id="t3", episode_seed=3, role_members=ROLE_MEMBERS, traffic_type="homogeneous"
    )
    for v in spec.vehicles.values():
        assert v.target_speed == pytest.approx(20.0)
        assert v.speed_class == "homogeneous"


def test_generate_scenario_has_one_front_one_rear_per_role():
    spec = generate_scenario(scenario_id="t4", episode_seed=4, role_members=ROLE_MEMBERS)
    for role, members in spec.role_members().items():
        slots = sorted(spec.vehicles[m].ttc_slot for m in members)
        assert slots == ["front", "rear"], f"role {role} slots: {slots}"


def test_spawn_speed_equals_target_speed():
    spec = generate_scenario(scenario_id="t5", episode_seed=5, role_members=ROLE_MEMBERS)
    for v in spec.vehicles.values():
        assert v.spawn_speed == pytest.approx(v.target_speed)


def test_same_scenario_id_and_seed_is_deterministic():
    a = generate_scenario(scenario_id="t6", episode_seed=42, role_members=ROLE_MEMBERS)
    b = generate_scenario(scenario_id="t6", episode_seed=42, role_members=ROLE_MEMBERS)
    for vid in a.vehicles:
        assert a.vehicles[vid] == b.vehicles[vid]


def test_matched_ttc_deltas_within_construction_tolerance():
    # By construction (individual jitter +/-0.05s on each side, shared
    # jitter cancels within a slot pair), |delta TTC| per slot pair must
    # never exceed 2 * 0.05 = 0.10s -- well inside the 0.5s Phase 0
    # acceptance threshold from new_research_plan.md.
    spec = generate_scenario(scenario_id="t7", episode_seed=7, role_members=ROLE_MEMBERS)
    deltas = matched_ttc_deltas(spec)
    assert set(deltas.keys()) == {"front", "rear"}
    for slot, delta in deltas.items():
        assert delta <= 0.10 + 1e-9, f"{slot} delta={delta}"


def test_matched_ttc_at_scale_95_percent_within_half_second():
    n = 2000
    ok = 0
    for seed in range(n):
        spec = generate_scenario(scenario_id=f"scale_{seed}", episode_seed=seed, role_members=ROLE_MEMBERS)
        deltas = matched_ttc_deltas(spec)
        if all(d <= 0.5 for d in deltas.values()):
            ok += 1
    assert ok / n >= 0.95


def test_same_lane_spawn_validity_at_scale():
    n = 2000
    for seed in range(n):
        spec = generate_scenario(scenario_id=f"valid_{seed}", episode_seed=seed, role_members=ROLE_MEMBERS)
        by_role: dict[str, list[float]] = {}
        for v in spec.vehicles.values():
            by_role.setdefault(v.role, []).append(v.route_position)
        for positions in by_role.values():
            assert abs(positions[0] - positions[1]) >= MIN_SAME_LANE_GAP_M


def test_role_member_count_validation():
    with pytest.raises(ValueError):
        generate_scenario(
            scenario_id="bad",
            episode_seed=1,
            role_members={"ramp": ["V0", "V1", "V2"], "mainline": ["V3"]},
        )


def test_route_position_matches_nominal_ttc_formula():
    spec = generate_scenario(scenario_id="t8", episode_seed=8, role_members=ROLE_MEMBERS, merge_start=200.0)
    for v in spec.vehicles.values():
        expected = 200.0 - v.target_speed * v.nominal_ttc
        assert v.route_position == pytest.approx(expected)
