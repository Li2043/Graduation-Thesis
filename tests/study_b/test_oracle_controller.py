"""Unit tests for oracle_controller.py -- pure-logic tests, no environment
required. VDN_Conditional_Amendment_Protocol.md sec 8 (Diagnostic 3)."""

from __future__ import annotations

import pytest

from thesis.study_b.oracle_controller import (
    ACCELERATE,
    DECELERATE,
    MAINTAIN,
    oracle_action_for_vehicle,
    oracle_actions,
    partner_id,
)
from thesis.study_b.scenario_generator import generate_scenario

MERGE_START = 200.0
MERGE_END = 300.0


def test_both_far_from_merge_start_maintains():
    action = oracle_action_for_vehicle(
        self_position=100.0, partner_position=110.0, merge_start=MERGE_START, merge_end=MERGE_END,
    )
    assert action == MAINTAIN


def test_self_clearly_closer_accelerates():
    action = oracle_action_for_vehicle(
        self_position=190.0, partner_position=150.0, merge_start=MERGE_START, merge_end=MERGE_END,
    )
    assert action == ACCELERATE


def test_partner_clearly_closer_decelerates():
    action = oracle_action_for_vehicle(
        self_position=150.0, partner_position=190.0, merge_start=MERGE_START, merge_end=MERGE_END,
    )
    assert action == DECELERATE


def test_self_already_past_merge_end_always_accelerates():
    action = oracle_action_for_vehicle(
        self_position=310.0, partner_position=150.0, merge_start=MERGE_START, merge_end=MERGE_END,
    )
    assert action == ACCELERATE


def test_partner_already_past_merge_end_frees_self_to_accelerate():
    # self is right at the interaction window and would normally yield,
    # but partner has already cleared merge_end -- no more conflict.
    action = oracle_action_for_vehicle(
        self_position=190.0, partner_position=305.0, merge_start=MERGE_START, merge_end=MERGE_END,
    )
    assert action == ACCELERATE


def test_tie_breaks_toward_closer_vehicle():
    # self_position=185 -> self_dist=15; partner_position=184 -> partner_dist=16
    # (route_position increases toward merge_start, so the LARGER position is closer).
    # self is closer (15 < 16) -> self gets priority.
    close_wins = oracle_action_for_vehicle(
        self_position=185.0, partner_position=184.0, merge_start=MERGE_START, merge_end=MERGE_END,
    )
    assert close_wins == ACCELERATE

    equal_distance = oracle_action_for_vehicle(
        self_position=185.0, partner_position=185.0, merge_start=MERGE_START, merge_end=MERGE_END,
    )
    assert equal_distance == ACCELERATE  # tie-break: self<=partner is True on exact equality


def test_tight_same_lane_gap_forces_decelerate_even_if_merge_priority_says_go():
    # self is clearly closer to merge_start than partner (would normally
    # ACCELERATE per the merge-conflict rule), but a same-lane vehicle
    # ahead is dangerously close -- safety rule 1 must win.
    action = oracle_action_for_vehicle(
        self_position=190.0, partner_position=150.0, merge_start=MERGE_START, merge_end=MERGE_END,
        same_lane_ahead_gap=5.0,
    )
    assert action == DECELERATE


def test_comfortable_same_lane_gap_does_not_override_merge_rule():
    action = oracle_action_for_vehicle(
        self_position=190.0, partner_position=150.0, merge_start=MERGE_START, merge_end=MERGE_END,
        same_lane_ahead_gap=100.0,  # comfortably above DEFAULT_MIN_FOLLOWING_GAP_M=60
    )
    assert action == ACCELERATE


def test_no_same_lane_vehicle_ahead_does_not_trigger_safety_rule():
    action = oracle_action_for_vehicle(
        self_position=190.0, partner_position=150.0, merge_start=MERGE_START, merge_end=MERGE_END,
        same_lane_ahead_gap=None,
    )
    assert action == ACCELERATE


def _make_scenario():
    return generate_scenario(
        scenario_id="oracle-test", episode_seed=7,
        role_members={"ramp": ["V0", "V1"], "mainline": ["V2", "V3"]},
        traffic_type="heterogeneous",
    )


def test_partner_id_pairs_opposite_role_same_ttc_slot():
    scenario = _make_scenario()
    for vid, spec in scenario.vehicles.items():
        partner = partner_id(scenario, vid)
        partner_spec = scenario.vehicles[partner]
        assert partner_spec.role != spec.role
        assert partner_spec.ttc_slot == spec.ttc_slot
        assert partner_id(scenario, partner) == vid  # symmetric


def test_oracle_actions_covers_every_requested_vehicle():
    scenario = _make_scenario()
    positions = {vid: spec.route_position for vid, spec in scenario.vehicles.items()}
    actions = oracle_actions(scenario=scenario, positions=positions, merge_start=MERGE_START, merge_end=MERGE_END)
    assert set(actions) == set(scenario.vehicles)
    assert all(a in (MAINTAIN, ACCELERATE, DECELERATE) for a in actions.values())


def test_oracle_actions_forces_trailing_same_lane_vehicle_to_decelerate():
    scenario = _make_scenario()
    # Manually place a same-role pair (whichever two share "ramp") 3m apart
    # -- well under min_following_gap -- with the leading one already past
    # everything so only the same-lane rule can explain a DECELERATE.
    ramp_ids = [vid for vid, spec in scenario.vehicles.items() if spec.role == "ramp"]
    mainline_ids = [vid for vid, spec in scenario.vehicles.items() if spec.role == "mainline"]
    positions = {vid: spec.route_position for vid, spec in scenario.vehicles.items()}
    positions[ramp_ids[0]] = 350.0  # leading ramp vehicle, already past merge_end
    positions[ramp_ids[1]] = 347.0  # trailing ramp vehicle, 3m behind -- unsafe gap
    positions[mainline_ids[0]] = 10.0
    positions[mainline_ids[1]] = 10.0
    actions = oracle_actions(scenario=scenario, positions=positions, merge_start=MERGE_START, merge_end=MERGE_END)
    assert actions[ramp_ids[1]] == DECELERATE


def test_oracle_actions_respects_active_filter():
    scenario = _make_scenario()
    positions = {vid: spec.route_position for vid, spec in scenario.vehicles.items()}
    active = {vid: (vid != "V0") for vid in scenario.vehicles}
    actions = oracle_actions(
        scenario=scenario, positions=positions, merge_start=MERGE_START, merge_end=MERGE_END, active_vehicle_ids=active,
    )
    assert "V0" not in actions
    assert set(actions) == {"V1", "V2", "V3"}


def test_oracle_controller_passes_environment_feasibility_on_q_bank():
    """End-to-end, real-physics regression test for VDN_Conditional_Amendment_Protocol.md
    sec 8's environment-feasibility gate (completion > 0.90). Locks in the
    empirically-tuned default thresholds against future accidental
    regressions (e.g. someone "simplifying" the defaults back down)."""
    from thesis.study_b.heterogeneous_env import StudyBEnvConfig, StudyBHeterogeneousEnv
    from thesis.study_b.training_common import load_scenario_bank

    scenario_bank = (
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "experiments/pilots/study_b_fairness_mappo/scenario_banks/Q.json"
    )
    if not scenario_bank.exists():
        pytest.skip("requires the frozen Q scenario bank on disk")

    scenarios = load_scenario_bank(scenario_bank)
    env = StudyBHeterogeneousEnv(StudyBEnvConfig(episode_max_steps=200))
    n_success = 0
    for scenario in scenarios:
        obs, _info = env.reset(seed=0, scenario=scenario)
        term_reason = "truncation"
        for _t in range(200):
            positions = {vid: env._env._vehicles[vid].route_position for vid in env.active_vehicle_ids}  # noqa: SLF001
            actions = oracle_actions(
                scenario=scenario, positions=positions, merge_start=MERGE_START, merge_end=MERGE_END,
                active_vehicle_ids={vid: True for vid in env.active_vehicle_ids},
            )
            obs, _r, terminated, truncated, step_info = env.step(actions)
            if terminated or truncated:
                term_reason = step_info["term_reason"]
                break
        if term_reason == "success":
            n_success += 1
    completion_rate = n_success / len(scenarios)
    assert completion_rate > 0.90, f"environment-feasibility gate failed: completion_rate={completion_rate:.3f}"


def test_lateral_positions_default_none_is_byte_identical_to_role_based_grouping():
    """2026-08-16 CONTROL_AUTHORITY amendment: lateral_positions is
    optional and defaults to None -- every existing caller (legacy
    backend, direct_accel HighwayEnv oracle) must see EXACTLY the same
    role-based same-lane grouping as before this parameter existed."""
    scenario = _make_scenario()
    ramp_ids = [vid for vid, spec in scenario.vehicles.items() if spec.role == "ramp"]
    mainline_ids = [vid for vid, spec in scenario.vehicles.items() if spec.role == "mainline"]
    positions = {vid: spec.route_position for vid, spec in scenario.vehicles.items()}
    positions[ramp_ids[0]] = 350.0
    positions[ramp_ids[1]] = 347.0  # unsafe gap, but only same-ROLE (ramp) vehicles compared
    positions[mainline_ids[0]] = 348.5  # would ALSO be an unsafe gap if lateral proximity were used
    positions[mainline_ids[1]] = 10.0
    actions = oracle_actions(scenario=scenario, positions=positions, merge_start=MERGE_START, merge_end=MERGE_END)
    # Role-based grouping: ramp_ids[1] only compares against ramp_ids[0] (3m, unsafe) -> DECELERATE.
    assert actions[ramp_ids[1]] == DECELERATE


def test_lateral_positions_groups_by_real_physical_proximity_not_role():
    """The new behavior: a ramp vehicle physically alongside (small |dy|)
    a mainline vehicle must be treated as same-lane, even though their
    scenario `role` fields differ -- this is exactly the gap that caused
    2 residual collisions under the HighwayEnv meta_speed oracle
    (a merged ramp vehicle rear-ended by a faster mainline vehicle that
    the role-based check never compared it against)."""
    scenario = _make_scenario()
    ramp_ids = [vid for vid, spec in scenario.vehicles.items() if spec.role == "ramp"]
    mainline_ids = [vid for vid, spec in scenario.vehicles.items() if spec.role == "mainline"]
    positions = {vid: spec.route_position for vid, spec in scenario.vehicles.items()}
    # Ramp vehicle 0 has already merged (same lateral position as mainline
    # vehicle 0, y=4.0) and is being closed on from behind by it.
    positions[ramp_ids[0]] = 400.0
    positions[mainline_ids[0]] = 397.0  # 3m behind, same physical lane -> unsafe
    positions[ramp_ids[1]] = 50.0
    positions[mainline_ids[1]] = 10.0
    lateral_positions = {
        ramp_ids[0]: 4.0, mainline_ids[0]: 4.0,  # physically merged, same lane
        ramp_ids[1]: 8.0, mainline_ids[1]: 4.0,  # not yet merged -- different lane
    }
    actions = oracle_actions(
        scenario=scenario, positions=positions, merge_start=MERGE_START, merge_end=MERGE_END,
        lateral_positions=lateral_positions,
    )
    assert actions[mainline_ids[0]] == DECELERATE


def test_partner_id_raises_if_no_match_in_malformed_scenario():
    scenario = _make_scenario()
    # Corrupt one vehicle's ttc_slot so no partner exists.
    import dataclasses

    broken_vehicles = dict(scenario.vehicles)
    v0 = broken_vehicles["V0"]
    broken_vehicles["V0"] = dataclasses.replace(v0, ttc_slot="nonexistent-slot")
    broken_scenario = dataclasses.replace(scenario, vehicles=broken_vehicles)
    with pytest.raises(ValueError):
        partner_id(broken_scenario, "V0")
