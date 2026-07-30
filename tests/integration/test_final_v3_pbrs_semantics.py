"""Stage 5A-0 — four-stakeholder PBRS semantics on V3."""

from __future__ import annotations

from thesis.rewards.pbrs_v2 import STAKEHOLDER_ORDER
from thesis.training.final_experiment_runtime import (
    compute_telescoping_errors,
    scripted_a_accel_b_maintain,
    scripted_accelerate,
)
from thesis.training.final_lock_loader import load_final_locks
from thesis.training.final_reward_conditions import IntegrationPBRSConfig
from thesis.training.final_v3_pipeline import (
    potential_state_from_v3_vehicles,
    run_final_v3_episode,
)
from thesis.rewards.pbrs_v2 import compute_potential_breakdown


def test_four_stakeholder_registry_and_potentials():
    bundle = load_final_locks()
    ep = run_final_v3_episode(
        bundle,
        reward_condition="mean_pbrs",
        scripted_actions=scripted_accelerate(5),
        pbrs_config=IntegrationPBRSConfig(),
        episode_id="pbrs4",
    )
    row = ep["transitions"][0]
    assert set(row["experiences_t"]) == set(STAKEHOLDER_ORDER)
    assert len(STAKEHOLDER_ORDER) == 4
    mean = sum(row["experiences_t"][s] for s in STAKEHOLDER_ORDER) / 4.0
    assert abs(mean - row["raw_mean_t"]) < 1e-12
    assert row["raw_min_t"] == min(row["experiences_t"][s] for s in STAKEHOLDER_ORDER)


def test_active_experience_and_exit_experience():
    bundle = load_final_locks()
    ep = run_final_v3_episode(
        bundle,
        reward_condition="mean_pbrs",
        scripted_actions=scripted_a_accel_b_maintain(70),
        pbrs_config=IntegrationPBRSConfig(),
        episode_id="exit_e",
    )
    exit_rows = [r for r in ep["transitions"] if r["exit_event"]["A"] >= 1.0]
    assert exit_rows
    # Exited learner at t1 has E=1
    for r in exit_rows:
        assert r["experiences_t1"]["A"] == 1.0
        # At t was on-road (not yet completed)
        assert r["vehicles_t"]["A"]["completed"] is False
        assert r["vehicles_t1"]["A"]["completed"] is True


def test_true_terminal_vs_truncation_successor_potential():
    bundle = load_final_locks()
    coll = run_final_v3_episode(
        bundle,
        reward_condition="mean_pbrs",
        scripted_actions=scripted_accelerate(30),
        pbrs_config=IntegrationPBRSConfig(),
        episode_id="term_coll",
    )
    last = coll["transitions"][-1]
    assert last["terminated"] is True
    assert last["actual_mean_t1"] == 0.0
    assert last["actual_min_t1"] == 0.0

    trunc = run_final_v3_episode(
        bundle,
        reward_condition="mean_pbrs",
        scripted_actions=scripted_accelerate(10),
        pbrs_config=IntegrationPBRSConfig(),
        max_policy_steps=3,
        episode_id="term_trunc",
    )
    tlast = trunc["transitions"][-1]
    assert tlast["truncated"] is True
    assert tlast["terminated"] is False
    # Not forcibly zero
    assert tlast["actual_mean_t1"] == tlast["raw_mean_t1"]
    assert tlast["actual_mean_t1"] > 0.0 or tlast["raw_mean_t1"] >= 0.0


def test_background_stakeholders_affect_potential():
    # Synthetic local check using V3 snapshot conversion
    vehicles = {
        sid: {
            "speed": 10.0 if sid != "B_rear" else 0.0,
            "completed": False,
        }
        for sid in STAKEHOLDER_ORDER
    }
    targets = {sid: 20.0 for sid in STAKEHOLDER_ORDER}
    pot = potential_state_from_v3_vehicles(
        vehicles, targets, terminated=False, truncated=False
    )
    bd = compute_potential_breakdown(pot, "min")
    assert bd.raw_potential == 0.0  # B_rear worst
    assert "B_rear" in bd.worst_off_ids


def test_pbrs_telescoping_terminal_and_truncation():
    bundle = load_final_locks()
    coll = run_final_v3_episode(
        bundle,
        reward_condition="mean_pbrs",
        scripted_actions=scripted_accelerate(30),
        pbrs_config=IntegrationPBRSConfig(),
        episode_id="tele_coll",
    )
    tele = compute_telescoping_errors(coll["transitions"])
    assert tele["mean_error"] < 1e-10
    assert tele["min_error"] < 1e-10
    assert tele["phi_mean_T"] == 0.0
    assert abs(tele["mean_sum"] - (-tele["phi_mean_0"])) < 1e-10

    trunc = run_final_v3_episode(
        bundle,
        reward_condition="min_pbrs",
        scripted_actions=scripted_accelerate(10),
        pbrs_config=IntegrationPBRSConfig(),
        max_policy_steps=4,
        episode_id="tele_trunc",
    )
    tele_t = compute_telescoping_errors(trunc["transitions"])
    assert tele_t["mean_error"] < 1e-10
    assert tele_t["min_error"] < 1e-10
    assert tele_t["phi_mean_T"] != 0.0 or tele_t["phi_min_T"] >= 0.0
