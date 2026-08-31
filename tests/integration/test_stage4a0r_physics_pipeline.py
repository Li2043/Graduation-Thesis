"""Stage 4A-0R physics hardening integration tests (no DQN / no reselection)."""

from __future__ import annotations

from thesis.certification.choice_state_scenarios import (
    GEOMETRY,
    IDM_PROFILES,
    build_ic_blocks,
)
from thesis.certification.environment_candidate_selection import select_environment_candidate
from thesis.certification.holdout_signatures import (
    find_duplicate_signatures,
    physical_block_signature,
)
from thesis.envs.final_environment_config import EnvironmentCandidate, InitialConditionBlock
from thesis.envs.final_observation import OBSERVATION_DIM
from thesis.envs.final_route_geometry import build_final_route_geometry
from thesis.envs.merge_env_candidate_v3 import MergeEnvCandidateV3, MergeEnvCandidateV3Config
from thesis.envs.vehicle_dynamics import integrate_longitudinal


def test_no_dqn_execution_on_env_step():
    cal, _ = build_ic_blocks()
    cand = EnvironmentCandidate("G1-I1", GEOMETRY[0], IDM_PROFILES[0], 1)
    env = MergeEnvCandidateV3(MergeEnvCandidateV3Config(candidate=cand, block=cal[0]))
    obs, _ = env.reset(seed=1)
    assert obs["A"].shape == (OBSERVATION_DIM,)
    # Stepping must not create optimiser / learner attributes
    env.step({"A": 0, "B": 0})
    assert not hasattr(env, "optimizer")
    assert not hasattr(env, "optimiser")
    assert not hasattr(env, "learner")


def test_duplicate_holdout_signatures_detected():
    cal, val = build_ic_blocks()
    src = cal[0]
    dup = InitialConditionBlock(
        block_id="validation_dup",
        block_set="validation",
        seed=9999,
        role_A=src.role_A,
        role_B=src.role_B,
        spawn_route_mainline=src.spawn_route_mainline,
        spawn_route_ramp=src.spawn_route_ramp,
        spawn_speed_mainline=src.spawn_speed_mainline,
        spawn_speed_ramp=src.spawn_speed_ramp,
        spawn_route_B_front=src.spawn_route_B_front,
        spawn_route_B_rear=src.spawn_route_B_rear,
        spawn_speed_B_front=src.spawn_speed_B_front,
        spawn_speed_B_rear=src.spawn_speed_B_rear,
        delta_arrival=src.delta_arrival,
        arrival_category=src.arrival_category,
        background_time_headway=src.background_time_headway,
        target_speeds=src.target_speeds,
    )
    assert physical_block_signature(dup) == physical_block_signature(src)
    found = find_duplicate_signatures(cal, [dup, val[0]])
    assert any(d["validation_block_id"] == "validation_dup" for d in found)


def test_selection_uses_calibration_only_with_mock_candidates(monkeypatch):
    """Call the real selection function on a tiny deterministic mock set."""
    from thesis.certification import environment_candidate_selection as ecs

    cal, val = build_ic_blocks()
    cal = cal[:2]
    val = val[:2]
    cands = [
        EnvironmentCandidate("G1-I1", GEOMETRY[0], IDM_PROFILES[0], 1),
        EnvironmentCandidate("G1-I2", GEOMETRY[0], IDM_PROFILES[1], 2),
    ]

    def fake_eval(candidate, blocks, *, block_set):
        n = len(blocks)
        certs = [
            {
                "candidate_id": candidate.candidate_id,
                "block_set": block_set,
                "block_id": b.block_id,
                "arrival_category": b.arrival_category,
                "certified": True,
                "rejection_reasons": [],
                "normalised_order_gap": 0.01,
                "background_relevant": True,
                "label_swap_max_error": 0.0,
            }
            for b in blocks
        ]
        # Fail G1-I1 entirely so lowest-priority feasible is G1-I2.
        if candidate.priority_rank == 1 and block_set == "calibration":
            for c in certs:
                c["certified"] = False
            n_cert = 0
        else:
            n_cert = n
        return {
            "candidate_id": candidate.candidate_id,
            "priority_rank": candidate.priority_rank,
            "block_set": block_set,
            "n_blocks": n,
            "n_certified": n_cert,
            "certification_rate": n_cert / n,
            "certified_arrival_categories": ["mainline_lead", "ramp_lead", "near_simultaneous"],
            "background_relevance_rate": 1.0,
            "label_swap_max_error": 0.0,
            "median_normalised_order_gap": 0.01,
            "maximum_normalised_order_gap": 0.02,
            "certifications": certs,
            "matrices": [],
            "traces": [],
            "background_rows": [],
        }

    monkeypatch.setattr(ecs, "evaluate_candidate_on_blocks", fake_eval)
    monkeypatch.setattr(
        ecs,
        "run_background_safety_audit",
        lambda *_a, **_k: {"spontaneous_collision_count": 0, "details": []},
    )
    def _cal_ok(eval_result, **kw):
        if eval_result["n_certified"] >= 1 and eval_result["background_relevance_rate"] >= 0.75:
            return True, []
        return False, ["certified_low"]

    monkeypatch.setattr(ecs, "calibration_feasible", _cal_ok)
    monkeypatch.setattr(
        ecs,
        "validation_pass",
        lambda eval_result, **kw: (eval_result["n_certified"] >= 1, []),
    )

    result = select_environment_candidate(
        candidates=cands, calibration_blocks=cal, validation_blocks=val
    )
    assert result["selection_used_validation"] is False
    # G1-I1 fails mock calibration; lowest feasible is G1-I2
    assert result["selected_candidate"]["candidate_id"] == "G1-I2"
    assert result["selected_candidate"]["priority_rank"] == 2


def test_stopping_and_route_errors_bounded():
    s1, v1, a = integrate_longitudinal(
        route_position=0.0, speed=0.15, acceleration=-4.0, dt=0.05
    )
    t_stop = 0.15 / 4.0
    s_exp = 0.15 * t_stop + 0.5 * (-4.0) * t_stop**2
    assert abs(s1 - s_exp) < 1e-12
    geom = build_final_route_geometry(GEOMETRY[0])
    pose = geom.pose("ramp", geom.ramp_straight_length + 0.5)
    s_rec = geom.recover_route_position("ramp", pose.x, pose.y)
    assert abs(s_rec - (geom.ramp_straight_length + 0.5)) < 0.05


def test_prior_stage4a_run_not_deleted():
    from pathlib import Path

    p = Path(
        "experiments/pre_impl/stage4a_environment_choice_state/artifacts/"
        "20260729T231946Z_c8d92bc3/manifest.json"
    )
    assert p.is_file()
    text = p.read_text(encoding="utf-8")
    assert "20260729T231946Z_c8d92bc3" in text
