"""Deterministic Stage-2A PBRS pure-function tests (Tests 1–30)."""

from __future__ import annotations

import math

import pytest

from thesis.rewards.pbrs_v2 import (
    LEARNING_CONTROLLERS,
    STAKEHOLDER_ORDER,
    PBRSConfig,
    PotentialState,
    StakeholderState,
    apply_pbrs_to_base_rewards,
    compute_active_experience,
    compute_actual_potential,
    compute_pbrs_signal,
    compute_potential_breakdown,
    compute_raw_mean_potential,
    compute_raw_min_potential,
    compute_stakeholder_experiences,
    potential_state_from_experiences,
    telescoping_sum,
)


def E(a: float, b: float, front: float, rear: float) -> dict[str, float]:
    return {"A": a, "B": b, "B_front": front, "B_rear": rear}


def _cfg(**kw) -> PBRSConfig:
    defaults = dict(
        learner_gamma=0.995,
        shaping_gamma=0.995,
        lambda_mean=0.5,
        lambda_min=0.5,
    )
    defaults.update(kw)
    return PBRSConfig(**defaults)


# ---------------------------------------------------------------------------
# Tests 1–5: active experience
# ---------------------------------------------------------------------------
def test_01_active_experience_at_target_speed():
    assert compute_active_experience(20.0, 20.0) == 1.0


def test_02_active_experience_below_target():
    assert compute_active_experience(15.0, 20.0) == pytest.approx(0.75)


def test_03_overspeed_clipping():
    assert compute_active_experience(30.0, 20.0) == 1.0


def test_04_negative_speed_clipping():
    assert compute_active_experience(-5.0, 20.0) == 0.0


@pytest.mark.parametrize("vt", [0.0, -1.0, float("nan"), float("inf")])
def test_05_invalid_target_speed(vt: float):
    with pytest.raises(ValueError):
        compute_active_experience(10.0, vt)


# ---------------------------------------------------------------------------
# Test 6 — Completed stakeholder absorbing status
# ---------------------------------------------------------------------------
def test_06_completed_stakeholder_absorbing_status():
    stakeholders = {
        "A": StakeholderState(speed=0.0, target_speed=20.0, completed=True),
        "B": StakeholderState(speed=10.0, target_speed=20.0),
        "B_front": StakeholderState(speed=10.0, target_speed=20.0),
        "B_rear": StakeholderState(speed=10.0, target_speed=20.0),
    }
    exp = compute_stakeholder_experiences(stakeholders)
    assert exp["A"] == 1.0


# ---------------------------------------------------------------------------
# Test 7 — Fixed stakeholder set
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad",
    [
        {"B": StakeholderState(explicit_experience=0.5), "B_front": StakeholderState(explicit_experience=0.5), "B_rear": StakeholderState(explicit_experience=0.5)},
        {"A": StakeholderState(explicit_experience=0.5), "B_front": StakeholderState(explicit_experience=0.5), "B_rear": StakeholderState(explicit_experience=0.5)},
        {"A": StakeholderState(explicit_experience=0.5), "B": StakeholderState(explicit_experience=0.5), "B_rear": StakeholderState(explicit_experience=0.5)},
        {"A": StakeholderState(explicit_experience=0.5), "B": StakeholderState(explicit_experience=0.5), "B_front": StakeholderState(explicit_experience=0.5)},
        {
            "A": StakeholderState(explicit_experience=0.5),
            "B": StakeholderState(explicit_experience=0.5),
            "B_front": StakeholderState(explicit_experience=0.5),
            "B_rear": StakeholderState(explicit_experience=0.5),
            "C": StakeholderState(explicit_experience=0.5),
        },
    ],
)
def test_07_fixed_stakeholder_set(bad):
    with pytest.raises(ValueError):
        compute_stakeholder_experiences(bad)


# ---------------------------------------------------------------------------
# Tests 8–9 — Mean / Min potential
# ---------------------------------------------------------------------------
def test_08_mean_potential():
    exp = E(0.2, 0.4, 0.6, 0.8)
    assert compute_raw_mean_potential(exp) == pytest.approx(0.5)


def test_09_minimum_potential():
    exp = E(0.2, 0.4, 0.6, 0.8)
    assert compute_raw_min_potential(exp) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Test 10 — Mean permits compensation
# ---------------------------------------------------------------------------
def test_10_mean_permits_compensation():
    s1 = E(0.2, 0.8, 0.8, 0.8)
    s2 = E(0.2, 1.0, 1.0, 1.0)
    d_mean = compute_raw_mean_potential(s2) - compute_raw_mean_potential(s1)
    d_min = compute_raw_min_potential(s2) - compute_raw_min_potential(s1)
    assert d_mean > 0.0
    assert d_min == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 11 — Worst-off improvement
# ---------------------------------------------------------------------------
def test_11_worst_off_improvement():
    s1 = E(0.2, 0.8, 0.8, 0.8)
    s2 = E(0.3, 0.8, 0.8, 0.8)
    assert compute_raw_min_potential(s2) - compute_raw_min_potential(s1) == pytest.approx(0.1)
    assert compute_raw_mean_potential(s2) - compute_raw_mean_potential(s1) == pytest.approx(0.025)


# ---------------------------------------------------------------------------
# Test 12 — Worst-off identity switching
# ---------------------------------------------------------------------------
def test_12_worst_off_identity_switching():
    s1 = E(0.30, 0.31, 0.80, 0.80)
    s2 = E(0.32, 0.29, 0.80, 0.80)
    bd1 = compute_potential_breakdown(
        potential_state_from_experiences(s1), "min", experiences=s1
    )
    bd2 = compute_potential_breakdown(
        potential_state_from_experiences(s2), "min", experiences=s2
    )
    assert bd1.raw_potential == pytest.approx(0.30)
    assert bd2.raw_potential == pytest.approx(0.29)
    assert bd1.worst_off_ids == ("A",)
    assert bd2.worst_off_ids == ("B",)
    assert set(bd1.worst_off_ids) != set(bd2.worst_off_ids)

    # Ties represented as a set / ordered tuple of all tied IDs
    tied = E(0.2, 0.2, 0.8, 0.8)
    bd_tie = compute_potential_breakdown(
        potential_state_from_experiences(tied), "min", experiences=tied
    )
    assert bd_tie.worst_off_ids == ("A", "B")
    assert set(bd_tie.worst_off_ids) == {"A", "B"}


# ---------------------------------------------------------------------------
# Test 13 — True terminal potential
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("label", ["success", "collision"])
def test_13_true_terminal_potential(label: str):
    exp = E(0.5, 0.6, 0.7, 0.8)
    state = potential_state_from_experiences(
        exp, terminated=True, truncated=False, terminal_label=label
    )
    for cond in ("mean", "min"):
        bd = compute_potential_breakdown(state, cond, experiences=exp)
        assert bd.actual_potential == 0.0
        assert bd.raw_potential > 0.0


# ---------------------------------------------------------------------------
# Test 14 — External truncation preserves potential
# ---------------------------------------------------------------------------
def test_14_external_truncation_preserves_potential():
    exp = E(0.5, 0.6, 0.7, 0.8)
    state = potential_state_from_experiences(exp, terminated=False, truncated=True)
    for cond in ("mean", "min"):
        bd = compute_potential_breakdown(state, cond, experiences=exp)
        assert bd.actual_potential == pytest.approx(bd.raw_potential)
        assert bd.actual_potential != 0.0


# ---------------------------------------------------------------------------
# Test 15 — Invalid simultaneous flags
# ---------------------------------------------------------------------------
def test_15_invalid_simultaneous_flags():
    exp = E(0.5, 0.5, 0.5, 0.5)
    state = potential_state_from_experiences(exp, terminated=True, truncated=True)
    with pytest.raises(ValueError, match="simultaneously"):
        compute_potential_breakdown(state, "mean", experiences=exp)
    with pytest.raises(ValueError):
        compute_actual_potential(0.5, terminated=True, truncated=True)


# ---------------------------------------------------------------------------
# Test 16 — Ordinary PBRS signal
# ---------------------------------------------------------------------------
def test_16_ordinary_pbrs_signal():
    gamma = 0.995
    cfg = _cfg(learner_gamma=gamma, shaping_gamma=gamma)
    e_t = E(0.2, 0.2, 0.2, 0.2)
    e_t1 = E(0.4, 0.4, 0.4, 0.4)
    sig = compute_pbrs_signal(
        potential_state_from_experiences(e_t),
        potential_state_from_experiences(e_t1),
        "mean",
        cfg,
        experiences_t=e_t,
        experiences_t1=e_t1,
    )
    assert sig.shaping_signal == pytest.approx(gamma * sig.phi_t1 - sig.phi_t)


# ---------------------------------------------------------------------------
# Test 17 — Terminal PBRS signal
# ---------------------------------------------------------------------------
def test_17_terminal_pbrs_signal():
    gamma = 0.995
    cfg = _cfg(learner_gamma=gamma, shaping_gamma=gamma)
    e_t = E(0.8, 0.8, 0.8, 0.8)
    e_t1 = E(0.9, 0.9, 0.9, 0.9)
    sig = compute_pbrs_signal(
        potential_state_from_experiences(e_t),
        potential_state_from_experiences(
            e_t1, terminated=True, truncated=False, terminal_label="collision"
        ),
        "mean",
        cfg,
        experiences_t=e_t,
        experiences_t1=e_t1,
    )
    assert sig.phi_t1 == 0.0
    assert sig.shaping_signal == pytest.approx(-sig.phi_t)


# ---------------------------------------------------------------------------
# Test 18 — Truncation PBRS signal
# ---------------------------------------------------------------------------
def test_18_truncation_pbrs_signal():
    gamma = 0.995
    cfg = _cfg(learner_gamma=gamma, shaping_gamma=gamma)
    e_t = E(0.4, 0.4, 0.4, 0.4)
    e_t1 = E(0.6, 0.6, 0.6, 0.6)
    sig = compute_pbrs_signal(
        potential_state_from_experiences(e_t),
        potential_state_from_experiences(e_t1, terminated=False, truncated=True),
        "mean",
        cfg,
        experiences_t=e_t,
        experiences_t1=e_t1,
    )
    assert sig.phi_t1 == pytest.approx(sig.raw_phi_t1)
    assert sig.phi_t1 != 0.0
    assert sig.shaping_signal == pytest.approx(gamma * sig.phi_t1 - sig.phi_t)
    assert sig.shaping_signal != pytest.approx(-sig.phi_t)


# ---------------------------------------------------------------------------
# Tests 19–21 — Base reward shaping
# ---------------------------------------------------------------------------
def test_19_base_reward_not_rescaled():
    gamma = 0.995
    lam = 0.5
    cfg = _cfg(learner_gamma=gamma, shaping_gamma=gamma, lambda_mean=lam)
    e_t = E(0.2, 0.2, 0.2, 0.2)
    e_t1 = E(0.4, 0.4, 0.4, 0.4)
    base = {"A": 0.04, "B": -1.0}
    out = apply_pbrs_to_base_rewards(
        base,
        potential_state_from_experiences(e_t),
        potential_state_from_experiences(e_t1),
        "mean",
        cfg,
        experiences_t=e_t,
        experiences_t1=e_t1,
    )
    f = out["A"].shaping_signal
    assert out["A"].shaped_reward == pytest.approx(base["A"] + lam * f)
    assert out["B"].shaped_reward == pytest.approx(base["B"] + lam * f)
    assert out["A"].shaped_reward != pytest.approx((1 - lam) * base["A"])
    assert out["B"].shaped_reward != pytest.approx((1 - lam) * base["B"])


def test_20_common_shaping_individual_base():
    cfg = _cfg()
    e_t = E(0.1, 0.1, 0.1, 0.1)
    e_t1 = E(0.5, 0.5, 0.5, 0.5)
    base = {"A": 0.1, "B": 0.7}
    out = apply_pbrs_to_base_rewards(
        base,
        potential_state_from_experiences(e_t),
        potential_state_from_experiences(e_t1),
        "min",
        cfg,
        experiences_t=e_t,
        experiences_t1=e_t1,
    )
    assert out["A"].scaled_shaping_component == pytest.approx(
        out["B"].scaled_shaping_component
    )
    assert out["A"].shaped_reward != pytest.approx(out["B"].shaped_reward)


def test_21_lambda_zero():
    cfg = _cfg(lambda_mean=0.0, lambda_min=0.0)
    e_t = E(0.2, 0.2, 0.2, 0.2)
    e_t1 = E(0.9, 0.9, 0.9, 0.9)
    base = {"A": 0.123, "B": -0.456}
    out = apply_pbrs_to_base_rewards(
        base,
        potential_state_from_experiences(e_t),
        potential_state_from_experiences(e_t1),
        "mean",
        cfg,
        experiences_t=e_t,
        experiences_t1=e_t1,
    )
    assert out["A"].shaped_reward == pytest.approx(base["A"])
    assert out["B"].shaped_reward == pytest.approx(base["B"])
    assert out["A"].scaled_shaping_component == 0.0


# ---------------------------------------------------------------------------
# Tests 22–24 — Invalid config
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lam", [-0.1, float("nan"), float("inf")])
def test_22_invalid_lambda(lam: float):
    with pytest.raises(ValueError):
        _cfg(lambda_mean=lam).validate()


@pytest.mark.parametrize("g", [-0.1, 1.0, 1.5, float("nan"), float("inf")])
def test_23_invalid_gamma(g: float):
    with pytest.raises(ValueError):
        # Keep learner/shaping equal so mismatch isn't the first error when both invalid
        if math.isfinite(g) and g == g:
            _cfg(learner_gamma=g, shaping_gamma=g).validate()
        else:
            _cfg(learner_gamma=0.995, shaping_gamma=g).validate()


def test_24_gamma_mismatch():
    with pytest.raises(ValueError, match="must equal"):
        _cfg(learner_gamma=0.995, shaping_gamma=0.99).validate()


# ---------------------------------------------------------------------------
# Tests 25–26 — Telescoping identities
# ---------------------------------------------------------------------------
def test_25_terminal_telescoping_identity():
    gamma = 0.995
    # Mean-labelled sequence ending at 0
    mean_phis = [0.5, 0.6, 0.7, 0.8, 0.0]
    total, _ = telescoping_sum(mean_phis, gamma)
    assert total == pytest.approx(-mean_phis[0], abs=1e-12)

    # Min-labelled sequence ending at 0
    min_phis = [0.2, 0.25, 0.3, 0.35, 0.0]
    total2, _ = telescoping_sum(min_phis, gamma)
    assert total2 == pytest.approx(-min_phis[0], abs=1e-12)


def test_26_truncated_segment_telescoping_identity():
    gamma = 0.995
    phis = [0.4, 0.45, 0.5, 0.55]  # phi_K != 0
    k = len(phis) - 1
    total, _ = telescoping_sum(phis, gamma)
    expected = -phis[0] + (gamma**k) * phis[k]
    assert total == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# Tests 27–28 — Completed stakeholder membership
# ---------------------------------------------------------------------------
def test_27_completed_stakeholder_remains_in_denominator():
    stakeholders = {
        "A": StakeholderState(completed=True, speed=0.0, target_speed=20.0),
        "B": StakeholderState(explicit_experience=0.4),
        "B_front": StakeholderState(explicit_experience=0.6),
        "B_rear": StakeholderState(explicit_experience=0.8),
    }
    exp = compute_stakeholder_experiences(stakeholders)
    assert set(exp.keys()) == set(STAKEHOLDER_ORDER)
    mean = compute_raw_mean_potential(exp)
    assert mean == pytest.approx((1.0 + 0.4 + 0.6 + 0.8) / 4.0)
    assert mean != pytest.approx((0.4 + 0.6 + 0.8) / 3.0)


def test_28_completed_stakeholder_remains_eligible_for_minimum():
    stakeholders = {
        "A": StakeholderState(completed=True),
        "B": StakeholderState(explicit_experience=0.2),
        "B_front": StakeholderState(explicit_experience=0.3),
        "B_rear": StakeholderState(explicit_experience=0.4),
    }
    exp = compute_stakeholder_experiences(stakeholders)
    assert "A" in exp and exp["A"] == 1.0
    assert len(exp) == 4
    bd = compute_potential_breakdown(
        PotentialState(stakeholders=stakeholders), "min"
    )
    assert set(bd.stakeholder_experiences.keys()) == set(STAKEHOLDER_ORDER)
    assert bd.worst_off_ids == ("B",)


# ---------------------------------------------------------------------------
# Test 29 — Non-finite state validation
# ---------------------------------------------------------------------------
def test_29_non_finite_state_validation():
    with pytest.raises(ValueError):
        compute_active_experience(float("nan"), 20.0)
    with pytest.raises(ValueError):
        compute_active_experience(10.0, float("inf"))
    with pytest.raises(ValueError):
        compute_raw_mean_potential(E(float("nan"), 0.5, 0.5, 0.5))
    with pytest.raises(ValueError):
        compute_actual_potential(float("inf"), terminated=False, truncated=False)
    with pytest.raises(ValueError):
        apply_pbrs_to_base_rewards(
            {"A": float("nan"), "B": 0.0},
            potential_state_from_experiences(E(0.2, 0.2, 0.2, 0.2)),
            potential_state_from_experiences(E(0.3, 0.3, 0.3, 0.3)),
            "mean",
            _cfg(),
        )
    with pytest.raises(ValueError):
        compute_stakeholder_experiences(
            {
                "A": StakeholderState(explicit_experience=float("nan")),
                "B": StakeholderState(explicit_experience=0.5),
                "B_front": StakeholderState(explicit_experience=0.5),
                "B_rear": StakeholderState(explicit_experience=0.5),
            }
        )


# ---------------------------------------------------------------------------
# Test 30 — Deterministic decomposition
# ---------------------------------------------------------------------------
def test_30_deterministic_decomposition():
    cfg = _cfg(lambda_mean=0.3, lambda_min=0.7)
    e_t = E(0.25, 0.35, 0.45, 0.55)
    e_t1 = E(0.30, 0.40, 0.50, 0.60)
    base = {"A": 0.04, "B": -0.025}
    for cond in ("mean", "min"):
        out = apply_pbrs_to_base_rewards(
            base,
            potential_state_from_experiences(e_t),
            potential_state_from_experiences(e_t1),
            cond,
            cfg,
            experiences_t=e_t,
            experiences_t1=e_t1,
        )
        for aid in LEARNING_CONTROLLERS:
            br = out[aid]
            assert br.shaped_reward == pytest.approx(
                br.base_reward + br.scaled_shaping_component, abs=1e-12
            )
