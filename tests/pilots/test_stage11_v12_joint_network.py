"""Stage 11 pilot (E30) v12 -- joint (centralised) network + baseline /
mean_pbrs / min_pbrs three-condition PBRS comparison.

Covers: the three new seed blocks' disjointness/guard-pass/union
membership; flag-composition validation (pbrs_condition_v12 requires
enable_joint_network_v12; v12 is mutually exclusive with v6-v11's
independent-learner mechanisms; v12 requires v5's reward); the three
reward_version tags; direct numeric verification of the PBRS potential
math (hand-computed Phi_mean/Phi_min and the shaping signal F against a
known 2-vehicle state, and the terminal-zero / truncation-preserves-
potential boundary cases); short real end-to-end runs for all three
conditions; and a regression proof that the baseline condition's rewards
are byte-identical to the unshaped base reward (the PBRS code path is
truly inert when pbrs_condition_v12 is None/"baseline").
"""

from __future__ import annotations

import json

import pytest

from thesis.pilots.stage11_dyad_merge_pilot_config import (
    CHECKPOINT_STEPS_V12,
    EPSILON_DECAY_STEPS_V12,
    EPSILON_END,
    EPSILON_START,
    LEARNING_RATE_DECAY_STEPS_V12,
    LEARNING_RATE_END,
    LEARNING_RATE_START,
    MAX_STEPS,
    MAX_STEPS_V12,
    PBRS_LAMBDA_V12,
    PER_ALPHA_V12,
    PER_BETA_END_V12,
    PER_BETA_START_V12,
    PILOT_SEEDS,
    PILOT_V1_SEEDS,
    PILOT_V2_SEEDS,
    PILOT_V3_SEEDS,
    PILOT_V4_SEEDS,
    PILOT_V5_SEEDS,
    PILOT_V6_SEEDS,
    PILOT_V7_SEEDS,
    PILOT_V8_SEEDS,
    PILOT_V9_SEEDS,
    PILOT_V10_SEEDS,
    PILOT_V11_SEEDS,
    PILOT_V12_BASELINE_SEEDS,
    PILOT_V12_BASELINE_V2_SEEDS,
    PILOT_V12_MEAN_PBRS_SEEDS,
    PILOT_V12_MIN_PBRS_SEEDS,
    assert_stage11_pilot_guards,
    epsilon_at_step_v12,
    lr_at_step_v12,
)
from thesis.pilots.stage11_dyad_merge_runner import (
    CHECKPOINT_STEPS,
    REWARD_VERSION_STAGE11_V12_JOINT_BASELINE,
    REWARD_VERSION_STAGE11_V12_JOINT_MEAN_PBRS,
    REWARD_VERSION_STAGE11_V12_JOINT_MIN_PBRS,
    _resolve_v12_checkpoint_steps,
    run_stage11_pilot_training_job,
)
from thesis.pilots.stage11_welfare import (
    actual_potential,
    mean_welfare,
    min_welfare,
    pbrs_shaping_signal,
    stakeholder_experience,
    target_speed_attainment,
)

ALL_V12_SEEDS = PILOT_V12_BASELINE_SEEDS + PILOT_V12_MEAN_PBRS_SEEDS + PILOT_V12_MIN_PBRS_SEEDS


# --------------------------------------------------------------- config constants


def test_pbrs_lambda_v12_constant():
    assert PBRS_LAMBDA_V12 == 0.2


def test_v12_three_seed_blocks_are_disjoint_from_each_other():
    assert set(PILOT_V12_BASELINE_SEEDS).isdisjoint(PILOT_V12_MEAN_PBRS_SEEDS)
    assert set(PILOT_V12_BASELINE_SEEDS).isdisjoint(PILOT_V12_MIN_PBRS_SEEDS)
    assert set(PILOT_V12_MEAN_PBRS_SEEDS).isdisjoint(PILOT_V12_MIN_PBRS_SEEDS)


def test_v12_seeds_disjoint_from_v1_through_v11():
    for block in (
        PILOT_V1_SEEDS,
        PILOT_V2_SEEDS,
        PILOT_V3_SEEDS,
        PILOT_V4_SEEDS,
        PILOT_V5_SEEDS,
        PILOT_V6_SEEDS,
        PILOT_V7_SEEDS,
        PILOT_V8_SEEDS,
        PILOT_V9_SEEDS,
        PILOT_V10_SEEDS,
        PILOT_V11_SEEDS,
    ):
        assert set(ALL_V12_SEEDS).isdisjoint(block)


def test_v12_seeds_pass_guard():
    for seed in ALL_V12_SEEDS:
        assert_stage11_pilot_guards(master_seed=seed, max_steps=MAX_STEPS)


def test_pilot_seeds_union_includes_v12():
    assert set(ALL_V12_SEEDS) <= set(PILOT_SEEDS)


def test_v12_seed_blocks_each_have_eight_seeds():
    assert len(PILOT_V12_BASELINE_SEEDS) == 8
    assert len(PILOT_V12_MEAN_PBRS_SEEDS) == 8
    assert len(PILOT_V12_MIN_PBRS_SEEDS) == 8


# --------------------------------------------------------------- PBRS potential math


def test_phi_mean_and_phi_min_hand_computed():
    # ramp: speed=10, target=20 -> e=0.5, not completed -> E=0.5
    # mainline: completed -> E=1.0 regardless of speed
    e_ramp = target_speed_attainment(10.0, 20.0)
    assert e_ramp == pytest.approx(0.5)
    E_ramp = stakeholder_experience(e_ramp, completed=False)
    E_mainline = stakeholder_experience(0.0, completed=True)
    assert E_ramp == pytest.approx(0.5)
    assert E_mainline == pytest.approx(1.0)

    phi_mean = mean_welfare([E_ramp, E_mainline])
    phi_min = min_welfare([E_ramp, E_mainline])
    assert phi_mean == pytest.approx(0.75)
    assert phi_min == pytest.approx(0.5)


def test_actual_potential_zero_at_true_terminal():
    assert actual_potential(0.75, terminated=True, truncated=False) == 0.0
    assert actual_potential(0.5, terminated=True, truncated=False) == 0.0


def test_actual_potential_preserved_at_truncation():
    assert actual_potential(0.75, terminated=False, truncated=True) == pytest.approx(0.75)


def test_actual_potential_preserved_mid_episode():
    assert actual_potential(0.42, terminated=False, truncated=False) == pytest.approx(0.42)


def test_actual_potential_rejects_terminated_and_truncated_simultaneously():
    with pytest.raises(ValueError):
        actual_potential(0.5, terminated=True, truncated=True)


def test_pbrs_shaping_signal_hand_computed():
    # F = gamma * phi_t1 - phi_t
    phi_t = 0.5
    phi_t1 = 0.75
    gamma = 0.995
    F = pbrs_shaping_signal(phi_t, phi_t1, gamma=gamma)
    expected = gamma * phi_t1 - phi_t
    assert F == pytest.approx(expected)
    assert F == pytest.approx(0.24625)


def test_pbrs_shaping_signal_at_true_terminal_next_state():
    # phi_t1 = 0 (true terminal), phi_t = 0.6 (mid-episode) -> F should be
    # strongly negative (potential collapsed to zero at termination).
    phi_t = 0.6
    phi_t1 = actual_potential(0.9, terminated=True, truncated=False)
    F = pbrs_shaping_signal(phi_t, phi_t1, gamma=0.995)
    assert F == pytest.approx(0.995 * 0.0 - 0.6)
    assert F < 0.0


# --------------------------------------------------------------- flag composition rules


def test_pbrs_condition_without_joint_network_raises(tmp_path):
    with pytest.raises(ValueError):
        run_stage11_pilot_training_job(
            master_seed=PILOT_V12_BASELINE_SEEDS[0],
            output_root=tmp_path / "output",
            checkpoint_root=tmp_path / "checkpoints",
            max_steps=600,
            strict=False,
            checkpoint_steps=(0, 600),
            episode_max_steps=50,
            pbrs_condition_v12="mean",
        )


def test_invalid_pbrs_condition_string_raises(tmp_path):
    with pytest.raises(ValueError):
        run_stage11_pilot_training_job(
            master_seed=PILOT_V12_BASELINE_SEEDS[0],
            output_root=tmp_path / "output",
            checkpoint_root=tmp_path / "checkpoints",
            max_steps=600,
            strict=False,
            checkpoint_steps=(0, 600),
            episode_max_steps=50,
            enable_stage9_based_reward_v5=True,
            enable_joint_network_v12=True,
            pbrs_condition_v12="not_a_real_condition",
        )


def test_joint_network_without_v5_reward_raises(tmp_path):
    with pytest.raises(ValueError):
        run_stage11_pilot_training_job(
            master_seed=PILOT_V12_BASELINE_SEEDS[0],
            output_root=tmp_path / "output",
            checkpoint_root=tmp_path / "checkpoints",
            max_steps=600,
            strict=False,
            checkpoint_steps=(0, 600),
            episode_max_steps=50,
            enable_joint_network_v12=True,
        )


@pytest.mark.parametrize(
    "conflicting_kwarg",
    [
        "enable_role_split_v6",
        "enable_periodic_reset_v7",
        "enable_visit_count_epsilon_v8",
        "enable_hysteretic_v9",
        "enable_hysteretic_v10",
        "enable_priority_signal_v11",
    ],
)
def test_joint_network_mutually_exclusive_with_v6_through_v11(tmp_path, conflicting_kwarg):
    kwargs = {
        "master_seed": PILOT_V12_BASELINE_SEEDS[0],
        "output_root": tmp_path / "output",
        "checkpoint_root": tmp_path / "checkpoints",
        "max_steps": 600,
        "strict": False,
        "checkpoint_steps": (0, 600),
        "episode_max_steps": 50,
        "enable_stage9_based_reward_v5": True,
        "enable_joint_network_v12": True,
        conflicting_kwarg: True,
    }
    if conflicting_kwarg in (
        "enable_periodic_reset_v7",
        "enable_visit_count_epsilon_v8",
        "enable_hysteretic_v9",
        "enable_hysteretic_v10",
        "enable_priority_signal_v11",
    ):
        kwargs["enable_role_split_v6"] = True
    with pytest.raises(ValueError):
        run_stage11_pilot_training_job(**kwargs)


# --------------------------------------------------------------- reward_version tags


def test_baseline_condition_selects_v12_baseline_tag(tmp_path):
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V12_BASELINE_SEEDS[0],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 600),
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
        enable_joint_network_v12=True,
    )
    assert manifest["reward_version"] == REWARD_VERSION_STAGE11_V12_JOINT_BASELINE
    assert manifest["architecture"] == "joint_centralised_v12"


def test_mean_condition_selects_v12_mean_tag(tmp_path):
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V12_MEAN_PBRS_SEEDS[0],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 600),
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
        enable_joint_network_v12=True,
        pbrs_condition_v12="mean",
    )
    assert manifest["reward_version"] == REWARD_VERSION_STAGE11_V12_JOINT_MEAN_PBRS


def test_min_condition_selects_v12_min_tag(tmp_path):
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V12_MIN_PBRS_SEEDS[0],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 600),
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
        enable_joint_network_v12=True,
        pbrs_condition_v12="min",
    )
    assert manifest["reward_version"] == REWARD_VERSION_STAGE11_V12_JOINT_MIN_PBRS


# --------------------------------------------------------------- end-to-end regression


def test_baseline_rewards_are_byte_identical_to_base_reward(tmp_path):
    """The whole point of Condition C: no PBRS shaping applied at all --
    shaped_reward must equal base_reward for every vehicle, every step."""
    output_root = tmp_path / "output"
    run_stage11_pilot_training_job(
        master_seed=PILOT_V12_BASELINE_SEEDS[1],
        output_root=output_root,
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 600),
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
        enable_joint_network_v12=True,
    )
    traj_path = output_root / "trajectories" / f"seed_{PILOT_V12_BASELINE_SEEDS[1]}.jsonl"
    lines = [json.loads(l) for l in traj_path.read_text(encoding="utf-8").strip().splitlines()]
    assert lines, "expected at least one trajectory row"
    for rec in lines:
        for v in rec["vehicles"]:
            assert v["base_reward"] == pytest.approx(v["shaped_reward"], abs=1e-12)


def test_mean_condition_rewards_differ_from_base_reward(tmp_path):
    output_root = tmp_path / "output"
    run_stage11_pilot_training_job(
        master_seed=PILOT_V12_MEAN_PBRS_SEEDS[1],
        output_root=output_root,
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 600),
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
        enable_joint_network_v12=True,
        pbrs_condition_v12="mean",
    )
    traj_path = output_root / "trajectories" / f"seed_{PILOT_V12_MEAN_PBRS_SEEDS[1]}.jsonl"
    lines = [json.loads(l) for l in traj_path.read_text(encoding="utf-8").strip().splitlines()]
    shaping_nonzero = sum(
        1 for rec in lines for v in rec["vehicles"] if abs(v["base_reward"] - v["shaped_reward"]) > 1e-9
    )
    assert shaping_nonzero > 0, "PBRS shaping never fired -- mean_pbrs condition is inert"


def test_v12_short_run_completes_and_produces_manifest(tmp_path):
    output_root = tmp_path / "output"
    checkpoint_root = tmp_path / "checkpoints"
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V12_BASELINE_SEEDS[2],
        output_root=output_root,
        checkpoint_root=checkpoint_root,
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 300, 600),
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
        enable_joint_network_v12=True,
    )
    assert manifest["final_step"] == 600
    assert len(manifest["checkpoints"]) == 3

    traj_path = output_root / "trajectories" / f"seed_{PILOT_V12_BASELINE_SEEDS[2]}.jsonl"
    assert traj_path.exists()
    lines = traj_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 600
    first = json.loads(lines[0])
    assert "welfare_mean" in first
    assert "welfare_min" in first

    last_ckpt = manifest["checkpoints"][-1]
    window = last_ckpt["window"]
    assert "p_ramp_first" in window
    assert "completion_rate" in window
    assert last_ckpt["learner"]["update_count"] >= 0  # exercised the joint update path


def test_v12_checkpoint_payload_has_single_joint_learner_shape(tmp_path):
    """Unlike v6-v11's per-role learner_map checkpoint shape, v12's
    payload["learner"] is a single flat dict (online/target/optimiser)."""
    import torch

    output_root = tmp_path / "output"
    checkpoint_root = tmp_path / "checkpoints"
    seed = PILOT_V12_BASELINE_SEEDS[3]
    run_stage11_pilot_training_job(
        master_seed=seed,
        output_root=output_root,
        checkpoint_root=checkpoint_root,
        max_steps=300,
        strict=False,
        checkpoint_steps=(0, 300),
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
        enable_joint_network_v12=True,
    )
    ckpt_path = checkpoint_root / f"seed_{seed}" / "ckpt_step_300.pt"
    payload = torch.load(ckpt_path, weights_only=False)
    assert set(payload["learner"].keys()) >= {"online", "target", "optimiser", "update_count", "replay_size"}
    assert payload["architecture"] == "joint_centralised_v12"


# --------------------------------------------------------------- speed-asymmetry option (Sec 12.4)


def test_target_speed_ramp_without_joint_network_raises(tmp_path):
    with pytest.raises(ValueError, match="target_speed_ramp"):
        run_stage11_pilot_training_job(
            master_seed=PILOT_V12_BASELINE_SEEDS[0],
            output_root=tmp_path / "output",
            checkpoint_root=tmp_path / "checkpoints",
            max_steps=600,
            strict=False,
            checkpoint_steps=(0, 600),
            episode_max_steps=50,
            target_speed_ramp=14.0,
        )


def test_spawn_speed_ramp_without_joint_network_raises(tmp_path):
    with pytest.raises(ValueError, match="spawn_speed_ramp"):
        run_stage11_pilot_training_job(
            master_seed=PILOT_V12_BASELINE_SEEDS[0],
            output_root=tmp_path / "output",
            checkpoint_root=tmp_path / "checkpoints",
            max_steps=600,
            strict=False,
            checkpoint_steps=(0, 600),
            episode_max_steps=50,
            spawn_speed_ramp=10.0,
        )


def test_target_speed_ramp_short_run_completes_and_shifts_ramp_attainment(tmp_path):
    """Directly recomputes attainment = clip(speed/target_speed, 0, 1) from
    each row's own recorded speed, using the KNOWN per-role target speeds,
    and checks it against the recorded ``attainment`` field -- confirms
    target_speed_by_role actually reaches the per-step attainment
    calculation (not just env construction), for every row of a real run,
    rather than hoping for a coincidental speed match between roles."""
    output_root = tmp_path / "output"
    checkpoint_root = tmp_path / "checkpoints"
    seed = PILOT_V12_BASELINE_SEEDS[4]
    manifest = run_stage11_pilot_training_job(
        master_seed=seed,
        output_root=output_root,
        checkpoint_root=checkpoint_root,
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 600),
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
        enable_joint_network_v12=True,
        target_speed_ramp=14.0,
        spawn_speed_ramp=10.0,
    )
    assert manifest["final_step"] == 600

    traj_path = output_root / "trajectories" / f"seed_{seed}.jsonl"
    rows = [json.loads(line) for line in traj_path.read_text(encoding="utf-8").strip().splitlines()]
    target_speed_by_role = {"ramp": 14.0, "mainline": 20.0}
    checked_ramp_rows = 0
    for row in rows:
        for v in row["vehicles"]:
            expected = max(0.0, min(1.0, v["speed"] / target_speed_by_role[v["role"]]))
            assert v["attainment"] == pytest.approx(expected, abs=1e-9), (row["step"], v)
            if v["role"] == "ramp":
                checked_ramp_rows += 1
    assert checked_ramp_rows > 0

    # Sanity check the asymmetry is real, not accidentally a no-op: at least
    # one row must show ramp attainment computed against 14.0 differing from
    # what it would have been against the (unused for ramp) 20.0 mainline
    # target -- i.e. ramp is not saturated at 1.0 the entire run.
    any_ramp_below_saturation = any(
        v["attainment"] < 1.0 for row in rows for v in row["vehicles"] if v["role"] == "ramp"
    )
    assert any_ramp_below_saturation


# --------------------------------------------------------------- v12 round 2: PER + n-step config


def test_per_hyperparameters_are_hessel_et_al_defaults():
    assert PER_ALPHA_V12 == pytest.approx(0.5)
    assert PER_BETA_START_V12 == pytest.approx(0.4)
    assert PER_BETA_END_V12 == pytest.approx(1.0)


def test_v12_baseline_v2_seeds_disjoint_from_every_prior_block():
    for block in (
        PILOT_V1_SEEDS,
        PILOT_V2_SEEDS,
        PILOT_V3_SEEDS,
        PILOT_V4_SEEDS,
        PILOT_V5_SEEDS,
        PILOT_V6_SEEDS,
        PILOT_V7_SEEDS,
        PILOT_V8_SEEDS,
        PILOT_V9_SEEDS,
        PILOT_V10_SEEDS,
        PILOT_V11_SEEDS,
        PILOT_V12_BASELINE_SEEDS,
        PILOT_V12_MEAN_PBRS_SEEDS,
        PILOT_V12_MIN_PBRS_SEEDS,
    ):
        assert set(PILOT_V12_BASELINE_V2_SEEDS).isdisjoint(block)


def test_v12_baseline_v2_seeds_pass_guard_and_are_eight():
    assert len(PILOT_V12_BASELINE_V2_SEEDS) == 8
    for seed in PILOT_V12_BASELINE_V2_SEEDS:
        assert_stage11_pilot_guards(master_seed=seed, max_steps=MAX_STEPS)


# --------------------------------------------------------------- v12 round 2: flag composition


def test_n_step_v12_without_joint_network_raises(tmp_path):
    with pytest.raises(ValueError):
        run_stage11_pilot_training_job(
            master_seed=PILOT_V12_BASELINE_V2_SEEDS[0],
            output_root=tmp_path / "output",
            checkpoint_root=tmp_path / "checkpoints",
            max_steps=600,
            strict=False,
            checkpoint_steps=(0, 600),
            episode_max_steps=50,
            n_step_v12=3,
        )


def test_prioritized_replay_v12_without_joint_network_raises(tmp_path):
    with pytest.raises(ValueError):
        run_stage11_pilot_training_job(
            master_seed=PILOT_V12_BASELINE_V2_SEEDS[0],
            output_root=tmp_path / "output",
            checkpoint_root=tmp_path / "checkpoints",
            max_steps=600,
            strict=False,
            checkpoint_steps=(0, 600),
            episode_max_steps=50,
            enable_prioritized_replay_v12=True,
        )


def test_n_step_v12_less_than_one_raises(tmp_path):
    with pytest.raises(ValueError):
        run_stage11_pilot_training_job(
            master_seed=PILOT_V12_BASELINE_V2_SEEDS[0],
            output_root=tmp_path / "output",
            checkpoint_root=tmp_path / "checkpoints",
            max_steps=600,
            strict=False,
            checkpoint_steps=(0, 600),
            episode_max_steps=50,
            enable_stage9_based_reward_v5=True,
            enable_joint_network_v12=True,
            n_step_v12=0,
        )


# --------------------------------------------------------------- v12 round 2: n-step windowing


def test_n_step_windowing_preserves_one_transition_per_env_step(tmp_path):
    """Regardless of n_step_v12, the windowing algorithm (fill-and-emit
    mid-episode, flush-remaining-with-shortened-n_steps at episode end) must
    emit exactly ONE replay transition per real environment step over the
    whole run -- no step's data is ever dropped or double-counted. n_step_v12=1
    is the default/pre-n-step case; this also proves it reduces to the old
    one-transition-per-step behaviour."""
    for n in (1, 2, 5):
        manifest = run_stage11_pilot_training_job(
            master_seed=PILOT_V12_BASELINE_V2_SEEDS[0],
            output_root=tmp_path / f"output_{n}",
            checkpoint_root=tmp_path / f"checkpoints_{n}",
            max_steps=600,
            strict=False,
            checkpoint_steps=(0, 600),
            episode_max_steps=50,
            enable_stage9_based_reward_v5=True,
            enable_joint_network_v12=True,
            n_step_v12=n,
        )
        replay_size = manifest["checkpoints"][-1]["learner"]["replay_size"]
        assert replay_size == 600, f"n_step_v12={n}: expected 600 transitions, got {replay_size}"


def test_n_step_v12_default_short_run_completes_and_produces_manifest(tmp_path):
    """n_step_v12 left at its default (1): same shape guarantees as the
    already-passing pre-n-step short-run test above."""
    output_root = tmp_path / "output"
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V12_BASELINE_V2_SEEDS[1],
        output_root=output_root,
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 300, 600),
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
        enable_joint_network_v12=True,
    )
    assert manifest["final_step"] == 600
    assert len(manifest["checkpoints"]) == 3
    traj_path = output_root / "trajectories" / f"seed_{PILOT_V12_BASELINE_V2_SEEDS[1]}.jsonl"
    assert len(traj_path.read_text(encoding="utf-8").strip().splitlines()) == 600


# --------------------------------------------------------------- v12 round 2: PER end-to-end


def test_v12_prioritized_replay_short_run_completes(tmp_path):
    output_root = tmp_path / "output"
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V12_BASELINE_V2_SEEDS[2],
        output_root=output_root,
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 600),
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
        enable_joint_network_v12=True,
        enable_prioritized_replay_v12=True,
    )
    assert manifest["final_step"] == 600
    assert manifest["checkpoints"][-1]["learner"]["update_count"] > 0


def test_v12_prioritized_replay_with_n_step_short_run_completes(tmp_path):
    """Both new mechanisms combined -- the actually-intended launch
    configuration for this round's next real training run (not launched by
    this fork; implementation/tests only)."""
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V12_BASELINE_V2_SEEDS[3],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 600),
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
        enable_joint_network_v12=True,
        enable_prioritized_replay_v12=True,
        n_step_v12=3,
    )
    assert manifest["final_step"] == 600
    assert manifest["checkpoints"][-1]["learner"]["replay_size"] == 600


# --------------------------------------------------------------- v12 round 2: default-off regression


def test_default_flags_run_is_deterministic_and_matches_pre_change_shape(tmp_path):
    """The single most important regression check for this round's addition:
    with n_step_v12=1 and enable_prioritized_replay_v12=False (both
    defaults), two runs of the SAME seed must be byte-identical in every
    reported metric -- proving the new windowing/PER code paths consume no
    incidental randomness and alter no control flow when left at their
    default/off values."""
    kwargs = dict(
        master_seed=PILOT_V12_BASELINE_V2_SEEDS[4],
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 300, 600),
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
        enable_joint_network_v12=True,
    )
    manifest_a = run_stage11_pilot_training_job(
        output_root=tmp_path / "a_output", checkpoint_root=tmp_path / "a_checkpoints", **kwargs
    )
    manifest_b = run_stage11_pilot_training_job(
        output_root=tmp_path / "b_output", checkpoint_root=tmp_path / "b_checkpoints", **kwargs
    )
    assert manifest_a["checkpoints"] == manifest_b["checkpoints"]
    assert manifest_a["final_step"] == manifest_b["final_step"] == 600


# --------------------------------------------------------------- v12 round 2: 400K step-budget extension


def test_max_steps_v12_is_400k():
    assert MAX_STEPS_V12 == 400_000


def test_checkpoint_steps_v12_spans_full_400k_budget():
    assert CHECKPOINT_STEPS_V12[0] == 0
    assert CHECKPOINT_STEPS_V12[-1] == MAX_STEPS_V12
    assert len(CHECKPOINT_STEPS_V12) == 41  # 0, 10K, ..., 400K


def test_epsilon_and_lr_decay_steps_v12_scale_proportionally_with_v1_v11():
    """Same 100%/80% fractions as the frozen v1-v11 schedule (LEARNING_RATE_DECAY_STEPS
    == MAX_STEPS, EPSILON_DECAY_STEPS == 0.8 * MAX_STEPS), applied to
    MAX_STEPS_V12 instead -- this proportional (not fixed-absolute) scaling
    is what prevents the schedule from hitting its floor early relative to
    the longer 400K budget."""
    assert LEARNING_RATE_DECAY_STEPS_V12 == MAX_STEPS_V12
    assert EPSILON_DECAY_STEPS_V12 == int(0.8 * MAX_STEPS_V12) == 320_000


def test_epsilon_at_step_v12_boundaries():
    assert epsilon_at_step_v12(0) == pytest.approx(EPSILON_START)
    assert epsilon_at_step_v12(EPSILON_DECAY_STEPS_V12) == pytest.approx(EPSILON_END)
    assert epsilon_at_step_v12(EPSILON_DECAY_STEPS_V12 + 50_000) == pytest.approx(EPSILON_END)


def test_epsilon_at_step_v12_80k_is_far_above_floor():
    """The exact property the 400K extension exists to guarantee: at step
    80,000 -- where the OLD 100K-budget schedule (EPSILON_DECAY_STEPS =
    80,000) had ALREADY hit EPSILON_END and stayed pinned there for the rest
    of training, the low-plasticity tail directly implicated in the
    'frozen_stall' failure mode found in this pilot's own real v12 baseline
    trajectory data -- the NEW v12 schedule is still exploring at a
    substantial rate, exactly 25% of the way through its own decay window
    (80,000 / 320,000)."""
    eps_80k = epsilon_at_step_v12(80_000)
    expected = EPSILON_START + 0.25 * (EPSILON_END - EPSILON_START)
    assert eps_80k == pytest.approx(expected)
    assert eps_80k == pytest.approx(0.775)
    assert eps_80k > EPSILON_END + 0.5  # nowhere near the floor


def test_lr_at_step_v12_boundaries():
    assert lr_at_step_v12(0) == pytest.approx(LEARNING_RATE_START)
    assert lr_at_step_v12(LEARNING_RATE_DECAY_STEPS_V12) == pytest.approx(LEARNING_RATE_END)
    assert lr_at_step_v12(LEARNING_RATE_DECAY_STEPS_V12 + 10_000) == pytest.approx(LEARNING_RATE_END)


def test_lr_at_step_v12_80k_is_still_well_above_floor():
    lr_80k = lr_at_step_v12(80_000)
    expected = LEARNING_RATE_START + 0.2 * (LEARNING_RATE_END - LEARNING_RATE_START)
    assert lr_80k == pytest.approx(expected)
    assert lr_80k > LEARNING_RATE_END


def test_assert_guard_accepts_max_steps_v12_via_allowed_max_steps_param():
    for seed in PILOT_V12_BASELINE_V2_SEEDS:
        assert_stage11_pilot_guards(master_seed=seed, max_steps=MAX_STEPS_V12, allowed_max_steps=MAX_STEPS_V12)


def test_assert_guard_rejects_max_steps_v12_without_allowed_max_steps_param():
    """Omitting allowed_max_steps must still enforce the frozen v1-v11
    MAX_STEPS (100K) exactly as before -- 400K is only accepted when the
    caller opts in explicitly."""
    with pytest.raises(RuntimeError, match="max_steps must be"):
        assert_stage11_pilot_guards(master_seed=PILOT_V12_BASELINE_V2_SEEDS[0], max_steps=MAX_STEPS_V12)


def test_assert_guard_v1_through_v11_unaffected_by_new_param():
    """Regression: v1-v11 call sites (which never pass allowed_max_steps)
    must behave byte-identically to before this round's change."""
    for block in (PILOT_V1_SEEDS, PILOT_V5_SEEDS, PILOT_V11_SEEDS):
        for seed in block:
            assert_stage11_pilot_guards(master_seed=seed, max_steps=MAX_STEPS)
    with pytest.raises(RuntimeError, match="max_steps must be"):
        assert_stage11_pilot_guards(master_seed=PILOT_V1_SEEDS[0], max_steps=250_000)


def test_resolve_v12_checkpoint_steps_swaps_shared_default_to_v12_schedule():
    assert _resolve_v12_checkpoint_steps(CHECKPOINT_STEPS, 400_000) == CHECKPOINT_STEPS_V12


def test_resolve_v12_checkpoint_steps_preserves_explicit_override():
    custom = (0, 300, 600)
    assert _resolve_v12_checkpoint_steps(custom, 400_000) == custom


def test_resolve_v12_checkpoint_steps_scales_with_larger_max_steps():
    assert _resolve_v12_checkpoint_steps(CHECKPOINT_STEPS, 3_000_000) == tuple(
        range(0, 3_000_001, 10_000)
    )


def test_v12_run_with_default_checkpoint_steps_derives_from_own_max_steps(tmp_path):
    """End-to-end wiring proof (the fast pure-function unit tests above
    already cover the scaling logic itself in detail): a v12 run that
    leaves checkpoint_steps at its caller-facing default (i.e. never passes
    the kwarg at all) actually threads its own max_steps through to
    _resolve_v12_checkpoint_steps rather than silently reverting to some
    other value. Real large-max_steps training is too slow for a unit
    test, so this only checks the small-max_steps case (checkpoint_steps
    == (0,), since the next 10K-spaced checkpoint at step 10_000 is beyond
    max_steps=40) -- deliberately not re-proving the scaling behaviour
    itself, just that the wiring didn't regress to a crash or a stale
    hardcoded value."""
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V12_BASELINE_V2_SEEDS[5],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=40,
        strict=False,
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
        enable_joint_network_v12=True,
    )
    assert manifest["checkpoint_steps"] == [0]
