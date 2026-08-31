"""Stage 10 pilot v4 (E28) -- pre-training audit for the 2->4->6
threshold-triggered curriculum, the scene-vehicle-count observation feature,
and the easier baseline geometry (wider pre-merge zone + spawn spacing +
longer per-episode step budget).

Covers: env n_vehicles in {2,4,6} with correct N-per-side queue spacing,
scene_vehicle_count feature correctness at all three counts, OBS_DIM==9
everywhere, stage_index_for_advance's threshold/safety-valve decision logic
(pure, no RNG/env needed), the runner's actual stage-advancement behaviour
end-to-end on tiny windows, LR decay's new parameterised decay_steps, and the
v4 seed guard -- without running the full ~180K-step pilot.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from thesis.agents.independent_dqn_v2 import DQNConfig
from thesis.envs.stage10_symmetric_merge_env import (
    OBS_DIM,
    HighLevelAction,
    Stage10MergeEnvConfig,
    Stage10RouteGeometry,
    Stage10SymmetricMergeEnv,
    encode_scene_vehicle_count,
)
from thesis.pilots.stage10_role_phase_subpolicy_config import (
    CURRICULUM_V4_ADVANCE_THRESHOLD,
    CURRICULUM_V4_ROLLING_WINDOW_EPISODES,
    CURRICULUM_V4_STAGE_MAX_STEPS,
    CURRICULUM_V4_STAGE_VEHICLE_COUNTS,
    EPSILON_END,
    EPSILON_STAGE_BUMP_DECAY_FRACTION,
    EPSILON_STAGE_BUMP_VALUE,
    LEARNING_RATE_END,
    LEARNING_RATE_START,
    MAX_STEPS_V4,
    OBS_DIM as CONFIG_OBS_DIM,
    PILOT_V1_SEEDS,
    PILOT_V2_SEEDS,
    PILOT_V3_SEEDS,
    PILOT_V4_SEEDS,
    assert_stage10_pilot_guards,
    epsilon_at_step,
    epsilon_for_stage_transition,
    epsilon_for_step,
    lr_at_step,
    stage_index_for_advance,
)
from thesis.pilots.stage10_role_phase_subpolicy_runner import (
    build_dqn_config,
    run_pilot_training_job,
)


# ------------------------------------------------------------------- geometry
def test_pre_merge_lengthened_others_unchanged():
    g = Stage10RouteGeometry()
    assert g.merge_start == 200.0  # v1-v3: 150.0 -> v4: 200.0 (easier-baseline lever)
    assert g.merge_end == 300.0
    assert g.route_exit == 400.0
    pre_length = g.merge_start - g.route_start
    merging_length = g.merge_end - g.merge_start
    post_length = g.route_exit - g.merge_end
    assert pre_length == 200.0
    assert merging_length == 100.0  # unchanged since v3's revert
    assert post_length == 100.0  # unchanged since v3's revert


def test_env_default_max_steps_and_spawn_gap_widened():
    cfg = Stage10MergeEnvConfig()
    assert cfg.max_steps == 600  # v1-v3: 400 -> v4: 600
    assert cfg.spawn_queue_gap == 60.0  # v1-v3 implicit gap was 40.0 -> v4: +50%


# ------------------------------------------------------------------ n_vehicles
def test_env_accepts_2_4_6_rejects_others():
    for n in (2, 4, 6):
        Stage10MergeEnvConfig(n_vehicles=n).validate()  # must not raise
    for bad in (1, 3, 5, 7, 8):
        with pytest.raises(ValueError):
            Stage10MergeEnvConfig(n_vehicles=bad).validate()


@pytest.mark.parametrize("n_vehicles,expected_per_role", [(2, 1), (4, 2), (6, 3)])
@pytest.mark.parametrize("seed", range(5))
def test_role_counts_correct_at_each_vehicle_count(n_vehicles, expected_per_role, seed):
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=seed, n_vehicles=n_vehicles))
    obs, info = env.reset(seed=seed)
    roles = info["roles"]
    assert len(roles) == n_vehicles
    assert sum(1 for r in roles.values() if r == "ramp") == expected_per_role
    assert sum(1 for r in roles.values() if r == "mainline") == expected_per_role
    assert set(env.active_vehicle_ids) == set(roles.keys())
    assert set(obs.keys()) == set(roles.keys())


@pytest.mark.parametrize("n_vehicles,expected_per_role", [(2, 1), (4, 2), (6, 3)])
def test_spawn_queue_spacing_correct_at_each_vehicle_count(n_vehicles, expected_per_role):
    """Each role's queue members must be spaced exactly spawn_queue_gap apart
    (up to the shared per-episode jitter, which shifts the whole queue
    together and so does not affect relative spacing)."""
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=42, n_vehicles=n_vehicles))
    _, info = env.reset(seed=42)
    roles = info["roles"]
    gap = env.config.spawn_queue_gap
    for role in ("ramp", "mainline"):
        members = sorted(vid for vid, r in roles.items() if r == role)
        assert len(members) == expected_per_role
        positions = sorted((env._vehicles[vid].route_position for vid in members), reverse=True)
        for i in range(1, len(positions)):
            assert positions[i - 1] - positions[i] == pytest.approx(gap)


def test_six_vehicle_mode_step_and_collision_work():
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=7, n_vehicles=6))
    env.reset(seed=7)
    active = env.active_vehicle_ids
    assert len(active) == 6
    actions = {vid: HighLevelAction.MAINTAIN for vid in active}
    obs, reward, terminated, truncated, info = env.step(actions)
    assert set(obs.keys()) == set(active)
    assert set(reward.keys()) == set(active)
    assert len(info["zone_t"]) == 6


# ----------------------------------------------------------- scene-context feature
def test_encode_scene_vehicle_count_values():
    assert encode_scene_vehicle_count(2) == 0.0
    assert encode_scene_vehicle_count(4) == 0.5
    assert encode_scene_vehicle_count(6) == 1.0


@pytest.mark.parametrize("n_vehicles,expected", [(2, 0.0), (4, 0.5), (6, 1.0)])
def test_scene_vehicle_count_feature_in_observation(n_vehicles, expected):
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=1, n_vehicles=n_vehicles))
    obs, _ = env.reset(seed=1)
    for vid, o in obs.items():
        assert o[8] == pytest.approx(expected)  # scene_vehicle_count is the 9th (index 8) feature
    actions = {vid: HighLevelAction.MAINTAIN for vid in env.active_vehicle_ids}
    obs2, *_ = env.step(actions)
    for vid, o in obs2.items():
        assert o[8] == pytest.approx(expected)


# --------------------------------------------------------------- OBS_DIM everywhere
def test_obs_dim_is_9_everywhere():
    assert OBS_DIM == 9
    assert CONFIG_OBS_DIM == 9
    cfg: DQNConfig = build_dqn_config()
    assert cfg.obs_dim == 9


def test_reset_and_step_observations_have_9_dims():
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=1))
    obs, _ = env.reset(seed=1)
    for vid, o in obs.items():
        assert o.shape == (9,)
    actions = {vid: HighLevelAction.MAINTAIN for vid in env.active_vehicle_ids}
    obs2, *_ = env.step(actions)
    for vid, o in obs2.items():
        assert o.shape == (9,)


# -------------------------------------------------------- LR decay (parameterised)
def test_lr_at_step_default_unchanged_from_v1_v3():
    # decay_steps defaults to the frozen v1-v3 value (100_000) -- v1-v3
    # behaviour/tests must be completely unaffected by v4's new parameter.
    assert lr_at_step(0) == pytest.approx(LEARNING_RATE_START)
    assert lr_at_step(100_000) == pytest.approx(LEARNING_RATE_END)


def test_lr_at_step_v4_decay_steps_override():
    assert lr_at_step(0, decay_steps=MAX_STEPS_V4) == pytest.approx(LEARNING_RATE_START)
    assert lr_at_step(MAX_STEPS_V4, decay_steps=MAX_STEPS_V4) == pytest.approx(LEARNING_RATE_END)
    mid = lr_at_step(MAX_STEPS_V4 // 2, decay_steps=MAX_STEPS_V4)
    assert LEARNING_RATE_END < mid < LEARNING_RATE_START


# ----------------------------------------------- epsilon stage-transition bump
def test_epsilon_stage_transition_bumps_to_0_5_at_stage_start():
    assert epsilon_for_stage_transition(0, stage_max_steps=60_000) == pytest.approx(
        EPSILON_STAGE_BUMP_VALUE
    )


def test_epsilon_stage_transition_decays_linearly_to_floor():
    stage_max_steps = 60_000
    decay_steps = EPSILON_STAGE_BUMP_DECAY_FRACTION * stage_max_steps  # 12_000
    half = epsilon_for_stage_transition(int(decay_steps // 2), stage_max_steps=stage_max_steps)
    assert EPSILON_END < half < EPSILON_STAGE_BUMP_VALUE
    at_end = epsilon_for_stage_transition(int(decay_steps), stage_max_steps=stage_max_steps)
    assert at_end == pytest.approx(EPSILON_END)


def test_epsilon_stage_transition_stays_at_floor_after_decay_window():
    stage_max_steps = 60_000
    decay_steps = EPSILON_STAGE_BUMP_DECAY_FRACTION * stage_max_steps
    well_past = epsilon_for_stage_transition(
        int(decay_steps) + 30_000, stage_max_steps=stage_max_steps
    )
    assert well_past == pytest.approx(EPSILON_END)
    at_stage_end = epsilon_for_stage_transition(stage_max_steps, stage_max_steps=stage_max_steps)
    assert at_stage_end == pytest.approx(EPSILON_END)


def test_epsilon_for_step_stage_zero_matches_original_unchanged_schedule():
    # Stage 1 (stage_idx=0) must be bit-for-bit identical to the original
    # global epsilon_at_step -- v1-v3's characterised Stage-1 dynamics must
    # not shift at all from this fix.
    for step in (0, 1, 25_000, 49_999, 50_000, 75_000):
        assert epsilon_for_step(
            step=step, stage_idx=0, steps_in_stage=step, stage_max_steps=CURRICULUM_V4_STAGE_MAX_STEPS
        ) == pytest.approx(epsilon_at_step(step))


@pytest.mark.parametrize("stage_idx", [1, 2])
def test_epsilon_for_step_nonzero_stage_uses_bump_not_global_schedule(stage_idx):
    # At the instant a non-initial stage begins, epsilon must be the bump
    # value regardless of how large the global step count already is (e.g.
    # entering Stage 3 around global step ~100,000, long past where the
    # original epsilon_at_step schedule would have floored at 0.10).
    global_step_at_stage_start = 100_000
    eps = epsilon_for_step(
        step=global_step_at_stage_start,
        stage_idx=stage_idx,
        steps_in_stage=0,
        stage_max_steps=CURRICULUM_V4_STAGE_MAX_STEPS,
    )
    assert eps == pytest.approx(EPSILON_STAGE_BUMP_VALUE)
    assert eps > epsilon_at_step(global_step_at_stage_start)  # strictly more exploration than before the fix


def test_epsilon_for_step_frozen_constants_give_sensible_decay_windows():
    # Sanity-check the actual frozen per-stage decay windows implied by
    # EPSILON_STAGE_BUMP_DECAY_FRACTION against CURRICULUM_V4_STAGE_MAX_STEPS
    # (40_000, 60_000, 80_000): Stage 2 -> 12_000-step window, Stage 3 ->
    # 16_000-step window -- both comfortably shorter than each stage's own
    # safety valve, not eating most of the stage's budget.
    stage2_window = EPSILON_STAGE_BUMP_DECAY_FRACTION * CURRICULUM_V4_STAGE_MAX_STEPS[1]
    stage3_window = EPSILON_STAGE_BUMP_DECAY_FRACTION * CURRICULUM_V4_STAGE_MAX_STEPS[2]
    assert stage2_window == pytest.approx(12_000)
    assert stage3_window == pytest.approx(16_000)
    assert stage2_window < CURRICULUM_V4_STAGE_MAX_STEPS[1]
    assert stage3_window < CURRICULUM_V4_STAGE_MAX_STEPS[2]


# --------------------------------------------------- stage_index_for_advance (pure)
def test_stage_advances_on_threshold():
    assert (
        stage_index_for_advance(
            current_stage_idx=0,
            rolling_completion_rate=0.95,
            steps_in_current_stage=100,
            n_stages=3,
            advance_threshold=0.90,
            stage_max_steps=(40_000, 60_000, 80_000),
        )
        == 1
    )


def test_stage_does_not_advance_below_threshold_and_below_safety_valve():
    assert (
        stage_index_for_advance(
            current_stage_idx=0,
            rolling_completion_rate=0.50,
            steps_in_current_stage=100,
            n_stages=3,
            advance_threshold=0.90,
            stage_max_steps=(40_000, 60_000, 80_000),
        )
        == 0
    )


def test_stage_advances_on_safety_valve_even_if_threshold_never_met():
    assert (
        stage_index_for_advance(
            current_stage_idx=0,
            rolling_completion_rate=0.0,  # threshold never met
            steps_in_current_stage=40_000,  # exactly at the safety valve
            n_stages=3,
            advance_threshold=0.90,
            stage_max_steps=(40_000, 60_000, 80_000),
        )
        == 1
    )


def test_stage_does_not_advance_when_rolling_rate_is_none():
    """None means the trailing window isn't full yet -- must not accidentally
    be treated as 0.0 (which would never trigger) or any other sentinel that
    could misbehave; explicit None handling is what's being tested here."""
    assert (
        stage_index_for_advance(
            current_stage_idx=0,
            rolling_completion_rate=None,
            steps_in_current_stage=10,
            n_stages=3,
            advance_threshold=0.90,
            stage_max_steps=(40_000, 60_000, 80_000),
        )
        == 0
    )


def test_terminal_stage_never_advances():
    assert (
        stage_index_for_advance(
            current_stage_idx=2,
            rolling_completion_rate=1.0,
            steps_in_current_stage=999_999,
            n_stages=3,
            advance_threshold=0.90,
            stage_max_steps=(40_000, 60_000, 80_000),
        )
        == 2
    )


def test_stage_index_for_advance_defaults_match_frozen_v4_constants():
    assert (
        stage_index_for_advance(
            current_stage_idx=0,
            rolling_completion_rate=None,
            steps_in_current_stage=0,
        )
        == 0
    )
    assert len(CURRICULUM_V4_STAGE_VEHICLE_COUNTS) == 3
    assert CURRICULUM_V4_STAGE_VEHICLE_COUNTS == (2, 4, 6)
    assert CURRICULUM_V4_STAGE_MAX_STEPS == (40_000, 60_000, 80_000)
    assert MAX_STEPS_V4 == 180_000
    assert CURRICULUM_V4_ADVANCE_THRESHOLD == pytest.approx(0.90)
    assert CURRICULUM_V4_ROLLING_WINDOW_EPISODES == 100


# ------------------------------------------------------ runner integration (tiny)
def test_runner_advances_stage_via_safety_valve_end_to_end():
    """Tiny per-stage safety valves guarantee a safety-valve-triggered
    advance regardless of actual completion rate (threshold is set to an
    unreachable 2.0 so only the safety valve can fire) -- verifies the
    runner's stage bookkeeping (env.config.n_vehicles, checkpoint
    curriculum_stage_idx) advances correctly end-to-end, not just the pure
    decision function in isolation. episode_max_steps=10 forces frequent
    episode boundaries deterministically (the env's own per-episode default
    is 600 -- far too long to guarantee even one boundary within this tiny
    150-step total budget by chance alone; advancement can only be checked
    at an episode boundary, so this must not be left to luck)."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        result = run_pilot_training_job(
            master_seed=PILOT_V4_SEEDS[0],
            output_root=d / "out",
            checkpoint_root=d / "ckpt",
            max_steps=150,
            strict=False,
            stage_vehicle_counts=(2, 4, 6),
            stage_max_steps=(30, 30, 90),  # tiny safety valves
            advance_threshold=2.0,  # unreachable -- forces safety-valve-only advancement
            rolling_window_episodes=1,
            checkpoint_steps=(0, 30, 60, 150),
            episode_max_steps=10,
        )
        assert result["final_step"] == 150
        assert result["final_curriculum_stage_idx"] == 2
        assert result["final_curriculum_stage_vehicles"] == 6

        traj_path = d / "out" / "trajectories" / f"seed_{PILOT_V4_SEEDS[0]}.jsonl"
        lines = [json.loads(l) for l in traj_path.read_text(encoding="utf-8").strip().splitlines()]
        # Well past both safety valves, the tail must be Stage 3 (6 vehicles).
        tail = [rec for rec in lines if rec["step"] > 90]
        assert tail, "expected trajectory rows past both safety valves"
        assert all(len(rec["vehicles"]) == 6 for rec in tail)
        assert all(rec["curriculum_stage_idx"] == 2 for rec in tail)


def test_runner_stays_in_stage_one_when_max_steps_below_first_safety_valve():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        result = run_pilot_training_job(
            master_seed=PILOT_V4_SEEDS[1],
            output_root=d / "out",
            checkpoint_root=d / "ckpt",
            max_steps=20,
            strict=False,
            stage_vehicle_counts=(2, 4, 6),
            stage_max_steps=(30, 30, 90),
            advance_threshold=2.0,
            rolling_window_episodes=1,
            checkpoint_steps=(0, 20),
        )
        assert result["final_curriculum_stage_idx"] == 0
        assert result["final_curriculum_stage_vehicles"] == 2


def test_runner_disabling_trajectory_logging_writes_no_file():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        run_pilot_training_job(
            master_seed=PILOT_V4_SEEDS[2],
            output_root=d / "out",
            checkpoint_root=d / "ckpt",
            max_steps=20,
            strict=False,
            stage_vehicle_counts=(2, 4, 6),
            stage_max_steps=(30, 30, 90),
            checkpoint_steps=(0, 20),
            enable_trajectory_logging=False,
        )
        assert not (d / "out" / "trajectories").exists()


def test_runner_no_duplicate_final_checkpoint():
    """Incidental bug fix (not one of the four confirmed round-4 changes,
    found while rewriting the runner): v1-v3's finally-block unconditionally
    re-saved the checkpoint at the final step even when the main loop had
    already saved it that same iteration, producing a spurious duplicate
    all-zero-window entry at the end of every manifest. Verifies the fix:
    exactly one record per checkpoint step, not two."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        result = run_pilot_training_job(
            master_seed=PILOT_V4_SEEDS[3],
            output_root=d / "out",
            checkpoint_root=d / "ckpt",
            max_steps=20,
            strict=False,
            stage_vehicle_counts=(2, 4, 6),
            stage_max_steps=(30, 30, 90),
            checkpoint_steps=(0, 20),
        )
        steps_seen = [c["step"] for c in result["checkpoints"]]
        assert steps_seen == sorted(set(steps_seen)), "duplicate checkpoint step entries found"


# --------------------------------------------------------------- seed guards
def test_v4_seeds_pass_the_guard_with_new_ceiling_and_are_disjoint_from_v1_v2_v3():
    assert set(PILOT_V4_SEEDS).isdisjoint(set(PILOT_V1_SEEDS))
    assert set(PILOT_V4_SEEDS).isdisjoint(set(PILOT_V2_SEEDS))
    assert set(PILOT_V4_SEEDS).isdisjoint(set(PILOT_V3_SEEDS))
    for seed in PILOT_V4_SEEDS:
        assert_stage10_pilot_guards(master_seed=seed, max_steps=MAX_STEPS_V4)  # must not raise


def test_v4_seeds_reject_the_old_100k_ceiling():
    """v4 seeds must use the NEW 180_000 ceiling, not v1-v3's frozen 100_000
    -- each arm's budget is immutable and arm-specific."""
    for seed in PILOT_V4_SEEDS:
        with pytest.raises(RuntimeError):
            assert_stage10_pilot_guards(master_seed=seed, max_steps=100_000)


def test_v1_v2_v3_seeds_still_require_the_old_100k_ceiling():
    for seed in (*PILOT_V1_SEEDS, *PILOT_V2_SEEDS, *PILOT_V3_SEEDS):
        assert_stage10_pilot_guards(master_seed=seed, max_steps=100_000)  # must not raise
        with pytest.raises(RuntimeError):
            assert_stage10_pilot_guards(master_seed=seed, max_steps=MAX_STEPS_V4)


def test_guard_rejects_seed_outside_all_six_pilot_blocks():
    # 68017-68020 are legitimate v4 seeds; 68021-68028 are legitimate v5 seeds
    # (shared-parameter architecture, 2026-08-06); 68029-68036 are legitimate
    # v6 seeds (reward-function revision, 2026-08-06); 68037-68044 are now
    # legitimate v7 seeds (reward-magnitude curriculum, 2026-08-06) -- 68045
    # is the first value genuinely past the whole E28 Stage 10 reserved block
    # (68001-68044). Boundary value bumped again for the same reason it was
    # bumped at the v3->v4, v4->v5, and v5->v6/v7 transitions: name kept
    # (not renamed to ...seven... every time) since the point of the test
    # ("first seed past the WHOLE reserved block") doesn't change, only the
    # boundary number does.
    with pytest.raises(RuntimeError):
        assert_stage10_pilot_guards(master_seed=68045, max_steps=MAX_STEPS_V4)
