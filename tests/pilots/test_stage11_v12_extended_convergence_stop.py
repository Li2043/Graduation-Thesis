"""Stage 11 v12 -- auto-stop (converge / fail / timeout) for the N=4
extended-training run, plus the epsilon/LR schedule rescaling that makes a
longer-than-400K run behave sensibly instead of flooring exploration/LR at
the old absolute step counts.

Covers: ``_stage11_extended_stop_check`` in isolation (converged, not-
converged-due-to-variance, frozen-stall failure, collision failure, the
"failure streak must be genuinely consecutive" requirement, zero-episode
windows excluded, not-enough-history-yet); ``epsilon_at_step_v12``/
``lr_at_step_v12``'s ``decay_steps`` override actually stretching the
decay instead of flooring early; and one real, short end-to-end run
proving the stop-check is actually wired into the training loop (not just
correct as a standalone function) by forcing a trivial always-true
convergence rule and confirming the run stops before ``max_steps`` with
``stop_reason == "converged"`` in the manifest.
"""

from __future__ import annotations

from thesis.pilots.stage11_dyad_merge_pilot_config import (
    EPSILON_DECAY_STEPS_V12,
    EPSILON_END,
    EPSILON_START,
    LEARNING_RATE_DECAY_STEPS_V12,
    LEARNING_RATE_END,
    LEARNING_RATE_START,
    PILOT_V12_BASELINE_V2_SEEDS,
    epsilon_at_step_v12,
    lr_at_step_v12,
)
from thesis.pilots.stage11_dyad_merge_runner import (
    _stage11_extended_stop_check,
    run_stage11_pilot_training_job,
)


def _window(completion_rate: float, collision_free_rate: float = 1.0, episodes: int = 20) -> dict:
    return {
        "episodes": episodes,
        "completion_rate": completion_rate,
        "collision_free_rate": collision_free_rate,
    }


def _records(windows: list[dict], start_step: int = 10_000, spacing: int = 10_000) -> list[dict]:
    return [{"step": start_step + i * spacing, "window": w} for i, w in enumerate(windows)]


def test_stop_check_returns_none_with_too_little_history():
    records = _records([_window(0.99)] * 3)  # short of the default converge_window=5
    assert _stage11_extended_stop_check(records) is None


def test_stop_check_converged_on_high_stable_completion():
    records = _records([_window(0.96), _window(0.97), _window(0.95), _window(0.96), _window(0.98)])
    assert _stage11_extended_stop_check(records) == "converged"


def test_stop_check_high_mean_but_unstable_does_not_converge():
    # mean == 0.96 (>= 0.95 threshold) but pstdev ~0.049 (> 0.02 threshold)
    rates = [1.0, 0.90, 1.0, 0.90, 1.0]
    records = _records([_window(r) for r in rates])
    assert _stage11_extended_stop_check(records) is None


def test_stop_check_frozen_stall_failure():
    records = _records([_window(0.0, collision_free_rate=1.0)] * 10)
    assert _stage11_extended_stop_check(records) == "failure_frozen_stall"


def test_stop_check_collision_failure():
    # completion_rate=0.5 is well above the frozen-stall floor (0.05), so
    # only the collision-floor rule can fire here.
    records = _records([_window(0.5, collision_free_rate=0.2)] * 10)
    assert _stage11_extended_stop_check(records) == "failure_collision"


def test_stop_check_failure_streak_must_be_consecutive():
    # 9 near-zero windows plus one healthy window breaks the streak -- must
    # NOT trigger failure_frozen_stall, since the rule requires ALL of the
    # last fail_window windows to qualify, not most of them.
    windows = [_window(0.0)] * 9 + [_window(0.9)]
    records = _records(windows)
    assert _stage11_extended_stop_check(records) is None


def test_stop_check_ignores_zero_episode_windows():
    # A step-0-style record with no episodes yet must not count toward
    # either window length or be mistaken for a 0.0 completion_rate.
    zero_episode = _window(0.0, episodes=0)
    healthy = [_window(0.96)] * 5
    records = [{"step": 0, "window": zero_episode}] + _records(healthy)
    assert _stage11_extended_stop_check(records) == "converged"


def test_stop_check_thresholds_are_overridable():
    records = _records([_window(0.80)] * 3)
    assert (
        _stage11_extended_stop_check(
            records,
            converge_window=3,
            converge_completion_mean=0.75,
            converge_completion_std=0.05,
        )
        == "converged"
    )


def test_epsilon_at_step_v12_default_matches_400k_constant():
    for step in (0, 50_000, 320_000, 400_000):
        assert epsilon_at_step_v12(step) == epsilon_at_step_v12(step, decay_steps=EPSILON_DECAY_STEPS_V12)


def test_epsilon_at_step_v12_rescale_keeps_annealing_past_old_floor_point():
    step = EPSILON_DECAY_STEPS_V12  # 320,000: floored under the old 400K-budget schedule
    assert epsilon_at_step_v12(step) == EPSILON_END
    # Rescaled to a 3,000,000-step ceiling (eps_decay_steps = 0.8 * 3,000,000
    # = 2,400,000), the same absolute step is still early in decay.
    rescaled = epsilon_at_step_v12(step, decay_steps=2_400_000)
    assert EPSILON_END < rescaled < EPSILON_START


def test_lr_at_step_v12_default_matches_400k_constant():
    for step in (0, 200_000, 400_000):
        assert lr_at_step_v12(step) == lr_at_step_v12(step, decay_steps=LEARNING_RATE_DECAY_STEPS_V12)


def test_lr_at_step_v12_rescale_keeps_annealing_past_old_floor_point():
    step = LEARNING_RATE_DECAY_STEPS_V12  # 400,000: floored under the old schedule
    assert lr_at_step_v12(step) == LEARNING_RATE_END
    rescaled = lr_at_step_v12(step, decay_steps=3_000_000)
    assert LEARNING_RATE_END < rescaled < LEARNING_RATE_START


def test_extended_run_stops_early_when_converge_rule_is_trivially_satisfied(tmp_path):
    """Real, short end-to-end proof that the stop-check is actually wired
    into the training loop's break, not just correct in isolation. Uses an
    explicit small checkpoint_steps override (bypassing the normal 10,000
    spacing) to keep this fast, and a converge rule any non-empty window
    trivially satisfies (converge_window=1, mean floor 0.0, std ceiling
    1.0) so the run must stop at the first checkpoint with episodes > 0,
    well before max_steps."""
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V12_BASELINE_V2_SEEDS[6],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=300,
        checkpoint_steps=(0, 100, 200, 300),
        strict=False,
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
        enable_joint_network_v12=True,
        converge_window=1,
        converge_completion_mean=0.0,
        converge_completion_std=1.0,
    )
    assert manifest["stop_reason"] == "converged"
    assert manifest["final_step"] < 300
    assert manifest["final_step"] in (100, 200)


def test_extended_run_reaches_max_steps_with_default_thresholds(tmp_path):
    """Sanity companion to the above: with the real (non-trivial) default
    thresholds, a run this short cannot possibly converge or fail, so it
    must run all the way to max_steps and record stop_reason accordingly."""
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V12_BASELINE_V2_SEEDS[7],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=300,
        checkpoint_steps=(0, 100, 200, 300),
        strict=False,
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
        enable_joint_network_v12=True,
    )
    assert manifest["stop_reason"] == "max_steps_reached"
    assert manifest["final_step"] == 300
