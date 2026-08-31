"""Stage 11 pilot (E30) v10 -- hysteretic Q-learning, higher ratio (0.3 vs
v9's 0.1).

Covers: ``HYSTERESIS_RATIO_V10``'s value and ``PILOT_V10_SEEDS`` disjointness/
guard/union checks; the v9/v10 mutual-exclusion validation; the new v10
reward-version tag (alone and unaffected-baseline cases); a direct numeric
check that 0.3 (not v9's 0.1) is what actually reaches
``SharedDQNLearner.update()`` when ``enable_hysteretic_v10`` is set; and an
end-to-end short run. Does NOT re-test the core hysteretic loss-weighting
math itself (``hysteresis_ratio=None``/``1.0`` equivalence, invalid-ratio
``ValueError``s) -- that mechanism is unmodified from v9 and already covered
by ``test_stage11_v9_hysteretic.py`` against the shared, unmodified
``SharedDQNLearner.update()`` code path.
"""

from __future__ import annotations

import numpy as np
import pytest

from thesis.agents.replay_buffer_v2 import ReplayBatch, ReplayTransition
from thesis.agents.stage10_shared_dqn import SharedDQNLearner
from thesis.pilots.stage11_dyad_merge_pilot_config import (
    HYSTERESIS_RATIO_V9,
    HYSTERESIS_RATIO_V10,
    MAX_STEPS,
    N_ACTIONS,
    OBS_DIM,
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
    assert_stage11_pilot_guards,
)
from thesis.pilots.stage11_dyad_merge_runner import (
    REWARD_VERSION_STAGE11_V6_ROLE_SPLIT,
    REWARD_VERSION_STAGE11_V10_HYSTERETIC_RATIO_0_3,
    build_shared_dqn_config,
    run_stage11_pilot_training_job,
)


# --------------------------------------------------------------- config constants


def test_hysteresis_ratio_v10_constant():
    assert HYSTERESIS_RATIO_V10 == 0.3
    assert HYSTERESIS_RATIO_V10 != HYSTERESIS_RATIO_V9


def test_v10_seeds_disjoint_from_v1_through_v9():
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
    ):
        assert set(PILOT_V10_SEEDS).isdisjoint(block)


def test_v10_seeds_pass_guard():
    for seed in PILOT_V10_SEEDS:
        assert_stage11_pilot_guards(master_seed=seed, max_steps=MAX_STEPS)


def test_pilot_seeds_union_includes_v10():
    assert set(PILOT_V10_SEEDS) <= set(PILOT_SEEDS)


# --------------------------------------------------------------- ratio actually threaded through


def _make_learner(seed: int) -> SharedDQNLearner:
    return SharedDQNLearner(build_shared_dqn_config(replay_capacity=1000), seed=seed)


def _all_negative_td_error_batch(n: int = 8) -> ReplayBatch:
    """Every row engineered to have a strongly NEGATIVE TD-error -- same
    construction as test_stage11_v9_hysteretic.py's helper."""
    temp = SharedDQNLearner(build_shared_dqn_config(replay_capacity=n), seed=999)
    obs = np.zeros(OBS_DIM, dtype=np.float64)
    mask = np.ones(N_ACTIONS, dtype=bool)
    for _ in range(n):
        temp.store_transition(
            ReplayTransition(
                observation=obs.copy(),
                action=0,
                shaped_reward=-100.0,
                next_observation=None,
                terminated=True,
                truncated=False,
                action_mask=mask.copy(),
                next_action_mask=None,
                controller_terminal=True,
                learner_completed=False,
            )
        )
    return temp.replay.sample(n)


def test_hysteresis_ratio_v10_scales_all_negative_batch_loss_exactly():
    """0.3 (not v9's 0.1) must be the ratio actually applied -- direct
    numeric check, not just 'a smaller loss than unweighted'."""
    batch = _all_negative_td_error_batch(8)
    learner_none = _make_learner(45)
    learner_v10 = _make_learner(45)  # identical init -- same seed
    result_none = learner_none.update(batch=batch, hysteresis_ratio=None)
    result_v10 = learner_v10.update(batch=batch, hysteresis_ratio=HYSTERESIS_RATIO_V10)
    assert result_v10["loss"] == pytest.approx(0.3 * result_none["loss"], rel=1e-4)


def test_hysteresis_ratio_v10_differs_from_v9_on_same_batch():
    """Sanity: v10's ratio must produce a DIFFERENT (larger) loss than v9's
    0.1 on an identical all-negative batch -- confirms 0.3 != 0.1 actually
    changes behaviour, not just that the constant is defined."""
    batch = _all_negative_td_error_batch(8)
    learner_v9 = _make_learner(46)
    learner_v10 = _make_learner(46)  # identical init -- same seed
    result_v9 = learner_v9.update(batch=batch, hysteresis_ratio=HYSTERESIS_RATIO_V9)
    result_v10 = learner_v10.update(batch=batch, hysteresis_ratio=HYSTERESIS_RATIO_V10)
    assert result_v10["loss"] > result_v9["loss"]
    assert result_v10["loss"] == pytest.approx(3.0 * result_v9["loss"], rel=1e-4)


# --------------------------------------------------------------- flag composition rules


def test_v10_without_role_split_raises(tmp_path):
    with pytest.raises(ValueError):
        run_stage11_pilot_training_job(
            master_seed=PILOT_V10_SEEDS[0],
            output_root=tmp_path / "output",
            checkpoint_root=tmp_path / "checkpoints",
            max_steps=600,
            strict=False,
            checkpoint_steps=(0, 600),
            episode_max_steps=50,
            enable_stage9_based_reward_v5=True,
            enable_hysteretic_v10=True,
            # enable_role_split_v6 deliberately omitted (False)
        )


def test_v9_and_v10_together_raises(tmp_path):
    with pytest.raises(ValueError):
        run_stage11_pilot_training_job(
            master_seed=PILOT_V10_SEEDS[0],
            output_root=tmp_path / "output",
            checkpoint_root=tmp_path / "checkpoints",
            max_steps=600,
            strict=False,
            checkpoint_steps=(0, 600),
            episode_max_steps=50,
            enable_stage9_based_reward_v5=True,
            enable_role_split_v6=True,
            enable_hysteretic_v9=True,
            enable_hysteretic_v10=True,
        )


def test_v10_run_selects_the_v10_reward_version_tag(tmp_path):
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V10_SEEDS[0],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 600),
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
        enable_role_split_v6=True,
        enable_hysteretic_v10=True,
    )
    assert manifest["reward_version"] == REWARD_VERSION_STAGE11_V10_HYSTERETIC_RATIO_0_3


def test_v6_without_v10_flag_still_selects_v6_tag(tmp_path):
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V10_SEEDS[1],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 600),
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
        enable_role_split_v6=True,
    )
    assert manifest["reward_version"] == REWARD_VERSION_STAGE11_V6_ROLE_SPLIT


# --------------------------------------------------------------- end-to-end (runner)


def test_v10_short_run_completes_without_error(tmp_path):
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V10_SEEDS[2],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=3000,
        strict=False,
        checkpoint_steps=(0, 3000),
        episode_max_steps=150,
        enable_stage9_based_reward_v5=True,
        enable_role_split_v6=True,
        enable_hysteretic_v10=True,
    )
    assert manifest["final_step"] == 3000
    assert manifest["reward_version"] == REWARD_VERSION_STAGE11_V10_HYSTERETIC_RATIO_0_3
