"""Stage 11 pilot (E30) v9 -- hysteretic Q-learning (Matignon et al. 2007;
deep-RL extension Omidshafiei et al. 2017).

Covers: ``SharedDQNLearner.update``'s ``hysteresis_ratio`` loss weighting
(``None``/``1.0`` match the pre-v9 unweighted loss exactly; an all-negative-
TD-error batch scales the loss by exactly the ratio; invalid ratios raise);
``enable_hysteretic_v9`` requiring v6's role-split flag; the new v9
reward-version tag (alone and unaffected-baseline cases); and
``PILOT_V9_SEEDS`` being disjoint from every prior version's seeds and every
other stage's forbidden blocks.
"""

from __future__ import annotations

import numpy as np
import pytest

from thesis.agents.replay_buffer_v2 import ReplayBatch, ReplayTransition
from thesis.agents.stage10_shared_dqn import SharedDQNLearner
from thesis.pilots.stage11_dyad_merge_pilot_config import (
    HYSTERESIS_RATIO_V9,
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
    assert_stage11_pilot_guards,
)
from thesis.pilots.stage11_dyad_merge_runner import (
    REWARD_VERSION_STAGE11_V6_ROLE_SPLIT,
    REWARD_VERSION_STAGE11_V9_HYSTERETIC,
    build_shared_dqn_config,
    run_stage11_pilot_training_job,
)


# --------------------------------------------------------------- config constants


def test_hysteresis_ratio_v9_constant():
    assert HYSTERESIS_RATIO_V9 == 0.1


def test_v9_seeds_disjoint_from_v1_through_v8():
    for block in (
        PILOT_V1_SEEDS,
        PILOT_V2_SEEDS,
        PILOT_V3_SEEDS,
        PILOT_V4_SEEDS,
        PILOT_V5_SEEDS,
        PILOT_V6_SEEDS,
        PILOT_V7_SEEDS,
        PILOT_V8_SEEDS,
    ):
        assert set(PILOT_V9_SEEDS).isdisjoint(block)


def test_v9_seeds_pass_guard():
    for seed in PILOT_V9_SEEDS:
        assert_stage11_pilot_guards(master_seed=seed, max_steps=MAX_STEPS)


def test_pilot_seeds_union_includes_v9():
    assert set(PILOT_V9_SEEDS) <= set(PILOT_SEEDS)


# --------------------------------------------------------------- hysteretic loss weighting


def _make_learner(seed: int) -> SharedDQNLearner:
    return SharedDQNLearner(build_shared_dqn_config(replay_capacity=1000), seed=seed)


def _all_negative_td_error_batch(n: int = 8) -> ReplayBatch:
    """Every row is engineered to have a strongly NEGATIVE TD-error: a
    controller-terminal transition (target = shaped_reward exactly, no
    bootstrap) with a huge negative reward that overwhelms any freshly-
    initialised network's small Q-value output, regardless of random seed."""
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


def test_hysteresis_ratio_none_matches_pre_v9_unweighted_loss():
    """hysteresis_ratio=None must be EXACTLY the old (pre-v9) code path --
    every other caller of update() across Stage 10/11 is unaffected."""
    batch = _all_negative_td_error_batch(8)
    learner_none = _make_learner(42)
    learner_full = _make_learner(42)  # identical init -- same seed
    result_none = learner_none.update(batch=batch, hysteresis_ratio=None)
    result_full = learner_full.update(batch=batch, hysteresis_ratio=1.0)
    assert result_none["loss"] == pytest.approx(result_full["loss"], rel=1e-5)


def test_hysteresis_ratio_scales_all_negative_batch_loss_exactly():
    """A batch with ONLY negative-TD-error rows: hysteresis_ratio=0.1 must
    produce a loss exactly 0.1x the hysteresis_ratio=None loss on that same
    batch, since every row's weight is 0.1 -- a direct numeric check, not
    just 'loss is smaller'."""
    batch = _all_negative_td_error_batch(8)
    learner_none = _make_learner(43)
    learner_scaled = _make_learner(43)  # identical init -- same seed
    result_none = learner_none.update(batch=batch, hysteresis_ratio=None)
    result_scaled = learner_scaled.update(batch=batch, hysteresis_ratio=0.1)
    assert result_scaled["loss"] == pytest.approx(0.1 * result_none["loss"], rel=1e-4)


@pytest.mark.parametrize("bad_ratio", [0.0, -0.1, 1.5])
def test_invalid_hysteresis_ratio_raises(bad_ratio):
    batch = _all_negative_td_error_batch(4)
    learner = _make_learner(44)
    with pytest.raises(ValueError):
        learner.update(batch=batch, hysteresis_ratio=bad_ratio)


# --------------------------------------------------------------- flag composition rules


def test_v9_without_role_split_raises(tmp_path):
    with pytest.raises(ValueError):
        run_stage11_pilot_training_job(
            master_seed=PILOT_V9_SEEDS[0],
            output_root=tmp_path / "output",
            checkpoint_root=tmp_path / "checkpoints",
            max_steps=600,
            strict=False,
            checkpoint_steps=(0, 600),
            episode_max_steps=50,
            enable_stage9_based_reward_v5=True,
            enable_hysteretic_v9=True,
            # enable_role_split_v6 deliberately omitted (False)
        )


def test_v9_run_selects_the_v9_reward_version_tag(tmp_path):
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V9_SEEDS[0],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 600),
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
        enable_role_split_v6=True,
        enable_hysteretic_v9=True,
    )
    assert manifest["reward_version"] == REWARD_VERSION_STAGE11_V9_HYSTERETIC


def test_v6_without_v9_flag_still_selects_v6_tag(tmp_path):
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V9_SEEDS[1],
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


def test_v9_short_run_completes_without_error(tmp_path):
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V9_SEEDS[2],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=3000,
        strict=False,
        checkpoint_steps=(0, 3000),
        episode_max_steps=150,
        enable_stage9_based_reward_v5=True,
        enable_role_split_v6=True,
        enable_hysteretic_v9=True,
    )
    assert manifest["final_step"] == 3000
    assert manifest["reward_version"] == REWARD_VERSION_STAGE11_V9_HYSTERETIC
