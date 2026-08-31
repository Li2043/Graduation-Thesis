"""Stage 11 pilot (E30) v11 -- per-episode "priority signal" observation
feature (symmetry-breaking device, Patil et al. 2026 "Diamond Attention" /
arXiv:2605.06825, adapted for this single-conflict DQN setting), combined
with v9's hysteretic Q-learning (ratio=0.1) as this round's tested launch.

Covers: env-level correctness of ``Stage10MergeEnvConfig.include_priority_signal``
(default-False regression, observation dim/range, the STRUCTURAL own/peer
consistency the mechanism depends on, per-episode-not-per-step stability,
determinism given a reset seed); ``enable_priority_signal_v11`` requiring
v6's role-split flag; the new v11 reward-version tag (alone and combined
with v9, and the unaffected-baseline case); an end-to-end short run
confirming the network genuinely accepts a 15-dim observation. Does NOT
re-test the core hysteretic loss-weighting math (unmodified from v9,
already covered by test_stage11_v9_hysteretic.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from thesis.envs.stage10_symmetric_merge_env import (
    OBS_DIM_WITH_ROLE_ZONE,
    OBS_DIM_WITH_ROLE_ZONE_AND_PRIORITY,
    Stage10MergeEnvConfig,
    Stage10SymmetricMergeEnv,
)
from thesis.pilots.stage11_dyad_merge_pilot_config import (
    MAX_STEPS,
    OBS_DIM_V11,
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
    assert_stage11_pilot_guards,
)
from thesis.agents.stage10_shared_dqn import SharedDQNLearner
from thesis.pilots.stage11_dyad_merge_runner import (
    REWARD_VERSION_STAGE11_V6_ROLE_SPLIT,
    REWARD_VERSION_STAGE11_V11_PRIORITY_SIGNAL,
    build_shared_dqn_config,
    run_stage11_pilot_training_job,
)


# --------------------------------------------------------------- config constants


def test_obs_dim_v11_constant():
    assert OBS_DIM_V11 == 15
    assert OBS_DIM_V11 == OBS_DIM_WITH_ROLE_ZONE_AND_PRIORITY


def test_v11_seeds_disjoint_from_v1_through_v10():
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
    ):
        assert set(PILOT_V11_SEEDS).isdisjoint(block)


def test_v11_seeds_pass_guard():
    for seed in PILOT_V11_SEEDS:
        assert_stage11_pilot_guards(master_seed=seed, max_steps=MAX_STEPS)


def test_pilot_seeds_union_includes_v11():
    assert set(PILOT_V11_SEEDS) <= set(PILOT_SEEDS)


# --------------------------------------------------------------- env-level: priority signal


def _dyad_env(*, include_priority_signal: bool, seed: int = 1) -> Stage10SymmetricMergeEnv:
    return Stage10SymmetricMergeEnv(
        Stage10MergeEnvConfig(
            seed=seed,
            n_vehicles=2,
            spawn_route_lead=0.0,
            include_role_zone_features=True,
            include_priority_signal=include_priority_signal,
        )
    )


def test_default_omits_priority_signal_byte_identical_to_explicit_false():
    env_default = Stage10SymmetricMergeEnv(
        Stage10MergeEnvConfig(seed=7, n_vehicles=2, spawn_route_lead=0.0, include_role_zone_features=True)
    )
    env_explicit_false = _dyad_env(include_priority_signal=False, seed=7)
    obs_default, _ = env_default.reset(seed=7)
    obs_false, _ = env_explicit_false.reset(seed=7)
    for vid in obs_default:
        assert obs_default[vid].shape == (OBS_DIM_WITH_ROLE_ZONE,)
        np.testing.assert_array_equal(obs_default[vid], obs_false[vid])


def test_priority_signal_adds_two_features_in_expected_range():
    env = _dyad_env(include_priority_signal=True, seed=2)
    obs, _ = env.reset(seed=2)
    for vid in obs:
        assert obs[vid].shape == (OBS_DIM_WITH_ROLE_ZONE_AND_PRIORITY,)
        own_priority, peer_priority = obs[vid][-2], obs[vid][-1]
        assert 0.0 <= own_priority < 1.0
        assert 0.0 <= peer_priority < 1.0


def test_priority_signal_is_structurally_consistent_between_vehicles():
    """The key property the mechanism depends on: both vehicles' observations
    encode a CONSISTENT, shared comparison, not independent private noise --
    vehicle A's own-priority must exactly equal vehicle B's peer-priority,
    and vice versa."""
    env = _dyad_env(include_priority_signal=True, seed=3)
    obs, info = env.reset(seed=3)
    vids = list(obs.keys())
    assert len(vids) == 2
    a, b = vids
    own_a, peer_a = obs[a][-2], obs[a][-1]
    own_b, peer_b = obs[b][-2], obs[b][-1]
    assert own_a == pytest.approx(peer_b)
    assert own_b == pytest.approx(peer_a)
    assert own_a != pytest.approx(own_b)  # two independent draws, not the same value


def test_priority_signal_fixed_within_episode_not_redrawn_per_step():
    env = _dyad_env(include_priority_signal=True, seed=4)
    obs0, _ = env.reset(seed=4)
    first_own = {vid: obs0[vid][-2] for vid in obs0}
    actions = {vid: 1 for vid in obs0}
    for _ in range(5):
        obs, _reward, terminated, truncated, _info = env.step(actions)
        if terminated or truncated:
            break
        for vid in obs:
            assert obs[vid][-2] == pytest.approx(first_own[vid])


def test_priority_signal_differs_across_seeds():
    draws = []
    for seed in range(10, 15):
        env = _dyad_env(include_priority_signal=True, seed=seed)
        obs, _ = env.reset(seed=seed)
        vid = next(iter(obs))
        draws.append(float(obs[vid][-2]))
    assert len(set(draws)) > 1  # not all identical across 5 different seeds


def test_priority_signal_deterministic_given_reset_seed():
    env1 = _dyad_env(include_priority_signal=True, seed=42)
    env2 = _dyad_env(include_priority_signal=True, seed=42)
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    for vid in obs1:
        np.testing.assert_array_equal(obs1[vid], obs2[vid])


# --------------------------------------------------------------- flag composition rules


def test_v11_without_role_split_raises(tmp_path):
    with pytest.raises(ValueError):
        run_stage11_pilot_training_job(
            master_seed=PILOT_V11_SEEDS[0],
            output_root=tmp_path / "output",
            checkpoint_root=tmp_path / "checkpoints",
            max_steps=600,
            strict=False,
            checkpoint_steps=(0, 600),
            episode_max_steps=50,
            enable_stage9_based_reward_v5=True,
            enable_priority_signal_v11=True,
            # enable_role_split_v6 deliberately omitted (False)
        )


def test_v11_run_selects_the_v11_reward_version_tag(tmp_path):
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V11_SEEDS[0],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 600),
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
        enable_role_split_v6=True,
        enable_hysteretic_v9=True,
        enable_priority_signal_v11=True,
    )
    assert manifest["reward_version"] == REWARD_VERSION_STAGE11_V11_PRIORITY_SIGNAL


def test_v6_without_v11_flag_still_selects_v6_tag(tmp_path):
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V11_SEEDS[1],
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


def test_v11_short_run_completes_without_error_and_network_accepts_15dim_obs(tmp_path):
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V11_SEEDS[2],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=3000,
        strict=False,
        checkpoint_steps=(0, 3000),
        episode_max_steps=150,
        enable_stage9_based_reward_v5=True,
        enable_role_split_v6=True,
        enable_hysteretic_v9=True,
        enable_priority_signal_v11=True,
    )
    assert manifest["final_step"] == 3000
    assert manifest["reward_version"] == REWARD_VERSION_STAGE11_V11_PRIORITY_SIGNAL

    # Directly confirm the NETWORK ITSELF (not just the env) genuinely
    # accepts a 15-dim observation, not just that the flag/tag plumbing ran
    # without crashing.
    env = _dyad_env(include_priority_signal=True, seed=99)
    obs, _ = env.reset(seed=99)
    vid = next(iter(obs))
    assert obs[vid].shape == (OBS_DIM_V11,)
    learner = SharedDQNLearner(build_shared_dqn_config(obs_dim=OBS_DIM_V11), seed=1)
    q = learner.q_values(obs[vid])
    assert q.shape == (3,)
    assert np.all(np.isfinite(q))
