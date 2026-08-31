"""Stage 10 pilot v5 (E28) -- pre-training audit for the shared-parameter
architecture (protocol S0.1 pilot v5 pivot): ONE shared Q-network for every
vehicle instead of six independent (role,zone)-keyed networks, with role and
zone now explicit input features since routing no longer implicitly encodes
them.

Covers: OBS_DIM_WITH_ROLE_ZONE / include_role_zone_features flag correctness
(and that v1-v4's default behaviour is completely unaffected), encode_role /
encode_zone_onehot correctness for all combinations, that the shared network
is genuinely singular (one gradient step changes the SAME parameters
regardless of which vehicle/role/zone/curriculum-stage produced the
transition -- no per-role/per-zone weight duplication anywhere), the v5 seed
guard, and the runner's end-to-end wiring on tiny windows (curriculum/LR/
epsilon reused unchanged from v4, only the learner/env wiring differs) --
without running the full ~180K-step pilot.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from thesis.agents.independent_dqn_v2 import DQNConfig
from thesis.agents.replay_buffer_v2 import ReplayTransition
from thesis.agents.stage10_shared_dqn import SharedDQNLearner
from thesis.envs.stage10_symmetric_merge_env import (
    OBS_DIM,
    OBS_DIM_WITH_ROLE_ZONE,
    HighLevelAction,
    Stage10MergeEnvConfig,
    Stage10SymmetricMergeEnv,
    encode_role,
    encode_zone_onehot,
)
from thesis.pilots.stage10_role_phase_subpolicy_config import (
    ARCHITECTURE_SHARED_PARAMETER,
    MAX_STEPS_V4,
    OBS_DIM_V5,
    PILOT_V4_SEEDS,
    PILOT_V5_SEEDS,
    assert_stage10_pilot_guards,
)
from thesis.pilots.stage10_shared_dqn_runner import (
    build_shared_dqn_config,
    run_pilot_training_job_v5,
)


# --------------------------------------------------------------------- OBS_DIM


def test_obs_dim_with_role_zone_is_base_plus_four():
    assert OBS_DIM_WITH_ROLE_ZONE == OBS_DIM + 4 == 13
    assert OBS_DIM_V5 == OBS_DIM_WITH_ROLE_ZONE


def test_default_env_config_unaffected_v1_v4_backward_compat():
    """v1-v4 build Stage10MergeEnvConfig() with no include_role_zone_features
    kwarg at all -- must still get the original 9-dim observation, untouched."""
    env = Stage10SymmetricMergeEnv(Stage10MergeEnvConfig(seed=1, n_vehicles=4))
    obs, _ = env.reset(seed=1)
    for vid in env.active_vehicle_ids:
        assert obs[vid].shape == (OBS_DIM,)


def test_include_role_zone_features_bumps_obs_dim():
    env = Stage10SymmetricMergeEnv(
        Stage10MergeEnvConfig(seed=1, n_vehicles=4, include_role_zone_features=True)
    )
    obs, _ = env.reset(seed=1)
    for vid in env.active_vehicle_ids:
        assert obs[vid].shape == (OBS_DIM_WITH_ROLE_ZONE,)


@pytest.mark.parametrize("n_vehicles", [2, 4, 6])
def test_role_zone_features_populated_for_all_vehicle_counts(n_vehicles):
    env = Stage10SymmetricMergeEnv(
        Stage10MergeEnvConfig(seed=7, n_vehicles=n_vehicles, include_role_zone_features=True)
    )
    obs, info = env.reset(seed=7)
    roles = info["roles"]
    for vid in env.active_vehicle_ids:
        role_val = obs[vid][9]
        zone_onehot = obs[vid][10:13]
        expected_role = encode_role(roles[vid])
        assert role_val == pytest.approx(expected_role)
        assert zone_onehot.sum() == pytest.approx(1.0)  # exactly one-hot
        expected_zone = np.asarray(encode_zone_onehot(env.zone_of(vid)))
        assert np.allclose(zone_onehot, expected_zone)


# ---------------------------------------------------------------- encode_* fns


def test_encode_role_convention_matches_project_style():
    assert encode_role("mainline") == 1.0
    assert encode_role("ramp") == -1.0
    with pytest.raises(ValueError):
        encode_role("bogus")


def test_encode_zone_onehot_exactly_one_hot_each_zone():
    assert encode_zone_onehot("pre") == (1.0, 0.0, 0.0)
    assert encode_zone_onehot("merging") == (0.0, 1.0, 0.0)
    assert encode_zone_onehot("post") == (0.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        encode_zone_onehot("bogus")


# ---------------------------------------------------------- shared singularity


def _make_transition(obs_dim: int, *, terminal: bool = False) -> ReplayTransition:
    rng = np.random.default_rng(0)
    obs = rng.normal(size=obs_dim)
    mask = np.array([True, True, True])
    if terminal:
        return ReplayTransition(
            observation=obs,
            action=0,
            shaped_reward=1.0,
            next_observation=None,
            terminated=True,
            truncated=False,
            action_mask=mask,
            next_action_mask=None,
            controller_terminal=True,
            learner_completed=True,
        )
    next_obs = rng.normal(size=obs_dim)
    return ReplayTransition(
        observation=obs,
        action=1,
        shaped_reward=0.1,
        next_observation=next_obs,
        terminated=False,
        truncated=False,
        action_mask=mask,
        next_action_mask=mask,
        controller_terminal=False,
        learner_completed=False,
    )


def test_shared_learner_is_genuinely_singular_not_six_hidden_copies():
    """A gradient step from ANY vehicle's experience changes the SAME
    parameters that govern every other vehicle's action selection --
    there is exactly one online network object, not six with tied initial
    weights that could then silently diverge."""
    cfg = build_shared_dqn_config()
    cfg.replay_capacity = 200
    cfg.batch_size = 8
    learner = SharedDQNLearner(cfg, seed=1)

    # Populate replay with a mix of "roles/zones" -- but the learner has no
    # concept of role/zone at all; it just sees observation vectors.
    for _ in range(50):
        learner.store_transition(_make_transition(cfg.obs_dim))
    for _ in range(10):
        learner.store_transition(_make_transition(cfg.obs_dim, terminal=True))

    before = torch.nn.utils.parameters_to_vector(learner.online.parameters()).clone()
    id_before = id(learner.online)
    stats = learner.update()
    after = torch.nn.utils.parameters_to_vector(learner.online.parameters())

    assert id(learner.online) == id_before  # same object, not replaced
    assert not torch.allclose(before, after)  # actually changed
    assert stats["update_count"] == 1
    # Only one online/target pair exists at all -- SharedDQNLearner has no
    # per-role/per-zone dict of learners the way SubPolicyManager does.
    assert isinstance(learner.online, torch.nn.Module)
    assert not hasattr(learner, "learners")  # no hidden multi-network dict


def test_shared_learner_replay_is_one_buffer_across_arbitrary_vehicle_labels():
    """Nothing about SharedDQNLearner.store_transition partitions by role,
    zone, or vehicle identity -- confirm the same replay buffer object
    receives everything regardless of caller-side labels (which the runner
    never even passes to the learner -- see stage10_shared_dqn_runner.py)."""
    cfg = build_shared_dqn_config()
    cfg.replay_capacity = 100
    learner = SharedDQNLearner(cfg, seed=2)
    replay_id_before = id(learner.replay)
    for _ in range(20):
        learner.store_transition(_make_transition(cfg.obs_dim))
    assert id(learner.replay) == replay_id_before
    assert len(learner.replay) == 20


# ------------------------------------------------------------------ seed guard


def test_v5_seeds_pass_guard_with_v4_budget():
    for seed in PILOT_V5_SEEDS:
        assert_stage10_pilot_guards(master_seed=seed, max_steps=MAX_STEPS_V4)


def test_v5_seeds_reject_wrong_budget():
    with pytest.raises(RuntimeError):
        assert_stage10_pilot_guards(master_seed=PILOT_V5_SEEDS[0], max_steps=100_000)


def test_v5_seeds_disjoint_from_all_prior_arms():
    assert set(PILOT_V5_SEEDS).isdisjoint(PILOT_V4_SEEDS)


def test_seed_just_outside_v5_block_rejected():
    """68029 was one past PILOT_V5_SEEDS' last seed (68028) when this test
    was written -- but 68029 is now a legitimate pilot v6 seed (reward-
    function revision, 2026-08-06), so it no longer demonstrates "outside
    every reserved block". Then updated to 68037 (one past v6's last seed)
    for the v5->v6 transition -- but 68037 is now itself a legitimate pilot
    v7 seed (reward-magnitude curriculum, 2026-08-06). Updated again to
    68045 (one past v7's last seed, 68044), the same fix this test's sibling
    in test_stage10_pilot_v4.py needed at the v5->v6 and v6->v7 transitions.
    68020 (v4's own last seed) is NOT outside anything and must keep
    passing -- checked separately so this test can't be confused with a
    v4/v5 boundary mixup."""
    with pytest.raises(RuntimeError):
        assert_stage10_pilot_guards(master_seed=68045, max_steps=MAX_STEPS_V4)
    assert_stage10_pilot_guards(master_seed=68020, max_steps=MAX_STEPS_V4)  # still valid (v4 seed)


# --------------------------------------------------------------- end-to-end


def test_runner_end_to_end_tiny_window_smoke():
    """Full run_pilot_training_job_v5 on a tiny window (small stage budgets,
    small max_steps, forced small episode_max_steps for deterministic episode
    boundaries) -- confirms the whole wiring (env with role/zone features,
    shared learner, curriculum stage advancement, checkpointing, manifest,
    trajectory logging with architecture tag) works end-to-end without
    running anything close to the real 180K-step budget."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        manifest = run_pilot_training_job_v5(
            master_seed=PILOT_V5_SEEDS[0],
            output_root=tmp_path / "output",
            checkpoint_root=tmp_path / "checkpoints",
            max_steps=300,
            strict=False,
            stage_vehicle_counts=(2, 4),
            stage_max_steps=(100, 200),
            advance_threshold=0.9,
            rolling_window_episodes=2,
            checkpoint_steps=(0, 150, 300),
            lr_decay_steps=300,
            episode_max_steps=15,
        )
        assert manifest["final_step"] == 300
        assert manifest["architecture"] == ARCHITECTURE_SHARED_PARAMETER
        assert manifest["checkpoints"][0]["step"] == 0
        assert manifest["checkpoints"][-1]["step"] == 300
        for ckpt in manifest["checkpoints"]:
            assert "learner" in ckpt
            assert "per_subpolicy" not in ckpt  # v4-only key must NOT leak into v5's manifest

        ckpt_dir = tmp_path / "checkpoints" / f"seed_{PILOT_V5_SEEDS[0]}"
        saved_steps = sorted(int(p.stem.split("_")[-1]) for p in ckpt_dir.glob("ckpt_step_*.pt"))
        assert saved_steps == [0, 150, 300]
        payload = torch.load(ckpt_dir / "ckpt_step_300.pt", weights_only=False)
        assert payload["architecture"] == ARCHITECTURE_SHARED_PARAMETER
        assert "learner" in payload and "learners" not in payload

        traj_path = tmp_path / "output" / "trajectories" / f"seed_{PILOT_V5_SEEDS[0]}.jsonl"
        assert traj_path.exists()
        lines = traj_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 300
