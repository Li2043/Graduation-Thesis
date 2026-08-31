"""Stage 11 pilot (E30) v7 -- periodic reset (Nikishin et al. 2022 primacy-bias fix).

Covers: ``reset_last_layers`` actually changes the last-two-layers' weights
while leaving the first layer and the replay buffer untouched, rebuilds the
optimiser at the current (not initial) learning rate, and hard-syncs the
target network; ``enable_periodic_reset_v7`` requiring v6's role-split flag;
the new v7 reward-version tag; resets actually firing at the configured
schedule during a real run; and PILOT_V7_SEEDS being disjoint from every
prior version's seeds and every other stage's forbidden blocks.
"""

from __future__ import annotations

import copy

import pytest
import torch

from thesis.pilots.stage11_dyad_merge_pilot_config import (
    MAX_STEPS,
    PILOT_SEEDS,
    PILOT_V1_SEEDS,
    PILOT_V2_SEEDS,
    PILOT_V3_SEEDS,
    PILOT_V4_SEEDS,
    PILOT_V5_SEEDS,
    PILOT_V6_SEEDS,
    PILOT_V7_SEEDS,
    RESET_INTERVAL_V7,
    assert_stage11_pilot_guards,
)
from thesis.pilots.stage11_dyad_merge_runner import (
    REWARD_VERSION_STAGE11_V6_ROLE_SPLIT,
    REWARD_VERSION_STAGE11_V7_PERIODIC_RESET,
    build_shared_dqn_config,
    reset_last_layers,
    run_stage11_pilot_training_job,
)
from thesis.agents.stage10_shared_dqn import SharedDQNLearner


# --------------------------------------------------------------- config constants


def test_reset_interval_v7_constant():
    assert RESET_INTERVAL_V7 == 20_000


def test_v7_seeds_disjoint_from_v1_through_v6():
    for block in (PILOT_V1_SEEDS, PILOT_V2_SEEDS, PILOT_V3_SEEDS, PILOT_V4_SEEDS, PILOT_V5_SEEDS, PILOT_V6_SEEDS):
        assert set(PILOT_V7_SEEDS).isdisjoint(block)


def test_v7_seeds_pass_guard():
    for seed in PILOT_V7_SEEDS:
        assert_stage11_pilot_guards(master_seed=seed, max_steps=MAX_STEPS)


def test_pilot_seeds_union_includes_v7():
    assert set(PILOT_V7_SEEDS) <= set(PILOT_SEEDS)


# --------------------------------------------------------------- reset_last_layers


def _make_learner(seed: int) -> SharedDQNLearner:
    return SharedDQNLearner(build_shared_dqn_config(replay_capacity=1000), seed=seed)


def test_reset_changes_last_two_layers_but_not_first():
    learner = _make_learner(1)
    before_layer0 = copy.deepcopy(learner.online.net[0].weight.data)
    before_layer2 = copy.deepcopy(learner.online.net[2].weight.data)
    before_layer4 = copy.deepcopy(learner.online.net[4].weight.data)

    reset_last_layers(learner)

    assert torch.equal(learner.online.net[0].weight.data, before_layer0)
    assert not torch.equal(learner.online.net[2].weight.data, before_layer2)
    assert not torch.equal(learner.online.net[4].weight.data, before_layer4)


def test_reset_rebuilds_optimiser_at_current_lr():
    learner = _make_learner(2)
    learner.set_learning_rate(0.00013)
    old_optimiser = learner.optimiser

    reset_last_layers(learner)

    assert learner.optimiser is not old_optimiser
    assert learner.optimiser.param_groups[0]["lr"] == pytest.approx(0.00013)


def test_reset_hard_syncs_target_to_match_reset_online():
    learner = _make_learner(3)
    reset_last_layers(learner)
    online_state = learner.online.state_dict()
    target_state = learner.target.state_dict()
    for key in online_state:
        assert torch.equal(online_state[key], target_state[key])


def test_reset_never_touches_replay_buffer():
    learner = _make_learner(4)
    replay_before = learner.replay
    reset_last_layers(learner)
    assert learner.replay is replay_before  # exact same object, untouched


# --------------------------------------------------------------- runner integration


def test_v7_without_role_split_raises(tmp_path):
    with pytest.raises(ValueError):
        run_stage11_pilot_training_job(
            master_seed=PILOT_V7_SEEDS[0],
            output_root=tmp_path / "output",
            checkpoint_root=tmp_path / "checkpoints",
            max_steps=600,
            strict=False,
            checkpoint_steps=(0, 600),
            episode_max_steps=50,
            enable_stage9_based_reward_v5=True,
            enable_periodic_reset_v7=True,
            # enable_role_split_v6 deliberately omitted (False)
        )


def test_v7_run_selects_the_v7_reward_version_tag(tmp_path):
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V7_SEEDS[0],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=600,
        strict=False,
        checkpoint_steps=(0, 600),
        episode_max_steps=50,
        enable_stage9_based_reward_v5=True,
        enable_role_split_v6=True,
        enable_periodic_reset_v7=True,
    )
    assert manifest["reward_version"] == REWARD_VERSION_STAGE11_V7_PERIODIC_RESET


def test_v6_without_reset_flag_still_selects_v6_tag(tmp_path):
    manifest = run_stage11_pilot_training_job(
        master_seed=PILOT_V7_SEEDS[1],
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


def test_v7_run_calls_reset_exactly_at_the_configured_steps(tmp_path, monkeypatch):
    """End-to-end, verified directly via mock rather than inferred from
    weight diffs (which would be confounded by ordinary gradient updates
    happening every step regardless of reset): patch reset_last_layers to
    record which env step it was called at, run past two reset boundaries
    (20,000 and 40,000), and confirm it fired exactly at those steps -- once
    per role learner each time -- and not at any other step."""
    import thesis.pilots.stage11_dyad_merge_runner as runner_module

    call_log: list[int] = []
    current_step_box = {"step": None}

    def fake_reset(lnr):
        call_log.append(current_step_box["step"])

    # Wrap the real loop's step counter by patching reset_last_layers AND
    # tracking step via a thin wrapper around epsilon_at_step (called once
    # per loop iteration with the current step) to know "step" at call time.
    real_epsilon_at_step = runner_module.epsilon_at_step

    def tracking_epsilon_at_step(step):
        current_step_box["step"] = step
        return real_epsilon_at_step(step)

    monkeypatch.setattr(runner_module, "reset_last_layers", fake_reset)
    monkeypatch.setattr(runner_module, "epsilon_at_step", tracking_epsilon_at_step)

    run_stage11_pilot_training_job(
        master_seed=PILOT_V7_SEEDS[2],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=2 * RESET_INTERVAL_V7 + 500,
        strict=False,
        checkpoint_steps=(0, 2 * RESET_INTERVAL_V7 + 500),
        episode_max_steps=150,
        enable_stage9_based_reward_v5=True,
        enable_role_split_v6=True,
        enable_periodic_reset_v7=True,
    )
    # Fired at real step 20,000 and 40,000 (post-increment, matching
    # RESET_INTERVAL_V7's own semantics), twice each (once per role
    # learner). The tracked value is 1 less than that in each case --
    # epsilon_at_step(step) is called with the PRE-increment step at the
    # top of the loop, while `step += 1` happens before the reset check
    # later in the SAME iteration, so the box captured here is always
    # exactly (real reset step - 1). This is a property of the proxy this
    # test uses to observe step, not of the reset schedule itself (verified
    # separately: RESET_INTERVAL_V7 is applied via `step % RESET_INTERVAL_V7
    # == 0` against the post-increment step).
    assert call_log.count(RESET_INTERVAL_V7 - 1) == 2
    assert call_log.count(2 * RESET_INTERVAL_V7 - 1) == 2
    assert all(s in (RESET_INTERVAL_V7 - 1, 2 * RESET_INTERVAL_V7 - 1) for s in call_log)


def test_v6_run_never_calls_reset(tmp_path, monkeypatch):
    import thesis.pilots.stage11_dyad_merge_runner as runner_module

    call_log: list[int] = []
    monkeypatch.setattr(runner_module, "reset_last_layers", lambda lnr: call_log.append(1))

    run_stage11_pilot_training_job(
        master_seed=PILOT_V7_SEEDS[3],
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        max_steps=RESET_INTERVAL_V7 + 500,
        strict=False,
        checkpoint_steps=(0, RESET_INTERVAL_V7 + 500),
        episode_max_steps=150,
        enable_stage9_based_reward_v5=True,
        enable_role_split_v6=True,
        # enable_periodic_reset_v7 deliberately omitted (False)
    )
    assert call_log == []
