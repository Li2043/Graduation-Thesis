"""Training budget tests."""

from __future__ import annotations

from thesis.pilots.stage7a1_config import (
    CHECKPOINT_STEPS,
    EPSILON_DECAY_STEPS,
    MAX_STEPS,
)
from thesis.pilots.stage7a1_runner import build_pilot_formal_config


def test_maximum_steps_300k():
    cfg = build_pilot_formal_config(max_steps=MAX_STEPS)
    assert cfg.duration.environment_steps_per_run == 300_000
    assert cfg.duration.early_stopping is False
    assert cfg.exploration.epsilon_decay_environment_steps == EPSILON_DECAY_STEPS == 50_000
    assert cfg.exploration.epsilon_after_decay == 0.10


def test_evaluation_checkpoints_complete():
    assert CHECKPOINT_STEPS == (
        0,
        10_000,
        25_000,
        50_000,
        75_000,
        100_000,
        150_000,
        200_000,
        250_000,
        300_000,
    )
    assert len(CHECKPOINT_STEPS) == 10


def test_episode_counts():
    n_seeds = 20
    n_ckpt = 10
    n_ep = 16
    assert n_seeds * n_ep == 320
    assert n_ckpt * 320 == 3200
