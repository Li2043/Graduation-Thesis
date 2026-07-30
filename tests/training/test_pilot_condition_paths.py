"""Stage 5B-0 — condition path shaping checks."""

from __future__ import annotations

from thesis.training.final_lock_loader import load_final_locks
from thesis.training.pilot_config import (
    PilotConfig,
    PilotDQNConfig,
    PilotDurationConfig,
    PilotExplorationConfig,
)
from thesis.training.pilot_training_loop import PilotTrainer


def _cfg(steps: int = 50) -> PilotConfig:
    return PilotConfig(
        dqn=PilotDQNConfig(
            replay_warmup_per_controller=64,
            batch_size=8,
            target_sync_interval_updates=1000,
            replay_capacity_per_controller=5000,
        ),
        duration=PilotDurationConfig(
            environment_steps_per_run=steps,
            checkpoint_steps=(steps,),
            evaluation_steps=(),
        ),
        exploration=PilotExplorationConfig(epsilon_decay_environment_steps=30),
    )


def test_baseline_zero_mean_min_nonzero_and_decomposition():
    bundle = load_final_locks()
    cfg = _cfg(60)
    base = PilotTrainer(
        bundle, condition="baseline", pilot_seed=51001, config=cfg, write_traces=True
    )
    mean = PilotTrainer(
        bundle, condition="mean_pbrs", pilot_seed=51001, config=cfg, write_traces=True
    )
    mn = PilotTrainer(
        bundle, condition="min_pbrs", pilot_seed=51001, config=cfg, write_traces=True
    )
    base.run()
    mean.run()
    mn.run()
    assert all(r["shaping_component"] == 0.0 for r in base.diag.transition_trace)
    assert mean.diag.non_zero_shaping_count >= 1
    assert mn.diag.non_zero_shaping_count >= 1
    assert base.diag.max_decomp_error <= 1e-12
    assert mean.diag.max_decomp_error <= 1e-12
    # Find a step where mean and min scaled shaping differ
    mean_by = {
        (r["env_step"], r["controller_id"]): r["shaping_component"]
        for r in mean.diag.transition_trace
    }
    min_by = {
        (r["env_step"], r["controller_id"]): r["shaping_component"]
        for r in mn.diag.transition_trace
    }
    diffs = [
        abs(mean_by[k] - min_by[k])
        for k in mean_by
        if k in min_by
    ]
    assert any(d > 1e-12 for d in diffs)
