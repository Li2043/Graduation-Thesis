"""Stage 5B-0 — evaluation isolation."""

from __future__ import annotations

from thesis.training.final_lock_loader import load_final_locks
from thesis.training.pilot_checkpoint import learner_fingerprint
from thesis.training.pilot_config import (
    PilotConfig,
    PilotDQNConfig,
    PilotDurationConfig,
    PilotExplorationConfig,
)
from thesis.training.pilot_evaluation import run_isolated_evaluation
from thesis.training.pilot_training_loop import PilotTrainer


def test_evaluation_does_not_mutate_training():
    bundle = load_final_locks()
    cfg = PilotConfig(
        dqn=PilotDQNConfig(
            replay_warmup_per_controller=8,
            batch_size=8,
            target_sync_interval_updates=10,
            replay_capacity_per_controller=2000,
        ),
        duration=PilotDurationConfig(
            environment_steps_per_run=25,
            checkpoint_steps=(25,),
            evaluation_steps=(),
        ),
        exploration=PilotExplorationConfig(epsilon_decay_environment_steps=20),
    )
    t = PilotTrainer(
        bundle, condition="baseline", pilot_seed=51001, config=cfg, write_traces=False
    )
    t.run(n_steps=25)
    eps_before = t.current_epsilon()
    eps_steps_before = t.epsilon_env_steps
    fp_before = {aid: learner_fingerprint(t.learners[aid]) for aid in ("A", "B")}
    replay_before = {aid: len(t.learners[aid].replay) for aid in ("A", "B")}
    updates_before = {aid: t.learners[aid]._update_count for aid in ("A", "B")}

    result = run_isolated_evaluation(
        bundle, t.learners, eval_seed=t.seeds["evaluation"]
    )
    assert result["n_episodes"] == 16
    assert result["mutation"]["any"] is False
    assert result["optimiser_updates"] is False
    assert result["replay_writes"] is False
    assert result["target_syncs"] is False
    assert t.current_epsilon() == eps_before
    assert t.epsilon_env_steps == eps_steps_before
    assert {aid: len(t.learners[aid].replay) for aid in ("A", "B")} == replay_before
    assert {aid: t.learners[aid]._update_count for aid in ("A", "B")} == updates_before
    assert {aid: learner_fingerprint(t.learners[aid]) for aid in ("A", "B")} == fp_before
