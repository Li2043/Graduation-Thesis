"""Evaluation isolation tests."""

from __future__ import annotations

from thesis.pilots.stage7a1_eval import evaluate_checkpoint_rich
from thesis.pilots.stage7a1_runner import make_trainer


def test_evaluation_does_not_mutate_training_state():
    t = make_trainer(master_seed=62001, protocol_hash="t", checkpoint_dir=None)
    t.run(n_steps=20)
    before_steps = t.env_steps
    before_eps = t.epsilon_env_steps
    before_upd = (
        int(t.learners["A"]._update_count),
        int(t.learners["B"]._update_count),
    )
    before_replay = (len(t.learners["A"].replay), len(t.learners["B"].replay))
    result = evaluate_checkpoint_rich(
        t.bundle,
        t.learners,
        master_seed=62001,
        evaluation_seed=t.seeds["evaluation_seed"],
        checkpoint_step=0,
        checkpoint_index=0,
        checkpoint_sha256="x",
        collect_trajectories=False,
    )
    assert result["n_episodes"] == 16
    assert t.env_steps == before_steps
    assert t.epsilon_env_steps == before_eps
    assert (
        int(t.learners["A"]._update_count),
        int(t.learners["B"]._update_count),
    ) == before_upd
    assert (len(t.learners["A"].replay), len(t.learners["B"].replay)) == before_replay
    assert all(v == 0 for v in result["evaluation_guard"].values())
