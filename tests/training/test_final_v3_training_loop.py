"""Stage 5B-0 — reduced training-loop behavioural tests."""

from __future__ import annotations

import ast
from pathlib import Path

from thesis.training.final_lock_loader import load_final_locks
from thesis.training.pilot_config import (
    PilotConfig,
    PilotDQNConfig,
    PilotDurationConfig,
    PilotExplorationConfig,
)
from thesis.training.pilot_training_loop import PilotTrainer


def _small_cfg(**kw) -> PilotConfig:
    dqn = PilotDQNConfig(
        replay_warmup_per_controller=8,
        batch_size=8,
        target_sync_interval_updates=10,
        replay_capacity_per_controller=2000,
        hidden_sizes=(64, 64),
    )
    dur = PilotDurationConfig(
        environment_steps_per_run=40,
        checkpoint_steps=(20, 40),
        evaluation_steps=(0, 40),
    )
    exp = PilotExplorationConfig(epsilon_decay_environment_steps=20)
    return PilotConfig(dqn=dqn, duration=dur, exploration=exp, **kw)


def test_v3_only_no_v2_imports_in_pilot_modules():
    for rel in (
        "src/thesis/training/pilot_training_loop.py",
        "src/thesis/training/pilot_evaluation.py",
        "src/thesis/training/pilot_ic_schedule.py",
    ):
        tree = ast.parse(Path(rel).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
                for a in node.names:
                    imported.add(a.name)
        assert "MergeEnvV2" not in imported
        assert "scripted_scenarios" not in imported


def test_warmup_then_updates_and_target_sync(tmp_path):
    bundle = load_final_locks()
    cfg = _small_cfg()
    t = PilotTrainer(
        bundle,
        condition="baseline",
        pilot_seed=51001,
        config=cfg,
        checkpoint_dir=tmp_path / "ckpt",
        write_traces=True,
    )
    # Before warmup: no updates
    while t.env_steps < 5:
        t.step_once()
    assert t.learners["A"]._update_count == 0
    diag = t.run(n_steps=40)
    assert t.env_steps == 40
    assert t.learners["A"]._update_count >= 1
    assert t.learners["B"]._update_count >= 1
    assert t.diag.target_syncs["A"] >= 1
    assert t.diag.target_syncs["B"] >= 1
    assert all(u["environment_step"] >= 8 for u in t.diag.update_trace)
    assert t._obs is None or t._obs["A"].shape == (27,)


def test_completed_learner_stops_replay_rows():
    bundle = load_final_locks()
    cfg = _small_cfg()
    # Longer run to allow an exit
    cfg = PilotConfig(
        dqn=PilotDQNConfig(
            replay_warmup_per_controller=64,
            batch_size=8,
            target_sync_interval_updates=1000,
            replay_capacity_per_controller=5000,
        ),
        duration=PilotDurationConfig(
            environment_steps_per_run=120,
            checkpoint_steps=(120,),
            evaluation_steps=(),
        ),
        exploration=PilotExplorationConfig(
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay_environment_steps=1,
            epsilon_after_decay=0.0,
        ),
    )
    t = PilotTrainer(
        bundle, condition="baseline", pilot_seed=51001, config=cfg, write_traces=True
    )
    t.run(n_steps=120)
    # If any controller terminal exit occurred, later rows for that controller stop
    by_c = {"A": [], "B": []}
    for row in t.diag.transition_trace:
        by_c[row["controller_id"]].append(row)
    for aid, rows in by_c.items():
        exits = [r for r in rows if r["controller_terminal"] and not r["terminated"]]
        if not exits:
            continue
        exit_step = exits[0]["env_step"]
        assert all(r["env_step"] <= exit_step for r in rows)


def test_obs_dim_27_throughout():
    bundle = load_final_locks()
    t = PilotTrainer(
        bundle,
        condition="baseline",
        pilot_seed=51002,
        config=_small_cfg(),
        write_traces=True,
    )
    t.run(n_steps=20)
    assert all(r["obs_dim"] == 27 for r in t.diag.transition_trace)
