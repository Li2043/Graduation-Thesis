"""Deterministic uninterrupted vs checkpoint-resume equivalence (Stage 5B-0)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from thesis.training.final_lock_loader import FinalLockBundle
from thesis.training.pilot_checkpoint import load_checkpoint
from thesis.training.pilot_config import PilotConfig, PilotDurationConfig, PilotDQNConfig, PilotExplorationConfig
from thesis.training.pilot_training_loop import PilotTrainer


def run_resume_equivalence(
    bundle: FinalLockBundle,
    *,
    work_dir: Path,
    comparison_length: int = 1000,
    interruption_step: int = 500,
    pilot_seed: int = 51001,
) -> dict[str, Any]:
    """Path A uninterrupted vs Path B interrupted/resumed on baseline."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Use full pilot DQN hyperparams but limited duration for this comparison.
    cfg = PilotConfig(
        duration=PilotDurationConfig(
            environment_steps_per_run=comparison_length,
            maximum_runs=6,
            checkpoint_steps=(interruption_step, comparison_length),
            evaluation_steps=(),  # disable eval during resume comparison
        ),
    )
    cfg.validate()

    # Path A
    a = PilotTrainer(
        bundle,
        condition="baseline",
        pilot_seed=pilot_seed,
        config=cfg,
        checkpoint_dir=work_dir / "path_a",
        write_traces=True,
    )
    actions_a: list[dict[str, int]] = []
    rewards_a: list[dict[str, float]] = []

    def on_a(step: int, info: dict[str, Any]) -> None:
        actions_a.append(dict(info["actions"]))

    a.run(n_steps=comparison_length, on_step=on_a)
    online_a = {
        "A": a.learners["A"].parameter_vector(network="online"),
        "B": a.learners["B"].parameter_vector(network="online"),
    }
    target_a = {
        "A": a.learners["A"].parameter_vector(network="target"),
        "B": a.learners["B"].parameter_vector(network="target"),
    }
    losses_a_after = [
        float(u["loss"])
        for u in a.diag.update_trace
        if int(u["environment_step"]) > interruption_step
    ]

    # Path B: run to interruption, save, fresh restore, continue
    b = PilotTrainer(
        bundle,
        condition="baseline",
        pilot_seed=pilot_seed,
        config=cfg,
        checkpoint_dir=work_dir / "path_b",
        write_traces=True,
    )
    actions_b: list[dict[str, int]] = []

    def on_b(step: int, info: dict[str, Any]) -> None:
        actions_b.append(dict(info["actions"]))

    while b.env_steps < interruption_step:
        info = b.step_once()
        on_b(b.env_steps, info)
        b.maybe_checkpoint(b.env_steps)
    from thesis.training.pilot_checkpoint import atomic_torch_save

    ckpt_path = work_dir / "path_b" / f"ckpt_step_{interruption_step:05d}.pt"
    if not ckpt_path.exists():
        atomic_torch_save(ckpt_path, b.export_checkpoint(step=interruption_step))

    payload = load_checkpoint(ckpt_path)
    b2 = PilotTrainer(
        bundle,
        condition="baseline",
        pilot_seed=pilot_seed,
        config=cfg,
        checkpoint_dir=work_dir / "path_b_resume",
        write_traces=True,
    )
    b2.import_checkpoint(payload)
    while b2.env_steps < comparison_length:
        info = b2.step_once()
        on_b(b2.env_steps, info)

    online_b = {
        "A": b2.learners["A"].parameter_vector(network="online"),
        "B": b2.learners["B"].parameter_vector(network="online"),
    }
    target_b = {
        "A": b2.learners["A"].parameter_vector(network="target"),
        "B": b2.learners["B"].parameter_vector(network="target"),
    }
    losses_b_after = [float(u["loss"]) for u in b2.diag.update_trace]

    action_mismatches = sum(1 for x, y in zip(actions_a, actions_b) if x != y)
    if len(actions_a) != len(actions_b):
        action_mismatches += abs(len(actions_a) - len(actions_b))

    param_errs = {
        f"online_{aid}": float(np.max(np.abs(online_a[aid] - online_b[aid])))
        for aid in ("A", "B")
    }
    param_errs.update(
        {
            f"target_{aid}": float(np.max(np.abs(target_a[aid] - target_b[aid])))
            for aid in ("A", "B")
        }
    )
    max_param_err = max(param_errs.values()) if param_errs else 0.0

    counts_ok = (
        a.learners["A"]._update_count == b2.learners["A"]._update_count
        and a.learners["B"]._update_count == b2.learners["B"]._update_count
        and len(a.learners["A"].replay) == len(b2.learners["A"].replay)
        and len(a.learners["B"].replay) == len(b2.learners["B"].replay)
        and a.diag.target_syncs == b2.diag.target_syncs
        and abs(a.current_epsilon() - b2.current_epsilon()) < 1e-15
    )

    loss_err = 0.0
    if len(losses_a_after) != len(losses_b_after):
        loss_err = float("inf")
    elif losses_a_after:
        loss_err = max(abs(x - y) for x, y in zip(losses_a_after, losses_b_after))

    passed = (
        action_mismatches == 0
        and max_param_err <= 1e-12
        and loss_err <= 1e-12
        and counts_ok
        and a.env_steps == b2.env_steps == comparison_length
    )
    return {
        "passed": passed,
        "action_mismatch_count": action_mismatches,
        "transition_mismatch_count": action_mismatches,  # proxy in this harness
        "max_parameter_abs_diff": max_param_err,
        "max_loss_diff": loss_err,
        "param_errors": param_errs,
        "counts_ok": counts_ok,
        "env_steps_a": a.env_steps,
        "env_steps_b": b2.env_steps,
        "updates_a": {
            "A": a.learners["A"]._update_count,
            "B": a.learners["B"]._update_count,
        },
        "updates_b": {
            "A": b2.learners["A"]._update_count,
            "B": b2.learners["B"]._update_count,
        },
        "epsilon_a": a.current_epsilon(),
        "epsilon_b": b2.current_epsilon(),
        "comparison_length": comparison_length,
        "interruption_step": interruption_step,
    }


__all__ = ["run_resume_equivalence"]
