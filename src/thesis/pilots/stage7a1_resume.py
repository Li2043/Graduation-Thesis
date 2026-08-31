"""Stage 7A-1 resume equivalence: Path A uninterrupted vs Path B interrupted."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from thesis.formal.formal_trainer import FormalTrainer
from thesis.pilots.stage7a1_checkpoint import atomic_hashed_torch_save, sha256_file
from thesis.pilots.stage7a1_eval import evaluate_checkpoint_rich
from thesis.pilots.stage7a1_config import CHECKPOINT_STEPS
from thesis.pilots.stage7a1_runner import make_trainer
from thesis.training.pilot_checkpoint import load_checkpoint


def _param_max_abs_diff(a: FormalTrainer, b: FormalTrainer) -> float:
    diffs = []
    for aid in ("A", "B"):
        for net in ("online", "target"):
            va = a.learners[aid].parameter_vector(network=net)
            vb = b.learners[aid].parameter_vector(network=net)
            diffs.append(float(np.max(np.abs(va - vb))))
    return max(diffs) if diffs else 0.0


def _replay_mismatch(a: FormalTrainer, b: FormalTrainer) -> int:
    mism = 0
    for aid in ("A", "B"):
        sa = a.learners[aid].replay.export_full_state()
        sb = b.learners[aid].replay.export_full_state()
        if sa.keys() != sb.keys():
            mism += 1
            continue
        for k in sa:
            va, vb = sa[k], sb[k]
            if isinstance(va, np.ndarray):
                if not np.array_equal(va, vb):
                    mism += 1
            elif va != vb:
                mism += 1
    return mism


def _rng_mismatch(a: FormalTrainer, b: FormalTrainer) -> int:
    mism = 0
    for aid in ("A", "B"):
        if a.learners[aid]._rng.bit_generator.state != b.learners[aid]._rng.bit_generator.state:
            mism += 1
        if a.learners[aid].replay.seed != b.learners[aid].replay.seed:
            mism += 1
    if a.schedule.export_state() != b.schedule.export_state():
        mism += 1
    if a.env_steps != b.env_steps:
        mism += 1
    if a.epsilon_env_steps != b.epsilon_env_steps:
        mism += 1
    if a.diag.target_syncs != b.diag.target_syncs:
        mism += 1
    if int(a.learners["A"]._update_count) != int(b.learners["A"]._update_count):
        mism += 1
    if int(a.learners["B"]._update_count) != int(b.learners["B"]._update_count):
        mism += 1
    return mism


def run_resume_equivalence(
    *,
    work_dir: Path,
    protocol_hash: str = "stage7a1-resume",
    master_seed: int = 62001,
    interruption_step: int = 25_000,
    comparison_step: int = 50_000,
) -> dict[str, Any]:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Path A: uninterrupted
    a = make_trainer(
        master_seed=master_seed,
        protocol_hash=protocol_hash,
        checkpoint_dir=work_dir / "path_a",
        max_steps=comparison_step,
    )
    a.run(n_steps=comparison_step)

    # Path B: interrupt at 25K, fresh process restore, continue
    b = make_trainer(
        master_seed=master_seed,
        protocol_hash=protocol_hash,
        checkpoint_dir=work_dir / "path_b",
        max_steps=comparison_step,
    )
    while b.env_steps < interruption_step:
        b.step_once()
    ckpt_path = work_dir / "path_b" / f"ckpt_step_{interruption_step}_full.pt"
    atomic_hashed_torch_save(ckpt_path, b.export_checkpoint(step=interruption_step))

    payload = load_checkpoint(ckpt_path)
    b2 = make_trainer(
        master_seed=master_seed,
        protocol_hash=protocol_hash,
        checkpoint_dir=work_dir / "path_b_resume",
        max_steps=comparison_step,
    )
    b2.import_checkpoint(payload)
    while b2.env_steps < comparison_step:
        b2.step_once()

    network_mismatch = _param_max_abs_diff(a, b2)
    replay_mismatch = _replay_mismatch(a, b2)
    rng_mismatch = _rng_mismatch(a, b2)

    try:
        ckpt_idx = list(CHECKPOINT_STEPS).index(int(comparison_step))
    except ValueError:
        ckpt_idx = 0
    # Evaluation comparison (greedy outcomes)
    eval_a = evaluate_checkpoint_rich(
        a.bundle,
        a.learners,
        master_seed=master_seed,
        evaluation_seed=a.seeds["evaluation_seed"],
        checkpoint_step=comparison_step,
        checkpoint_index=ckpt_idx,
        checkpoint_sha256="path_a",
        collect_trajectories=False,
    )
    eval_b = evaluate_checkpoint_rich(
        b2.bundle,
        b2.learners,
        master_seed=master_seed,
        evaluation_seed=b2.seeds["evaluation_seed"],
        checkpoint_step=comparison_step,
        checkpoint_index=ckpt_idx,
        checkpoint_sha256="path_b",
        collect_trajectories=False,
    )
    eval_mismatch = 0
    for ea, eb in zip(eval_a["episodes"], eval_b["episodes"]):
        for key in (
            "success",
            "collision",
            "truncated",
            "termination_reason",
            "episode_length",
            "passing_order",
        ):
            if ea.get(key) != eb.get(key):
                eval_mismatch += 1

    # Optimiser state equality (keys + tensor equality)
    opt_mismatch = 0
    for aid in ("A", "B"):
        oa = a.learners[aid].optimiser.state_dict()
        ob = b2.learners[aid].optimiser.state_dict()
        if oa.keys() != ob.keys():
            opt_mismatch += 1
            continue
        # compare recursively via torch.save bytes of state
        import io
        import torch

        ba = io.BytesIO()
        bb = io.BytesIO()
        torch.save(oa, ba)
        torch.save(ob, bb)
        if ba.getvalue() != bb.getvalue():
            opt_mismatch += 1

    passed = (
        network_mismatch == 0.0
        and replay_mismatch == 0
        and rng_mismatch == 0
        and eval_mismatch == 0
        and opt_mismatch == 0
        and a.env_steps == comparison_step
        and b2.env_steps == comparison_step
    )
    report = {
        "seed": master_seed,
        "interruption_step": interruption_step,
        "comparison_step": comparison_step,
        "passed": passed,
        "network_parameter_max_abs_diff": network_mismatch,
        "replay_row_mismatch": replay_mismatch,
        "rng_state_mismatch": rng_mismatch,
        "evaluation_outcome_mismatch": eval_mismatch,
        "optimizer_state_mismatch": opt_mismatch,
        "schedule_cursor_a": a.schedule.export_state(),
        "schedule_cursor_b": b2.schedule.export_state(),
        "env_steps_a": a.env_steps,
        "env_steps_b": b2.env_steps,
        "checkpoint_sha256_interrupt": sha256_file(ckpt_path),
        "update_counts_a": {
            "A": int(a.learners["A"]._update_count),
            "B": int(a.learners["B"]._update_count),
        },
        "update_counts_b": {
            "A": int(b2.learners["A"]._update_count),
            "B": int(b2.learners["B"]._update_count),
        },
        "target_syncs_a": dict(a.diag.target_syncs),
        "target_syncs_b": dict(b2.diag.target_syncs),
    }
    out = work_dir / "resume_equivalence_report.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


__all__ = ["run_resume_equivalence"]
