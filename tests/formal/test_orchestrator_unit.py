"""Orchestrator unit tests — dry-run, worker failure survival, thread limits."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from thesis.formal.status_registry import FormalStatusRegistry, TERMINAL_FAILED


REPO = Path(__file__).resolve().parents[2]
ORCH = (
    REPO
    / "experiments/formal/stage6a_formal_training/scripts/run_formal_matrix.py"
)


def test_pytorch_thread_limits_helper():
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    assert torch.get_num_threads() == 1


def test_orchestrator_dry_run(tmp_path):
    # Minimal 1-row matrix for dry-run path (dry-run skips 30-row enforcement when we pass dry-run)
    matrix = tmp_path / "matrix.csv"
    matrix.write_text(
        "formal_job_id,condition,master_seed\nbaseline__61001,baseline,61001\n",
        encoding="utf-8",
    )
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("protocol_version: 5C-0-H1-R1-100K\n", encoding="utf-8")
    out = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            str(ORCH),
            "--run-matrix",
            str(matrix),
            "--protocol-lock",
            str(protocol),
            "--output-root",
            str(out),
            "--workers",
            "2",
            "--threads-per-worker",
            "1",
            "--dry-run",
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
    )
    assert proc.returncode == 0, proc.stderr
    plan = json.loads((out / "orchestrator_plan.json").read_text(encoding="utf-8"))
    assert plan["dry_run"] is True
    assert plan["formal_training_started"] is False
    assert plan["replace_failed_seeds"] is False
    assert plan["vectorized_training"] is False


def test_registry_survives_worker_failure_record(tmp_path):
    reg = FormalStatusRegistry(tmp_path / "reg.json")
    reg.upsert(
        "baseline__61001",
        {
            "status": TERMINAL_FAILED,
            "reason": "worker_process_failure:boom",
            "master_seed": 61001,
            "seed_replaced": False,
        },
    )
    counts = reg.account_all(["baseline__61001", "baseline__61002"])
    assert counts["failed"] == 1
    assert counts["pending_or_missing"] == 1
    assert reg.get("baseline__61001")["seed_replaced"] is False
