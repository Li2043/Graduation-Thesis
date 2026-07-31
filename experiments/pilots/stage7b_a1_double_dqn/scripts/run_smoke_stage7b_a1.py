#!/usr/bin/env python3
"""Local smoke: 2 conditions × 1 temporary seed × 1000 steps (not formal)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve()
PILOT_ROOT = SCRIPT.parents[1]
REPO_ROOT = SCRIPT.parents[4]


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    os.environ["PYTHONPATH"] = str(REPO_ROOT / "src")
    from thesis.pilots.stage7b_a1_runner import run_training_job
    from thesis.training.pilot_checkpoint import load_checkpoint

    protocol = PILOT_ROOT / "configs" / "stage7b_a1_protocol.yaml"
    smoke_seed = 63999  # outside frozen block
    with tempfile.TemporaryDirectory(prefix="stage7b_a1_smoke_") as td:
        td_path = Path(td)
        out_root = td_path / "output"
        ckpt_root = td_path / "checkpoints_external"
        results = {}
        for cond in ("vanilla_dqn", "double_dqn"):
            results[cond] = run_training_job(
                condition=cond,
                master_seed=smoke_seed,
                protocol_path=protocol,
                output_root=out_root,
                checkpoint_root=ckpt_root,
                max_steps=1000,
                resume=False,
                device="cpu",
                strict=False,
                allow_smoke=True,
            )
            ckpt = ckpt_root / cond / f"seed_{smoke_seed}" / "ckpt_step_0_full.pt"
            assert ckpt.is_file(), ckpt
            payload = load_checkpoint(ckpt)
            assert payload["algorithm_mode"] == cond
            assert payload["condition"] == "baseline"
        # paths separated
        assert (out_root / "runs" / "vanilla_dqn" / f"seed_{smoke_seed}").is_dir()
        assert (out_root / "runs" / "double_dqn" / f"seed_{smoke_seed}").is_dir()
        report = {
            "smoke": "PASS",
            "seed": smoke_seed,
            "max_steps": 1000,
            "results": {k: {"success": v.get("success"), "final_step": v.get("final_step")} for k, v in results.items()},
        }
        dest = PILOT_ROOT / "logs" / "smoke_test_report.json"
        dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
