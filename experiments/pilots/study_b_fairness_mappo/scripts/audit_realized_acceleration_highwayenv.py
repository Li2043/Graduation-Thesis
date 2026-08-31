#!/usr/bin/env python3
"""M4-M realized-acceleration audit (2026-08-16 CONTROL_AUTHORITY
amendment). Samples random-policy rollouts under the ``meta_speed``
representation, records the REALIZED (post-clip) physical acceleration
broken down by which discrete command (ACCELERATE/HOLD/BRAKE) produced
it, and reports the required percentile/probability statistics plus the
hard bound check (min>=-3.0, max<=2.0)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(REPO_SRC))

from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig  # noqa: E402
from thesis.study_b.envs.highwayenv_wrapper import (  # noqa: E402
    StudyBHeterogeneousHighwayEnv,
    StudyBHighwayWrapperConfig,
)

LABELS = {0: "HOLD", 1: "ACCELERATE", 2: "BRAKE"}


def run(*, n_steps: int, master_seed: int) -> dict:
    cfg = ThesisHighwayMergeEnvConfig(action_representation="meta_speed")
    env = StudyBHeterogeneousHighwayEnv(StudyBHighwayWrapperConfig(env_config=cfg))
    rng = np.random.default_rng(master_seed)
    by_label: dict[str, list[float]] = {"HOLD": [], "ACCELERATE": [], "BRAKE": []}

    obs, _info = env.reset(seed=int(rng.integers(0, 1_000_000)))
    for _ in range(n_steps):
        actions = {vid: int(rng.integers(0, 3)) for vid in env.active_vehicle_ids}
        _obs, _rew, term, trunc, _info = env.step(actions)
        for vid, a in actions.items():
            vehicle = env._env._vehicle_by_id[vid]  # noqa: SLF001
            if vehicle.frozen:
                continue  # frozen vehicles' residual accel isn't policy-attributable to a fresh command
            by_label[LABELS[a]].append(float(vehicle.action["acceleration"]))
        if term or trunc:
            obs, _info = env.reset(seed=int(rng.integers(0, 1_000_000)))

    report = {}
    all_values: list[float] = []
    for label, values in by_label.items():
        all_values.extend(values)
        if not values:
            report[label] = {"n": 0}
            continue
        arr = np.array(values)
        report[label] = {
            "n": int(arr.size),
            "min": float(arr.min()), "max": float(arr.max()),
            "mean": float(arr.mean()), "median": float(np.median(arr)),
            "p1": float(np.percentile(arr, 1)), "p5": float(np.percentile(arr, 5)),
            "p25": float(np.percentile(arr, 25)), "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)), "p99": float(np.percentile(arr, 99)),
        }

    arr_all = np.array(all_values)
    report["ALL"] = {
        "n": int(arr_all.size),
        "min": float(arr_all.min()), "max": float(arr_all.max()),
        "P(a<=-1)": float((arr_all <= -1).mean()), "P(a<=-2)": float((arr_all <= -2).mean()),
        "P(a<=-3)": float((arr_all <= -3).mean()), "P(a>=1)": float((arr_all >= 1).mean()),
        "P(a>=2)": float((arr_all >= 2).mean()),
        "hard_bound_min_ge_neg3": bool(arr_all.min() >= -3.0 - 1e-9),
        "hard_bound_max_le_pos2": bool(arr_all.max() <= 2.0 + 1e-9),
    }
    return report


def main() -> int:
    report = run(n_steps=2000, master_seed=900101)
    out_dir = REPO_SRC.parent / "output" / "highwayenv_migration" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "M4_M_realized_acceleration_audit.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"-> {out_path}")
    return 0 if (report["ALL"]["hard_bound_min_ge_neg3"] and report["ALL"]["hard_bound_max_le_pos2"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
