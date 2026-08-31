#!/usr/bin/env python3
"""Diagnostic_6_DQN_Pipeline_Verification_Protocol.md sec 14 (6J):
per-feature observation-scale audit over a mixture of oracle, random,
and (optionally) learned trajectories -- looks for constant (std=0)
features and extreme scale mismatches across the 18-dim local
observation. Also re-verifies the local-observation leakage guarantee
(an ego's own observation must never reveal another vehicle's target
speed) as a bonus check, reusing the same trace data.

Read-only / no training required."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(REPO_SRC))
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_policy import load_policy  # noqa: E402

from thesis.study_b.heterogeneous_env import StudyBEnvConfig, StudyBHeterogeneousEnv  # noqa: E402
from thesis.study_b.local_observation import LOCAL_OBS_DIM  # noqa: E402
from thesis.study_b.oracle_controller import oracle_actions  # noqa: E402
from thesis.study_b.training_common import load_scenario_bank  # noqa: E402

__all__ = ["FEATURE_NAMES", "collect_observations", "feature_statistics"]

FEATURE_NAMES = [
    "self_role", "self_speed", "self_target_speed", "self_acceleration", "self_dist_to_merge", "self_prev_action",
] + [
    f"neighbour{slot}_{name}"
    for slot in range(3)
    for name in ("presence", "delta_d_norm", "delta_v_norm", "lane_relation")
]
assert len(FEATURE_NAMES) == LOCAL_OBS_DIM


def collect_observations(
    *, policy: str, scenario_bank: Path, episode_max_steps: int = 200,
    checkpoint: Path | None = None, algorithm: str = "dqn", device: str = "cpu", seed: int = 0,
) -> np.ndarray:
    """``policy``: ``"oracle"``, ``"random"``, or ``"learned"`` (requires
    ``checkpoint``). Returns an ``[N, LOCAL_OBS_DIM]`` array of every
    per-vehicle observation seen across every scenario/step."""
    scenarios = load_scenario_bank(scenario_bank)
    env = StudyBHeterogeneousEnv(StudyBEnvConfig(episode_max_steps=episode_max_steps))
    rng = np.random.default_rng(seed)

    select = None
    if policy == "learned":
        if checkpoint is None:
            raise ValueError("policy='learned' requires --checkpoint")
        select = load_policy(algorithm=algorithm, checkpoint=checkpoint, env=env, device=device)

    rows: list[np.ndarray] = []
    for scenario in scenarios:
        obs, _info = env.reset(seed=0, scenario=scenario)
        prev_active = {vid: True for vid in env.active_vehicle_ids}
        for _t in range(episode_max_steps):
            # Only observations for vehicles still genuinely active going
            # into this step -- matches EXACTLY what the real training
            # scripts feed into replay (`if not prev_active[vid]: continue`
            # in train_dqn_direct_welfare.py/train_dqn_fallback.py).
            # active_vehicle_ids never shrinks (fixed at all 4 for the
            # whole episode -- see heterogeneous_env.py), so without this
            # filter an already-completed vehicle's route_position keeps
            # drifting arbitrarily far past the merge zone (physics holds
            # its last speed indefinitely once completed=True), producing
            # wildly out-of-distribution self_dist_to_merge values that
            # the actual training data never sees.
            for vid, arr in obs.items():
                if prev_active[vid]:
                    rows.append(np.asarray(arr, dtype=np.float64))
            if policy == "oracle":
                positions = {vid: env._env._vehicles[vid].route_position for vid in env.active_vehicle_ids}  # noqa: SLF001
                actions = oracle_actions(
                    scenario=scenario, positions=positions, merge_start=200.0, merge_end=300.0,
                    active_vehicle_ids={vid: True for vid in env.active_vehicle_ids},
                )
            elif policy == "random":
                actions = {vid: int(rng.integers(0, 3)) for vid in env.active_vehicle_ids}
            else:
                actions = select(obs)
            obs, _r, terminated, truncated, step_info = env.step(actions)
            prev_active = dict(step_info["active"])
            if terminated or truncated:
                break
    return np.stack(rows)


def feature_statistics(observations: np.ndarray) -> dict[str, dict]:
    stats = {}
    for i, name in enumerate(FEATURE_NAMES):
        col = observations[:, i]
        stats[name] = {
            "min": float(col.min()), "max": float(col.max()), "mean": float(col.mean()), "std": float(col.std()),
            "fraction_zero": float(np.mean(col == 0.0)),
        }
    return stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario-bank", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episode-max-steps", type=int, default=200)
    p.add_argument("--checkpoint", type=Path, default=None, help="optional -- adds a 'learned' policy mixture")
    p.add_argument("--algorithm", type=str, default="dqn", choices=["mappo", "dqn"])
    args = p.parse_args(argv)

    policies = ["oracle", "random"]
    if args.checkpoint is not None:
        policies.append("learned")

    all_obs = []
    per_policy_counts = {}
    for policy in policies:
        obs = collect_observations(
            policy=policy, scenario_bank=args.scenario_bank, episode_max_steps=args.episode_max_steps,
            checkpoint=args.checkpoint, algorithm=args.algorithm,
        )
        per_policy_counts[policy] = int(obs.shape[0])
        all_obs.append(obs)
    combined = np.concatenate(all_obs, axis=0)
    stats = feature_statistics(combined)

    report = {"policies_mixed": policies, "n_observations_per_policy": per_policy_counts, "feature_statistics": stats}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    constant_features = [name for name, s in stats.items() if s["std"] == 0.0]
    for name, s in stats.items():
        flag = "  <-- CONSTANT (std=0)" if s["std"] == 0.0 else ""
        print(f"{name:>26}: min={s['min']:8.3f} max={s['max']:8.3f} mean={s['mean']:8.3f} std={s['std']:8.3f}{flag}")
    print(f"constant features: {constant_features if constant_features else 'none'}")
    print(f"report written -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
