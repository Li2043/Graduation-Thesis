#!/usr/bin/env python3
"""VDN_Conditional_Amendment_Protocol.md sec 6 (Diagnostic 1): greedy
(epsilon=0, deterministic) action-selection frequency, tallied separately
for each of the 4 vehicle classes (Ramp-Fast / Ramp-Slow / Mainline-Fast
/ Mainline-Slow), across a frozen scenario bank. Also checks whether all
4 classes collapse to a near-identical behavioural distribution --
evidence that parameter sharing + local observations made the policy
insensitive to which class it's actually controlling.

Read-only against an already-finished checkpoint -- runs no training."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(REPO_SRC))
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import torch  # noqa: E402

from evaluate_policy import load_policy  # noqa: E402
from train_joint_dqn_diagnostic import (  # noqa: E402
    ALWAYS_LEGAL_ACTION_MASK,
    build_joint_dqn_config,
    reorder_joint_observation,
    role_major_slot_order,
)

from thesis.agents.joint_dqn import JointDQNLearner  # noqa: E402
from thesis.study_b.heterogeneous_env import StudyBEnvConfig, StudyBHeterogeneousEnv  # noqa: E402
from thesis.study_b.training_common import load_scenario_bank  # noqa: E402

__all__ = ["ACTION_NAMES", "class_label", "tally_greedy_actions", "tally_greedy_actions_joint", "cross_class_similarity"]

ACTION_NAMES = {0: "MAINTAIN", 1: "ACCELERATE", 2: "DECELERATE"}
CLASS_LABELS = ("ramp-fast", "ramp-slow", "mainline-fast", "mainline-slow")


def class_label(role: str, speed_class: str) -> str:
    return f"{role}-{speed_class}"


def _distribution(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {name: 0.0 for name in ACTION_NAMES.values()}
    return {ACTION_NAMES[a]: counter[a] / total for a in ACTION_NAMES}


def cross_class_similarity(distributions: dict[str, dict[str, float]], *, threshold: float = 0.10) -> dict:
    """Total-variation distance between every pair of class distributions
    (max over action names of |p(a) - q(a)|, summed and halved). Flags
    collapse-to-one-mode if EVERY pair is within ``threshold``."""
    labels = list(distributions)
    pair_distances = {}
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            tv = 0.5 * sum(abs(distributions[a][name] - distributions[b][name]) for name in ACTION_NAMES.values())
            pair_distances[f"{a} vs {b}"] = tv
    max_distance = max(pair_distances.values()) if pair_distances else 0.0
    return {"pair_distances": pair_distances, "max_distance": max_distance, "near_identical_across_classes": max_distance <= threshold}


def tally_greedy_actions(
    *, algorithm: str, checkpoint: Path, scenario_bank: Path, episode_max_steps: int = 200, device: str = "cpu",
) -> dict:
    scenarios = load_scenario_bank(scenario_bank)
    env = StudyBHeterogeneousEnv(StudyBEnvConfig(episode_max_steps=episode_max_steps))
    select = load_policy(algorithm=algorithm, checkpoint=checkpoint, env=env, device=device)

    class_counters: dict[str, Counter] = {label: Counter() for label in CLASS_LABELS}
    for scenario in scenarios:
        obs, _info = env.reset(seed=0, scenario=scenario)
        for _t in range(episode_max_steps):
            actions = select(obs)
            for vid, action in actions.items():
                label = class_label(scenario.vehicles[vid].role, scenario.vehicles[vid].speed_class)
                class_counters.setdefault(label, Counter())[action] += 1
            obs, _base_reward, terminated, truncated, _step_info = env.step(actions)
            if terminated or truncated:
                break

    distributions = {label: _distribution(counter) for label, counter in class_counters.items()}
    similarity = cross_class_similarity(distributions)
    return {
        "counts": {label: dict(counter) for label, counter in class_counters.items()},
        "distributions": distributions,
        "similarity": similarity,
    }


def tally_greedy_actions_joint(
    *, checkpoint: Path, scenario_bank: Path, episode_max_steps: int = 200, device: str = "cpu",
) -> dict:
    """Same as ``tally_greedy_actions`` but for a joint-DQN checkpoint
    (``train_joint_dqn_diagnostic.py``, Diagnostic 5) -- not routable
    through ``evaluate_policy.load_policy`` since that function's
    ``--algorithm dqn`` path assumes ``SharedLocalDQNAgent``'s local,
    18-dim-obs architecture, which does not match ``JointQNetwork``'s
    state_dict shape."""
    vid_order = ("V0", "V1", "V2", "V3")
    scenarios = load_scenario_bank(scenario_bank)
    env = StudyBHeterogeneousEnv(StudyBEnvConfig(episode_max_steps=episode_max_steps))

    dqn_config = build_joint_dqn_config(device=device)
    learner = JointDQNLearner(dqn_config, seed=0)
    ckpt = torch.load(checkpoint, map_location=device)
    learner.online.load_state_dict(ckpt["online"])

    class_counters: dict[str, Counter] = {label: Counter() for label in CLASS_LABELS}
    masks = [ALWAYS_LEGAL_ACTION_MASK] * 4
    for scenario in scenarios:
        _obs, info = env.reset(seed=0, scenario=scenario)
        slot_order = role_major_slot_order(info["roles"])
        joint_obs = reorder_joint_observation(env.global_state(), vehicle_id_order=vid_order, slot_order=slot_order)
        for _t in range(episode_max_steps):
            actions_tuple = learner.select_action(joint_obs, masks, greedy=True)
            for vid, action in zip(slot_order, actions_tuple):
                label = class_label(scenario.vehicles[vid].role, scenario.vehicles[vid].speed_class)
                class_counters.setdefault(label, Counter())[action] += 1
            actions_dict = dict(zip(slot_order, actions_tuple))
            _next_obs, _base_reward, terminated, truncated, _step_info = env.step(actions_dict)
            if terminated or truncated:
                break
            joint_obs = reorder_joint_observation(env.global_state(), vehicle_id_order=vid_order, slot_order=slot_order)

    distributions = {label: _distribution(counter) for label, counter in class_counters.items()}
    similarity = cross_class_similarity(distributions)
    return {
        "counts": {label: dict(counter) for label, counter in class_counters.items()},
        "distributions": distributions,
        "similarity": similarity,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", type=str, default="dqn", choices=["mappo", "dqn", "joint_dqn"])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--scenario-bank", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episode-max-steps", type=int, default=200)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args(argv)

    if args.algorithm == "joint_dqn":
        report = tally_greedy_actions_joint(
            checkpoint=args.checkpoint, scenario_bank=args.scenario_bank,
            episode_max_steps=args.episode_max_steps, device=args.device,
        )
    else:
        report = tally_greedy_actions(
            algorithm=args.algorithm, checkpoint=args.checkpoint, scenario_bank=args.scenario_bank,
            episode_max_steps=args.episode_max_steps, device=args.device,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for label, dist in report["distributions"].items():
        pct = "  ".join(f"{name}={p:.1%}" for name, p in dist.items())
        print(f"{label:>15}: {pct}")
    print(f"max cross-class distance: {report['similarity']['max_distance']:.3f}  "
          f"near_identical_across_classes={report['similarity']['near_identical_across_classes']}")
    print(f"report written -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
