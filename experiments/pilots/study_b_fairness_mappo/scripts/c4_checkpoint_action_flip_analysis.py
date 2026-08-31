#!/usr/bin/env python3
"""C4 checkpoint action-flip + Q-margin analysis (RUNBOOK Diversity
Recovery diagnostic addendum, 2026-08-17): for every (seed, scenario)
whose greedy outcome flips between two frozen checkpoints, replays
both checkpoints in lockstep against the identical deterministic
scenario (since dynamics + observations are deterministic given a
fixed scenario and a fixed action sequence), finds the EARLIEST
timestep where their greedy argmax actions diverge, and records both
checkpoints' full Q(ACCELERATE)/Q(HOLD)/Q(BRAKE) at that state plus the
resulting margin (Q_best - Q_second_best). Also records a short window
of states before/after the divergence, and continues each checkpoint's
own trajectory to its own eventual outcome.

No training occurs anywhere in this script."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(REPO_SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from thesis.study_b.envs.highwayenv_action import ACCELERATE, BRAKE, HOLD  # noqa: E402
from thesis.study_b.envs.highwayenv_merge import ThesisHighwayMergeEnvConfig  # noqa: E402
from thesis.study_b.envs.highwayenv_wrapper import StudyBHeterogeneousHighwayEnv, StudyBHighwayWrapperConfig  # noqa: E402
from thesis.study_b.shared_local_dqn import SharedLocalDQNAgent, build_study_b_dqn_config  # noqa: E402
from thesis.study_b.training_common import load_scenario_bank  # noqa: E402

_ACTION_NAME = {HOLD: "HOLD", ACCELERATE: "ACCELERATE", BRAKE: "BRAKE"}
_ACTION_ORDER = [HOLD, ACCELERATE, BRAKE]  # matches n_actions indexing (0,1,2)


def _load_agent(checkpoint: Path, device: str) -> SharedLocalDQNAgent:
    dqn_config = build_study_b_dqn_config(device=device)
    agent = SharedLocalDQNAgent(dqn_config, seed=0)
    ckpt = torch.load(checkpoint, map_location=device)
    agent.learner.online.load_state_dict(ckpt["online"])
    return agent


def _make_env(episode_max_steps: int) -> StudyBHeterogeneousHighwayEnv:
    cfg = ThesisHighwayMergeEnvConfig(episode_max_steps=episode_max_steps, action_representation="meta_speed")
    return StudyBHeterogeneousHighwayEnv(StudyBHighwayWrapperConfig(env_config=cfg))


def _q_values_all_vehicles(agent: SharedLocalDQNAgent, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {vid: agent.learner.q_values(o, network="online") for vid, o in obs.items()}


def _run_to_outcome(agent: SharedLocalDQNAgent, env: StudyBHeterogeneousHighwayEnv, scenario, episode_max_steps: int) -> tuple[str, int]:
    obs, _info = env.reset(seed=0, scenario=scenario)
    for t in range(episode_max_steps):
        actions = agent.select_actions(obs, epsilon=0.0, greedy=True)
        obs, _r, terminated, truncated, step_info = env.step(actions)
        if step_info["collision_event"]:
            return "collision", t + 1
        if terminated:
            return "success", t + 1
        if truncated:
            return "truncation", t + 1
    return "truncation", episode_max_steps


def find_divergence_and_trace(
    *, seed: int, scenario_id: str, checkpoint_a: Path, checkpoint_b: Path, label_a: str, label_b: str,
    scenario_bank_path: Path, episode_max_steps: int, device: str, context_window: int,
) -> dict:
    scenarios = load_scenario_bank(scenario_bank_path)
    scenario = next(s for s in scenarios if s.scenario_id == scenario_id)

    agent_a = _load_agent(checkpoint_a, device)
    agent_b = _load_agent(checkpoint_b, device)
    env_a = _make_env(episode_max_steps)
    env_b = _make_env(episode_max_steps)

    obs_a, _ = env_a.reset(seed=0, scenario=scenario)
    obs_b, _ = env_b.reset(seed=0, scenario=scenario)

    history: list[dict] = []
    divergence: dict | None = None

    for t in range(episode_max_steps):
        vids = env_a.active_vehicle_ids
        actions_a = agent_a.select_actions(obs_a, epsilon=0.0, greedy=True)
        actions_b = agent_b.select_actions(obs_b, epsilon=0.0, greedy=True)
        q_a = _q_values_all_vehicles(agent_a, obs_a)
        q_b = _q_values_all_vehicles(agent_b, obs_b)

        step_record = {"t": t, "vehicles": {}}
        any_diff = False
        for vid in vids:
            qa = q_a[vid]
            qb = q_b[vid]
            diff = int(actions_a[vid]) != int(actions_b[vid])
            any_diff = any_diff or diff
            qa_sorted = sorted(qa, reverse=True)
            qb_sorted = sorted(qb, reverse=True)
            step_record["vehicles"][vid] = {
                "action_a": int(actions_a[vid]), "action_a_name": _ACTION_NAME[int(actions_a[vid])],
                "action_b": int(actions_b[vid]), "action_b_name": _ACTION_NAME[int(actions_b[vid])],
                "differs": diff,
                "q_a": {_ACTION_NAME[a]: float(qa[i]) for i, a in enumerate(_ACTION_ORDER)},
                "q_b": {_ACTION_NAME[a]: float(qb[i]) for i, a in enumerate(_ACTION_ORDER)},
                "margin_a": float(qa_sorted[0] - qa_sorted[1]),
                "margin_b": float(qb_sorted[0] - qb_sorted[1]),
                "obs_a": obs_a[vid].tolist(), "obs_b": obs_b[vid].tolist(),
            }
        history.append(step_record)

        if any_diff and divergence is None:
            divergence = {"t": t, "step_record": step_record}

        obs_a, _r, term_a, trunc_a, info_a = env_a.step(actions_a)
        obs_b, _r, term_b, trunc_b, info_b = env_b.step(actions_b)

        if term_a or trunc_a or info_a["collision_event"]:
            outcome_a = "collision" if info_a["collision_event"] else ("success" if term_a else "truncation")
            step_a_end = t + 1
            break
    else:
        outcome_a, step_a_end = "truncation", episode_max_steps

    if term_b or trunc_b or info_b["collision_event"]:
        outcome_b = "collision" if info_b["collision_event"] else ("success" if term_b else "truncation")
        step_b_end = t + 1
    elif divergence is not None:
        # env_b may terminate at a different step than env_a; finish it out.
        outcome_b, extra = _run_to_outcome(agent_b, env_b, scenario, episode_max_steps - (t + 1))
        step_b_end = (t + 1) + extra
    else:
        outcome_b, step_b_end = "truncation", episode_max_steps

    context_start = max(0, (divergence["t"] if divergence else 0) - context_window)
    context_end = min(len(history), (divergence["t"] if divergence else 0) + context_window + 1)

    return {
        "seed": seed, "scenario_id": scenario_id, "label_a": label_a, "label_b": label_b,
        "outcome_a": outcome_a, "outcome_b": outcome_b, "step_a_end": step_a_end, "step_b_end": step_b_end,
        "divergence_t": (divergence["t"] if divergence else None),
        "divergence_detail": (divergence["step_record"] if divergence else None),
        "context_window_steps": history[context_start:context_end],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario-bank", type=Path, required=True)
    p.add_argument("--pairs", type=str, nargs="+", required=True,
                    help="seed:scenario_id:label_a:ckpt_a:label_b:ckpt_b (colon-separated, 6 fields)")
    p.add_argument("--episode-max-steps", type=int, default=200)
    p.add_argument("--context-window", type=int, default=10)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    csv_rows = []

    for pair in args.pairs:
        fields = pair.split(":")
        seed_str, scenario_id, label_a, ckpt_a, label_b, ckpt_b = fields
        seed = int(seed_str)
        result = find_divergence_and_trace(
            seed=seed, scenario_id=scenario_id, checkpoint_a=Path(ckpt_a), checkpoint_b=Path(ckpt_b),
            label_a=label_a, label_b=label_b, scenario_bank_path=args.scenario_bank,
            episode_max_steps=args.episode_max_steps, device=args.device, context_window=args.context_window,
        )
        results.append(result)

        out_path = args.output_dir / f"FLIP_seed{seed}_{scenario_id}_{label_a}_vs_{label_b}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        dv = result["divergence_detail"]
        if dv is not None:
            for vid, v in dv["vehicles"].items():
                if v["differs"]:
                    csv_rows.append({
                        "seed": seed, "scenario_id": scenario_id, "label_a": label_a, "label_b": label_b,
                        "divergence_t": result["divergence_t"], "vehicle_id": vid,
                        "action_a": v["action_a_name"], "action_b": v["action_b_name"],
                        "q_a_HOLD": v["q_a"]["HOLD"], "q_a_ACCELERATE": v["q_a"]["ACCELERATE"], "q_a_BRAKE": v["q_a"]["BRAKE"],
                        "q_b_HOLD": v["q_b"]["HOLD"], "q_b_ACCELERATE": v["q_b"]["ACCELERATE"], "q_b_BRAKE": v["q_b"]["BRAKE"],
                        "margin_a": v["margin_a"], "margin_b": v["margin_b"],
                        "outcome_a": result["outcome_a"], "outcome_b": result["outcome_b"],
                    })
        print(f"[FLIP] seed={seed} scenario={scenario_id} {label_a}(outcome={result['outcome_a']}) vs "
              f"{label_b}(outcome={result['outcome_b']}) divergence_t={result['divergence_t']}")

    if csv_rows:
        csv_path = args.output_dir / "C4_CHECKPOINT_ACTION_FLIP_ANALYSIS.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"wrote {csv_path}")

    (args.output_dir / "C4_CHECKPOINT_ACTION_FLIP_ANALYSIS.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
