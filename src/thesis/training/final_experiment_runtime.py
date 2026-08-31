"""Stage 5A-0 experiment runtime helpers (integration regression only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from thesis.agents.independent_dqn_v2 import IndependentDQNLearner
from thesis.agents.replay_buffer_v2 import ReplayBatch, ReplayTransition
from thesis.envs.final_observation import OBSERVATION_DIM
from thesis.rewards.pbrs_v2 import telescoping_sum
from thesis.training.final_lock_loader import FinalLockBundle, load_final_locks
from thesis.training.final_reward_conditions import (
    FINAL_REWARD_CONDITIONS,
    IntegrationPBRSConfig,
    RewardConditionName,
)
from thesis.training.final_v3_pipeline import (
    N_ACTIONS,
    build_integration_learners,
    run_final_v3_episode,
)


def scripted_accelerate(n: int) -> list[dict[str, int]]:
    return [{"A": 1, "B": 1} for _ in range(n)]


def scripted_a_accel_b_maintain(n: int) -> list[dict[str, int]]:
    return [{"A": 1, "B": 0} for _ in range(n)]


def physical_fingerprint(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "policy_step": row["policy_step"],
        "controller_id": row["controller_id"],
        "action": row["action"],
        "observation": row["observation"],
        "vehicles_t": row["vehicles_t"],
        "vehicles_t1": row["vehicles_t1"],
        "exit_event": row["exit_event"],
        "collision_pairs": row["collision_pairs"],
        "terminated": row["terminated"],
        "truncated": row["truncated"],
        "term_reason": row["term_reason"],
        "base_reward": row["base_reward"],
        "progress_component": row["progress_component"],
        "exit_component": row["exit_component"],
        "collision_component": row["collision_component"],
        "hard_braking_cost": row["hard_braking_cost"],
        "hard_braking_component": row["hard_braking_component"],
        "policy_level_acceleration": row["policy_level_acceleration"],
    }


def max_physical_diff(
    rows_a: Sequence[Mapping[str, Any]], rows_b: Sequence[Mapping[str, Any]]
) -> float:
    if len(rows_a) != len(rows_b):
        return float(abs(len(rows_a) - len(rows_b)))
    max_diff = 0.0
    for a, b in zip(rows_a, rows_b):
        fa, fb = physical_fingerprint(a), physical_fingerprint(b)
        if fa.keys() != fb.keys():
            return float("inf")
        for k in fa:
            va, vb = fa[k], fb[k]
            if isinstance(va, (int, float, bool)) and isinstance(vb, (int, float, bool)):
                max_diff = max(max_diff, abs(float(va) - float(vb)))
            elif va != vb:
                # nested structures compared via JSON for float tolerance
                sa = json.dumps(va, sort_keys=True)
                sb = json.dumps(vb, sort_keys=True)
                if sa != sb:
                    # try numeric walk
                    diff = _nested_max_abs_diff(va, vb)
                    if diff is None:
                        return float("inf")
                    max_diff = max(max_diff, diff)
    return float(max_diff)


def _nested_max_abs_diff(a: Any, b: Any) -> float | None:
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return None
        m = 0.0
        for k in a:
            d = _nested_max_abs_diff(a[k], b[k])
            if d is None:
                return None
            m = max(m, d)
        return m
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return None
        m = 0.0
        for x, y in zip(a, b):
            d = _nested_max_abs_diff(x, y)
            if d is None:
                return None
            m = max(m, d)
        return m
    if isinstance(a, (int, float, bool)) and isinstance(b, (int, float, bool)):
        return abs(float(a) - float(b))
    return 0.0 if a == b else None


def compute_telescoping_errors(transitions: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Use controller A rows as the physical timeline (one row per step)."""
    a_rows = [r for r in transitions if r["controller_id"] == "A"]
    if not a_rows:
        a_rows = list(transitions)
    # Deduplicate by policy_step keeping first
    by_step: dict[int, Mapping[str, Any]] = {}
    for r in a_rows:
        by_step.setdefault(int(r["policy_step"]), r)
    ordered = [by_step[k] for k in sorted(by_step)]
    if len(ordered) < 1:
        return {"mean_error": 0.0, "min_error": 0.0}

    mean_phis = [float(ordered[0]["actual_mean_t"])] + [
        float(r["actual_mean_t1"]) for r in ordered
    ]
    min_phis = [float(ordered[0]["actual_min_t"])] + [
        float(r["actual_min_t1"]) for r in ordered
    ]
    gamma = 0.995
    mean_sum, _ = telescoping_sum(mean_phis, gamma)
    min_sum, _ = telescoping_sum(min_phis, gamma)
    T = len(ordered)
    mean_closed = (gamma**T) * mean_phis[-1] - mean_phis[0]
    min_closed = (gamma**T) * min_phis[-1] - min_phis[0]
    return {
        "mean_error": abs(mean_sum - mean_closed),
        "min_error": abs(min_sum - min_closed),
        "mean_sum": float(mean_sum),
        "min_sum": float(min_sum),
        "mean_closed_form": float(mean_closed),
        "min_closed_form": float(min_closed),
        "phi_mean_0": float(mean_phis[0]),
        "phi_mean_T": float(mean_phis[-1]),
        "phi_min_0": float(min_phis[0]),
        "phi_min_T": float(min_phis[-1]),
        "n_transitions": float(T),
        "terminated": float(bool(ordered[-1]["terminated"])),
        "truncated": float(bool(ordered[-1]["truncated"])),
    }


def _stack_batch(transitions: Sequence[ReplayTransition]) -> ReplayBatch:
    return ReplayBatch(
        observations=np.stack([t.observation for t in transitions]),
        actions=np.asarray([t.action for t in transitions], dtype=np.int64),
        shaped_rewards=np.asarray([t.shaped_reward for t in transitions], dtype=np.float64),
        next_observations=[t.next_observation for t in transitions],
        terminated=np.asarray([t.terminated for t in transitions], dtype=bool),
        truncated=np.asarray([t.truncated for t in transitions], dtype=bool),
        controller_terminal=np.asarray(
            [t.controller_terminal for t in transitions], dtype=bool
        ),
        learner_completed=np.asarray(
            [t.learner_completed for t in transitions], dtype=bool
        ),
        action_masks=np.stack([t.action_mask for t in transitions]),
        next_action_masks=[t.next_action_mask for t in transitions],
        base_rewards=np.asarray([t.base_reward for t in transitions], dtype=np.float64),
        shaping_components=np.asarray(
            [t.shaping_component for t in transitions], dtype=np.float64
        ),
        reward_conditions=[t.reward_condition for t in transitions],
        indices=np.arange(len(transitions), dtype=np.int64),
        transitions=list(transitions),
    )


def isolated_dqn_update(
    learner: IndependentDQNLearner,
    transitions: Sequence[ReplayTransition],
) -> dict[str, Any]:
    """Exactly one optimiser update on a provided batch (no training loop)."""
    if not transitions:
        raise ValueError("need at least one transition for isolated update")
    batch = _stack_batch(transitions)
    if batch.observations.shape[1] != OBSERVATION_DIM:
        raise RuntimeError(f"obs dim {batch.observations.shape[1]} != {OBSERVATION_DIM}")

    calls = {"n": 0}
    real_forward = learner.target.forward

    def counting_forward(x):
        calls["n"] += 1
        return real_forward(x)

    learner.target.forward = counting_forward  # type: ignore[method-assign]
    target_before = {
        k: v.detach().clone() for k, v in learner.target.state_dict().items()
    }
    online_before = torch.nn.utils.parameters_to_vector(
        learner.online.parameters()
    ).detach().clone()

    stats = learner.update(batch)

    online_after = torch.nn.utils.parameters_to_vector(
        learner.online.parameters()
    ).detach().clone()
    target_unchanged = all(
        torch.equal(target_before[k], learner.target.state_dict()[k])
        for k in target_before
    )
    with torch.no_grad():
        q = learner.online(
            torch.as_tensor(batch.observations, dtype=torch.float32, device=learner.device)
        )
    return {
        "controller_id": learner.controller_id,
        "batch_size": int(batch.observations.shape[0]),
        "obs_shape": list(batch.observations.shape),
        "q_shape": list(q.shape),
        "loss": float(stats["loss"]),
        "online_param_changed": bool(
            not torch.allclose(online_before, online_after, atol=0.0, rtol=0.0)
        ),
        "target_unchanged_without_sync": bool(target_unchanged),
        "target_network_forward_calls": int(calls["n"]),
        "target_network_forward_calls_reported": int(
            stats.get("target_network_forward_calls", -1)
        ),
        "n_bootstrap_rows": int(np.sum(~batch.controller_terminal)),
        "n_terminal_rows": int(np.sum(batch.controller_terminal)),
        "max_abs_q": float(torch.max(torch.abs(q)).item()),
        "max_abs_target": float(np.max(np.abs(batch.shaped_rewards))),
        "finite_loss": bool(np.isfinite(stats["loss"])),
        "finite_q": bool(torch.isfinite(q).all().item()),
        "isolated_optimizer_updates_only": True,
        "sustained_training_invoked": False,
        "policy_training_started": False,
    }


def collect_mixed_batch_transitions(
    bundle: FinalLockBundle,
    *,
    condition: RewardConditionName = "baseline",
) -> list[ReplayTransition]:
    """Collect terminal + ongoing + truncation rows from real V3 paths."""
    pcfg = IntegrationPBRSConfig()
    # Ongoing + early exit / success path
    ep_exit = run_final_v3_episode(
        bundle,
        reward_condition=condition,
        scripted_actions=scripted_a_accel_b_maintain(70),
        pbrs_config=pcfg,
        episode_id=f"mix_exit_{condition}",
    )
    # Truncation path
    ep_trunc = run_final_v3_episode(
        bundle,
        reward_condition=condition,
        scripted_actions=scripted_accelerate(10),
        pbrs_config=pcfg,
        max_policy_steps=3,
        episode_id=f"mix_trunc_{condition}",
    )
    # Collision terminal
    ep_coll = run_final_v3_episode(
        bundle,
        reward_condition=condition,
        scripted_actions=scripted_accelerate(30),
        pbrs_config=pcfg,
        episode_id=f"mix_coll_{condition}",
    )

    selected: list[ReplayTransition] = []
    # one ongoing bootstrap (early step, non-terminal)
    for tr in ep_exit["stored_transitions"]:
        if (not tr.controller_terminal) and (not tr.truncated) and (not tr.terminated):
            selected.append(tr)
            break
    # one controller-terminal (exit or collision)
    for tr in ep_exit["stored_transitions"] + ep_coll["stored_transitions"]:
        if tr.controller_terminal:
            selected.append(tr)
            break
    # one truncation bootstrap
    for tr in ep_trunc["stored_transitions"]:
        if tr.truncated and (not tr.controller_terminal):
            selected.append(tr)
            break
    kinds = {
        "terminal": any(t.controller_terminal for t in selected),
        "ongoing": any(
            (not t.controller_terminal) and (not t.truncated) for t in selected
        ),
        "truncation": any(t.truncated and (not t.controller_terminal) for t in selected),
    }
    if not all(kinds.values()):
        raise RuntimeError(f"mixed batch incomplete: {kinds}")
    return selected


def run_condition_suite(bundle: FinalLockBundle) -> dict[str, Any]:
    """Run core Stage 5A-0 integration scenarios for all reward conditions."""
    pcfg = IntegrationPBRSConfig()
    pcfg_zero = pcfg.with_lambda_zero()
    results: dict[str, Any] = {"conditions": {}, "invariance": {}, "telescoping": {}}

    # Physical invariance across conditions (same scripted collision trajectory)
    actions_coll = scripted_accelerate(25)
    by_cond: dict[str, Any] = {}
    for cond in FINAL_REWARD_CONDITIONS:
        by_cond[cond] = run_final_v3_episode(
            bundle,
            reward_condition=cond,
            scripted_actions=actions_coll,
            pbrs_config=pcfg,
            episode_id=f"inv_{cond}",
        )
    base_rows = by_cond["baseline"]["transitions"]
    max_phys = 0.0
    max_base_diff = 0.0
    for cond in ("mean_pbrs", "min_pbrs"):
        max_phys = max(
            max_phys, max_physical_diff(base_rows, by_cond[cond]["transitions"])
        )
        for a, b in zip(base_rows, by_cond[cond]["transitions"]):
            max_base_diff = max(
                max_base_diff, abs(float(a["base_reward"]) - float(b["base_reward"]))
            )
    results["invariance"] = {
        "max_physical_diff": max_phys,
        "max_base_reward_diff": max_base_diff,
        "n_physical_transitions": by_cond["baseline"]["n_physical_transitions"],
    }
    results["conditions"] = {
        cond: {
            "n_transitions": len(ep["transitions"]),
            "n_physical": ep["n_physical_transitions"],
            "term_reason": ep["transitions"][-1]["term_reason"] if ep["transitions"] else None,
        }
        for cond, ep in by_cond.items()
    }

    # Early-exit continuation
    exit_ep = run_final_v3_episode(
        bundle,
        reward_condition="mean_pbrs",
        scripted_actions=scripted_a_accel_b_maintain(70),
        pbrs_config=pcfg,
        episode_id="early_exit",
    )
    results["early_exit"] = exit_ep

    # Truncation
    trunc_ep = run_final_v3_episode(
        bundle,
        reward_condition="min_pbrs",
        scripted_actions=scripted_accelerate(10),
        pbrs_config=pcfg,
        max_policy_steps=4,
        episode_id="truncation",
    )
    results["truncation"] = trunc_ep

    # Lambda=0 equals baseline
    zero_mean = run_final_v3_episode(
        bundle,
        reward_condition="mean_pbrs",
        scripted_actions=actions_coll,
        pbrs_config=pcfg_zero,
        episode_id="lambda0_mean",
    )
    zero_min = run_final_v3_episode(
        bundle,
        reward_condition="min_pbrs",
        scripted_actions=actions_coll,
        pbrs_config=pcfg_zero,
        episode_id="lambda0_min",
    )
    lam0_err = 0.0
    for a, b, c in zip(
        by_cond["baseline"]["transitions"],
        zero_mean["transitions"],
        zero_min["transitions"],
    ):
        lam0_err = max(
            lam0_err,
            abs(float(a["learner_reward"]) - float(b["learner_reward"])),
            abs(float(a["learner_reward"]) - float(c["learner_reward"])),
        )
    results["lambda0_max_reward_diff"] = lam0_err

    results["telescoping"] = {
        "collision_mean_pbrs": compute_telescoping_errors(
            by_cond["mean_pbrs"]["transitions"]
        ),
        "early_exit_mean": compute_telescoping_errors(exit_ep["transitions"]),
        "truncation_min": compute_telescoping_errors(trunc_ep["transitions"]),
    }

    # Isolated updates per condition
    update_rows: list[dict[str, Any]] = []
    for cond in FINAL_REWARD_CONDITIONS:
        mixed = collect_mixed_batch_transitions(bundle, condition=cond)
        learners = build_integration_learners(reward_condition=cond, seed_A=11, seed_B=22)
        for aid in ("A", "B"):
            # Prefer rows for this controller; fall back to all
            own = [t for t in mixed if t.controller_id == aid]
            batch_trs = own if len(own) >= 2 else list(mixed)
            # Ensure terminal+bootstrap present in batch
            if not any(t.controller_terminal for t in batch_trs):
                batch_trs = list(mixed)
            upd = isolated_dqn_update(learners[aid], batch_trs)
            upd["reward_condition"] = cond
            update_rows.append(upd)
    results["isolated_updates"] = update_rows
    results["by_condition_episodes"] = by_cond
    return results


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=_json_default) + "\n")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


__all__ = [
    "collect_mixed_batch_transitions",
    "compute_telescoping_errors",
    "isolated_dqn_update",
    "load_final_locks",
    "max_physical_diff",
    "physical_fingerprint",
    "run_condition_suite",
    "scripted_a_accel_b_maintain",
    "scripted_accelerate",
    "write_jsonl",
]
