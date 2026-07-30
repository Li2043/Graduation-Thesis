"""Authoritative Final V3 → reward/PBRS → Independent DQN pipeline (Stage 5A-0).

Uses MergeEnvCandidateV3 exclusively. Historical V2 pipelines remain out of the
final training-facing factory path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from thesis.agents.action_masking import role_action_mask, validate_action_mask
from thesis.agents.dqn_targets import compute_dqn_target
from thesis.agents.independent_dqn_v2 import DQNConfig, IndependentDQNLearner
from thesis.agents.replay_buffer_v2 import ReplayTransition
from thesis.envs.final_observation import OBSERVATION_DIM
from thesis.envs.merge_env_candidate_v3 import HighLevelAction, MergeEnvCandidateV3
from thesis.rewards.pbrs_v2 import (
    STAKEHOLDER_ORDER,
    PotentialState,
    StakeholderState,
    apply_pbrs_to_base_rewards,
    compute_potential_breakdown,
)
from thesis.training.final_lock_loader import FinalLockBundle
from thesis.training.final_reward_conditions import (
    FINAL_REWARD_CONDITIONS,
    IntegrationPBRSConfig,
    RewardConditionName,
    select_condition_reward,
)

N_ACTIONS = 3
DQN_TARGET_VERSION = "hardened_controller_terminal_v2b2r"
REWARD_VERSION = "final_base_reward_plus_locked_comfort"
ENVIRONMENT_CLASS = "MergeEnvCandidateV3"


@dataclass(frozen=True)
class FinalRuntimeAssertion:
    environment_class: str = ENVIRONMENT_CLASS
    observation_dimension: int = OBSERVATION_DIM
    route_geometry_version: str = ""
    reward_version: str = REWARD_VERSION
    dqn_target_version: str = DQN_TARGET_VERSION
    integration_test_architecture: bool = True
    policy_training_started: bool = False
    pilot_training_started: bool = False
    sustained_training_invoked: bool = False
    isolated_optimizer_updates_only: bool = True
    pbrs_parameters_final: bool = False
    integration_test_only: bool = True


def runtime_assertion_from_locks(bundle: FinalLockBundle) -> FinalRuntimeAssertion:
    return FinalRuntimeAssertion(
        environment_class=ENVIRONMENT_CLASS,
        observation_dimension=int(bundle.observation_dimension),
        route_geometry_version=str(bundle.route_geometry_version),
        reward_version=REWARD_VERSION,
        dqn_target_version=DQN_TARGET_VERSION,
        integration_test_architecture=True,
        policy_training_started=False,
        pilot_training_started=False,
        sustained_training_invoked=False,
        isolated_optimizer_updates_only=True,
        pbrs_parameters_final=False,
        integration_test_only=True,
    )


def assert_final_v3_runtime(env: MergeEnvCandidateV3, bundle: FinalLockBundle) -> FinalRuntimeAssertion:
    if type(env).__name__ != ENVIRONMENT_CLASS:
        raise RuntimeError(f"expected {ENVIRONMENT_CLASS}, got {type(env).__name__}")
    if type(env).__module__ != "thesis.envs.merge_env_candidate_v3":
        raise RuntimeError(
            f"final pipeline must use merge_env_candidate_v3, got {type(env).__module__}"
        )
    meta = runtime_assertion_from_locks(bundle)
    if meta.observation_dimension != OBSERVATION_DIM:
        raise RuntimeError("observation_dimension must be 27")
    return meta


def potential_state_from_v3_vehicles(
    vehicles: Mapping[str, Mapping[str, Any]],
    target_speeds: Mapping[str, float],
    *,
    terminated: bool,
    truncated: bool,
    terminal_label: str | None = None,
) -> PotentialState:
    """Authoritative V3 vehicle snapshot → PotentialState.

    Completion is taken from the supplied snapshot only (state_t vs state_t1).
    Do not infer state_t completion from state_t1 flags.
    Target speeds come from the locked IC block (not inferred from peers).
    """
    stakeholders: dict[str, StakeholderState] = {}
    for sid in STAKEHOLDER_ORDER:
        if sid not in vehicles:
            raise ValueError(f"vehicles missing stakeholder {sid!r}")
        if sid not in target_speeds:
            raise ValueError(f"target_speeds missing stakeholder {sid!r}")
        v = vehicles[sid]
        stakeholders[sid] = StakeholderState(
            speed=float(v["speed"]),
            target_speed=float(target_speeds[sid]),
            completed=bool(v["completed"]),
        )
    return PotentialState(
        stakeholders=stakeholders,
        terminated=bool(terminated),
        truncated=bool(truncated),
        terminal_label=terminal_label,
    )


def build_integration_learners(
    *,
    reward_condition: RewardConditionName = "baseline",
    seed_A: int = 0,
    seed_B: int = 1,
) -> dict[str, IndependentDQNLearner]:
    cfg = DQNConfig(
        obs_dim=OBSERVATION_DIM,
        n_actions=N_ACTIONS,
        gamma=0.995,
        hidden_sizes=(32, 32),  # integration_test_architecture only
        reward_condition=reward_condition,
        batch_size=8,
        replay_capacity=2048,
    )
    return {
        "A": IndependentDQNLearner("A", cfg, seed=seed_A),
        "B": IndependentDQNLearner("B", cfg, seed=seed_B),
    }


@dataclass
class TransitionRecord:
    """One physical transition with full reward / PBRS / replay diagnostics."""

    payload: dict[str, Any] = field(default_factory=dict)


def _mask_for_role(role: str) -> np.ndarray:
    return validate_action_mask(role_action_mask(str(role), n_actions=N_ACTIONS), N_ACTIONS)


def _decomposition_error(comp: Mapping[str, float], total: float) -> float:
    recon = (
        float(comp["progress_component"])
        + float(comp["exit_component"])
        + float(comp["collision_component"])
        + float(comp["hard_braking_component"])
    )
    return abs(recon - float(total))


def run_final_v3_episode(
    bundle: FinalLockBundle,
    *,
    reward_condition: RewardConditionName,
    scripted_actions: Sequence[Mapping[str, int]],
    pbrs_config: IntegrationPBRSConfig | None = None,
    block_id: str = "calibration_001",
    block_set: str = "calibration",
    max_policy_steps: int | None = None,
    episode_id: str = "ep0",
    store_in_learners: bool = False,
    learners: dict[str, IndependentDQNLearner] | None = None,
) -> dict[str, Any]:
    """Run one scripted episode under a single reward condition.

    Physical dynamics are independent of ``reward_condition``.
    """
    if reward_condition not in FINAL_REWARD_CONDITIONS:
        raise ValueError(f"unknown condition {reward_condition!r}")
    pcfg = pbrs_config or IntegrationPBRSConfig()
    pbrs = pcfg.to_pbrs_config()
    env = bundle.build_env(
        block_id=block_id,
        block_set=block_set,
        max_policy_steps=max_policy_steps,
    )
    meta = assert_final_v3_runtime(env, bundle)
    obs, _reset_info = env.reset(seed=env.config.block.seed)
    target_speeds = env.config.block.target_speeds.as_map()
    for aid in ("A", "B"):
        if obs[aid].shape != (OBSERVATION_DIM,):
            raise RuntimeError(f"obs[{aid}] shape {obs[aid].shape} != {(OBSERVATION_DIM,)}")

    active = {"A": True, "B": True}
    transitions: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    stored: list[ReplayTransition] = []

    for step_i, scripted in enumerate(scripted_actions):
        roles_t = {
            aid: str(env._vehicles[aid].role) for aid in ("A", "B")  # noqa: SLF001
        }
        masks = {aid: _mask_for_role(roles_t[aid]) for aid in ("A", "B")}
        actions: dict[str, int] = {}
        for aid in ("A", "B"):
            if not active[aid]:
                actions[aid] = int(HighLevelAction.MAINTAIN)
            else:
                actions[aid] = int(scripted[aid])

        next_obs, base_rewards, terminated, truncated, info = env.step(actions)
        next_masks = {
            aid: _mask_for_role(str(info["vehicles_t1"][aid]["role"])) for aid in ("A", "B")
        }

        # Potentials: completion from the matching snapshot only.
        pot_t = potential_state_from_v3_vehicles(
            info["vehicles_t"],
            target_speeds,
            terminated=False,
            truncated=False,
        )
        term_label = None
        if terminated:
            term_label = (
                "collision"
                if float(info["events"]["stakeholder_collision_event"]) >= 1.0
                else "success"
            )
        pot_t1 = potential_state_from_v3_vehicles(
            info["vehicles_t1"],
            target_speeds,
            terminated=bool(terminated),
            truncated=bool(truncated),
            terminal_label=term_label,
        )
        mean_bd_t = compute_potential_breakdown(pot_t, "mean")
        mean_bd_t1 = compute_potential_breakdown(pot_t1, "mean")
        min_bd_t = compute_potential_breakdown(pot_t, "min")
        min_bd_t1 = compute_potential_breakdown(pot_t1, "min")

        base_map = {
            aid: float(info["components"][aid]["total_base_reward"]) for aid in ("A", "B")
        }
        for aid in ("A", "B"):
            if abs(base_map[aid] - float(base_rewards[aid])) > 1e-12:
                raise RuntimeError("env reward / components mismatch")

        mean_shaped = apply_pbrs_to_base_rewards(base_map, pot_t, pot_t1, "mean", pbrs)
        min_shaped = apply_pbrs_to_base_rewards(base_map, pot_t, pot_t1, "min", pbrs)

        for aid in ("A", "B"):
            if not active[aid]:
                continue
            comp = info["components"][aid]
            base = float(comp["total_base_reward"])
            scaled_mean = float(mean_shaped[aid].scaled_shaping_component)
            scaled_min = float(min_shaped[aid].scaled_shaping_component)
            learner_r, shaping = select_condition_reward(
                condition=reward_condition,
                base_reward=base,
                scaled_mean_shaping=scaled_mean,
                scaled_min_shaping=scaled_min,
            )
            exit_now = float(info["events"]["exit_event"].get(aid, 0.0)) >= 1.0
            controller_terminal = bool(exit_now or terminated)
            learner_completed = bool(exit_now or info["completion"].get(aid, False))
            decomp_err = _decomposition_error(comp, base)
            if decomp_err > 1e-12:
                raise RuntimeError(f"base decomposition error {decomp_err}")

            tr = ReplayTransition(
                observation=np.asarray(obs[aid], dtype=np.float64),
                action=int(actions[aid]),
                shaped_reward=float(learner_r),
                next_observation=None
                if controller_terminal
                else np.asarray(next_obs[aid], dtype=np.float64),
                terminated=bool(terminated),
                truncated=bool(truncated),
                controller_terminal=controller_terminal,
                learner_completed=learner_completed,
                action_mask=masks[aid],
                next_action_mask=None if controller_terminal else next_masks[aid],
                base_reward=base,
                shaping_component=float(shaping),
                reward_condition=reward_condition,
                episode_id=episode_id,
                step=int(info["policy_step"]),
                controller_id=aid,
                traffic_role=str(info["vehicles_t"][aid]["role"]),
            )
            # Validate against 27D / 3-action contract
            tr.validate(n_actions=N_ACTIONS, obs_dim=OBSERVATION_DIM)
            stored.append(tr)
            if store_in_learners and learners is not None:
                learners[aid].store_transition(tr)

            target_bd = compute_dqn_target(
                float(learner_r),
                controller_terminal=controller_terminal,
                truncated=bool(truncated),
                gamma=0.995,
                next_q_values=None if controller_terminal else np.zeros(N_ACTIONS),
                next_action_mask=None if controller_terminal else next_masks[aid],
                terminated=bool(terminated),
            )
            # For bootstrap scalar checks without a network, use zero Q (trace only).
            if not controller_terminal:
                target_bd = compute_dqn_target(
                    float(learner_r),
                    controller_terminal=False,
                    truncated=bool(truncated),
                    gamma=0.995,
                    next_q_values=np.zeros(N_ACTIONS),
                    next_action_mask=next_masks[aid],
                    terminated=bool(terminated),
                )

            row = {
                "episode_id": episode_id,
                "reward_condition": reward_condition,
                "policy_step": int(info["policy_step"]),
                "controller_id": aid,
                "action": int(actions[aid]),
                "action_mask": masks[aid].astype(bool).tolist(),
                "observation": np.asarray(obs[aid], dtype=np.float64).tolist(),
                "next_observation": None
                if tr.next_observation is None
                else tr.next_observation.tolist(),
                "next_action_mask": None
                if tr.next_action_mask is None
                else tr.next_action_mask.astype(bool).tolist(),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "controller_terminal": controller_terminal,
                "learner_completed": learner_completed,
                "term_reason": info["term_reason"],
                "progress_component": float(comp["progress_component"]),
                "exit_component": float(comp["exit_component"]),
                "collision_component": float(comp["collision_component"]),
                "hard_braking_cost": float(comp["hard_braking_cost"]),
                "hard_braking_component": float(comp["hard_braking_component"]),
                "policy_level_acceleration": float(comp["policy_level_acceleration"]),
                "base_reward": base,
                "mean_shaping_signal": float(mean_shaped[aid].shaping_signal),
                "min_shaping_signal": float(min_shaped[aid].shaping_signal),
                "scaled_mean_shaping": scaled_mean,
                "scaled_min_shaping": scaled_min,
                "shaping_component": float(shaping),
                "learner_reward": float(learner_r),
                "decomposition_error": decomp_err,
                "experiences_t": dict(mean_bd_t.stakeholder_experiences),
                "experiences_t1": dict(mean_bd_t1.stakeholder_experiences),
                "raw_mean_t": float(mean_bd_t.raw_potential),
                "raw_mean_t1": float(mean_bd_t1.raw_potential),
                "actual_mean_t": float(mean_bd_t.actual_potential),
                "actual_mean_t1": float(mean_bd_t1.actual_potential),
                "raw_min_t": float(min_bd_t.raw_potential),
                "raw_min_t1": float(min_bd_t1.raw_potential),
                "actual_min_t": float(min_bd_t.actual_potential),
                "actual_min_t1": float(min_bd_t1.actual_potential),
                "vehicles_t": {
                    sid: {
                        "route_position": float(info["vehicles_t"][sid]["route_position"]),
                        "speed": float(info["vehicles_t"][sid]["speed"]),
                        "realised_acceleration": float(
                            info["vehicles_t"][sid]["realised_acceleration"]
                        ),
                        "completed": bool(info["vehicles_t"][sid]["completed"]),
                        "active_on_road": bool(info["vehicles_t"][sid]["active_on_road"]),
                    }
                    for sid in STAKEHOLDER_ORDER
                },
                "vehicles_t1": {
                    sid: {
                        "route_position": float(info["vehicles_t1"][sid]["route_position"]),
                        "speed": float(info["vehicles_t1"][sid]["speed"]),
                        "realised_acceleration": float(
                            info["vehicles_t1"][sid]["realised_acceleration"]
                        ),
                        "completed": bool(info["vehicles_t1"][sid]["completed"]),
                        "active_on_road": bool(info["vehicles_t1"][sid]["active_on_road"]),
                    }
                    for sid in STAKEHOLDER_ORDER
                },
                "exit_event": {
                    k: float(v) for k, v in info["events"]["exit_event"].items()
                },
                "collision_pairs": list(info["events"]["collision_pairs"]),
                "target": float(target_bd.target),
                "bootstrap_multiplier": float(target_bd.bootstrap_multiplier),
                "observation_dim": int(obs[aid].shape[0]),
                "lambda_mean": float(pcfg.lambda_mean),
                "lambda_min": float(pcfg.lambda_min),
                "integration_test_only": True,
                "pbrs_parameters_final": False,
            }
            transitions.append(row)
            replay_rows.append(
                {
                    "episode_id": episode_id,
                    "reward_condition": reward_condition,
                    "controller_id": aid,
                    "policy_step": row["policy_step"],
                    "controller_terminal": controller_terminal,
                    "truncated": bool(truncated),
                    "terminated": bool(terminated),
                    "obs_dim": int(obs[aid].shape[0]),
                    "next_observation_is_none": tr.next_observation is None,
                    "learner_reward": float(learner_r),
                    "target": float(target_bd.target),
                }
            )

            if exit_now:
                active[aid] = False

        obs = next_obs
        if terminated or truncated:
            break
        if step_i + 1 >= len(scripted_actions):
            break

    return {
        "meta": meta.__dict__,
        "environment_class": ENVIRONMENT_CLASS,
        "observation_dimension": OBSERVATION_DIM,
        "reward_condition": reward_condition,
        "transitions": transitions,
        "replay_rows": replay_rows,
        "stored_transitions": stored,
        "n_physical_transitions": len({r["policy_step"] for r in transitions}),
        "final_active": dict(active),
        "pbrs_config": {
            "lambda_mean": pcfg.lambda_mean,
            "lambda_min": pcfg.lambda_min,
            "integration_test_only": True,
            "pbrs_parameters_final": False,
        },
    }


__all__ = [
    "DQN_TARGET_VERSION",
    "ENVIRONMENT_CLASS",
    "FinalRuntimeAssertion",
    "N_ACTIONS",
    "REWARD_VERSION",
    "assert_final_v3_runtime",
    "build_integration_learners",
    "potential_state_from_v3_vehicles",
    "runtime_assertion_from_locks",
    "run_final_v3_episode",
]
