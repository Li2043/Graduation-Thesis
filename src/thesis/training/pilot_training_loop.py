"""Stage 5B-0 bounded engineering pilot training loop."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from thesis.agents.action_masking import role_action_mask, validate_action_mask
from thesis.agents.independent_dqn_v2 import DQNConfig, IndependentDQNLearner
from thesis.agents.replay_buffer_v2 import ReplayTransition
from thesis.envs.final_observation import OBSERVATION_DIM
from thesis.envs.merge_env_candidate_v3 import HighLevelAction
from thesis.rewards.pbrs_v2 import (
    apply_pbrs_to_base_rewards,
    compute_potential_breakdown,
)
from thesis.rewards.pbrs_v2 import PBRSConfig
from thesis.training.final_lock_loader import FinalLockBundle
from thesis.training.final_reward_conditions import RewardConditionName, select_condition_reward
from thesis.training.final_v3_pipeline import (
    ENVIRONMENT_CLASS,
    potential_state_from_v3_vehicles,
)
from thesis.training.pilot_checkpoint import (
    atomic_torch_save,
    capture_global_rng_states,
    load_checkpoint,
    restore_global_rng_states,
)
from thesis.training.pilot_config import (
    PilotConfig,
    derive_run_seeds,
    epsilon_at_step,
)
from thesis.training.pilot_evaluation import run_isolated_evaluation
from thesis.training.pilot_ic_schedule import PilotICSchedule, build_env_for_block


class PilotEngineeringError(RuntimeError):
    """Category-A engineering failure — abort the affected run."""


@dataclass
class PilotRunDiagnostics:
    episode_trace: list[dict[str, Any]] = field(default_factory=list)
    transition_trace: list[dict[str, Any]] = field(default_factory=list)
    update_trace: list[dict[str, Any]] = field(default_factory=list)
    evaluation_trace: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_trace: list[dict[str, Any]] = field(default_factory=list)
    non_zero_shaping_count: int = 0
    max_abs_loss: float = 0.0
    max_abs_q: float = 0.0
    max_abs_target: float = 0.0
    max_decomp_error: float = 0.0
    illegal_action_count: int = 0
    nan_inf_count: int = 0
    target_syncs: dict[str, int] = field(default_factory=lambda: {"A": 0, "B": 0})


class PilotTrainer:
    """Persistent A/B learners with sustained env interaction for one condition×seed."""

    def __init__(
        self,
        bundle: FinalLockBundle,
        *,
        condition: RewardConditionName,
        pilot_seed: int,
        config: PilotConfig,
        checkpoint_dir: Path | None = None,
        write_traces: bool = True,
    ):
        config.validate()
        self.bundle = bundle
        self.condition = condition
        self.pilot_seed = int(pilot_seed)
        self.config = config
        self.seeds = derive_run_seeds(self.pilot_seed)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.write_traces = write_traces

        torch.manual_seed(self.seeds["torch"])
        np.random.seed(self.seeds["environment"] % (2**31 - 1))

        dqn = config.dqn
        self.learners = {
            "A": IndependentDQNLearner(
                "A",
                DQNConfig(
                    obs_dim=dqn.observation_dimension,
                    n_actions=dqn.action_count,
                    hidden_sizes=tuple(dqn.hidden_sizes),
                    learning_rate=dqn.learning_rate,
                    gamma=dqn.gamma,
                    replay_capacity=dqn.replay_capacity_per_controller,
                    batch_size=dqn.batch_size,
                    device=dqn.device,
                    reward_condition=condition,
                ),
                seed=self.seeds["learner_A"],
            ),
            "B": IndependentDQNLearner(
                "B",
                DQNConfig(
                    obs_dim=dqn.observation_dimension,
                    n_actions=dqn.action_count,
                    hidden_sizes=tuple(dqn.hidden_sizes),
                    learning_rate=dqn.learning_rate,
                    gamma=dqn.gamma,
                    replay_capacity=dqn.replay_capacity_per_controller,
                    batch_size=dqn.batch_size,
                    device=dqn.device,
                    reward_condition=condition,
                ),
                seed=self.seeds["learner_B"],
            ),
        }
        # Override replay RNG seeds for condition-independent derivation
        self.learners["A"].replay._rng = np.random.default_rng(self.seeds["replay_A"])
        self.learners["B"].replay._rng = np.random.default_rng(self.seeds["replay_B"])
        self.learners["A"].replay.seed = self.seeds["replay_A"]
        self.learners["B"].replay.seed = self.seeds["replay_B"]

        self.schedule = PilotICSchedule(bundle, schedule_seed=self.seeds["ic_schedule"])
        self.pbrs = PBRSConfig(
            learner_gamma=config.dqn.gamma,
            shaping_gamma=config.dqn.gamma,
            lambda_mean=float(config.pbrs.lambda_mean),
            lambda_min=float(config.pbrs.lambda_min),
        )
        self.pbrs.validate()
        # PilotConfig.pbrs carries pilot_only / pbrs_parameters_final status flags.

        self.env_steps = 0
        self.episode_count = 0
        self.epsilon_env_steps = 0
        self.diag = PilotRunDiagnostics()
        self._env = None
        self._obs = None
        self._active = {"A": True, "B": True}
        self._episode_open = False
        self._current_block_id = ""
        self._current_assignment = 0
        self.pilot_training_started = False
        self.policy_training_started = False
        self.sustained_training_invoked = False

    def current_epsilon(self) -> float:
        return epsilon_at_step(self.epsilon_env_steps, self.config.exploration)

    def _require_finite(self, name: str, value: float) -> float:
        v = float(value)
        if not math.isfinite(v):
            self.diag.nan_inf_count += 1
            raise PilotEngineeringError(f"non-finite {name}: {v}")
        return v

    def _start_episode(self) -> None:
        block, assignment, block_id = self.schedule.peek()
        self._current_block_id = block_id
        self._current_assignment = assignment
        self._env = build_env_for_block(self.bundle, block)
        if type(self._env).__name__ != ENVIRONMENT_CLASS:
            raise PilotEngineeringError("pilot must use MergeEnvCandidateV3")
        self._obs, _ = self._env.reset(seed=int(block.seed))
        for aid in ("A", "B"):
            if self._obs[aid].shape != (OBSERVATION_DIM,):
                raise PilotEngineeringError("observation must be 27D")
        self._active = {"A": True, "B": True}
        self._episode_open = True
        self.episode_count += 1

    def _end_episode(self, info: dict[str, Any], terminated: bool, truncated: bool) -> None:
        self.diag.episode_trace.append(
            {
                "episode": self.episode_count,
                "block_id": self._current_block_id,
                "assignment": self._current_assignment,
                "env_steps": self.env_steps,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "term_reason": info.get("term_reason"),
            }
        )
        self.schedule.advance()
        self._episode_open = False
        self._env = None
        self._obs = None

    def _build_transition(
        self,
        aid: str,
        action: int,
        mask: np.ndarray,
        next_obs: dict,
        next_mask: np.ndarray,
        base: float,
        shaping: float,
        learner_r: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
        exit_now: bool,
    ) -> ReplayTransition:
        controller_terminal = bool(exit_now or terminated)
        learner_completed = bool(exit_now or info["completion"].get(aid, False))
        return ReplayTransition(
            observation=np.asarray(self._obs[aid], dtype=np.float64),
            action=int(action),
            shaped_reward=float(learner_r),
            next_observation=None
            if controller_terminal
            else np.asarray(next_obs[aid], dtype=np.float64),
            terminated=bool(terminated),
            truncated=bool(truncated),
            controller_terminal=controller_terminal,
            learner_completed=learner_completed,
            action_mask=mask,
            next_action_mask=None if controller_terminal else next_mask,
            base_reward=float(base),
            shaping_component=float(shaping),
            reward_condition=self.condition,
            episode_id=f"{self.condition}_{self.pilot_seed}_{self.episode_count}",
            step=int(info["policy_step"]),
            controller_id=aid,
            traffic_role=str(info["vehicles_t"][aid]["role"]),
        )

    def step_once(self) -> dict[str, Any]:
        if not self._episode_open:
            self._start_episode()
        assert self._env is not None and self._obs is not None

        self.pilot_training_started = True
        self.policy_training_started = True
        self.sustained_training_invoked = True

        eps = self.current_epsilon()
        actions: dict[str, int] = {}
        masks: dict[str, np.ndarray] = {}
        for aid in ("A", "B"):
            role = str(self._env._vehicles[aid].role)  # noqa: SLF001
            mask = validate_action_mask(role_action_mask(role, 3), 3)
            masks[aid] = mask
            if not self._active[aid]:
                actions[aid] = int(HighLevelAction.MAINTAIN)
            else:
                a = self.learners[aid].select_action(
                    self._obs[aid], mask, epsilon=eps, greedy=False
                )
                if not bool(mask[a]):
                    self.diag.illegal_action_count += 1
                    raise PilotEngineeringError(f"illegal action {a} for {aid}")
                actions[aid] = a

        next_obs, base_rewards, terminated, truncated, info = self._env.step(actions)
        target_speeds = self._env.config.block.target_speeds.as_map()
        pot_t = potential_state_from_v3_vehicles(
            info["vehicles_t"], target_speeds, terminated=False, truncated=False
        )
        pot_t1 = potential_state_from_v3_vehicles(
            info["vehicles_t1"],
            target_speeds,
            terminated=bool(terminated),
            truncated=bool(truncated),
            terminal_label=info.get("term_reason"),
        )
        base_map = {
            aid: float(info["components"][aid]["total_base_reward"]) for aid in ("A", "B")
        }
        mean_shaped = apply_pbrs_to_base_rewards(
            base_map, pot_t, pot_t1, "mean", self.pbrs
        )
        min_shaped = apply_pbrs_to_base_rewards(
            base_map, pot_t, pot_t1, "min", self.pbrs
        )

        step_info: dict[str, Any] = {
            "env_step": self.env_steps + 1,
            "epsilon": eps,
            "actions": dict(actions),
            "updates": {},
        }

        for aid in ("A", "B"):
            if not self._active[aid]:
                continue
            comp = info["components"][aid]
            base = float(comp["total_base_reward"])
            recon = (
                float(comp["progress_component"])
                + float(comp["exit_component"])
                + float(comp["collision_component"])
                + float(comp["hard_braking_component"])
            )
            decomp_err = abs(recon - base)
            self.diag.max_decomp_error = max(self.diag.max_decomp_error, decomp_err)
            if decomp_err > 1e-12:
                raise PilotEngineeringError(f"decomposition error {decomp_err}")

            scaled_mean = float(mean_shaped[aid].scaled_shaping_component)
            scaled_min = float(min_shaped[aid].scaled_shaping_component)
            learner_r, shaping = select_condition_reward(
                condition=self.condition,
                base_reward=base,
                scaled_mean_shaping=scaled_mean,
                scaled_min_shaping=scaled_min,
            )
            self._require_finite(f"{aid}.learner_reward", learner_r)
            self._require_finite(f"{aid}.shaping", shaping)
            if abs(shaping) > 0.0:
                self.diag.non_zero_shaping_count += 1
            if self.condition == "baseline" and shaping != 0.0:
                raise PilotEngineeringError("baseline shaping must be exactly zero")

            exit_now = float(info["events"]["exit_event"].get(aid, 0.0)) >= 1.0
            next_role = str(info["vehicles_t1"][aid]["role"])
            next_mask = validate_action_mask(role_action_mask(next_role, 3), 3)
            tr = self._build_transition(
                aid,
                actions[aid],
                masks[aid],
                next_obs,
                next_mask,
                base,
                shaping,
                learner_r,
                terminated,
                truncated,
                info,
                exit_now,
            )
            tr.validate(n_actions=3, obs_dim=OBSERVATION_DIM)
            self.learners[aid].store_transition(tr)

            if self.write_traces:
                self.diag.transition_trace.append(
                    {
                        "env_step": self.env_steps + 1,
                        "controller_id": aid,
                        "action": int(actions[aid]),
                        "epsilon": eps,
                        "base_reward": base,
                        "shaping_component": shaping,
                        "learner_reward": learner_r,
                        "controller_terminal": tr.controller_terminal,
                        "truncated": tr.truncated,
                        "terminated": tr.terminated,
                        "block_id": self._current_block_id,
                        "assignment": self._current_assignment,
                        "obs_dim": 27,
                    }
                )

            # Optimiser update after warmup
            warm = self.config.dqn.replay_warmup_per_controller
            upd_stats = None
            if len(self.learners[aid].replay) >= warm:
                batch = self.learners[aid].replay.sample(self.config.dqn.batch_size)
                batch_hash = hash(tuple(int(i) for i in batch.indices.tolist()))
                upd_stats = self.learners[aid].update(batch)
                loss = self._require_finite(f"{aid}.loss", upd_stats["loss"])
                self.diag.max_abs_loss = max(self.diag.max_abs_loss, abs(loss))
                self.diag.max_abs_q = max(
                    self.diag.max_abs_q, abs(float(upd_stats["mean_q_sa"]))
                )
                self.diag.max_abs_target = max(
                    self.diag.max_abs_target, abs(float(upd_stats["mean_target"]))
                )
                # Target sync
                if (
                    self.learners[aid]._update_count
                    % self.config.dqn.target_sync_interval_updates
                    == 0
                ):
                    self.learners[aid].hard_sync_target()
                    self.diag.target_syncs[aid] += 1
                if self.write_traces:
                    self.diag.update_trace.append(
                        {
                            "condition": self.condition,
                            "pilot_seed": self.pilot_seed,
                            "controller": aid,
                            "environment_step": self.env_steps + 1,
                            "learner_update_count": upd_stats["update_count"],
                            "replay_size": len(self.learners[aid].replay),
                            "batch_hash": batch_hash,
                            "epsilon": eps,
                            "loss": loss,
                            "mean_q_sa": float(upd_stats["mean_q_sa"]),
                            "mean_target": float(upd_stats["mean_target"]),
                            "gradient_finite": True,
                            "parameter_finite": True,
                            "target_network_forward_count": int(
                                upd_stats["target_network_forward_calls"]
                            ),
                            "n_bootstrap_rows": int(upd_stats["n_bootstrap_rows"]),
                            "n_controller_terminal_rows": int(upd_stats["n_terminal_rows"]),
                            "target_sync_event": (
                                self.learners[aid]._update_count
                                % self.config.dqn.target_sync_interval_updates
                                == 0
                            ),
                        }
                    )
            step_info["updates"][aid] = upd_stats

            if exit_now:
                self._active[aid] = False

        self._obs = next_obs
        self.env_steps += 1
        self.epsilon_env_steps += 1

        if terminated or truncated:
            self._end_episode(info, terminated, truncated)

        return step_info

    def maybe_checkpoint(self, step: int) -> Path | None:
        if self.checkpoint_dir is None:
            return None
        if step not in self.config.duration.checkpoint_steps:
            return None
        path = self.checkpoint_dir / f"ckpt_step_{step:05d}.pt"
        payload = self.export_checkpoint(step=step)
        atomic_torch_save(path, payload)
        self.diag.checkpoint_trace.append(
            {"step": step, "path": str(path), "atomic": True}
        )
        return path

    def maybe_evaluate(self, step: int) -> dict[str, Any] | None:
        if step not in self.config.duration.evaluation_steps:
            return None
        result = run_isolated_evaluation(
            self.bundle,
            self.learners,
            eval_seed=self.seeds["evaluation"] + step,
        )
        result["environment_step"] = step
        if result["mutation"]["any"]:
            raise PilotEngineeringError("evaluation mutated training state")
        self.diag.evaluation_trace.append(
            {
                "step": step,
                "n_episodes": result["n_episodes"],
                "mutation_any": result["mutation"]["any"],
            }
        )
        return result

    def export_checkpoint(self, *, step: int) -> dict[str, Any]:
        env_state = None
        if self._env is not None:
            env_state = {
                "rng": self._env._rng.bit_generator.state,  # noqa: SLF001
                "policy_step": int(self._env._policy_step),  # noqa: SLF001
                "episode_open": True,
                "active": dict(self._active),
                "block_id": self._current_block_id,
                "assignment": self._current_assignment,
                # Full vehicle snapshot for exact resume is heavy; for resume
                # equivalence we restart mid-run via trainer state restore that
                # reloads learners/schedule and continues with fresh episode
                # boundaries. Exact mid-episode resume stores obs + vehicles.
                "obs": {
                    aid: np.asarray(self._obs[aid], dtype=np.float64)
                    for aid in ("A", "B")
                }
                if self._obs is not None
                else None,
                "vehicles": {
                    sid: deepcopy_vehicle(self._env._vehicles[sid])  # noqa: SLF001
                    for sid in self._env._vehicles  # noqa: SLF001
                },
                "exit_count": dict(self._env._exit_count),  # noqa: SLF001
                "exit_time": dict(self._env._exit_time),  # noqa: SLF001
                "exit_substep": dict(self._env._exit_substep),  # noqa: SLF001
            }
        return {
            "condition": self.condition,
            "pilot_seed": self.pilot_seed,
            "environment_lock_hash": self.bundle.environment_lock_sha256_before,
            "comfort_lock_hash": self.bundle.comfort_lock_sha256_before,
            "pilot_config_hash": self.config.sha256(),
            "env_steps": int(self.env_steps),
            "epsilon_env_steps": int(self.epsilon_env_steps),
            "episode_count": int(self.episode_count),
            "episode_open": bool(self._episode_open),
            "current_block_id": self._current_block_id,
            "current_assignment": self._current_assignment,
            "ic_schedule": self.schedule.export_state(),
            "learners": {
                "A": self.learners["A"].export_state(),
                "B": self.learners["B"].export_state(),
            },
            "target_syncs": dict(self.diag.target_syncs),
            "epsilon": self.current_epsilon(),
            "global_rng": capture_global_rng_states(),
            "env_state": env_state,
            "diagnostics": {
                "non_zero_shaping_count": self.diag.non_zero_shaping_count,
                "max_abs_loss": self.diag.max_abs_loss,
                "max_decomp_error": self.diag.max_decomp_error,
                "illegal_action_count": self.diag.illegal_action_count,
                "nan_inf_count": self.diag.nan_inf_count,
            },
            "checkpoint_step": int(step),
        }

    def import_checkpoint(self, payload: dict[str, Any]) -> None:
        if payload["condition"] != self.condition:
            raise PilotEngineeringError("condition mismatch on resume")
        if int(payload["pilot_seed"]) != self.pilot_seed:
            raise PilotEngineeringError("pilot_seed mismatch on resume")
        self.learners["A"].import_state(payload["learners"]["A"])
        self.learners["B"].import_state(payload["learners"]["B"])
        self.schedule.import_state(payload["ic_schedule"])
        self.env_steps = int(payload["env_steps"])
        self.epsilon_env_steps = int(payload["epsilon_env_steps"])
        self.episode_count = int(payload["episode_count"])
        self.diag.target_syncs = dict(payload["target_syncs"])
        restore_global_rng_states(payload["global_rng"])
        self._episode_open = bool(payload["episode_open"])
        self._current_block_id = str(payload["current_block_id"])
        self._current_assignment = int(payload["current_assignment"])
        env_state = payload.get("env_state")
        if self._episode_open and env_state is not None:
            block = self.schedule.materialize(
                self._current_block_id, self._current_assignment
            )
            self._env = build_env_for_block(self.bundle, block)
            # Restore internal env state
            self._env.reset(seed=int(block.seed))
            restore_vehicles(self._env, env_state)
            self._env._rng.bit_generator.state = env_state["rng"]  # noqa: SLF001
            self._env._policy_step = int(env_state["policy_step"])  # noqa: SLF001
            self._active = dict(env_state["active"])
            self._obs = {
                aid: np.asarray(env_state["obs"][aid], dtype=np.float64)
                for aid in ("A", "B")
            }
        else:
            self._env = None
            self._obs = None
            self._episode_open = False
        self.pilot_training_started = True
        self.policy_training_started = True
        self.sustained_training_invoked = True

    def run(
        self,
        *,
        n_steps: int | None = None,
        start_step: int = 0,
        on_step: Callable[[int, dict[str, Any]], None] | None = None,
    ) -> PilotRunDiagnostics:
        target = int(
            n_steps if n_steps is not None else self.config.duration.environment_steps_per_run
        )
        # Evaluate at 0 before any training steps when starting fresh
        if start_step == 0 and 0 in self.config.duration.evaluation_steps:
            self.maybe_evaluate(0)
        while self.env_steps < target:
            info = self.step_once()
            step = self.env_steps
            self.maybe_checkpoint(step)
            self.maybe_evaluate(step)
            if on_step is not None:
                on_step(step, info)
        if self.env_steps != target:
            raise PilotEngineeringError(
                f"run ended at {self.env_steps} steps, expected {target}"
            )
        return self.diag


def deepcopy_vehicle(veh):
    from copy import deepcopy

    return deepcopy(veh)


def restore_vehicles(env, env_state: dict[str, Any]) -> None:
    from copy import deepcopy

    env._vehicles = {k: deepcopy(v) for k, v in env_state["vehicles"].items()}
    env._exit_count = dict(env_state["exit_count"])
    env._exit_time = dict(env_state["exit_time"])
    env._exit_substep = dict(env_state["exit_substep"])


__all__ = [
    "PilotEngineeringError",
    "PilotRunDiagnostics",
    "PilotTrainer",
    "load_checkpoint",
]
