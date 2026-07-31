"""Single-job FormalTrainer for Stage 6A-0 (authoritative formal runtime)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from thesis.agents.action_masking import role_action_mask, validate_action_mask
from thesis.agents.dqn_bootstrap import DQNTargetMode
from thesis.agents.independent_dqn_v2 import DQNConfig, IndependentDQNLearner
from thesis.agents.replay_buffer_v2 import ReplayTransition
from thesis.envs.final_observation import OBSERVATION_DIM
from thesis.envs.merge_env_candidate_v3 import HighLevelAction
from thesis.formal.formal_config import FormalConfig, assert_condition, epsilon_at_step
from thesis.formal.formal_evaluation import run_formal_isolated_evaluation
from thesis.formal.formal_schedule import FormalICSchedule
from thesis.rewards.pbrs_v2 import PBRSConfig, apply_pbrs_to_base_rewards
from thesis.training.final_lock_loader import FinalLockBundle
from thesis.training.final_reward_conditions import select_condition_reward
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
from thesis.training.pilot_ic_schedule import build_env_for_block
from thesis.training.pilot_training_loop import deepcopy_vehicle, restore_vehicles


class FormalEngineeringError(RuntimeError):
    """Numerical / integrity failure — FAILED_WITH_REASON."""


@dataclass
class FormalRunDiagnostics:
    episode_trace: list[dict[str, Any]] = field(default_factory=list)
    update_trace: list[dict[str, Any]] = field(default_factory=list)
    evaluation_trace: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_trace: list[dict[str, Any]] = field(default_factory=list)
    non_zero_shaping_count: int = 0
    max_abs_loss: float = 0.0
    max_decomp_error: float = 0.0
    illegal_action_count: int = 0
    nan_inf_count: int = 0
    target_syncs: dict[str, int] = field(default_factory=lambda: {"A": 0, "B": 0})


class FormalTrainer:
    """One condition × master_seed formal job — single MergeEnvCandidateV3."""

    def __init__(
        self,
        bundle: FinalLockBundle,
        *,
        condition: str,
        master_seed: int,
        seeds: dict[str, int],
        config: FormalConfig,
        checkpoint_dir: Path | None = None,
        protocol_hash: str = "",
        target_mode: str | DQNTargetMode = DQNTargetMode.VANILLA,
        algorithm_condition: str | None = None,
    ):
        config.validate()
        self.bundle = bundle
        self.condition = assert_condition(condition)
        self.master_seed = int(master_seed)
        self.seeds = dict(seeds)
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.protocol_hash = protocol_hash
        self.target_mode = DQNTargetMode(target_mode)
        # Algorithm label for pilots (vanilla_dqn / double_dqn); reward condition stays separate
        self.algorithm_condition = (
            str(algorithm_condition)
            if algorithm_condition is not None
            else self.target_mode.value
        )

        required = (
            "environment_seed",
            "learner_A_seed",
            "learner_B_seed",
            "replay_A_seed",
            "replay_B_seed",
            "evaluation_seed",
            "schedule_seed",
        )
        for k in required:
            if k not in self.seeds:
                raise FormalEngineeringError(f"missing formal seed {k}")

        torch.manual_seed(self.seeds["learner_A_seed"])
        np.random.seed(self.seeds["environment_seed"] % (2**31 - 1))

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
                    reward_condition=self.condition,
                    target_mode=self.target_mode,
                ),
                seed=self.seeds["learner_A_seed"],
                replay_seed=self.seeds["replay_A_seed"],
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
                    reward_condition=self.condition,
                    target_mode=self.target_mode,
                ),
                seed=self.seeds["learner_B_seed"],
                replay_seed=self.seeds["replay_B_seed"],
            ),
        }
        if self.learners["A"].replay.seed != self.seeds["replay_A_seed"]:
            raise FormalEngineeringError("learner_A.replay.seed mismatch")
        if self.learners["B"].replay.seed != self.seeds["replay_B_seed"]:
            raise FormalEngineeringError("learner_B.replay.seed mismatch")

        self.schedule = FormalICSchedule(
            bundle, schedule_seed=self.seeds["schedule_seed"]
        )
        self.pbrs = PBRSConfig(
            learner_gamma=config.pbrs.gamma,
            shaping_gamma=config.pbrs.gamma,
            lambda_mean=float(config.pbrs.lambda_mean),
            lambda_min=float(config.pbrs.lambda_min),
        )
        self.pbrs.validate()

        self.env_steps = 0
        self.episode_count = 0
        self.epsilon_env_steps = 0
        self.diag = FormalRunDiagnostics()
        self._env = None
        self._obs = None
        self._active = {"A": True, "B": True}
        self._episode_open = False
        self._current_block_id = ""
        self._current_assignment = 0
        self.formal_training_started = False

    def current_epsilon(self) -> float:
        return epsilon_at_step(self.epsilon_env_steps, self.config.exploration)

    def _require_finite(self, name: str, value: float) -> float:
        v = float(value)
        if not math.isfinite(v):
            self.diag.nan_inf_count += 1
            raise FormalEngineeringError(f"non-finite {name}: {v}")
        return v

    def _start_episode(self) -> None:
        block, assignment, block_id = self.schedule.peek()
        self._current_block_id = block_id
        self._current_assignment = assignment
        self._env = build_env_for_block(
            self.bundle, block, max_policy_steps=self.config.duration.max_policy_steps
        )
        if type(self._env).__name__ != ENVIRONMENT_CLASS:
            raise FormalEngineeringError("must use MergeEnvCandidateV3")
        # One training environment object per run — never vectorized
        self._obs, _ = self._env.reset(seed=int(self.seeds["environment_seed"]) + self.episode_count)
        for aid in ("A", "B"):
            if self._obs[aid].shape != (OBSERVATION_DIM,):
                raise FormalEngineeringError("observation must be 27D")
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
            episode_id=f"{self.condition}_{self.master_seed}_{self.episode_count}",
            step=int(info["policy_step"]),
            controller_id=aid,
            traffic_role=str(info["vehicles_t"][aid]["role"]),
        )

    def step_once(self) -> dict[str, Any]:
        if not self._episode_open:
            self._start_episode()
        assert self._env is not None and self._obs is not None
        self.formal_training_started = True

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
                    raise FormalEngineeringError(f"illegal action {a} for {aid}")
                actions[aid] = a

        next_obs, _rewards, terminated, truncated, info = self._env.step(actions)
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
                raise FormalEngineeringError(f"decomposition error {decomp_err}")

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
                raise FormalEngineeringError("baseline shaping must be exactly zero")

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

            warm = self.config.dqn.replay_warmup_per_controller
            upd_stats = None
            if len(self.learners[aid].replay) >= warm:
                batch = self.learners[aid].replay.sample(self.config.dqn.batch_size)
                upd_stats = self.learners[aid].update(batch)
                loss = self._require_finite(f"{aid}.loss", upd_stats["loss"])
                self.diag.max_abs_loss = max(self.diag.max_abs_loss, abs(loss))
                if (
                    self.learners[aid]._update_count
                    % self.config.dqn.target_sync_interval_updates
                    == 0
                ):
                    self.learners[aid].hard_sync_target()
                    self.diag.target_syncs[aid] += 1
                self.diag.update_trace.append(
                    {
                        "condition": self.condition,
                        "master_seed": self.master_seed,
                        "controller": aid,
                        "environment_step": self.env_steps + 1,
                        "loss": loss,
                        "replay_seed": int(self.learners[aid].replay.seed),
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

    def _eval_checkpoint_index(self, step: int) -> int:
        steps = list(self.config.duration.evaluation_steps)
        return steps.index(int(step))

    def maybe_checkpoint(self, step: int) -> Path | None:
        if self.checkpoint_dir is None:
            return None
        if step not in self.config.duration.checkpoint_steps:
            return None
        path = self.checkpoint_dir / f"ckpt_step_{step:06d}.pt"
        payload = self.export_checkpoint(step=step)
        atomic_torch_save(path, payload)
        self.diag.checkpoint_trace.append({"step": step, "path": str(path)})
        return path

    def maybe_evaluate(self, step: int) -> dict[str, Any] | None:
        if step not in self.config.duration.evaluation_steps:
            return None
        idx = self._eval_checkpoint_index(step)
        result = run_formal_isolated_evaluation(
            self.bundle,
            self.learners,
            evaluation_seed=self.seeds["evaluation_seed"],
            checkpoint_index=idx,
            max_policy_steps=self.config.duration.max_policy_steps,
        )
        result["environment_step"] = step
        if result["mutation"]["any"]:
            raise FormalEngineeringError("evaluation mutated training state")
        self.diag.evaluation_trace.append(
            {
                "step": step,
                "checkpoint_index": idx,
                "n_episodes": result["n_episodes"],
                "mutation_any": False,
            }
        )
        return result

    def export_checkpoint(self, *, step: int) -> dict[str, Any]:
        env_state = None
        if self._env is not None and self._obs is not None:
            env_state = {
                "rng": self._env._rng.bit_generator.state,  # noqa: SLF001
                "policy_step": int(self._env._policy_step),  # noqa: SLF001
                "active": dict(self._active),
                "obs": {
                    aid: np.asarray(self._obs[aid], dtype=np.float64) for aid in ("A", "B")
                },
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
            "algorithm_condition": self.algorithm_condition,
            "algorithm_mode": self.target_mode.value,
            "target_mode": self.target_mode.value,
            "master_seed": self.master_seed,
            "seeds": dict(self.seeds),
            "replay_seeds_actual": {
                "A": int(self.learners["A"].replay.seed),
                "B": int(self.learners["B"].replay.seed),
            },
            "protocol_hash": self.protocol_hash,
            "environment_lock_hash": self.bundle.environment_lock_sha256_before,
            "comfort_lock_hash": self.bundle.comfort_lock_sha256_before,
            "formal_config_hash": self.config.sha256(),
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
            "global_rng": capture_global_rng_states(),
            "env_state": env_state,
            "checkpoint_step": int(step),
            "num_envs": 1,
            "vectorized": False,
        }

    def import_checkpoint(self, payload: dict[str, Any]) -> None:
        if payload["condition"] != self.condition:
            raise FormalEngineeringError("condition mismatch on resume")
        payload_mode = payload.get("algorithm_mode") or payload.get("target_mode")
        if payload_mode is not None and str(payload_mode) != self.target_mode.value:
            raise FormalEngineeringError(
                f"algorithm mode mismatch on resume: "
                f"checkpoint={payload_mode!r} trainer={self.target_mode.value!r}"
            )
        payload_algo = payload.get("algorithm_condition")
        if payload_algo is not None and str(payload_algo) != self.algorithm_condition:
            raise FormalEngineeringError(
                f"algorithm condition mismatch on resume: "
                f"checkpoint={payload_algo!r} trainer={self.algorithm_condition!r}"
            )
        if int(payload["master_seed"]) != self.master_seed:
            raise FormalEngineeringError("master_seed mismatch on resume")
        required_learner_keys = ("optimiser", "replay", "learner_rng", "online", "target")
        for aid in ("A", "B"):
            la = payload.get("learners", {}).get(aid, {})
            for k in required_learner_keys:
                if k not in la:
                    raise FormalEngineeringError(
                        f"checkpoint missing resumable field learners[{aid}].{k}"
                    )
        if "ic_schedule" not in payload or "global_rng" not in payload:
            raise FormalEngineeringError("checkpoint missing schedule/RNG state")
        self.learners["A"].import_state(payload["learners"]["A"])
        self.learners["B"].import_state(payload["learners"]["B"])
        if int(self.learners["A"].replay.seed) != self.seeds["replay_A_seed"]:
            raise FormalEngineeringError("replay_A seed lost on resume")
        if int(self.learners["B"].replay.seed) != self.seeds["replay_B_seed"]:
            raise FormalEngineeringError("replay_B seed lost on resume")
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
            self._env = build_env_for_block(
                self.bundle, block, max_policy_steps=self.config.duration.max_policy_steps
            )
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
        self.formal_training_started = True

    def run(
        self,
        *,
        n_steps: int | None = None,
        on_step: Callable[[int, dict[str, Any]], None] | None = None,
    ) -> FormalRunDiagnostics:
        target = int(
            n_steps
            if n_steps is not None
            else self.config.duration.environment_steps_per_run
        )
        start = self.env_steps
        if start == 0 and 0 in self.config.duration.evaluation_steps:
            self.maybe_evaluate(0)
        while self.env_steps < target:
            info = self.step_once()
            step = self.env_steps
            self.maybe_checkpoint(step)
            self.maybe_evaluate(step)
            if on_step is not None:
                on_step(step, info)
        if self.env_steps != target:
            raise FormalEngineeringError(
                f"run ended at {self.env_steps}, expected {target}"
            )
        return self.diag

    def write_job_manifest(self, path: Path, *, status: str, reason: str = "") -> None:
        payload = {
            "condition": self.condition,
            "master_seed": self.master_seed,
            "seeds": self.seeds,
            "replay_seeds_actual": {
                "A": int(self.learners["A"].replay.seed),
                "B": int(self.learners["B"].replay.seed),
            },
            "env_steps": self.env_steps,
            "status": status,
            "reason": reason,
            "protocol_hash": self.protocol_hash,
            "num_parallel_training_envs_per_run": 1,
            "vectorized_training": False,
            "formal_training_started": bool(self.formal_training_started),
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = [
    "FormalEngineeringError",
    "FormalRunDiagnostics",
    "FormalTrainer",
    "load_checkpoint",
]
