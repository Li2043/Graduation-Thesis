"""Deterministic reconstruction of evaluation episodes from final weights.

Stage 6A published only evaluation *summaries* (n_episodes). Episode-level
endpoint fields are reconstructed at the preregistered primary endpoint
(step 100000) by re-running the locked greedy evaluation protocol on published
final online networks. This is not policy training.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from thesis.agents.action_masking import role_action_mask, validate_action_mask
from thesis.agents.independent_dqn_v2 import DQNConfig, IndependentDQNLearner
from thesis.analysis.endpoints import (
    classify_convention,
    episode_stakeholder_utilities,
)
from thesis.envs.merge_env_candidate_v3 import HighLevelAction
from thesis.formal.formal_schedule import evaluation_episode_seed
from thesis.rewards.pbrs_v2 import STAKEHOLDER_ORDER, compute_stakeholder_experiences
from thesis.training.final_lock_loader import FinalLockBundle, load_final_locks
from thesis.training.final_v3_pipeline import potential_state_from_v3_vehicles
from thesis.training.pilot_ic_schedule import (
    build_env_for_block,
    validation_blocks_with_assignments,
)
from thesis.audits.audit_metrics import discounted_return


GAMMA = 0.995
CHECKPOINT_INDEX_100K = 5  # EVALUATION_STEPS index of 100000


def _experience(speed: float, target: float, completed: bool) -> float:
    if completed:
        return 1.0
    if target <= 0:
        return 0.0
    return float(min(1.0, max(0.0, speed / target)))


def load_learners_from_final_weights(
    weights_path: Path,
    *,
    condition: str,
) -> dict[str, IndependentDQNLearner]:
    payload = torch.load(Path(weights_path), map_location="cpu", weights_only=False)
    cfg = DQNConfig(
        obs_dim=27,
        n_actions=3,
        hidden_sizes=(64, 64),
        reward_condition=condition,  # type: ignore[arg-type]
    )
    learners = {
        "A": IndependentDQNLearner("A", cfg, seed=0, replay_seed=0),
        "B": IndependentDQNLearner("B", cfg, seed=1, replay_seed=1),
    }
    learners["A"].online.load_state_dict(payload["A_online"])
    learners["A"].target.load_state_dict(payload["A_target"])
    learners["B"].online.load_state_dict(payload["B_online"])
    learners["B"].target.load_state_dict(payload["B_target"])
    learners["A"].online.eval()
    learners["B"].online.eval()
    return learners


def run_instrumented_evaluation_episode(
    bundle: FinalLockBundle,
    learners: dict[str, IndependentDQNLearner],
    *,
    block,
    assignment: int,
    block_id: str,
    block_index: int,
    episode_seed: int,
    max_policy_steps: int = 400,
) -> dict[str, Any]:
    env = build_env_for_block(bundle, block, max_policy_steps=max_policy_steps)
    obs, _ = env.reset(seed=int(episode_seed))
    target_speeds = env.config.block.target_speeds.as_map()

    base_returns: dict[str, list[float]] = {"A": [], "B": []}
    learner_returns: dict[str, list[float]] = {"A": [], "B": []}
    hard_brake_events = 0
    hard_brake_steps = 0
    min_gap = math.inf
    min_ttc = math.inf
    bg_max_brake = 0.0
    steps = 0
    term_reason = "ongoing"
    terminated = False
    truncated = False
    info: dict[str, Any] = {}

    while True:
        actions: dict[str, int] = {}
        for aid in ("A", "B"):
            role = str(env._vehicles[aid].role)  # noqa: SLF001
            mask = validate_action_mask(role_action_mask(role, 3), 3)
            if not env._vehicles[aid].active_on_road:  # noqa: SLF001
                actions[aid] = int(HighLevelAction.MAINTAIN)
            else:
                actions[aid] = learners[aid].select_action(obs[aid], mask, greedy=True)
        obs, rewards, terminated, truncated, info = env.step(actions)
        steps += 1
        hard_brake_steps += 1
        for aid in ("A", "B"):
            base = float(info["components"][aid]["total_base_reward"])
            base_returns[aid].append(base)
            learner_returns[aid].append(float(rewards[aid]))
            if float(info["components"][aid].get("hard_braking_component", 0.0)) < 0.0:
                hard_brake_events += 1
        gap = info.get("min_bumper_gap")
        if gap is not None and math.isfinite(float(gap)):
            min_gap = min(min_gap, float(gap))
        ttc = info.get("ttc")
        if ttc is not None and math.isfinite(float(ttc)):
            min_ttc = min(min_ttc, float(ttc))
        for bid in ("B_front", "B_rear"):
            acc = float(info["vehicles_t1"][bid]["realised_acceleration"])
            bg_max_brake = max(bg_max_brake, max(0.0, -acc))
        term_reason = str(info["term_reason"])
        if terminated or truncated:
            break

    # Final stakeholder experiences / utilities
    pot = potential_state_from_v3_vehicles(
        info["vehicles_t1"],
        target_speeds,
        terminated=bool(terminated),
        truncated=bool(truncated),
        terminal_label=term_reason,
    )
    experiences = compute_stakeholder_experiences(pot.stakeholders)
    utilities = episode_stakeholder_utilities(experiences)
    roles = {aid: str(info["vehicles_t1"][aid]["role"]) for aid in ("A", "B")}
    success = term_reason == "success"
    collision = term_reason == "collision" or float(
        info["events"]["stakeholder_collision_event"]
    ) >= 1.0
    convention = classify_convention(
        success=success,
        exit_time=info["exit_time"],
        roles=roles,
    )
    collision_type = "none"
    pairs = info["events"].get("collision_pairs") or []
    if pairs:
        collision_type = ",".join(f"{a}-{b}" for a, b in pairs)

    return {
        "block_id": block_id,
        "assignment": int(assignment),
        "block_index": int(block_index),
        "episode_seed": int(episode_seed),
        "steps": int(steps),
        "episode_length": int(steps),
        "term_reason": term_reason,
        "success": bool(success),
        "collision": bool(collision),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "roles": roles,
        "exit_time": {k: (None if v is None else int(v)) for k, v in info["exit_time"].items()},
        "convention": convention,
        "stakeholder_utilities": utilities,
        "experiences": experiences,
        "learner_A_utility": utilities["A"],
        "learner_B_utility": utilities["B"],
        "B_front_utility": utilities["B_front"],
        "B_rear_utility": utilities["B_rear"],
        "worst_off_stakeholder_identity": min(
            STAKEHOLDER_ORDER, key=lambda s: (utilities[s], s)
        ),
        "discounted_base_return_A": discounted_return(base_returns["A"], GAMMA),
        "discounted_base_return_B": discounted_return(base_returns["B"], GAMMA),
        "discounted_learner_reward_A": discounted_return(learner_returns["A"], GAMMA),
        "discounted_learner_reward_B": discounted_return(learner_returns["B"], GAMMA),
        "collision_type": collision_type,
        "minimum_bumper_gap": (None if min_gap is math.inf else float(min_gap)),
        "minimum_TTC": (None if min_ttc is math.inf else float(min_ttc)),
        "hard_braking_rate": float(hard_brake_events / max(1, hard_brake_steps * 2)),
        "background_maximum_braking": float(bg_max_brake),
    }


def reconstruct_primary_endpoint_evaluations(
    results_root: Path,
    *,
    bundle: FinalLockBundle | None = None,
    checkpoint_step: int = 100_000,
    checkpoint_index: int = CHECKPOINT_INDEX_100K,
) -> list[dict[str, Any]]:
    """Reconstruct 16 eval episodes × 30 jobs at the primary endpoint."""
    root = Path(results_root)
    bundle = bundle or load_final_locks()
    rows: list[dict[str, Any]] = []
    pairs = validation_blocks_with_assignments(bundle)
    val_ids = sorted({bid for bid, _, _ in pairs})
    block_index_map = {bid: i for i, bid in enumerate(val_ids)}

    job_dirs = sorted((root / "jobs").glob("*__*"))
    if len(job_dirs) != 30:
        raise RuntimeError(f"expected 30 job dirs, got {len(job_dirs)}")

    for job_dir in job_dirs:
        manifest = __import__("json").loads(
            (job_dir / "job_manifest.json").read_text(encoding="utf-8-sig")
        )
        condition = str(manifest["condition"])
        master = int(manifest["master_seed"])
        eval_seed = int(manifest["seeds"]["evaluation_seed"])
        weights = job_dir / "final_online_target_weights.pt"
        learners = load_learners_from_final_weights(weights, condition=condition)

        episodes: list[dict[str, Any]] = []
        for block_id, assignment, block in pairs:
            b_idx = block_index_map[block_id]
            ep_seed = evaluation_episode_seed(
                eval_seed,
                checkpoint_index=checkpoint_index,
                block_index=b_idx,
                assignment_index=int(assignment),
            )
            ep = run_instrumented_evaluation_episode(
                bundle,
                learners,
                block=block,
                assignment=int(assignment),
                block_id=block_id,
                block_index=b_idx,
                episode_seed=ep_seed,
            )
            ep.update(
                {
                    "formal_job_id": job_dir.name,
                    "condition": condition,
                    "master_seed": master,
                    "checkpoint_step": int(checkpoint_step),
                    "checkpoint_index": int(checkpoint_index),
                    "evaluation_seed_base": eval_seed,
                    "reconstruction_source": "final_online_target_weights.pt",
                    "published_episode_payload_available": False,
                }
            )
            # finite check
            for key in ("learner_A_utility", "discounted_base_return_A"):
                if not math.isfinite(float(ep[key])):
                    raise RuntimeError(f"non-finite {key} in {job_dir.name}")
            episodes.append(ep)
            rows.append(ep)

        if len(episodes) != 16:
            raise RuntimeError(f"{job_dir.name}: expected 16 episodes")
        # uniqueness of episode keys
        keys = {(e["block_id"], e["assignment"]) for e in episodes}
        if len(keys) != 16:
            raise RuntimeError(f"{job_dir.name}: duplicate episode keys")

    return rows


__all__ = [
    "load_learners_from_final_weights",
    "reconstruct_primary_endpoint_evaluations",
    "run_instrumented_evaluation_episode",
]
