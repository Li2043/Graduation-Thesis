"""Isolated greedy evaluation for Stage 5B-0 (no training mutation)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from thesis.agents.action_masking import role_action_mask, validate_action_mask
from thesis.agents.independent_dqn_v2 import IndependentDQNLearner
from thesis.envs.merge_env_candidate_v3 import HighLevelAction
from thesis.training.final_lock_loader import FinalLockBundle
from thesis.training.pilot_checkpoint import learner_fingerprint
from thesis.training.pilot_ic_schedule import (
    build_env_for_block,
    validation_blocks_with_assignments,
)


def run_isolated_evaluation(
    bundle: FinalLockBundle,
    learners: dict[str, IndependentDQNLearner],
    *,
    eval_seed: int,
    max_policy_steps: int = 400,
) -> dict[str, Any]:
    """Evaluate on 16 validation episodes without mutating training state."""
    before = {
        "A": learner_fingerprint(learners["A"]),
        "B": learner_fingerprint(learners["B"]),
    }
    # Snapshot learner RNGs to restore (greedy should not consume RNG;
    # restore defensively for isolation guarantees).
    rng_A = deepcopy(learners["A"]._rng.bit_generator.state)
    rng_B = deepcopy(learners["B"]._rng.bit_generator.state)
    replay_A = len(learners["A"].replay)
    replay_B = len(learners["B"].replay)
    updates_A = int(learners["A"]._update_count)
    updates_B = int(learners["B"]._update_count)

    episodes: list[dict[str, Any]] = []
    eval_rng = np.random.default_rng(int(eval_seed))
    for block_id, assignment, block in validation_blocks_with_assignments(bundle):
        env = build_env_for_block(bundle, block, max_policy_steps=max_policy_steps)
        # Separate env RNG from training
        obs, _ = env.reset(seed=int(block.seed) + int(eval_rng.integers(0, 10_000)))
        ep_return = {"A": 0.0, "B": 0.0}
        steps = 0
        term_reason = "ongoing"
        while True:
            actions = {}
            for aid in ("A", "B"):
                role = str(env._vehicles[aid].role)  # noqa: SLF001
                mask = validate_action_mask(role_action_mask(role, 3), 3)
                if not env._vehicles[aid].active_on_road:  # noqa: SLF001
                    actions[aid] = int(HighLevelAction.MAINTAIN)
                else:
                    actions[aid] = learners[aid].select_action(
                        obs[aid], mask, greedy=True
                    )
            obs, rewards, terminated, truncated, info = env.step(actions)
            for aid in ("A", "B"):
                ep_return[aid] += float(rewards[aid])
            steps += 1
            term_reason = str(info["term_reason"])
            if terminated or truncated:
                break
        episodes.append(
            {
                "block_id": block_id,
                "assignment": int(assignment),
                "steps": steps,
                "term_reason": term_reason,
                "return_A": ep_return["A"],
                "return_B": ep_return["B"],
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

    # Restore learner RNGs (greedy path should be unchanged; enforce isolation)
    learners["A"]._rng.bit_generator.state = rng_A
    learners["B"]._rng.bit_generator.state = rng_B
    after = {
        "A": learner_fingerprint(learners["A"]),
        "B": learner_fingerprint(learners["B"]),
    }
    mutation = {
        "fingerprint_changed": before != after,
        "replay_size_changed": (
            len(learners["A"].replay) != replay_A or len(learners["B"].replay) != replay_B
        ),
        "update_count_changed": (
            int(learners["A"]._update_count) != updates_A
            or int(learners["B"]._update_count) != updates_B
        ),
    }
    mutation["any"] = any(mutation.values())
    return {
        "n_episodes": len(episodes),
        "episodes": episodes,
        "mutation": mutation,
        "fingerprint_before": before,
        "fingerprint_after": after,
        "optimiser_updates": False,
        "replay_writes": False,
        "target_syncs": False,
        "greedy": True,
        "epsilon": 0.0,
    }


__all__ = ["run_isolated_evaluation"]
