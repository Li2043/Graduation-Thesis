"""Stage 7A-1 rich evaluation at a checkpoint (H1 utility semantics)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from thesis.diagnostics.stage7a0_trajectory_eval import run_diagnostic_evaluation_episode
from thesis.formal.formal_schedule import evaluation_episode_seed
from thesis.training.final_lock_loader import FinalLockBundle
from thesis.training.pilot_checkpoint import learner_fingerprint
from thesis.training.pilot_ic_schedule import validation_blocks_with_assignments


def evaluate_checkpoint_rich(
    bundle: FinalLockBundle,
    learners: dict[str, Any],
    *,
    master_seed: int,
    evaluation_seed: int,
    checkpoint_step: int,
    checkpoint_index: int,
    checkpoint_sha256: str,
    max_policy_steps: int = 400,
    collect_trajectories: bool = False,
) -> dict[str, Any]:
    """Greedy eval; must not mutate learners / replay / update counts."""
    before = {
        "A": learner_fingerprint(learners["A"]),
        "B": learner_fingerprint(learners["B"]),
    }
    rng_A = deepcopy(learners["A"]._rng.bit_generator.state)
    rng_B = deepcopy(learners["B"]._rng.bit_generator.state)
    replay_A = len(learners["A"].replay)
    replay_B = len(learners["B"].replay)
    updates_A = int(learners["A"]._update_count)
    updates_B = int(learners["B"]._update_count)

    pairs = validation_blocks_with_assignments(bundle)
    val_ids = sorted({bid for bid, _, _ in pairs})
    block_index_map = {bid: i for i, bid in enumerate(val_ids)}

    episodes: list[dict[str, Any]] = []
    steps_all: list[dict[str, Any]] = []
    for i, (block_id, assignment, block) in enumerate(pairs):
        b_idx = block_index_map[block_id]
        ep_seed = evaluation_episode_seed(
            evaluation_seed,
            checkpoint_index=checkpoint_index,
            block_index=b_idx,
            assignment_index=int(assignment),
        )
        ep, step_rows = run_diagnostic_evaluation_episode(
            bundle,
            learners,
            block=block,
            assignment=int(assignment),
            block_id=block_id,
            block_index=b_idx,
            episode_seed=ep_seed,
            master_seed=master_seed,
            checkpoint_sha256=checkpoint_sha256,
            max_policy_steps=max_policy_steps,
        )
        ep["checkpoint_step"] = int(checkpoint_step)
        ep["evaluation_episode_index"] = int(i)
        ep["classifiable_order"] = str(ep.get("passing_order", "unresolved")) not in {
            "unresolved",
            "none",
            "",
        }
        # background utilities already in derived fields via episode utility
        episodes.append(ep)
        if collect_trajectories:
            for row in step_rows:
                row["master_seed"] = int(master_seed)
                row["checkpoint_step"] = int(checkpoint_step)
                row["checkpoint_sha256"] = checkpoint_sha256
                row["validation_block_id"] = block_id
                row["assignment"] = int(assignment)
            steps_all.extend(step_rows)

    learners["A"]._rng.bit_generator.state = rng_A
    learners["B"]._rng.bit_generator.state = rng_B
    after = {
        "A": learner_fingerprint(learners["A"]),
        "B": learner_fingerprint(learners["B"]),
    }
    evaluation_guard = {
        "replay_writes": 0
        if (len(learners["A"].replay) == replay_A and len(learners["B"].replay) == replay_B)
        else 1,
        "optimizer_steps": 0
        if (
            int(learners["A"]._update_count) == updates_A
            and int(learners["B"]._update_count) == updates_B
        )
        else 1,
        "target_syncs": 0,
        "epsilon_updates": 0,
        "training_counter_updates": 0,
        "fingerprint_changed": int(before != after),
    }
    if any(evaluation_guard.values()):
        raise RuntimeError(f"evaluation isolation violated: {evaluation_guard}")

    return {
        "episodes": episodes,
        "trajectories": steps_all,
        "evaluation_guard": evaluation_guard,
        "n_episodes": len(episodes),
    }


__all__ = ["evaluate_checkpoint_rich"]
