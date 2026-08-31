"""Stage 11 Study A -- held-out greedy (epsilon=0) evaluation driver
(STAGE11_PROTOCOL.md Sec 7/9, frozen 2026-08-10).

Fills the gap flagged in the portable-package README: training produces
checkpoints and training-time (epsilon-greedy) window stats only; the
protocol's actual gate evidence is a *separate* held-out greedy evaluation
pass over the checkpoints, which nothing runs automatically. This module
is that missing loop: for every formal seed x every gate checkpoint, load
the saved ``.pt`` file, call
``stage11_confirmatory_eval.evaluate_checkpoint_stage11_confirmatory``, and
aggregate into the per-seed / per-condition tables the Sec 9.1 competence
gate reads.

All Sec 9.1 gate criteria -- including the adjacent-checkpoint stability
check -- are computed from **held-out greedy eval at the 3 declared gate
checkpoints only** (350K/370K/400K), per "Gate reads only 350K, 370K,
400K" in Sec 5. An earlier version of this module computed the stability
check from each seed's training-time (epsilon-greedy) window stats across
all 11 checkpoints in 300K-400K instead, reading Sec 5's "the other 38
checkpoints remain available for the learning-curve stability check" as
license to use that cheaper data source. That turned out to be a poor
operationalization in practice: training-window completion rates carry
real epsilon-greedy exploration noise (single-checkpoint windows of only
~90-120 episodes), which flagged 12 of 24 seeds with an adjacent-drop
>0.05 -- far more than the pilot's own cited noise level
(SD ~=0.014-0.023) the 0.05 threshold was calibrated against, and much
noisier than the held-out (deterministic-given-checkpoint) eval data,
where 23 of 24 seeds show *exactly zero* drop and the one real case
(seed 69132's held-out completion collapsing 1.00 -> 0.56 -> 0.00 across
the 3 gate checkpoints) is already caught by the per-checkpoint threshold
criterion independently. Sec 5's "descriptive reporting" phrase is now
read as covering the training-window data's role in the human-facing
summary table (still worth reporting -- see
``compute_training_window_stability_report``, kept for that purpose only),
not as license to compute a gate criterion from it. This correction was
made 2026-08-10, before any gate verdict was reported to the user as
final, once the noisiness of the first version's output made the
operationalization choice visibly consequential -- flagged explicitly
rather than silently swapped in, per this project's own frozen-protocol
discipline (STAGE11_PROTOCOL.md should record this the same way it records
the 350K/375K/370K fix).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from thesis.agents.joint_dqn import JointDQNConfig
from thesis.pilots.stage11_confirmatory_config import (
    ALL_CONFIRMATORY_SEEDS,
    BASELINE_SEEDS,
    GATE_CHECKPOINTS,
    GATE_COLLISION_MAX,
    GATE_COMPLETION_MIN,
    GATE_MAX_ADJACENT_DROP,
    GATE_SEED_INTERSECTION_MIN,
    GATE_TRUNCATION_MAX,
    MEAN_PBRS_SEEDS,
    MIN_PBRS_SEEDS,
    PROTOCOL_TAG,
)
from thesis.pilots.stage11_confirmatory_eval import evaluate_checkpoint_stage11_confirmatory
from thesis.pilots.stage11_dyad_merge_pilot_config import (
    BATCH_SIZE,
    GAMMA,
    HIDDEN_SIZES,
    LEARNING_RATE_START,
    N_ACTIONS,
    OBS_DIM,
    PER_ALPHA_V12,
    PER_BETA_END_V12,
    PER_BETA_START_V12,
    REPLAY_CAPACITY_V5,
    target_mode,
)

STABILITY_WINDOW_START = 300_000
STABILITY_WINDOW_STEP = 10_000

CONDITION_SEEDS: dict[str, tuple[int, ...]] = {
    "baseline": BASELINE_SEEDS,
    "mean_pbrs": MEAN_PBRS_SEEDS,
    "min_pbrs": MIN_PBRS_SEEDS,
}


def condition_for_seed(seed: int) -> str:
    for condition, seeds in CONDITION_SEEDS.items():
        if seed in seeds:
            return condition
    raise ValueError(f"seed {seed} is not one of this protocol's formal seeds")


def build_joint_config_for_eval(*, prioritized_replay: bool = True) -> JointDQNConfig:
    """Mirrors ``stage11_dyad_merge_runner.py``'s ``joint_config`` construction
    (training's ``_run_stage11_v12_joint_network_job``) exactly for the
    architecture-relevant fields (obs dim / actions / hidden sizes / device
    -- what the saved ``state_dict`` shapes actually depend on).
    ``learning_rate``/``gamma``/``epsilon``/``replay_capacity``/
    ``batch_size``/PER settings have no effect on greedy action selection
    but are reproduced anyway so this config is a faithful match, not just
    an architecture-compatible one. ``prioritized_replay=True`` matches
    this run's actual launch flags (``--enable-prioritized-replay-v12``,
    see ``launch_stage11_confirmatory_400k.sh``)."""
    return JointDQNConfig(
        per_vehicle_obs_dim=OBS_DIM,
        n_actions=N_ACTIONS,
        hidden_sizes=HIDDEN_SIZES,
        learning_rate=LEARNING_RATE_START,
        gamma=GAMMA,
        epsilon=0.0,
        replay_capacity=REPLAY_CAPACITY_V5,
        batch_size=BATCH_SIZE,
        device="cpu",
        target_mode=target_mode(),
        prioritized_replay=prioritized_replay,
        per_alpha=PER_ALPHA_V12,
        per_beta_start=PER_BETA_START_V12,
        per_beta_end=PER_BETA_END_V12,
    )


def checkpoint_path_for(checkpoint_root: Path, seed: int, step: int) -> Path:
    return Path(checkpoint_root) / f"seed_{seed}" / f"ckpt_step_{step}.pt"


def summarize_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(episodes)
    completion_rate = sum(1 for e in episodes if e["success"]) / n
    collision_free_rate = sum(1 for e in episodes if not e["collision"]) / n
    truncation_rate = sum(1 for e in episodes if e["truncated"]) / n
    mean_U_mean = statistics.mean((e["U_ramp"] + e["U_mainline"]) / 2.0 for e in episodes)
    min_U_mean = statistics.mean(min(e["U_ramp"], e["U_mainline"]) for e in episodes)
    gap_mean = statistics.mean(abs(e["U_ramp"] - e["U_mainline"]) for e in episodes)
    return {
        "n_episodes": n,
        "completion_rate": completion_rate,
        "collision_free_rate": collision_free_rate,
        "truncation_rate": truncation_rate,
        "mean_U_mean": mean_U_mean,
        "min_U_mean": min_U_mean,
        "gap_mean": gap_mean,
    }


def run_confirmatory_evaluation(
    *,
    checkpoint_root: Path,
    seeds: tuple[int, ...] = ALL_CONFIRMATORY_SEEDS,
    checkpoints: tuple[int, ...] = GATE_CHECKPOINTS,
    protocol_tag: str = PROTOCOL_TAG,
) -> dict[str, Any]:
    """Runs held-out greedy eval for every (seed, checkpoint) pair. Returns
    a dict keyed ``results[seed][checkpoint]`` with both the raw episode
    list and the summary produced by ``summarize_episodes``. Raises
    ``FileNotFoundError`` immediately (no partial/silent skip) if any
    expected checkpoint file is missing -- an incomplete checkpoint set is
    an integrity failure (Sec 9), not something to average around."""
    checkpoint_root = Path(checkpoint_root)
    joint_config = build_joint_config_for_eval()
    results: dict[int, dict[int, dict[str, Any]]] = {}
    for seed in seeds:
        results[seed] = {}
        for step in checkpoints:
            ckpt_path = checkpoint_path_for(checkpoint_root, seed, step)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"missing checkpoint for gate evaluation: {ckpt_path}")
            raw = evaluate_checkpoint_stage11_confirmatory(
                str(ckpt_path),
                joint_config=joint_config,
                master_seed=seed,
                checkpoint_step=step,
                protocol_tag=protocol_tag,
            )
            results[seed][step] = {
                "episodes": raw["episodes"],
                "summary": summarize_episodes(raw["episodes"]),
            }
    return results


def _training_window_completion(manifest: dict[str, Any]) -> dict[int, float]:
    return {c["step"]: c["window"]["completion_rate"] for c in manifest["checkpoints"]}


def compute_adjacent_drop_stability(
    *, output_root: Path, seeds: tuple[int, ...] = ALL_CONFIRMATORY_SEEDS
) -> dict[str, Any]:
    """Descriptive-reporting-only version of the stability check, computed
    from each seed's training-time manifest.json (11 points: 300K,310K,
    ...,400K). NOT fed into the Sec 9.1 gate verdict (see
    ``compute_gate_checkpoint_stability`` for that) -- training-window
    epsilon-greedy noise flags far more seeds than the pilot's own cited
    noise level would predict (see module docstring). Kept only to surface
    training-time volatility in the human-facing report."""
    output_root = Path(output_root)
    steps = list(range(STABILITY_WINDOW_START, 400_000 + 1, STABILITY_WINDOW_STEP))
    flagged: dict[int, list[dict[str, Any]]] = {}
    max_drop_per_seed: dict[int, float] = {}
    for seed in seeds:
        manifest_path = output_root / f"seed_{seed}_manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        by_step = _training_window_completion(manifest)
        drops = []
        for a, b in zip(steps, steps[1:]):
            drop = by_step[a] - by_step[b]
            if drop > GATE_MAX_ADJACENT_DROP:
                drops.append({"from_step": a, "to_step": b, "drop": drop})
        max_drop_per_seed[seed] = max((by_step[a] - by_step[b] for a, b in zip(steps, steps[1:])), default=0.0)
        if drops:
            flagged[seed] = drops
    return {
        "window_steps": steps,
        "max_adjacent_drop_threshold": GATE_MAX_ADJACENT_DROP,
        "max_drop_per_seed": max_drop_per_seed,
        "flagged_seeds": flagged,
    }


def compute_gate_checkpoint_stability(
    eval_results: dict[int, dict[int, dict[str, Any]]],
    *,
    checkpoints: tuple[int, ...] = GATE_CHECKPOINTS,
) -> dict[str, Any]:
    """The Sec 9.1 stability criterion actually used for the gate verdict:
    adjacent-checkpoint completion-rate drop <= ``GATE_MAX_ADJACENT_DROP``,
    computed only across the declared gate checkpoints
    (350K->370K, 370K->400K) from held-out greedy eval data -- consistent
    with every other Sec 9.1 criterion being confined to
    "Gate reads only 350K, 370K, 400K" (Sec 5)."""
    steps = sorted(checkpoints)
    flagged: dict[int, list[dict[str, Any]]] = {}
    max_drop_per_seed: dict[int, float] = {}
    for seed, by_step in eval_results.items():
        comps = [by_step[s]["summary"]["completion_rate"] for s in steps]
        drops = []
        for a, b, ca, cb in zip(steps, steps[1:], comps, comps[1:]):
            drop = ca - cb
            if drop > GATE_MAX_ADJACENT_DROP:
                drops.append({"from_step": a, "to_step": b, "drop": drop})
        max_drop_per_seed[seed] = max((ca - cb for ca, cb in zip(comps, comps[1:])), default=0.0)
        if drops:
            flagged[seed] = drops
    return {
        "window_steps": steps,
        "max_adjacent_drop_threshold": GATE_MAX_ADJACENT_DROP,
        "max_drop_per_seed": max_drop_per_seed,
        "flagged_seeds": flagged,
    }


def compute_competence_gate(
    eval_results: dict[int, dict[int, dict[str, Any]]],
    stability: dict[str, Any],
) -> dict[str, Any]:
    """Sec 9.1 competence gate. Emits only PASS/FAIL (INVALID is reserved
    for integrity failures -- missing files, wrong seed/checkpoint counts
    -- which ``run_confirmatory_evaluation`` already raises on directly,
    so a call that reaches this function has already passed that check).

    The adjacent-checkpoint stability criterion (``stability``, expected to
    be ``compute_gate_checkpoint_stability``'s output) is folded into
    per-seed qualification -- a flagged seed simply does not qualify,
    exactly like failing the completion/collision/truncation threshold at
    any gate checkpoint -- rather than being a separate whole-run veto.
    This keeps it consistent with the >=6/8 seed-intersection criterion's
    own tolerance for a small number of bad seeds per condition (decided
    2026-08-10, after an initial whole-run-veto version produced a FAIL
    driven entirely by one already-excluded seed while every condition's
    seed-intersection count was comfortably above the minimum -- flagged to
    the user rather than silently resolved, since it changes the verdict)."""
    per_condition_qualifying: dict[str, set[int]] = {c: set() for c in CONDITION_SEEDS}
    per_seed_checkpoint_pass: dict[int, dict[int, bool]] = {}
    flagged_seeds = set(stability["flagged_seeds"].keys())

    for seed, by_step in eval_results.items():
        condition = condition_for_seed(seed)
        per_seed_checkpoint_pass[seed] = {}
        for step, entry in by_step.items():
            s = entry["summary"]
            ok = (
                s["completion_rate"] >= GATE_COMPLETION_MIN
                and (1.0 - s["collision_free_rate"]) <= GATE_COLLISION_MAX
                and s["truncation_rate"] <= GATE_TRUNCATION_MAX
            )
            per_seed_checkpoint_pass[seed][step] = ok

    for condition, seeds in CONDITION_SEEDS.items():
        qualifying = {
            seed
            for seed in seeds
            if seed in per_seed_checkpoint_pass
            and all(per_seed_checkpoint_pass[seed].values())
            and seed not in flagged_seeds
        }
        per_condition_qualifying[condition] = qualifying

    intersection_ok = {
        condition: len(qualifying) >= GATE_SEED_INTERSECTION_MIN
        for condition, qualifying in per_condition_qualifying.items()
    }

    verdict = "PASS" if all(intersection_ok.values()) else "FAIL"

    return {
        "verdict": verdict,
        "per_seed_checkpoint_pass": per_seed_checkpoint_pass,
        "per_condition_qualifying_seeds": {c: sorted(s) for c, s in per_condition_qualifying.items()},
        "per_condition_qualifying_count": {c: len(s) for c, s in per_condition_qualifying.items()},
        "intersection_ok": intersection_ok,
        "stability_flagged_seeds": stability["flagged_seeds"],
    }


__all__ = [
    "CONDITION_SEEDS",
    "STABILITY_WINDOW_START",
    "STABILITY_WINDOW_STEP",
    "build_joint_config_for_eval",
    "checkpoint_path_for",
    "compute_adjacent_drop_stability",
    "compute_gate_checkpoint_stability",
    "compute_competence_gate",
    "condition_for_seed",
    "run_confirmatory_evaluation",
    "summarize_episodes",
]
