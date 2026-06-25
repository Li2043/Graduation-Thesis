"""Canonical V1 seed protocol — single source of truth.

This is a **config / reporting utility only**. It contains no environment,
policy, DQN, reward, or experience-function logic and imports none of those
modules. It defines the seed sets used by calibration (Optuna), validation, and
final evaluation, plus helpers to (a) record per-trial / per-run seed metadata,
(b) detect final-seed leakage, and (c) aggregate robustness metrics across
calibration seeds. The human-readable rules live in docs/V1_SEED_PROTOCOL.md.

The seed sets below are *run/trial seeds* — the value passed to
``train.py --seed``. A run with seed ``s`` derives its per-episode training
seeds as ``s * TRAIN_SEED_STRIDE + episode`` (mirroring v1/training/train.py)
and evaluates the trained policy on a held-out evaluation seed set.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

# --------------------------------------------------------------------- seed sets
# Calibration seeds may be used INSIDE the Optuna objective.
CALIBRATION_SEEDS: tuple[int, ...] = (1, 2, 3)
# Validation seeds may be used ONLY after Optuna selects top candidate configs.
VALIDATION_SEEDS: tuple[int, ...] = (4, 5)
# Final evaluation seeds are LOCKED: never used by Optuna calibration/validation.
FINAL_EVALUATION_SEEDS: tuple[int, ...] = (100, 101, 102, 103, 104)

# Fixed Optuna sampler seed for reproducible search.
OPTUNA_SAMPLER_SEED: int = 42

# Mirrors v1/training/train.py: train_seed = seed * TRAIN_SEED_STRIDE + episode.
# Episodes per run must stay < TRAIN_SEED_STRIDE so train-seed blocks never
# collide across different run seeds.
TRAIN_SEED_STRIDE: int = 100000

# Hard-constraint thresholds for accepting an Optuna candidate (same for both
# modes). Documented in docs/V1_SEED_PROTOCOL.md; tune only via explicit edit.
HARD_CONSTRAINTS = {
    "min_mean_safe_merge_success_rate": 0.6,
    "max_mean_collision_rate": 0.3,
    "max_mean_non_merge_failure_rate": 0.3,
    "min_worst_seed_safe_merge_success_rate": 0.3,
}


def all_seed_sets() -> dict:
    """Return the canonical seed sets and sampler seed as plain lists."""
    return {
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS),
        "final_evaluation_seeds": list(FINAL_EVALUATION_SEEDS),
        "sampler_seed": OPTUNA_SAMPLER_SEED,
    }


def validate_protocol() -> None:
    """Assert the three seed sets are pairwise disjoint (no overlap/leakage)."""
    cal = set(CALIBRATION_SEEDS)
    val = set(VALIDATION_SEEDS)
    fin = set(FINAL_EVALUATION_SEEDS)
    assert cal.isdisjoint(val), f"calibration/validation overlap: {cal & val}"
    assert cal.isdisjoint(fin), f"calibration/final overlap: {cal & fin}"
    assert val.isdisjoint(fin), f"validation/final overlap: {val & fin}"


def derive_train_seeds(seed: int, episodes: int) -> list[int]:
    """Per-episode training seeds for a run, matching train.py exactly."""
    return [seed * TRAIN_SEED_STRIDE + e for e in range(int(episodes))]


def is_final_seed(seed: int) -> bool:
    return int(seed) in set(FINAL_EVALUATION_SEEDS)


def assert_no_final_leakage(seeds_used: Iterable[int]) -> None:
    """Raise if any locked final-evaluation seed appears in ``seeds_used``.

    Call this from the Optuna calibration/validation loops with every run seed
    they intend to use, so final seeds can never leak into model selection.
    """
    used = {int(s) for s in seeds_used}
    leaked = used & set(FINAL_EVALUATION_SEEDS)
    if leaked:
        raise ValueError(
            f"Final-evaluation seeds {sorted(leaked)} must not be used during "
            "Optuna calibration or validation (see docs/V1_SEED_PROTOCOL.md)."
        )


def single_run_seed_metadata(
    seed: int,
    episodes: int,
    eval_seeds: Sequence[int],
    seed_phase: Optional[str] = None,
) -> dict:
    """Self-describing seed metadata embedded in each run's config JSON.

    Records the run seed, its phase, the exact train seeds used (compact form
    plus formula), and the eval seeds used, so any run is auditable in isolation.
    """
    train_seeds = derive_train_seeds(seed, episodes)
    return {
        "seed_phase": seed_phase,
        "run_seed": int(seed),
        "train_seed_formula": "seed * 100000 + episode",
        "train_seeds_count": len(train_seeds),
        "train_seeds_first": train_seeds[0] if train_seeds else None,
        "train_seeds_last": train_seeds[-1] if train_seeds else None,
        "eval_seeds_used": [int(s) for s in eval_seeds],
        "protocol_seed_sets": all_seed_sets(),
    }


def build_trial_seed_metadata(
    trial_number: int,
    train_seeds_used: Sequence[int],
    eval_seeds_used: Sequence[int],
    sampler_seed: int = OPTUNA_SAMPLER_SEED,
) -> dict:
    """Per-trial seed record for the (future) Optuna calibration script.

    Satisfies the protocol requirement that every trial logs trial_number,
    sampler_seed, all three protocol seed sets, and the exact train/eval seeds
    it consumed. Also asserts no final-seed leakage in the trial.
    """
    assert_no_final_leakage(list(train_seeds_used) + list(eval_seeds_used))
    return {
        "trial_number": int(trial_number),
        "sampler_seed": int(sampler_seed),
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS),
        "final_evaluation_seeds": list(FINAL_EVALUATION_SEEDS),
        "train_seeds_used": [int(s) for s in train_seeds_used],
        "eval_seeds_used": [int(s) for s in eval_seeds_used],
    }


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def _std(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    if len(vals) < 2:
        return 0.0
    mu = _mean(vals)
    return (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5


def aggregate_seed_metrics(per_seed: Sequence[Mapping[str, float]]) -> dict:
    """Aggregate per-seed eval dicts into mean + robustness metrics.

    Each element of ``per_seed`` is one run's eval metrics dict (as produced by
    ``v1.training.train.evaluate``). Robustness metrics (worst/max/std) make the
    Optuna objective prefer configs that are good *and* stable across seeds.
    """

    def col(key: str) -> list[float]:
        return [float(m[key]) for m in per_seed if key in m]

    safe = col("eval_safe_merge_success_rate")
    coll = col("eval_collision_rate")
    nonmerge = col("eval_non_merge_failure_rate")
    minexp = col("eval_min_experience")
    gini = col("eval_gini_experience")
    return {
        "n_seeds": len(per_seed),
        "mean_safe_merge_success_rate": _mean(safe),
        "worst_seed_safe_merge_success_rate": min(safe) if safe else 0.0,
        "std_safe_merge_success_rate": _std(safe),
        "mean_collision_rate": _mean(coll),
        "max_seed_collision_rate": max(coll) if coll else 0.0,
        "mean_non_merge_failure_rate": _mean(nonmerge),
        "mean_min_experience": _mean(minexp),
        "mean_gini_experience": _mean(gini),
    }


def check_hard_constraints(aggregate: Mapping[str, float]) -> dict:
    """Evaluate the suggested hard constraints against an aggregate dict.

    Returns per-constraint booleans plus an overall ``feasible`` flag. The
    worst-seed constraint is reported as ``"if_feasible"`` guidance, not a hard
    gate, because it may be infeasible early in calibration.
    """
    c = HARD_CONSTRAINTS
    checks = {
        "mean_safe_merge_success_rate>=%.2f" % c["min_mean_safe_merge_success_rate"]:
            aggregate.get("mean_safe_merge_success_rate", 0.0)
            >= c["min_mean_safe_merge_success_rate"],
        "mean_collision_rate<=%.2f" % c["max_mean_collision_rate"]:
            aggregate.get("mean_collision_rate", 1.0) <= c["max_mean_collision_rate"],
        "mean_non_merge_failure_rate<=%.2f" % c["max_mean_non_merge_failure_rate"]:
            aggregate.get("mean_non_merge_failure_rate", 1.0)
            <= c["max_mean_non_merge_failure_rate"],
    }
    feasible = all(checks.values())
    checks["worst_seed_safe_merge_success_rate>=%.2f (if feasible)"
           % c["min_worst_seed_safe_merge_success_rate"]] = (
        aggregate.get("worst_seed_safe_merge_success_rate", 0.0)
        >= c["min_worst_seed_safe_merge_success_rate"]
    )
    checks["feasible"] = feasible
    return checks


# Fail fast at import if the protocol is ever edited into an inconsistent state.
validate_protocol()
