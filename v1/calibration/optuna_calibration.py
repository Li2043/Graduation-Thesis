"""Optuna calibration for V1 delta-min Rawlsian reward shaping.

Runs controlled hyperparameter search over shared task/safety parameters and
Rawlsian-only shaping parameters. Each trial trains **both** egoistic and
Rawlsian conditions on calibration seeds only; validation seeds are used only
after the study completes. Final evaluation seeds are never consumed here.

See docs/V1_OPTUNA_CALIBRATION_PROTOCOL.md and docs/V1_SEED_PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import optuna  # noqa: E402
from optuna.samplers import TPESampler  # noqa: E402

from v1.calibration.seed_protocol import (  # noqa: E402
    CALIBRATION_SEEDS,
    FINAL_EVALUATION_SEEDS,
    HARD_CONSTRAINTS,
    OPTUNA_SAMPLER_SEED,
    VALIDATION_SEEDS,
    aggregate_seed_metrics,
    assert_no_final_leakage,
    build_trial_seed_metadata,
)
from v1.calibration.train_runner import run_condition  # noqa: E402
from v1.training.train import EVAL_SEEDS, RunConfig  # noqa: E402

DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "optuna"
CONSTRAINT_PENALTY = -1_000_000.0

# Reference configuration from the 300-episode pilot (not hard-coded winner).
REFERENCE_CONFIG = {
    "terminal_collision_penalty": 5.0,
    "merge_success_bonus": 2.0,
    "non_merge_failure_penalty": 3.0,
    "rawlsian_lambda": 1.0,
    "rawlsian_epsilon": 1e-6,
}

TRIAL_CSV_FIELDS = [
    "trial_number",
    "terminal_collision_penalty",
    "merge_success_bonus",
    "non_merge_failure_penalty",
    "rawlsian_lambda",
    "rawlsian_epsilon",
    "egoistic_safe_merge_success_rate_mean",
    "egoistic_collision_rate_mean",
    "egoistic_non_merge_failure_rate_mean",
    "egoistic_min_experience_mean",
    "rawlsian_safe_merge_success_rate_mean",
    "rawlsian_collision_rate_mean",
    "rawlsian_non_merge_failure_rate_mean",
    "rawlsian_min_experience_mean",
    "rawlsian_gini_experience_mean",
    "rawlsian_mean_experience_mean",
    "calibration_score",
    "constraint_passed",
]


def parse_int_list(text: str) -> list[int]:
    return [int(s.strip()) for s in text.split(",") if s.strip()]


def sample_hyperparameters(trial: optuna.Trial) -> dict[str, float]:
    """Sample shared task/safety + Rawlsian-only parameters for one trial."""
    return {
        "terminal_collision_penalty": trial.suggest_float(
            "terminal_collision_penalty", 3.0, 8.0
        ),
        "merge_success_bonus": trial.suggest_float("merge_success_bonus", 1.0, 5.0),
        "non_merge_failure_penalty": trial.suggest_float(
            "non_merge_failure_penalty", 2.0, 6.0
        ),
        "rawlsian_lambda": trial.suggest_float("rawlsian_lambda", 0.5, 1.5),
        "rawlsian_epsilon": trial.suggest_float(
            "rawlsian_epsilon", 1e-6, 1e-2, log=True
        ),
    }


def build_run_config(
    mode: str,
    seed: int,
    episodes: int,
    max_steps: int,
    params: dict[str, float],
    seed_phase: str,
    eval_seeds: list[int],
) -> RunConfig:
    return RunConfig(
        mode=mode,
        seed=seed,
        episodes=episodes,
        max_steps=max_steps,
        seed_phase=seed_phase,
        eval_seeds=list(eval_seeds),
        terminal_collision_penalty=params["terminal_collision_penalty"],
        merge_success_bonus=params["merge_success_bonus"],
        non_merge_failure_penalty=params["non_merge_failure_penalty"],
        rawlsian_lambda=params["rawlsian_lambda"],
        rawlsian_epsilon=params["rawlsian_epsilon"],
    )


def check_mode_constraints(aggregate: dict) -> bool:
    """Return True if aggregate metrics satisfy primary hard constraints."""
    return (
        aggregate["mean_safe_merge_success_rate"]
        >= HARD_CONSTRAINTS["min_mean_safe_merge_success_rate"]
        and aggregate["mean_collision_rate"]
        <= HARD_CONSTRAINTS["max_mean_collision_rate"]
        and aggregate["mean_non_merge_failure_rate"]
        <= HARD_CONSTRAINTS["max_mean_non_merge_failure_rate"]
    )


def compute_calibration_score(
    ego: dict,
    rawls: dict,
    constraint_passed: bool,
) -> float:
    """Scalar calibration objective (NOT a final research metric).

    Within valid trials, prefer higher Rawlsian safe-merge and experience,
    lower Rawlsian collision/non-merge/gini, and stable egoistic baseline.
    """
    if not constraint_passed:
        return CONSTRAINT_PENALTY

    r_safe = rawls["mean_safe_merge_success_rate"]
    r_coll = rawls["mean_collision_rate"]
    r_nonmerge = rawls["mean_non_merge_failure_rate"]
    r_min_exp = rawls["mean_min_experience"]
    r_mean_exp = rawls["mean_mean_experience"]
    r_gini = rawls["mean_gini_experience"]

    e_safe = ego["mean_safe_merge_success_rate"]
    e_coll = ego["mean_collision_rate"]
    e_nonmerge = ego["mean_non_merge_failure_rate"]
    e_min_exp = ego["mean_min_experience"]

    score = (
        2.0 * r_safe
        - 3.0 * r_coll
        - 2.0 * r_nonmerge
        + 0.2 * r_min_exp
        + 0.1 * r_mean_exp
        - 1.0 * r_gini
        + 0.5 * max(0.0, r_min_exp - e_min_exp)
        - 2.0 * max(0.0, 0.6 - e_safe)
        - 2.0 * max(0.0, e_coll - 0.3)
        - 2.0 * max(0.0, e_nonmerge - 0.3)
    )
    return float(score)


def aggregate_with_mean_experience(per_seed: list[dict]) -> dict:
    """Extend aggregate_seed_metrics with mean eval_mean_experience."""
    base = aggregate_seed_metrics(per_seed)
    mean_exps = [float(m["eval_mean_experience"]) for m in per_seed]
    base["mean_mean_experience"] = sum(mean_exps) / len(mean_exps) if mean_exps else 0.0
    return base


def run_mode_on_seeds_extended(
    mode: str,
    seeds: Sequence[int],
    episodes: int,
    max_steps: int,
    params: dict[str, float],
    seed_phase: str,
    eval_seeds: list[int],
) -> dict:
    per_seed = []
    for seed in seeds:
        config = build_run_config(
            mode=mode,
            seed=int(seed),
            episodes=episodes,
            max_steps=max_steps,
            params=params,
            seed_phase=seed_phase,
            eval_seeds=eval_seeds,
        )
        per_seed.append(run_condition(config))
    return aggregate_with_mean_experience(per_seed)


def trial_row_from_result(
    trial_number: int,
    params: dict[str, float],
    ego: dict,
    rawls: dict,
    score: float,
    constraint_passed: bool,
) -> dict:
    return {
        "trial_number": trial_number,
        "terminal_collision_penalty": params["terminal_collision_penalty"],
        "merge_success_bonus": params["merge_success_bonus"],
        "non_merge_failure_penalty": params["non_merge_failure_penalty"],
        "rawlsian_lambda": params["rawlsian_lambda"],
        "rawlsian_epsilon": params["rawlsian_epsilon"],
        "egoistic_safe_merge_success_rate_mean": ego["mean_safe_merge_success_rate"],
        "egoistic_collision_rate_mean": ego["mean_collision_rate"],
        "egoistic_non_merge_failure_rate_mean": ego["mean_non_merge_failure_rate"],
        "egoistic_min_experience_mean": ego["mean_min_experience"],
        "rawlsian_safe_merge_success_rate_mean": rawls["mean_safe_merge_success_rate"],
        "rawlsian_collision_rate_mean": rawls["mean_collision_rate"],
        "rawlsian_non_merge_failure_rate_mean": rawls["mean_non_merge_failure_rate"],
        "rawlsian_min_experience_mean": rawls["mean_min_experience"],
        "rawlsian_gini_experience_mean": rawls["mean_gini_experience"],
        "rawlsian_mean_experience_mean": rawls["mean_mean_experience"],
        "calibration_score": score,
        "constraint_passed": constraint_passed,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def create_objective(
    calibration_seeds: list[int],
    episodes: int,
    max_steps: int,
    eval_seeds: list[int],
    trial_rows: list[dict],
):
    assert_no_final_leakage(calibration_seeds)

    def objective(trial: optuna.Trial) -> float:
        params = sample_hyperparameters(trial)

        ego = run_mode_on_seeds_extended(
            "egoistic",
            calibration_seeds,
            episodes,
            max_steps,
            params,
            seed_phase="calibration",
            eval_seeds=eval_seeds,
        )
        rawls = run_mode_on_seeds_extended(
            "rawlsian",
            calibration_seeds,
            episodes,
            max_steps,
            params,
            seed_phase="calibration",
            eval_seeds=eval_seeds,
        )

        ego_ok = check_mode_constraints(ego)
        rawls_ok = check_mode_constraints(rawls)
        constraint_passed = ego_ok and rawls_ok
        score = compute_calibration_score(ego, rawls, constraint_passed)

        row = trial_row_from_result(
            trial.number, params, ego, rawls, score, constraint_passed
        )
        trial_rows.append(row)

        seed_meta = build_trial_seed_metadata(
            trial_number=trial.number,
            train_seeds_used=calibration_seeds,
            eval_seeds_used=eval_seeds,
            sampler_seed=OPTUNA_SAMPLER_SEED,
        )
        for key, value in row.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("seed_metadata", seed_meta)
        trial.set_user_attr("reference_config", REFERENCE_CONFIG)
        trial.set_user_attr(
            "final_evaluation_seeds_placeholder",
            list(FINAL_EVALUATION_SEEDS),
        )

        return score

    return objective


def select_top_trials(
    trial_rows: list[dict], top_k: int = 3
) -> list[dict]:
    passing = [r for r in trial_rows if r.get("constraint_passed")]
    passing.sort(key=lambda r: float(r["calibration_score"]), reverse=True)
    return passing[:top_k]


def run_validation(
    top_configs: list[dict],
    validation_seeds: list[int],
    episodes: int,
    max_steps: int,
    eval_seeds: list[int],
) -> list[dict]:
    assert_no_final_leakage(validation_seeds)
    results = []
    for rank, cfg in enumerate(top_configs, start=1):
        params = {
            "terminal_collision_penalty": cfg["terminal_collision_penalty"],
            "merge_success_bonus": cfg["merge_success_bonus"],
            "non_merge_failure_penalty": cfg["non_merge_failure_penalty"],
            "rawlsian_lambda": cfg["rawlsian_lambda"],
            "rawlsian_epsilon": cfg["rawlsian_epsilon"],
        }
        ego = run_mode_on_seeds_extended(
            "egoistic",
            validation_seeds,
            episodes,
            max_steps,
            params,
            seed_phase="validation",
            eval_seeds=eval_seeds,
        )
        rawls = run_mode_on_seeds_extended(
            "rawlsian",
            validation_seeds,
            episodes,
            max_steps,
            params,
            seed_phase="validation",
            eval_seeds=eval_seeds,
        )
        results.append(
            {
                "rank": rank,
                "source_trial_number": cfg["trial_number"],
                "calibration_score": cfg["calibration_score"],
                **{f"egoistic_{k}": v for k, v in ego.items()},
                **{f"rawlsian_{k}": v for k, v in rawls.items()},
                **params,
            }
        )
    return results


def write_validation_summary(
    path: Path,
    top_configs: list[dict],
    validation_results: list[dict],
    calibration_seeds: list[int],
    validation_seeds: list[int],
) -> None:
    lines = [
        "# Optuna Validation Summary",
        "",
        "Validation runs are **report-only**. They do not continue tuning.",
        "",
        f"- Calibration seeds used during search: `{calibration_seeds}`",
        f"- Validation seeds used here: `{validation_seeds}`",
        f"- Final evaluation seeds (locked, not used): `{list(FINAL_EVALUATION_SEEDS)}`",
        "",
        "## Top candidate configs (from calibration)",
        "",
    ]
    for cfg in top_configs:
        lines.append(
            f"- Trial {cfg['trial_number']}: score={cfg['calibration_score']:.4f}, "
            f"tcp={cfg['terminal_collision_penalty']}, msb={cfg['merge_success_bonus']}, "
            f"nmfp={cfg['non_merge_failure_penalty']}, "
            f"λ={cfg['rawlsian_lambda']}, ε={cfg['rawlsian_epsilon']}"
        )
    lines.extend(["", "## Validation results", ""])
    if not validation_results:
        lines.append("_No validation runs performed (no constraint-passing trials)._")
    else:
        for row in validation_results:
            lines.append(f"### Rank {row['rank']} (trial {row['source_trial_number']})")
            lines.append("")
            lines.append(
                f"- Egoistic safe_merge (mean): "
                f"{row['egoistic_mean_safe_merge_success_rate']:.3f}"
            )
            lines.append(
                f"- Egoistic collision (mean): "
                f"{row['egoistic_mean_collision_rate']:.3f}"
            )
            lines.append(
                f"- Rawlsian safe_merge (mean): "
                f"{row['rawlsian_mean_safe_merge_success_rate']:.3f}"
            )
            lines.append(
                f"- Rawlsian collision (mean): "
                f"{row['rawlsian_mean_collision_rate']:.3f}"
            )
            lines.append(
                f"- Rawlsian min_experience (mean): "
                f"{row['rawlsian_mean_min_experience']:.3f}"
            )
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna calibration for V1 delta-min Rawlsian shaping."
    )
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument(
        "--calibration-seeds",
        type=str,
        default=",".join(str(s) for s in CALIBRATION_SEEDS),
    )
    parser.add_argument(
        "--validation-seeds",
        type=str,
        default=",".join(str(s) for s in VALIDATION_SEEDS),
    )
    parser.add_argument("--study-name", type=str, default="v1_optuna_calibration")
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (default: sqlite under output-dir).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip post-study validation on top configs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top constraint-passing trials to validate.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_seeds = parse_int_list(args.calibration_seeds)
    validation_seeds = parse_int_list(args.validation_seeds)
    eval_seeds = list(EVAL_SEEDS)

    assert_no_final_leakage(calibration_seeds)
    assert_no_final_leakage(validation_seeds)

    storage = args.storage or f"sqlite:///{output_dir / 'study.db'}"
    trial_rows: list[dict] = []

    sampler = TPESampler(seed=OPTUNA_SAMPLER_SEED)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    objective = create_objective(
        calibration_seeds=calibration_seeds,
        episodes=args.episodes,
        max_steps=args.max_steps,
        eval_seeds=eval_seeds,
        trial_rows=trial_rows,
    )

    print(
        f"Starting Optuna study '{args.study_name}' "
        f"({args.n_trials} trials, episodes={args.episodes}, "
        f"calibration_seeds={calibration_seeds})"
    )
    study.optimize(objective, n_trials=args.n_trials)

    # Persist outputs
    write_csv(output_dir / "trials.csv", trial_rows, TRIAL_CSV_FIELDS)

    top_configs = select_top_trials(trial_rows, top_k=args.top_k)
    write_csv(
        output_dir / "top_configs.csv",
        top_configs,
        TRIAL_CSV_FIELDS,
    )

    best = study.best_trial if study.trials else None
    best_payload: dict[str, Any] = {
        "study_name": args.study_name,
        "sampler_seed": OPTUNA_SAMPLER_SEED,
        "calibration_seeds": calibration_seeds,
        "validation_seeds": validation_seeds,
        "final_evaluation_seeds_locked": list(FINAL_EVALUATION_SEEDS),
        "reference_config": REFERENCE_CONFIG,
        "n_trials_requested": args.n_trials,
        "n_trials_completed": len(trial_rows),
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "eval_seeds_held_out": eval_seeds,
    }
    if best is not None:
        best_payload["best_trial"] = {
            "number": best.number,
            "value": best.value,
            "params": best.params,
            "user_attrs": {
                k: v for k, v in best.user_attrs.items() if k != "seed_metadata"
            },
        }
    with (output_dir / "best_trial.json").open("w", encoding="utf-8") as handle:
        json.dump(best_payload, handle, indent=2)

    summary_rows = [
        {
            "metric": "n_trials_completed",
            "value": len(trial_rows),
        },
        {
            "metric": "n_constraint_passing",
            "value": sum(1 for r in trial_rows if r.get("constraint_passed")),
        },
        {
            "metric": "best_calibration_score",
            "value": best.value if best else "",
        },
        {
            "metric": "best_trial_number",
            "value": best.number if best else "",
        },
    ]
    write_csv(
        output_dir / "calibration_summary.csv",
        summary_rows,
        ["metric", "value"],
    )

    validation_results: list[dict] = []
    if not args.skip_validation and top_configs:
        print(f"Running validation on top {len(top_configs)} configs ...")
        validation_results = run_validation(
            top_configs,
            validation_seeds,
            args.episodes,
            args.max_steps,
            eval_seeds,
        )
        val_fields = sorted(
            {k for row in validation_results for k in row.keys()}
        )
        write_csv(
            output_dir / "validation_top_configs.csv",
            validation_results,
            val_fields,
        )
        write_validation_summary(
            output_dir / "validation_summary.md",
            top_configs,
            validation_results,
            calibration_seeds,
            validation_seeds,
        )

    # Study pickle for portability
    import pickle

    with (output_dir / "study.pkl").open("wb") as handle:
        pickle.dump(study, handle)

    print(f"\nOptuna outputs written to {output_dir}")
    if best:
        print(f"Best trial: #{best.number} score={best.value:.4f}")


if __name__ == "__main__":
    main()
