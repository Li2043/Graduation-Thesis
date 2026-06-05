"""
v0.6.3: One-factor-at-a-time sensitivity analysis for Rawlsian DQN.

Trains and evaluates Rawlsian DQN under perturbed xi, neighbourhood radius, or W_RISK
while keeping other parameters at project defaults.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import (
    DQN_TOTAL_TIMESTEPS,
    EVAL_EPISODES,
    MAX_STEPS,
    SEEDS,
    SENSITIVITY_LOG_DIR,
    SENSITIVITY_MODEL_DIR,
    SENSITIVITY_QUICK_EVAL_EPISODES,
    SENSITIVITY_QUICK_TIMESTEPS,
    SENSITIVITY_RAW_CSV,
)
from sensitivity_utils import (
    SensitivityVariant,
    append_result_row,
    build_sensitivity_variants,
    evaluate_rawlsian_variant,
    load_existing_keys,
    model_path_for_variant,
    train_rawlsian_variant,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run v0.6.3 Rawlsian sensitivity analysis (train + evaluate)."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(SEEDS),
        help="Training/evaluation seeds (default: config.SEEDS)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer timesteps and eval episodes for debugging",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Optional subset of variant_id values (e.g. xi_0.1 radius_50)",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only train models; skip evaluation",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate existing models; skip training",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip train/eval when model or CSV row already exists",
    )
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="Overwrite sensitivity_raw_results.csv instead of merging rows",
    )
    return parser.parse_args()


def filter_variants(
    variants: list[SensitivityVariant],
    variant_ids: list[str] | None,
) -> list[SensitivityVariant]:
    if not variant_ids:
        return variants
    allowed = set(variant_ids)
    selected = [v for v in variants if v.variant_id in allowed]
    missing = allowed - {v.variant_id for v in selected}
    if missing:
        print(f"Warning: unknown variant_id(s) ignored: {sorted(missing)}")
    if not selected:
        print("Error: no variants matched --variants filter.")
        sys.exit(1)
    return selected


def run_training(
    variants: list[SensitivityVariant],
    seeds: list[int],
    *,
    skip_existing: bool,
    total_timesteps: int,
) -> None:
    for variant in variants:
        for seed in seeds:
            save_path = model_path_for_variant(
                PROJECT_ROOT, SENSITIVITY_MODEL_DIR, variant.variant_id, seed
            )
            if skip_existing and save_path.exists():
                print(f"Skip train (exists): {save_path}")
                continue
            train_rawlsian_variant(
                variant,
                seed,
                PROJECT_ROOT,
                SENSITIVITY_MODEL_DIR,
                SENSITIVITY_LOG_DIR,
                total_timesteps=total_timesteps,
            )


def run_evaluation(
    variants: list[SensitivityVariant],
    seeds: list[int],
    *,
    skip_existing: bool,
    n_episodes: int,
    max_steps: int,
    results_csv: Path,
    existing_keys: set[tuple[str, int]],
) -> None:
    for variant in variants:
        for seed in seeds:
            key = (variant.variant_id, seed)
            if skip_existing and key in existing_keys:
                print(f"Skip eval (in CSV): {key}")
                continue

            model_path = model_path_for_variant(
                PROJECT_ROOT, SENSITIVITY_MODEL_DIR, variant.variant_id, seed
            )
            if not model_path.exists():
                print(f"Error: missing model for eval: {model_path}")
                sys.exit(1)

            print(f"Evaluating {variant.variant_id} seed={seed} ...")
            row = evaluate_rawlsian_variant(
                variant,
                seed,
                PROJECT_ROOT,
                SENSITIVITY_MODEL_DIR,
                n_episodes=n_episodes,
                max_steps=max_steps,
            )
            append_result_row(results_csv, row)
            existing_keys.add(key)
            print(f"  mean_min_experience={row['mean_min_experience']:.4f} "
                  f"collisions={row['total_collision_count']:.4f}")


def main() -> None:
    args = parse_args()
    if args.train_only and args.eval_only:
        print("Error: cannot use --train-only and --eval-only together.")
        sys.exit(1)

    variants = filter_variants(build_sensitivity_variants(), args.variants)
    results_csv = PROJECT_ROOT / SENSITIVITY_RAW_CSV

    if args.no_append and results_csv.exists() and not args.eval_only:
        results_csv.unlink()
        print(f"Removed existing results file: {results_csv}")

    timesteps = SENSITIVITY_QUICK_TIMESTEPS if args.quick else DQN_TOTAL_TIMESTEPS
    n_episodes = SENSITIVITY_QUICK_EVAL_EPISODES if args.quick else EVAL_EPISODES

    print("v0.6.3 sensitivity analysis")
    print(f"  variants: {[v.variant_id for v in variants]}")
    print(f"  seeds: {args.seeds}")
    print(f"  quick={args.quick} timesteps={timesteps} eval_episodes={n_episodes}")
    print(f"  results -> {results_csv}")
    print(f"  models  -> {PROJECT_ROOT / SENSITIVITY_MODEL_DIR}")

    existing_keys = load_existing_keys(results_csv)

    if not args.eval_only:
        run_training(
            variants,
            args.seeds,
            skip_existing=args.skip_existing,
            total_timesteps=timesteps,
        )

    if not args.train_only:
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        run_evaluation(
            variants,
            args.seeds,
            skip_existing=args.skip_existing,
            n_episodes=n_episodes,
            max_steps=MAX_STEPS,
            results_csv=results_csv,
            existing_keys=existing_keys,
        )

    print("\nSensitivity run complete.")
    if not args.train_only:
        print(f"Raw results: {results_csv}")
        print("Next: python summarize_sensitivity.py")


if __name__ == "__main__":
    main()
