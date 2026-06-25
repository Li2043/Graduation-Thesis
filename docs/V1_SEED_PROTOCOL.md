# V1 Seed Protocol

Status: **defined and plumbed; Optuna not yet implemented.** This document is
the human-readable contract for the seed protocol. The machine-readable source
of truth is `v1/calibration/seed_protocol.py`; the two must always agree.

---

## 1. Why seeds are split into calibration, validation, and final evaluation

Hyperparameter calibration (Optuna) repeatedly trains and evaluates the system
and selects the configuration that scores best. If the configuration is selected
using the same seeds that later produce the reported results, the reported
numbers are optimistically biased — the search has implicitly fit to those
seeds. To prevent this we split seeds into three disjoint roles, mirroring the
standard train / dev / test discipline:

- **calibration_seeds** — the only seeds the Optuna objective is allowed to see.
  Used to *search* the hyperparameter space.
- **validation_seeds** — used *once*, after Optuna finishes, to sanity-check the
  top candidate configuration(s) on seeds the search never optimised against.
- **final_evaluation_seeds** — **locked**. Used only for the final reported
  experiment, after the configuration is frozen. Never seen by calibration or
  validation.

This guarantees the final numbers are produced on seeds that played no role in
choosing the hyperparameters.

---

## 2. Exact seed sets

| role | seeds | may be used by |
|---|---|---|
| `calibration_seeds` | **1, 2, 3** | Optuna objective (search) |
| `validation_seeds` | **4, 5** | post-Optuna check of top candidates only |
| `final_evaluation_seeds` | **100, 101, 102, 103, 104** | final frozen experiment only |
| Optuna sampler seed | **42** (fixed) | the sampler RNG, for reproducible search |

These are **run/trial seeds** — the value passed to `train.py --seed`. A run with
seed `s`:

- seeds the global RNG via `seed_everything(s)`;
- derives per-episode training seeds as `s * 100000 + episode`
  (so different run seeds occupy disjoint 100000-blocks, and episodes must stay
  below 100000);
- evaluates the trained policy on a **held-out evaluation seed set** (the default
  `EVAL_SEEDS = [9001..9010]`, overridable with `--eval-seeds`).

The three sets are pairwise disjoint, and the training-seed blocks for
`{1,2,3}`, `{4,5}`, `{100..104}` never overlap each other or the held-out
evaluation seeds. `seed_protocol.validate_protocol()` asserts disjointness and
runs at import time.

### Held-out evaluation scenarios (`--eval-seeds`)

Each run is *measured* on a held-out evaluation seed set. By default this is the
fixed `EVAL_SEEDS = [9001..9010]`, which is disjoint from every training seed.
`train.py` now exposes `--eval-seeds` so that, if desired, calibration and the
final experiment can be measured on **distinct** evaluation scenarios (passing
calibration eval seeds during search and a separate locked set for the final
run). When `--eval-seeds` is omitted, the default held-out set is used and
behaviour is unchanged from previous runs.

---

## 3. Why poor seeds must not be removed

Dropping or replacing a seed because it performs badly is a form of cherry-
picking: it inflates the apparent performance and hides instability that a
reader (or a real deployment) would actually experience. The protocol therefore
**fixes the seed sets in advance** and reports results over *all* of them.
A configuration that only works on 2 of 3 calibration seeds is genuinely less
robust than one that works on all 3, and the metrics must reflect that. This is
exactly why the robustness metrics below include worst-seed / max-seed / std,
not just means.

---

## 4. How Optuna avoids final-seed leakage

1. The Optuna objective trains/evaluates **only** with `calibration_seeds`.
2. `seed_protocol.assert_no_final_leakage(seeds_used)` is called with every seed
   a trial intends to use; it raises if any `final_evaluation_seed` appears. The
   same guard is built into `build_trial_seed_metadata(...)`.
3. `validation_seeds` are used only after the search completes, to check the top
   candidate(s) — never inside the objective.
4. `final_evaluation_seeds` are not referenced anywhere in calibration or
   validation code paths; they are reserved for the frozen final experiment.
5. Egoistic and Rawlsian use the **identical** seed set within each trial (the
   training-seed formula is mode-independent and the eval seeds are shared), so
   the only difference between conditions is the objective, never the seeds.
6. Every trial records `trial_number`, `sampler_seed`, all three seed sets, and
   the exact `train_seeds_used` / `eval_seeds_used` (via
   `build_trial_seed_metadata`). Every single run records the same information in
   its config JSON under `seed_metadata` (via `single_run_seed_metadata`).

---

## 5. How robustness across seeds is assessed

The Optuna objective aggregates each candidate's per-seed eval metrics (over the
calibration seeds) into mean **and** robustness statistics
(`seed_protocol.aggregate_seed_metrics`):

- `mean_safe_merge_success_rate`
- `worst_seed_safe_merge_success_rate`
- `std_safe_merge_success_rate`
- `mean_collision_rate`
- `max_seed_collision_rate`
- `mean_non_merge_failure_rate`
- `mean_min_experience`
- `mean_gini_experience`

Suggested hard constraints (same for both modes;
`seed_protocol.check_hard_constraints`):

- `mean_safe_merge_success_rate >= 0.6`
- `mean_collision_rate <= 0.3`
- `mean_non_merge_failure_rate <= 0.3`
- `worst_seed_safe_merge_success_rate >= 0.3` (if feasible)

The primary comparison metrics across modes remain
`eval_safe_merge_success_rate`, `eval_collision_rate`,
`eval_non_merge_failure_rate`, `eval_min_experience`, `eval_mean_experience`,
`eval_gini_experience`, and `eval_mean_time_to_merge_success_only`.
**`eval_episode_reward` must not be compared across modes** (the two conditions
have different reward compositions).

---

## 6. Locking rule

`final_evaluation_seeds = [100, 101, 102, 103, 104]` are **locked**. They must
not be used during calibration or validation under any circumstances, and must
not be changed once the final experiment begins. Runs that belong to the final
experiment must be launched with `--seed-phase final` (and seed ∈ that set) so
they are clearly labelled in `seed_metadata`. Any change to these seeds requires
a new dated entry in `docs/V1_DECISION_LOG.md`.

---

## Usage

```bash
# Calibration run (Optuna will drive these): seed in {1,2,3}
python -m v1.training.train --mode rawlsian --seed 1 --seed-phase calibration ...

# Validation run (after Optuna): seed in {4,5}
python -m v1.training.train --mode rawlsian --seed 4 --seed-phase validation ...

# Final experiment (locked): seed in {100..104}
python -m v1.training.train --mode rawlsian --seed 100 --seed-phase final ...
```

Each run writes its seed metadata to `experiments/.../configs/<run_id>.json`
under the `seed_metadata` key.
