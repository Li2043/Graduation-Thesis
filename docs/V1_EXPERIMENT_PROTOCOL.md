# V1 Experiment Protocol

> Status: **DRAFT SKELETON** — structure only. No experiment has been run under
> this protocol yet. This document defines *how* V1 experiments must be run so
> they are controlled, reproducible, and auditable. Fill `_TBD_` placeholders
> and record each change in `V1_DECISION_LOG.md` before freezing.

---

## 1. Conditions

V1 compares exactly two training conditions under identical everything-else.

### 1.1 Condition A — Egoistic DQN baseline

- Training reward: original environment reward only.
- No Rawlsian shaping term.
- Registry `condition` value: `baseline` _(TBD confirm label)_.

### 1.2 Condition B — Rawlsian DQN

- Training reward: original environment reward + Rawlsian maximin shaping term.
- Shaping parameters (e.g. `xi`, scope, experience mode): _TBD_, frozen before
  final runs.
- Registry `condition` value: `rawlsian` _(TBD confirm label)_.

> Both conditions must share the same environment, algorithm, network, training
> budget, observation/action spaces, and evaluation procedure. The **only**
> intended difference is the presence/absence of the Rawlsian shaping term.

---

## 2. Controlled variables

Held fixed and identical across both conditions (record exact values before
freezing):

| Variable | Value | Notes |
| --- | --- | --- |
| Environment ID | _TBD_ | dual controlled-vehicle setting |
| Number of controlled vehicles | _TBD_ (target: 2) | |
| Algorithm | _TBD_ (e.g. DQN) | |
| Policy network / architecture | _TBD_ | |
| Total training steps | _TBD_ | |
| Learning rate / buffer / batch / gamma / etc. | _TBD_ | |
| Observation space | _TBD_ | |
| Action space | _TBD_ | |
| Episode length / max steps | _TBD_ | |
| Evaluation episodes per seed | _TBD_ | |
| Fairness scope | _TBD_ | |
| Experience function | _TBD_ (see `V1_EXPERIENCE_DEFINITION.md`) | |

---

## 3. Train seed protocol

_TBD_. Skeleton rules:

- A fixed list of **training seeds** is defined before final runs and never
  changed afterwards: `TRAIN_SEEDS = _TBD_`.
- Both conditions are trained on the **same** train seed set.
- Each (condition, train seed) pair is one registry row.
- Seeds, library versions, and git commit are recorded per run.

---

## 4. Evaluation seed protocol

_TBD_. Skeleton rules:

- A fixed list of **evaluation seeds** is defined and kept **disjoint** from the
  training seeds: `EVAL_SEEDS = _TBD_`.
- The evaluation seed set is identical across both conditions.
- Evaluation seeds are referenced by an `eval_seed_set` identifier in
  `experiments/registry.csv`.

---

## 5. Calibration phase

_TBD_. Purpose: choose any free design parameters (e.g. shaping strength,
scope, weights) **before** the frozen evaluation.

Skeleton rules:

- Calibration uses a **separate** calibration seed set (disjoint from final eval
  seeds).
- All calibration decisions are logged in `V1_DECISION_LOG.md`.
- Calibration outcomes do **not** count as final V1 evidence.
- Once calibration ends, all chosen parameters are frozen.

---

## 6. Frozen final evaluation phase

_TBD_. Skeleton rules:

- Runs only after all parameters are frozen and recorded.
- Uses the predefined `EVAL_SEEDS` only.
- No parameter, reward, metric, or code change is permitted during this phase.
- Each final run is registered with `status = final` (or equivalent) in the
  registry and tied to a git commit.

---

## 7. No test-set tuning rule

- The final evaluation seed set is treated as a held-out test set.
- No design choice (parameters, scope, weights, metrics, stopping) may be made
  or revised by inspecting final-evaluation results.
- Any inspection of final-eval data that influences a design choice invalidates
  the run; a new frozen evaluation with fresh seeds is required.
- All tuning must occur in the calibration phase on calibration/validation seeds
  only.
