# Rawlsian MARL Project

## Project goal

Compare standard highway-env rewards with a **Rawlsian Maximin** reward wrapper in the `merge-v0` scenario, to study whether fairness-oriented shaping improves the worst-off vehicle’s safety–mobility experience.

## Current stage

- Environment: **highway-env** `merge-v0` only
- Policies: **random** (v0.1–v0.2), **trained DQN** (v0.3), **ego-neighbourhood Rawlsian** (v0.4), **safety-mobility experience** (v0.5), **multi-seed robustness** (v0.6.1)
- Comparison: baseline vs. Rawlsian on **reward and fairness metrics**

---

## Prototype v0.3 — DQN Training with Rawlsian Reward Shaping

### Purpose

v0.3 introduces the first learning-based prototype. Unlike v0.1 and v0.2, which used random policies, v0.3 trains DQN agents so that the reward signal can influence policy learning.

### Added

- `train_dqn_baseline.py`
- `train_dqn_rawlsian.py`
- `trained_policy_utils.py`
- `evaluate_trained.py`
- `models/`
- `logs/`
- Trained evaluation CSV outputs and plots

### Experimental conditions

- **Baseline DQN**: trained on original highway-env reward.
- **Rawlsian DQN**: trained on original reward plus Rawlsian maximin shaping reward.
- **Environment**: `merge-v0`.
- **Evaluation**: deterministic policy evaluation using shared fairness metrics.

### How to run v0.3

```bash
python train_dqn_baseline.py
python train_dqn_rawlsian.py
python evaluate_trained.py
```

Training may take several minutes per agent (`DQN_TOTAL_TIMESTEPS = 20000`).

### Expected outputs

```text
models/dqn_baseline.zip
models/dqn_rawlsian.zip
results/trained_baseline_eval.csv
results/trained_rawlsian_eval.csv
results/trained_summary.csv
results/trained_comparison_total_reward.png
results/trained_comparison_min_experience.png
results/trained_comparison_gini.png
results/trained_comparison_collision.png
results/trained_comparison_steps.png
```

### Interpretation

- v0.3 tests whether Rawlsian reward shaping can influence **learned** policy.
- The key metric is **`mean_min_experience`**, not only total reward.
- If Rawlsian DQN improves minimum experience while maintaining acceptable collision count and reward, this is a promising preliminary result.
- If results are inconclusive, the likely next step is to improve the vehicle experience function.

### Limitations

- Still uses single controlled ego-agent setup in highway-env, not full MARL.
- Training timesteps are small and intended for prototype validation.
- Experience is still simplified as speed minus collision penalty.
- No SVO, no human-in-the-loop, no symbolic norm synthesis yet.

### Git (after verifying results)

```bash
git add .
git commit -m "Prototype v0.3: add DQN training and trained policy evaluation"
```

---

## Prototype v0.6.1 — Multi-seed Robustness Test

### Motivation

v0.5 showed promising improvements for Rawlsian DQN, but the result was based on a single seed. v0.6.1 tests whether the improvement is robust across multiple random seeds.

### Seeds

```python
SEEDS = [0, 1, 2, 3, 4]
```

- Seed controls randomness in environment initialisation, exploration, replay sampling, and neural network initialisation.
- Baseline and Rawlsian are trained and evaluated using the same seeds.

### What v0.6.1 does

- Train baseline DQN for each seed.
- Train Rawlsian DQN for each seed.
- Evaluate each trained model using shared safety-mobility fairness metrics.
- Aggregate results across seeds.

### How to run

```bash
python train_multiseed.py
python evaluate_multiseed.py
```

Training runs 10 models (5 seeds × 2 conditions) at `DQN_TOTAL_TIMESTEPS` each; expect roughly 1–2 hours total depending on hardware.

### Outputs

```text
models/v0.6.1/baseline_seed_0.zip
models/v0.6.1/rawlsian_seed_0.zip
...
results/v0.6.1/multiseed_episode_results.csv
results/v0.6.1/multiseed_aggregate_summary.csv
results/v0.6.1/multiseed_mean_min_experience.png
results/v0.6.1/multiseed_collision_count.png
results/v0.6.1/multiseed_least_advantaged_ego_ratio.png
```

### Interpretation

- If Rawlsian improves `mean_min_experience` across most seeds, the fairness improvement is more robust.
- If `total_collision_count` is lower across most seeds, the safety improvement is more robust.
- Check `rawlsian_better_seed_count` in the aggregate CSV: values near 5/5 suggest consistent Rawlsian advantage; values near 0 suggest instability.
- If results vary strongly by seed, the method is not yet stable and requires further tuning.

### Limitations

- Still single-agent DQN.
- Still merge-v0 only.
- Still no SVO, HITL, ILP, or full MARL.
- Five seeds are enough for prototype robustness, not final statistical validation.

---

## Prototype v0.5 — Safety-Mobility Experience Function

### Motivation

v0.4 showed that ego-neighbourhood scope improved Rawlsian learning signals, but the least advantaged vehicle was still usually a non-ego vehicle. The previous experience function was too simple because it only used speed and collision.

### Change

v0.5 introduces a safety-mobility experience function:

```text
mobility_score = clip(speed / target_speed, 0, 1)

experience =
    W_MOBILITY * mobility_score
  - W_COLLISION * collision_penalty
  - W_LOW_SPEED * low_speed_penalty
  - W_RISK * risk_penalty
```

### Least advantaged reason

v0.5 also records why a vehicle is least advantaged:

- `collision`
- `low_mobility`
- `low_speed`
- `risk`
- `none`
- `combined`

### How to run diagnostics

```bash
python diagnose_experience_components.py
```

### How to train and evaluate

```bash
python train_dqn_baseline.py
python train_dqn_rawlsian.py
python evaluate_trained.py
```

**Important:** Retrain both DQN agents after switching to v0.5 experience mode.

### Key outputs

- `results/v0.5/diagnostic/experience_components_diagnostics.csv`
- `results/v0.5/trained/trained_summary.csv`
- trained comparison plots for min experience, risk penalty, mobility score, and reason counts

### Limitations

- Still single-agent DQN.
- Still not full MARL.
- Risk is approximated using nearest vehicle distance.
- Mobility is still approximated using speed, not full travel delay or merge success.

---

## Prototype v0.4 — Ego-neighbourhood Rawlsian Reward

### Motivation

v0.3 diagnostics showed that the global least advantaged vehicle was usually a non-ego background vehicle. This made the Rawlsian reward weakly connected to the controlled DQN agent.

### Change

v0.4 introduces a configurable Rawlsian scope:

- `global`: all road vehicles
- `ego_neighbourhood`: ego vehicle plus nearby vehicles within a radius
- `controlled`: controlled vehicles only

Default in `config.py`:

```python
RAWLSIAN_SCOPE = "ego_neighbourhood"
EGO_NEIGHBOURHOOD_RADIUS = 50.0
```

### Interpretation

The ego-neighbourhood scope keeps the Rawlsian maximin idea but restricts it to vehicles whose experience is more likely to be affected by the ego vehicle’s actions.

### How to run diagnostics

```bash
python diagnose_neighbourhood_scope.py
```

### How to train and evaluate

```bash
python train_dqn_baseline.py
python train_dqn_rawlsian.py
python evaluate_trained.py
```

**Important:** Retrain Rawlsian DQN after switching to v0.4 scope (old models used global Rawlsian shaping).

### Key outputs

- `results/neighbourhood_scope_diagnostics.csv`
- `results/trained_summary.csv`
- `results/trained_comparison_min_experience.png`
- `results/trained_comparison_least_advantaged_ego_ratio.png`
- `results/trained_comparison_scoped_vehicle_count.png`

### Limitations

- Still single-agent DQN, not full MARL.
- Ego-neighbourhood is an approximation of interaction fairness.
- Radius is a hyperparameter.
- Experience is still simplified; v0.5 should add waiting time, merge delay, TTC, headway, or risk exposure.

---

## Diagnostic: Least Advantaged Vehicle Identity

### Why

- Current Rawlsian reward is based on the **minimum vehicle experience** across all vehicles.
- If the least advantaged vehicle is usually a **background vehicle** rather than the controllable **ego vehicle**, the DQN agent may have limited ability to improve this metric.
- This diagnostic records whether the least advantaged vehicle is ego or non-ego at each step.

### How to run

```bash
python diagnose_least_advantaged.py
```

To refresh random/trained summaries with the new fields, re-run:

```bash
python run_random_baseline.py
python run_random_rawlsian.py
python evaluate_random.py
python evaluate_trained.py
```

### Output

```text
results/least_advantaged_diagnostics.csv
```

### How to interpret `least_advantaged_ego_ratio`

| Value | Meaning |
|-------|---------|
| Close to **1.0** | Min experience usually refers to **ego** — DQN can plausibly influence it |
| Close to **0.0** | Min experience usually determined by **non-ego** background vehicles |
| Near **0.5** | Both ego and non-ego can be least advantaged |

Also check `results/trained_summary.csv` and `results/random_summary.csv` for aggregated `least_advantaged_ego_ratio`.

---

## Prototype v0.2 — Fairness Metrics Evaluation

### Added / changed in v0.2

- Added `metrics.py` to compute shared fairness and safety metrics.
- Baseline now records the same fairness metrics as Rawlsian runs.
- Rawlsian wrapper now reuses shared metrics functions.
- Evaluation compares total reward, minimum experience, Gini coefficient, and collision count.
- Fixed seeds; 100 random-policy episodes.

### How to run v0.2 (random policy)

```bash
python env_test.py
python run_random_baseline.py
python run_random_rawlsian.py
python evaluate_random.py
```

---

## Quick start

Use **Python 3.10–3.12**.

```powershell
cd C:\Users\HP\Desktop\thesis\rawlsian_project
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration (`config.py`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `ENV_ID` | `merge-v0` | |
| `N_EPISODES` | `100` | Random-policy runs (v0.2) |
| `MAX_STEPS` | `200` | |
| `RAWLSIAN_XI` | `0.2` | |
| `SPEED_NORMALIZER` | `30.0` | |
| `BASE_SEED` | `42` | |
| `DQN_TOTAL_TIMESTEPS` | `20000` | v0.3 prototype training |
| `EVAL_EPISODES` | `50` | v0.3 trained evaluation |
| `RAWLSIAN_SCOPE` | `ego_neighbourhood` | v0.4+ Rawlsian vehicle scope |
| `EGO_NEIGHBOURHOOD_RADIUS` | `50.0` | v0.4+ neighbourhood radius |
| `EXPERIENCE_MODE` | `safety_mobility` | v0.5 experience function |
| `TARGET_SPEED` | `30.0` | v0.5 mobility normalizer |
| `W_MOBILITY` / `W_COLLISION` / `W_LOW_SPEED` / `W_RISK` | `1.0` / `2.0` / `0.3` / `0.5` | v0.5 experience weights |
| `SEEDS` | `[0, 1, 2, 3, 4]` | v0.6.1 multi-seed robustness |

## Planned extensions (not in this repo yet)

- Social Value Orientation (SVO) rewards
- Human-in-the-loop feedback
- ILP / symbolic norm synthesis
- Richer vehicle experience (waiting time, TTC, merge delay)
