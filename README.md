# Rawlsian MARL Project

## Project goal

Compare standard highway-env rewards with a **Rawlsian Maximin** reward wrapper in the `merge-v0` scenario, to study whether fairness-oriented shaping improves the worst-off vehicle’s safety–mobility experience.

## Current stage

- Environment: **highway-env** `merge-v0` only
- Policies: **random** (v0.1–v0.2) and **trained DQN** (v0.3)
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

## Planned extensions (not in this repo yet)

- Social Value Orientation (SVO) rewards
- Human-in-the-loop feedback
- ILP / symbolic norm synthesis
- Richer vehicle experience (waiting time, TTC, merge delay)
