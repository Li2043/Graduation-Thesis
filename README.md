# Rawlsian MARL Project (v0.2)

## Project goal

Compare standard highway-env rewards with a **Rawlsian Maximin** reward wrapper in the `merge-v0` scenario, to study whether fairness-oriented shaping improves the worst-off vehicle’s safety–mobility experience.

## Current stage

- Environment: **highway-env** `merge-v0` only
- Policy: **random** (no RL training yet)
- Comparison: baseline vs. baseline + `RawlsianRewardWrapper` on **reward and fairness metrics**

## Prototype v0.2 — Fairness Metrics Evaluation

### Added / changed in v0.2

- Added `metrics.py` to compute shared fairness and safety metrics.
- Baseline now records the same fairness metrics as Rawlsian runs.
- Rawlsian wrapper now reuses shared metrics functions.
- Evaluation now compares total reward, minimum experience, Gini coefficient, and collision count.
- Added fixed seeds for more comparable random-policy runs.
- Increased default episodes from 10 to 100.
- Added `results/random_summary.csv`.
- Added separate plots for reward, minimum experience, Gini, and collisions.

### v0.2 limitations

- Still uses random policy.
- Rawlsian reward still does not influence behaviour until RL training is introduced.
- Vehicle experience is still simplified as speed minus collision penalty.
- This version is for **metric validation**, not final experimental evidence.

## Planned extensions (not in this repo yet)

- RL training (e.g. stable-baselines3 / DQN / PPO)
- Social Value Orientation (SVO) rewards
- Human-in-the-loop feedback
- ILP / symbolic norm synthesis

---

## Quick start

Use **Python 3.10–3.12** (see v0.1 install notes if needed).

```powershell
cd C:\Users\HP\Desktop\thesis\rawlsian_marl_project.v2
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python env_test.py
python run_random_baseline.py
python run_random_rawlsian.py
python evaluate_random.py
```

### Outputs (`results/`)

| File | Description |
|------|-------------|
| `random_baseline.csv` | Baseline random policy + fairness metrics |
| `random_rawlsian.csv` | Rawlsian wrapper + same fairness metrics |
| `random_summary.csv` | Mean/std and Rawlsian − baseline differences |
| `random_comparison_total_reward.png` | Average total reward |
| `random_comparison_min_experience.png` | Average mean min experience |
| `random_comparison_gini.png` | Average mean Gini |
| `random_comparison_collision.png` | Average total collision count |

---

## Run order

1. `env_test.py` — smoke test
2. `run_random_baseline.py` — 100 episodes, seed `BASE_SEED + episode_id`
3. `run_random_rawlsian.py` — same seeds, Rawlsian wrapper
4. `evaluate_random.py` — summaries, CSV, plots

## Configuration (`config.py`)

| Parameter | Value |
|-----------|-------|
| `ENV_ID` | `merge-v0` |
| `N_EPISODES` | `100` |
| `MAX_STEPS` | `200` |
| `RAWLSIAN_XI` | `0.2` |
| `SPEED_NORMALIZER` | `30.0` |
| `BASE_SEED` | `42` |
