# Rawlsian MARL Project

## Project goal

Compare standard highway-env rewards with a **Rawlsian Maximin** reward wrapper in the `merge-v0` scenario, to study whether fairness-oriented shaping improves the worst-off vehicle’s safety–mobility experience.

## Current stage: minimal prototype

- Environment: **highway-env** `merge-v0` only
- Policy: **random** (no RL training yet)
- Comparison: baseline reward vs. baseline + `RawlsianRewardWrapper`

## Planned extensions (not in this repo yet)

- RL training (e.g. stable-baselines3)
- Social Value Orientation (SVO) rewards
- Human-in-the-loop feedback
- ILP / symbolic norm synthesis

---

## Quick start（已装好环境时）

```powershell
cd C:\Users\HP\Desktop\thesis\rawlsian_marl_project
.\.venv\Scripts\Activate.ps1
python env_test.py
python run_random_baseline.py
python run_random_rawlsian.py
python evaluate_random.py
```

Outputs are written under `results/`.



# results

The initial random-policy comparison shows ==no substantial difference== between the baseline and the Rawlsian wrapper in average total reward. 

This is expected because the current experiment is only a smoke test: both agents use random actions, so the Rawlsian reward does not influence behaviour or policy learning. The wrapper only changes the reward signal returned by the environment. Moreover, the current Rawlsian shaping term is small and may be positive or negative depending on whether the minimum vehicle experience improves between consecutive steps. 

Therefore, the lack of difference in total reward should not be interpreted as evidence against the Rawlsian mechanism. 

The next step is to record fairness-specific metrics, especially minimum vehicle experience, for both baseline and Rawlsian conditions, increase the number of episodes, fix random seeds, and then train agents using the shaped reward.