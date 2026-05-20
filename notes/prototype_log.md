# Prototype Log

## v0.2 — Fairness Metrics Evaluation

Date: 2026-05-20

### Purpose

To make baseline and Rawlsian random-policy runs comparable on fairness-oriented metrics, not only total reward.

### Added

- ==Shared metrics module.==
- Minimum experience logging for ==both== baseline and Rawlsian conditions.
- Gini coefficient of vehicle experience.
- Collision count.
- Fixed random seeds.
- Summary CSV and multiple comparison plots.

### Interpretation

This version tests whether the evaluation pipeline can measure Rawlsian-relevant outcomes. It is not yet a test of whether Rawlsian reward improves learned behaviour, because the policy remains random.

### Next step

Prototype v0.3 should introduce RL training, likely with DQN or PPO, so that the Rawlsian reward can influence policy learning.

---

## v0.1 — Random Policy Smoke Test

Date: 2026-05-20

### Purpose

This version tests whether highway-env `merge-v0` can be executed locally and whether a Rawlsian reward wrapper can be added externally without modifying highway-env source code.

### Implemented

- `env_test.py`: runs `merge-v0` with random actions.
- `run_random_baseline.py`: runs random policy without Rawlsian reward.
- `rawlsian_wrapper.py`: adds Rawlsian maximin reward shaping.
- `run_random_rawlsian.py`: runs random policy with Rawlsian wrapper.
- `evaluate_random.py`: compares average total reward.

### Observation

The average total reward of the baseline and Rawlsian wrapper is similar. This is expected because both models use random actions.

### Limitation (addressed in v0.2)

- Baseline did not record fairness metrics.
- Only 10 episodes; no fixed seed.
- Evaluation compared total reward only.
