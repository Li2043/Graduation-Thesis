# Prototype Log

## v0.4 — Ego-neighbourhood Rawlsian Reward

Date: 2026-05-20

### Purpose

To address the v0.3 finding that global minimum experience was dominated by non-ego background vehicles.

### Added

- Configurable Rawlsian scope (`global`, `ego_neighbourhood`, `controlled`).
- Ego-neighbourhood vehicle selection by radius.
- Scoped fairness metrics (`metric_scope`, `scoped_vehicle_count`).
- `diagnose_neighbourhood_scope.py`.
- Trained evaluation using ego-neighbourhood metrics by default.

### Interpretation

This version tests whether Rawlsian reward shaping becomes more learnable when the fairness target is restricted to vehicles interacting with the ego vehicle.

### Next step

If v0.4 improves learnability, v0.5 should improve the experience function by adding waiting time, merge delay, TTC, headway, or risk exposure.

---

## Diagnostic — Least Advantaged Vehicle Identity

Date: 2026-05-20

### Purpose

To understand whether the Rawlsian minimum-experience signal is controlled by the ego vehicle or dominated by background vehicles.

### Added

- Least advantaged vehicle identity metrics.
- Ego/non-ego ratio per episode.
- `diagnose_least_advantaged.py` diagnostic script.
- Additional random and trained summary fields and plots.

### Interpretation

If the least advantaged vehicle is rarely ego, then Rawlsian reward based on all vehicles may be difficult for a single-agent DQN to optimise. This would motivate v0.4 changes such as ego-centred experience, controlled-agent-only experience, or improved multi-agent setup.

---

## v0.3 — DQN Training with Rawlsian Reward Shaping

Date: 2026-05-20

### Purpose

To move from random-policy evaluation to trained-policy evaluation, so that Rawlsian reward shaping can affect the learned policy.

### Added

- Baseline DQN training.
- Rawlsian DQN training.
- Trained policy evaluation using shared fairness metrics.
- Model saving.
- Trained comparison plots and summary CSV.

### Interpretation

This is the first learning-based prototype. Results should be interpreted cautiously because training timesteps are limited and the current vehicle experience function is simplified.

### Next step

Prototype v0.4 should improve the vehicle experience definition by adding more meaningful traffic fairness variables, such as waiting time, merge delay, time-to-collision, headway, or risk exposure.

---

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
