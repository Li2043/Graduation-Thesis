# Prototype Log

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

### Current Rawlsian reward
Vehicle experience is defined as:

`experience_i = normalized_speed_i - collision_penalty_i`

The Rawlsian reward is:

- `+xi` if the minimum vehicle experience improves.
- `-xi` if the minimum vehicle experience worsens.
- `0` if unchanged.

### Observation
The average total reward of the baseline and Rawlsian wrapper is similar. This is expected because both models use random actions. The Rawlsian reward changes the reward signal but does not yet affect behaviour or policy learning.

### Limitation
- Baseline does not yet record `min_experience`.
- Only 10 episodes were used.
- No fixed random seed.
- The current experience function is very simple.
- No RL training has been conducted.
- The current plot compares total reward, not fairness.

### Next step
Develop v0.2:
- Record fairness metrics for both baseline and Rawlsian conditions.
- Add fixed seeds.
- Increase episodes to 100 or more.
- Compare `mean_min_experience`, `final_min_experience`, and possibly Gini coefficient.