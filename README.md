# Rawlsian Reward Shaping for Fair Driving Behaviour in HighwayEnv



## 1. Research Goal

This project explores whether **Rawlsian maximin fairness** can be operationalised as a reward-shaping mechanism for autonomous driving agents.

The current test environment is `highway-env` `merge-v0`. The main comparison is between:

- **Baseline DQN**: trained with the original `highway-env` reward.
- **Rawlsian DQN**: trained with the original reward plus a Rawlsian fairness reward.


**Note:**
The current version is still a **single-agent DQN prototype**, not a full multi-agent reinforcement learning system. 

The controlled agent is the ego vehicle; other vehicles are background vehicles controlled by the environment.

---

## 2. Core Idea

### (1) Rawlsian maximin ethics

**RAWL·E** motivates the use of Rawlsian maximin ethics in reinforcement learning. The key idea is to reward actions that improve the minimum experience among agents and penalise actions that worsen it.

### (2) **SYMPLEX**

 **SYMPLEX** motivates the long-term goal of learning interpretable social norms through exploration, expert imitation, and human feedback. 
 
 This has not yet been implemented in the current code.

### (3) Agent-directed Runtime Norm Synthesis

**Agent-directed Runtime Norm Synthesis** motivates a future extension where norms can be revised during system operation when existing norms produce undesirable outcomes.

The current prototype implements only the first part: **Rawlsian reward shaping**.

---

## 3. Current Implementation

### (1) Environment

- `highway-env` environment: `merge-v0`
- Current learning algorithm: DQN from Stable-Baselines3
- Current setup: single controlled ego vehicle + background vehicles

### (2) Rawlsian reward

The Rawlsian reward is based on the minimum vehicle experience:

```text
if minimum experience improves:   +xi
if minimum experience worsens:    -xi
if unchanged:                      0
```

The shaped reward is:

```text
total_reward = original_highway_reward + rawlsian_reward
```

### (3) Current experience function

The current version uses a safety-mobility experience function:

```text
mobility_score = speed / target_speed

experience = W_MOBILITY  * mobility_score
  - W_COLLISION * collision_penalty
  - W_LOW_SPEED * low_speed_penalty
  - W_RISK      * risk_penalty
```

The least advantaged vehicle is the vehicle with the lowest experience within the ego-neighbourhood.

---

## 4. Prototype Progress

### v0.1 — Set up environment and Rawlsian Wrapper 

The first version tested whether `merge-v0` could run locally and whether a Rawlsian reward wrapper could be added without modifying `highway-env` source code.

Outcome:
- `merge-v0` ran successfully.
- Rawlsian Reward wrapper worked.

---
### v0.2 — Fairness Metrics Evaluation

v0.2 added shared fairness and safety metrics for both baseline and Rawlsian scenario.


Added metrics included:
  
| Metric            | Meaning                                                                  |
| ----------------- | ------------------------------------------------------------------------ |
| `min_experience`  | Experience of the worst-off vehicle; the main Rawlsian fairness metric.  |
| `mean_experience` | Average vehicle experience across the current scene.                     |
| `gini_experience` | Inequality of vehicle experience; lower values mean more equal outcomes. |
| `collision_count` | Number of crashed vehicles; used as a safety metric.                     |
| `n_vehicles`      | Number of vehicles in the environment at the current step.               |

#### Outcome:

v0.2 showed that the evaluation pipeline worked, but the Rawlsian wrapper did not improve fairness under a random policy. 
  
| Metric                  | Baseline | Rawlsian | Difference |
| ----------------------- | -------: | -------: | ---------: |
| `total_reward`          |    7.469 |    5.996 |     -1.474 |
| `mean_reward`           |    0.838 |    0.742 |     -0.097 |
| `mean_min_experience`   |    0.440 |    0.434 |     -0.006 |
| `final_min_experience`  |   -0.457 |   -0.515 |     -0.058 |
| `mean_gini_experience`  |    0.111 |    0.113 |     +0.002 |
| `total_collision_count` |    1.670 |    1.820 |     +0.150 |
| `steps`                 |    8.890 |    7.940 |     -0.950 |

---

### v0.3 — DQN Training with Rawlsian Reward Shaping

v0.3 introduced DQN training, so the reward signal could influence policy learning. 

#### Outcome:

However, the first trained comparison still showed **no difference** in fairness-related metrics between the baseline and Rawlsian DQN.  
  
| Metric                    | Baseline DQN | Rawlsian DQN | Difference |
| ------------------------- | -----------: | -----------: | ---------: |
| `total_reward`            |       10.094 |        9.234 |     -0.860 |
| `mean_reward`             |        0.866 |        0.775 |     -0.092 |
| `mean_min_experience`     |        0.458 |        0.458 |      0.000 |
| `final_min_experience`    |       -0.047 |       -0.047 |      0.000 |
| `mean_vehicle_experience` |        0.648 |        0.648 |      0.000 |
| `mean_gini_experience`    |        0.087 |        0.087 |      0.000 |
| `total_collision_count`   |        0.880 |        0.880 |      0.000 |
| `steps`                   |       11.580 |       11.580 |      0.000 |

The Rawlsian DQN had a lower total reward. However, all fairness and safety metrics were identical to the baseline. This suggested that the original global Rawlsian reward was not producing a useful behavioural difference.  

---

### v0.4 — Ego-Neighbourhood Rawlsian Reward

v0.4 changed the Rawlsian scope from all road vehicles to the ego-neighbourhood:

```text
Rawlsian set = ego vehicle + nearby vehicles within a radius
```

#### Outcome: 

| Metric | Baseline | Rawlsian | Difference |
|---|---:|---:|---:|
| `mean_min_experience` | 0.489 | 0.520 | +0.031 |
| `final_min_experience` | 0.060 | 0.578 | +0.518 |
| `least_advantaged_ego_ratio` | 0.020 | 0.053 | +0.033 |
| `total_collision_count` | 0.88 | 0.04 | -0.84 |

- Rawlsian DQN improved local minimum experience.
- Collision count dropped substantially.
- However, the least advantaged vehicle was still usually a nearby background vehicle, not the ego vehicle.

#### Conclusion:

- Ego-neighbourhood scope improved the signal but did not fully solve the controllability issue.

---

### v0.5 — Safety-Mobility Experience Function

v0.5 replaced the earlier speed-collision experience proxy with a safety-mobility experience function.

```
mobility_score = speed / target_speed
  
experience =  
W_MOBILITY * mobility_score  
- W_COLLISION * collision_penalty  
- W_LOW_SPEED * low_speed_penalty  
- W_RISK * risk_penalty
```

It also added `least_advantaged_reason` to records why a vehicle is least advantaged:

| Reason         | Meaning                                              |
| -------------- | ---------------------------------------------------- |
| `collision`    | The vehicle has crashed.                             |
| `low_mobility` | The vehicle is far below the target mobility level.  |
| `low_speed`    | The vehicle is nearly stopped or moving very slowly. |
| `risk`         | The vehicle is too close to another vehicle.         |
| `none`         | No clear reason is detected.                         |
| `combined`     | More than one reason contributes.                    |

#### Outcomes: 

| Metric                       | Baseline | Rawlsian | Difference |
| ---------------------------- | -------: | -------: | ---------: |
| `mean_min_experience`        |    0.069 |    0.157 |     +0.088 |
| `final_min_experience`       |    -0.78 |    -0.36 |      +0.42 |
| `mean_vehicle_experience`    |     0.30 |     0.35 |      +0.04 |
| `mean_gini_experience`       |    0.169 |    0.147 |     -0.023 |
| `total_collision_count`      |     0.88 |     0.56 |      -0.32 |
| `least_advantaged_ego_ratio` |    0.009 |    0.089 |     +0.080 |
| `mean_risk_penalty`          |    0.647 |    0.539 |     -0.108 |
| `mean_mobility_score`        |    0.688 |    0.643 |     -0.045 |
| `steps`                      |     11.6 |     16.1 |       +4.5 |

Interpretation:

- Rawlsian DQN improved the minimum safety-mobility experience.
- Collision and risk decreased.
- The least advantaged vehicle was more often the ego vehicle than before.
- Mobility decreased slightly, suggesting a safety/fairness versus speed trade-off.
- The model became more interpretable because the reason for disadvantage could be diagnosed.

Conclusion:

- v0.5 provides the strongest single-seed evidence so far that Rawlsian reward shaping can influence learned driving behaviour.

---

### v0.6.1 — Multi-Seed Robustness Test

v0.6.1 repeated training and evaluation over five seeds:

```python
SEEDS = [0, 1, 2, 3, 4]
```

The purpose is to test whether the v0.5 results were stable or only caused by a favourable random seed.

#### Multi-seed results

| Metric                       | Baseline | Rawlsian | Difference | Rawlsian better seeds |
| ---------------------------- | -------- | -------- | ---------- | --------------------- |
| `mean_min_experience`        | 0.119    | 0.171    | +0.053     | 3 out of  5           |
| `final_min_experience`       | -0.373   | -0.076   | +0.297     | 2 out of  5           |
| `mean_vehicle_experience`    | 0.329    | 0.359    | +0.030     | 3 / 5                 |
| `mean_gini_experience`       | 0.157    | 0.153    | -0.005     | 2 / 5                 |
| `total_collision_count`      | 0.544    | 0.296    | -0.248     | 2 / 5                 |
| `least_advantaged_ego_ratio` | 0.047    | 0.081    | +0.034     | 3 / 5                 |
| `mean_risk_penalty`          | 0.592    | 0.524    | -0.068     | 3 / 5                 |
| `mean_mobility_score`        | 0.668    | 0.642    | -0.026     | 0 / 5                 |
| `mean_collision_penalty`     | 0.020    | 0.008    | -0.012     | 3 / 5                 |
| `reason_risk_steps`          | 4.680    | 3.620    | -1.060     | 3 / 5                 |
| `steps`                      | 13.656   | 15.812   | +2.156     | 3 / 5                 |

- Rawlsian DQN still shows improvements on average, especially in local minimum experience, collision reduction, risk reduction, and least-advantaged ego ratio.
- However, the improvement is not consistent across all seeds.

Conclusion:

- More validation is needed.

---

## 6. Current Problems

The project currently has several limitations.

### Single-agent

The current system trains only one ego vehicle. Other vehicles are background vehicles. This is not yet a full multi-agent reinforcement learning system.

### Controllability

Even with ego-neighbourhood scope, the least advantaged vehicle is often a background vehicle. This means the DQN agent can only affect it indirectly.

### Experience function design

The safety-mobility experience function is still a proxy.
It may include:
- waiting time
- merge delay
- time-to-collision
- headway
- successful merge rate
- travel-time delay

### Robustness issue

v0.6.1 shows that results vary across seeds. The current findings are promising but preliminary.

### No human-in-the-loop component yet

The current code does not yet implement:

- expert feedback
- human-in-the-loop refinement
- runtime norm synthesis

---

## 7. Next Steps

### (1)  multi-agent extension

The current prototype is single-agent. A natural next step is to increase the number of controlled vehicles so that multiple agents participate in the fairness calculation.

This would make the Rawlsian maximin principle more consistent with the original multi-agent research goal.

### (2) Better fairness metrics

The vehicle experience function should be improved with more traffic-specific variables:

- waiting time
- merge delay
- time-to-collision
- headway
- distance progress
- successful merge rate
- risk exposure over time

### (3) Human-in-the-loop and symbolic norm learning


1. **Human-in-the-loop feedback**  
   Human participants could be recruited to provide feedback on traffic scenarios. Participants could judge whether a vehicle’s behaviour is fair, safe, too aggressive, or too conservative. These judgements could be used to adjust Rawlsian fairness weights.

2. **SYMPLEX-style symbolic norm learning**  
   Learned behaviours could be represented as interpretable rules using Logic Programming.



---

## 8. References

1. Woodgate, J., Marshall, P., & Ajmeri, N. (2025). *Operationalising Rawlsian Ethics for Fairness in Norm-Learning Agents*. Proceedings of the AAAI Conference on Artificial Intelligence.

2. Deane, O., & Ray, O. (2025). *SYMPLEX: A Human-in-the-Loop Approach to Learning Social Norms and Behavioural Policies*. World Conference on Explainable Artificial Intelligence.

3. Deane, O., & Ray, O. (2025). *Symplex: Learning Social Norm Hierarchies by Combining Autonomous Exploration and Expert Imitation*. AAMAS.

4. Morris-Martin, A., De Vos, M., Padget, J., & Ray, O. (2023). *Agent-directed Runtime Norm Synthesis*. AAMAS.

5. Leurent, E. (2018). *An Environment for Autonomous Driving Decision-Making*. highway-env.

6. Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
