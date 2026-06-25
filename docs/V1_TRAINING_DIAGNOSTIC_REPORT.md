# V1 Training Diagnostic Report

> Diagnostic audit of why the learned V1 agent often fails to complete the
> merge (`eval_time_to_merge` frequently equal to `max_steps`). This report is
> **diagnostic only**: no reward, environment, policy, experience-function, or
> training-loop behaviour was changed. A non-invasive diagnostics harness
> (`v1/diagnostics/run_diagnostics.py`) was added, and short experiments were
> run. Evidence is in `experiments/diagnostics/`.

---

## 1. Executive summary

The merge task is **physically feasible and action 3 works**: with purely random
actions the agent completes the merge in 88% of episodes (Part 3A). The DQN is
**training correctly** (replay buffer fills, non-trivial losses, evolving
Q-values, scheduled epsilon). The trained egoistic policy already merges in 70%
of evaluation episodes after only 25 episodes and selects action 3 frequently.

The failures are therefore **not** caused by infeasibility, action execution,
exploration, or a broken update loop. The root cause is the **objective**: the
egoistic reward (and, by extension, the Rawlsian experience signal) rewards
forward progress toward the goal **regardless of lane**, and imposes **no cost
for failing to merge**. The non-merging evaluation episodes (5, 8, 9) end with
`final_lane = 1` (still on the ramp) and `final_position ≈ 194–204` — i.e. the
agent simply drove forward along the ramp past the goal distance and the episode
truncated at `max_steps`. Because not merging is reward-neutral (the lone ramp
vehicle has high TTC ⇒ zero risk penalty, high speed ⇒ zero waiting penalty),
the merge behaviour is under-determined and varies with seed/training length.

A secondary, compounding problem is a **logging/metric defect**: in
`v1/training/train.py`, `time_to_merge` is `steps if not collision else
max_steps` — it is actually episode length, never measures the merge step, and
maps both collisions and non-merge truncations onto `max_steps`. This overstates
"failure" and hides what actually happened.

---

## 2. Is merge physically feasible? **YES**

Part 3A (50 random-action episodes): merge completion **0.88**, reached merge
zone **1.0**, entered main lane **1.0**, avg time-to-merge when successful
**19.3 steps** (well within `max_steps = 60`). Random control merges most of the
time, so geometry/kinematics/`max_steps` are consistent and sufficient.

---

## 3. Is action 3 executable? **YES**

The lane-change action is executable across a wide region (`x ≥ conflict − 30`,
no upper bound). Part 3A produced 50 successful lane changes across 50 episodes;
the trained policy (Part 3C) selects action 3 as its most frequent greedy action
(126 of 301 steps) and successfully merges. Illegal attempts (out of zone /
already merged) are silently treated as "maintain speed" and do not stall the
agent.

---

## 4. Does the agent explore action 3 enough? **YES**

Epsilon-greedy over 4 actions samples action 3 ~25% of explore steps (Part 3A
frequency 0.254 ≈ uniform). The trained greedy policy uses action 3 at 0.42
frequency (Part 3C). Exploration of the merge action is not the bottleneck.

---

## 5. Does the reward encourage merge completion? **NO** (primary cause)

- **No explicit merge incentive.** `EgoisticReward` = `progress − collision −
  risk − waiting`. `progress` is the reduction in distance-to-goal and is earned
  by moving forward **on either lane**. There is no reward for changing lane or
  for completing the merge, and no penalty for ending the episode unmerged.
- **Non-merge is costless.** Part 3D (trained greedy, 301 steps): avg progress
  reward **0.0207**, avg risk penalty **0.0**, avg waiting penalty **0.0087**,
  collision-penalty steps **0**. A lone fast vehicle on the empty ramp has high
  TTC (risk ≈ 0) and never goes slow (waiting ≈ 0), so it keeps collecting
  positive progress reward without merging.
- **Direct evidence of the failure mode.** Non-merging eval episodes end at
  `final_lane = 1`, `final_position ≈ 194–204` — the agent drove the full ramp
  length without merging and truncated.
- **Rawlsian inherits the same gap.** `min_i E_i` uses the same experience
  components (mobility + safety − waiting); none reward merge completion, so the
  Rawlsian objective also does not require merging.
- **Merging is even mildly risky.** Part 3B (scripted "merge as early/fast as
  possible") collides in 100% of episodes (`blocked_by_main = 20/20`): merging
  aggressively into the busy main lane near the conflict point hits M/background
  traffic. Nothing in the reward teaches safe gap selection, so a value-greedy
  agent can rationally avoid merging altogether.

---

## 6. Is the observation sufficient? **MOSTLY YES (minor gaps)**

The trained agent reaches 70% merges, so the 5-dim observation
`[rel_position, rel_velocity, lane, (conflict − x)/goal, ttc_norm]` carries
enough signal to learn merging. Minor, non-blocking gaps that may slow learning:
- Distance is given relative to the **conflict point (80)**, not the
  **merge-complete point (100)** used for the success test.
- No explicit merge-eligibility flag and no **own absolute velocity** (only
  relative velocity to the other agent).

These are improvement opportunities, not the cause of failure.

---

## 7. Is DQN actually updating? **YES**

From `training_diagnostics.csv` (egoistic): replay buffer grows 42 → 417;
`train_updates` is 0 only while the buffer is below `batch_size` (episodes 0–1)
then runs every step; `mean_loss` is present and non-trivial (≈ 5e-4 … 8e-3);
`q_value_mean` evolves across training; epsilon decays on schedule (1.0 → 0.43
over 25 episodes). The learning machinery is functioning.

---

## 8. Why is `eval_time_to_merge` often equal to `max_steps`?

Two compounding reasons:

1. **The agent genuinely does not merge in those episodes** — it drives forward
   on the ramp to ≈ the goal distance and the episode hits the step limit
   (`truncated`). Because the objective does not require merging (Section 5),
   this is reward-neutral behaviour, not a malfunction.
2. **The metric is mislabelled (logging defect).** In `train.py`,
   `time_to_merge = steps if not collision else max_steps`. It never inspects
   whether a merge occurred; non-merge truncations yield `steps = max_steps`, and
   collisions are also forced to `max_steps`. So `time_to_merge == max_steps`
   conflates "never merged" with "crashed early" and is not a true time-to-merge.
   (The diagnostics harness measures the real merge step from the env `merged`
   flag — e.g. 15–22 steps for successful eval episodes.)

---

## 9. Ranked likely causes

1. **Objective does not encode the merge task (reward signal issue).** Forward
   progress is rewarded on any lane; failing to merge is costless ⇒ merge
   behaviour is under-determined and seed/training dependent. *Primary.*
2. **`time_to_merge` metric defect (logging).** Overstates failure and hides
   collisions vs. truncations vs. real merges. *Clear logging bug.*
3. **Under-training amplified by (1).** With no merge gradient, convergence is
   slow/unstable (70% at 25 episodes; worse in shorter/earlier pilots).
4. **No guidance for safe merge timing.** Merging is feasible but
   timing-sensitive (Part 3B: aggressive merge → 100% collision); the reward
   gives no signal to pick a gap, so avoidance is an easy local optimum.
5. **Minor observation gaps** (merge-complete distance, own velocity,
   eligibility flag) — secondary.

**Ruled out:** environment infeasibility (Part 3A 88%), action execution
(Part 3A 50/50 lane changes), exploration (action-3 freq ~0.25), DQN update
mechanism (Section 7), and evaluation epsilon (eval uses ε = 0 correctly).

---

## 10. Recommended next fixes (proposals only — not applied)

These require changes to modules that are out of scope for this diagnostic task;
they are listed for a future, explicitly-authorised change.

1. **Encode merge completion in the objective.** E.g. a terminal merge-success
   bonus and/or an explicit penalty for ending unmerged (or for traversing past
   the conflict point still on the ramp). *(reward / experience change)*
2. **Treat "reached goal distance on ramp without merging" as task failure** in
   the environment's termination/labelling. *(environment change)*
3. **Fix the `time_to_merge` metric** in `train.py` to record the actual merge
   step (from the env `merged` signal) and to log `merge_success` and a separate
   `collision` outcome rather than overloading `max_steps`. *(logging fix — the
   one item that qualifies as a clear logging bug)*
4. **Minor observation enrichment**: add distance to `merge_complete_position`,
   own absolute velocity, and/or a merge-eligibility flag. *(observation change)*
5. **Train longer / multiple seeds** once the objective rewards merging, and
   report distributions.

Per the task rules, none of the above were applied.

---

## Appendix — diagnostic experiment results

| Experiment | merge rate | collision rate | reached zone | entered main | action-3 freq | notes |
|---|---|---|---|---|---|---|
| 3A random (50 ep) | 0.88 | 0.16 | 1.0 | 1.0 | 0.254 | avg merge time 19.3 |
| 3B scripted merge-ASAP (20 ep) | 0.05 | 1.00 | 1.0 | 1.0 | 0.187 | blocked/collided 20/20 |
| 3C trained egoistic greedy (10 ep) | 0.70 | 0.00 | 1.0 | 0.70 | 0.419 | greedy actions [0:83,1:24,2:68,3:126] |

Part 3D (egoistic reward components, trained greedy, 301 steps): avg progress
0.0207, avg risk 0.0, avg waiting 0.0087, collision-penalty steps 0. Reward
components ARE accessible via `EgoisticReward` methods, so step-level component
logging is feasible if desired later.

CSV artefacts: `experiments/diagnostics/action_distribution.csv`,
`merge_diagnostics.csv`, `reward_diagnostics.csv`, `training_diagnostics.csv`,
`part3_summary.json`.
