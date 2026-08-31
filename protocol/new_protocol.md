# Independent-Seed Replication and Post-Coordination Welfare Adaptation Protocol

## Protocol status

**FINAL PROTOCOL — VERSION 1**

This protocol defines an additional six-seed replication experiment.

It does not replace the original six-seed formal experiment.

The original formal results remain unchanged and remain part of the thesis.

The new experiment has three purposes:

1. test whether the strong training-seed dependence observed in the original experiment reproduces in new independently trained policies;
2. test whether GGI or Maximin produces a consistent change in worst-off welfare relative to Mean;
3. examine whether second-stage welfare adaptation changes the distribution and behavioural pattern of coordination under the same decentralized local-information constraint.

No original seed is removed or replaced.


# 1. Research motivation

The original formal experiment used six independently trained seed lineages.

The final policies showed substantial variation across training seeds.

One seed, 910102, showed near-total failure under Mean, while the matched GGI and Maximin branches performed substantially better.

Because the number of independent training seeds was only six, one extreme learned policy had a large effect on the condition-level estimates.

The new experiment therefore tests whether this seed dependence reproduces in a new independent sample of training lineages.


# 2. Two-stage interpretation of the study

The study treats fairness as a second-stage adaptation problem.

The first stage learns basic decentralized highway-merging coordination:

$$
\text{task learning}
$$

The second stage introduces a social-welfare objective:

$$
\text{welfare adaptation}
$$

The complete structure is therefore:

$$
\boxed{
\text{task coordination}
\rightarrow
\text{welfare adaptation}
}
$$

This structure is motivated by previous multi-agent work showing that prosocial or fairness-oriented behaviour can be introduced after basic coordination has stabilized.

However, the present study does not assume that late-stage fairness adaptation must improve performance.

The effect is tested empirically.


# 3. Important difference from direct inequity-aware methods

The decentralized information constraint from the original experiment is retained.

A vehicle does not observe:

- another vehicle's utility;
- another vehicle's accumulated mobility burden;
- another vehicle's target speed;
- another vehicle's welfare rank;
- the social-welfare value;
- another vehicle's reward;
- another vehicle's intended action.

Vehicles continue to act only from their original local observations.

Therefore, fairness information affects policy learning only through the training reward.

This differs from inequity-aware approaches in which an agent directly observes another agent's accumulated cost and uses that information to decide whether to concede.

The present replication therefore asks whether second-stage welfare adaptation is reproducible under a stricter local-information setting.


# 4. Diagnostic questions

## DQ1 — Training robustness

How much does final task competence vary across six new independently trained seed lineages?

This is evaluated separately for:

- Mean;
- GGI;
- Maximin.


## DQ2 — Worst-off welfare

Do GGI or Maximin show a consistent improvement in worst-off utility relative to Mean across new independent seeds?

The primary matched contrasts are:

$$
\Delta U_{\min}^{GGI}
=
U_{\min}^{GGI}
-
U_{\min}^{Mean}
$$

and:

$$
\Delta U_{\min}^{Maximin}
=
U_{\min}^{Maximin}
-
U_{\min}^{Mean}.
$$


## DQ3 — Replication of extreme Mean-policy failure

Does a new Mean lineage show severe task failure similar to the pattern observed for seed 910102?

If such a failure occurs, do its matched GGI or Maximin branches perform better?

This is a descriptive replication question.

No new seed is selected because it produces this pattern.


## DQ4 — Distributional fairness

Do the welfare objectives change:

- the welfare floor;
- utility inequality;
- mobility burden;
- the upper tail of mobility burden?

Fairness is not judged using a single metric.


## DQ5 — Behavioural coordination

When a vehicle has accumulated relatively high mobility burden during an episode, is this followed by a different coordination outcome under Mean, GGI, or Maximin?

This is an offline behavioural diagnostic.

It does not imply that the policy directly observes burden.


# 5. Original formal seeds

The original six formal seeds remain:

- 900101
- 900102
- 900103
- 900104
- 910101
- 910102

These runs are not retrained.

Their existing formal results remain the original primary experiment.


# 6. New replication seeds

The proposed new seed block is:

- 920101
- 920102
- 920103
- 920104
- 920105
- 920106

Before any new training begins, the existing project directories, manifests, logs, and run folders must be searched to confirm that these IDs have never been used for:

- smoke tests;
- pilot experiments;
- qualification;
- training;
- formal evaluation.

If all six IDs are unused, this seed set is frozen.

If any ID has previously been used, training must not begin.

A new contiguous six-seed block must then be documented in a protocol amendment before any result from the new block is observed.

Once training starts, the seed set cannot change because of performance.


# 7. Stage 0 — Reproduce and audit the existing formal experiment

**This stage must be completed before any new training begins.**

The previous formal training package is located at:

`F:\正式训练`

This directory previously contained the formal scripts, environment setup, README documentation, checkpoints, and launch procedure that were copied to the 32-core training machine.

The original directory must be treated as the reference implementation.


## 7.1 Do not modify the original directory

Do not directly edit:

`F:\正式训练`

Create a separate copy, for example:

`F:\正式训练_seed_replication_v1`

The original directory remains read-only reference material.


## 7.2 Audit the project package

Before adding any new seeds, identify and record:

- training entry script;
- evaluation entry script;
- behavioural evaluation script;
- environment or requirements file;
- README;
- configuration files;
- seed manifest or seed-list mechanism;
- formal scenario banks;
- checkpoint naming convention;
- output directory structure;
- environment version;
- Python version;
- package versions;
- git commit/hash if available.


## 7.3 Re-run existing evaluation first

Before training new policies, use the existing formal evaluation scripts on the original six formal checkpoints.

The expected welfare evaluation script is the existing formal script such as:

`evaluate_formal_welfare.py`

The existing behavioural analysis script should also be identified, such as:

`evaluate_formal_behavioral.py`

Use the scripts and command structure documented in the existing README.

Do not create a new evaluation implementation unless the original one cannot be reproduced.


## 7.4 Reproduction target

The reproduced evaluation must match the previously recorded formal results within deterministic numerical tolerance.

At minimum, verify for every original seed:

- completion;
- collision;
- timeout;
- $U_{\mathrm{mean}}$;
- $U_{\min}$.

If the reproduced evaluation differs materially from the current Results chapter, stop.

Resolve the discrepancy before new training.


## 7.5 Save the evaluation audit

Create:

`replication_preflight_audit.md`

It must record:

- machine used;
- date;
- environment;
- command used;
- evaluation scripts;
- checkpoint locations;
- evaluation-bank hashes if available;
- whether the original results were reproduced.


# 8. Stage 1 — Freeze the new training package

Once Stage 0 passes, use the copied project directory:

`F:\正式训练_seed_replication_v1`

Do not redesign the training implementation.

Reuse the original:

- scripts;
- configuration;
- environment;
- scenario generator;
- reward implementation;
- DQN implementation;
- checkpoint logic;
- evaluation pipeline.

The intention is to change only the seed IDs and output locations.


# 9. Seed manifest

Create a new manifest:

`new_seed_manifest.csv`

with at least:

| seed | status | curriculum_dir | Mean_dir | GGI_dir | Maximin_dir |
|---:|---|---|---|---|---|
| 920101 | frozen | ... | ... | ... | ... |
| 920102 | frozen | ... | ... | ... | ... |
| 920103 | frozen | ... | ... | ... | ... |
| 920104 | frozen | ... | ... | ... | ... |
| 920105 | frozen | ... | ... | ... | ... |
| 920106 | frozen | ... | ... | ... | ... |

All six seeds must be retained regardless of learned performance.


# 10. Technical smoke test

Before running the six formal replication seeds on the 32-core machine, perform one short technical smoke test.

The smoke-test seed must not be one of the six replication seeds.

For example:

`929999`

The smoke test checks only:

- script launches;
- environment loads;
- checkpoints save;
- logs write correctly;
- replay buffer works;
- evaluation script can locate outputs.

The smoke-test result is not used scientifically.

Delete or clearly isolate its outputs from the formal replication directory.


# 11. Task-only curriculum

Each of the six new seeds begins from a fresh random initialization.

Each follows the same curriculum as the original formal experiment:

$$
M6_{R50}
\rightarrow
C4_{R50}
\rightarrow
C16_{R50}
\rightarrow
C64_{R50}.
$$

The frozen budgets are:

| Stage | Stage steps | Cumulative steps |
|---|---:|---:|
| $M6_{R50}$ | 400,000 | 400,000 |
| $C4_{R50}$ | 300,000 | 700,000 |
| $C16_{R50}$ | 250,000 | 950,000 |
| $C64_{R50}$ | 250,000 | 1,200,000 |

Each new seed therefore receives:

$$
1{,}200{,}000
$$

task-training steps.

No performance-based extension is introduced.


# 12. Task checkpoint

At:

$$
S=1{,}200{,}000
$$

the final task-only checkpoint is frozen for each seed.

For a seed $s$:

$$
\theta_s^{task}
$$

becomes the shared starting point for all three welfare branches.


# 13. Matched welfare branching

For every seed:

$$
\theta_s^{task}
\rightarrow
\begin{cases}
\theta_s^{Mean}\\
\theta_s^{GGI}\\
\theta_s^{Maximin}
\end{cases}
$$

The three branches must start from exactly the same task-only checkpoint for that seed.

Each branch receives:

$$
800{,}000
$$

additional environment steps.

Therefore every welfare run ends at:

$$
2{,}000{,}000
$$

absolute environment steps.


# 14. Welfare objectives

The welfare coefficient remains:

$$
\lambda_W=0.5.
$$

Mean:

$$
W_{Mean}
=
\frac{1}{4}
\sum_i U_i.
$$

GGI:

$$
W_{GGI}
=
0.4U_{(1)}
+
0.3U_{(2)}
+
0.2U_{(3)}
+
0.1U_{(4)}.
$$

Maximin:

$$
W_{Maximin}
=
\min_i U_i.
$$

No welfare definition is modified.


# 15. Frozen training configuration

The new replication must reuse the original formal configuration.

This includes:

- 18-dimensional local observation;
- $R=50\,\mathrm{m}$ sensing radius;
- parameter-shared decentralized DQN;
- network architecture;
- optimizer;
- replay buffer;
- batch size;
- discount factor;
- Double DQN;
- target-network update;
- learning-rate schedule;
- epsilon schedule;
- action mapping;
- meta-speed control;
- task reward;
- terminal welfare reward;
- simulator timing;
- physical acceleration limits.

No hyperparameter tuning is performed using the new six seeds.


# 16. Environment reproduction on the 32-core machine

The copied replication directory should be transferred to the same 32-core machine in the same way as the previous formal training package.

Before running training:

1. install or activate the environment documented in the original README;
2. verify package versions;
3. verify that the original evaluation from Stage 0 can also run on the 32-core machine;
4. verify paths to the scenario banks;
5. verify output paths;
6. verify CPU-process parallelism.

If the old README contains the original formal launch command, reuse that command.

Only:

- seed list;
- new output root;

should change unless the old code requires a documented path adjustment.


# 17. Parallel execution

The six curriculum runs are independent and may run in parallel.

After each curriculum reaches the frozen $C64_{R50}$ checkpoint, its Mean, GGI, and Maximin branches are also independent.

The replication contains:

$$
6
$$

task curricula,

and:

$$
6\times3=18
$$

welfare fine-tuning runs.

The 18 welfare branches may use the same process-level parallel strategy as the original formal launch if machine memory and CPU capacity permit.


# 18. Total training volume

Task training:

$$
6\times1.2M
=
7.2M
$$

steps.

Welfare fine-tuning:

$$
18\times0.8M
=
14.4M
$$

steps.

Total:

$$
21.6M
$$

environment steps.


# 19. Technical failure rule

A technical failure includes examples such as:

- process crash;
- corrupted checkpoint;
- incorrect configuration loaded;
- missing evaluation bank;
- NaN corruption;
- truncated output caused by infrastructure failure.

Poor learned policy performance is not a technical failure.

If a technical failure occurs:

- rerun the same seed;
- use the same condition;
- use the same configuration.

Do not replace the seed.


# 20. Stage 2 — New-seed evaluation

After all welfare runs finish, evaluate the new six seeds.

Evaluation must use the same formal evaluation implementation as Stage 0.


# 21. Primary H1 evaluation

For every new seed, evaluate:

- Mean;
- GGI;
- Maximin;

on the frozen H1 held-out bank.

Use:

$$
\epsilon=0
$$

and:

$$
R=50\,\mathrm{m}.
$$


# 22. Checkpoint-Q ensemble

Use the same final checkpoint ensemble:

$$
K(2{,}000{,}000)
=
\{
1{,}850{,}000,
1{,}900{,}000,
1{,}950{,}000,
2{,}000{,}000
\}.
$$

Use equal checkpoint weight.

No best-checkpoint selection is allowed.


# 23. Secondary H0 evaluation

Because evaluation is inexpensive compared with training, also evaluate the new Mean policies on H0.

This provides a secondary replication of the original H0-versus-H1 Mean comparison.

It does not replace the original RQ1 analysis.

Therefore:

- Mean: H0 and H1;
- GGI: H1;
- Maximin: H1.


# 24. Primary evaluation metrics

For every seed and condition, report:

- completion rate;
- collision rate;
- timeout rate;
- mean utility $U_{\mathrm{mean}}$;
- worst-off utility $U_{\min}$;
- utility Gini;
- mean burden $C_{\mathrm{mean}}$;
- maximum burden $C_{\max}$.


# 25. Tail fairness metric

Previous fairness work motivates examining the tail of the outcome distribution rather than only its mean.

Therefore calculate a secondary exploratory burden-tail metric:

$$
C_{95}
=
Q_{0.95}(C_i).
$$

$C_{95}$ is the 95th percentile of vehicle-level mobility burden.

It asks:

> How severe is mobility loss among the upper tail of burden outcomes?

This metric is secondary.

It does not replace $U_{\min}$ as the primary RQ2 welfare outcome.


# 26. Burden inequality

Also report:

- burden range;
- burden Gini when mathematically defined.

Do not force an all-zero burden vector to have Gini equal to zero if the existing project convention defines the quantity as undefined.

Undefined cases must be reported explicitly.


# 27. Successful-episode sensitivity analysis

Repeat the principal burden measures using only episodes with:

`completion == 1`

Report at least:

- success-only $C_{\mathrm{mean}}$;
- success-only $C_{95}$;
- success-only burden by role-speed class.

This analysis is secondary.

Its purpose is to distinguish burden during successful coordination from burden accumulated in collision or timeout episodes.


# 28. Worst-off vehicle analysis

Use the corrected tie-handling rule.

If multiple vehicles share the minimum utility, divide worst-off credit fractionally among the tied vehicles.

Degenerate four-way near-perfect ties should be treated according to the existing final analysis implementation and documented tolerance.

Do not use arbitrary `min()` tie-breaking.


# 29. Behavioural diagnostic inspired by prosocial coordination

A secondary behavioural analysis examines whether a vehicle that has already accumulated relatively high mobility burden later receives a different coordination outcome.

This is an offline analysis.

The policy does not observe the computed burden directly.


## 29.1 High-burden vehicle

Using the existing trajectory logs, define a fixed pre-merge reference event using an already available geometric or merge-event variable.

The exact reference event must be defined from an existing logged state before analysing the new-seed results.

It must then be applied identically to:

- the original six seeds;
- the new six seeds;
- Mean;
- GGI;
- Maximin.

At that reference event, identify the vehicle with the largest accumulated burden.


## 29.2 Subsequent coordination outcomes

For that vehicle, record:

- later merge order;
- whether it enters the merge before or after its principal conflict vehicle;
- subsequent burden increment;
- whether another vehicle performs a strong slowing action before its merge.

Use the project's existing hard-brake or physical-acceleration definition if available.

Do not invent a new threshold after observing results.


## 29.3 Interpretation

The behavioural diagnostic asks:

> Is high accumulated burden followed by coordination behaviour that gives the affected vehicle greater priority?

A positive association does not mean the policy explicitly reasons about burden.

The agents do not receive burden as an observation.


# 30. Multi-metric fairness interpretation

Fairness is not judged from one metric.

A welfare condition may:

- improve $U_{\min}$;
- reduce utility Gini;
- increase mobility burden;
- change the burden tail;
- change task competence;

at the same time.

Therefore, no condition is labelled simply as "fairer" because one metric improves.

The final interpretation must distinguish:

1. welfare floor;
2. welfare inequality;
3. mobility burden;
4. burden tail;
5. task competence;
6. behavioural distribution.


# 31. New six-seed replication analysis

The new six seeds must first be analysed separately from the original six.

Training seed is the statistical replication unit.

Evaluation episodes are not treated as independent RL training replicates.


# 32. Primary matched contrasts

For every new seed compute:

$$
\Delta U_{\min}^{GGI}
=
U_{\min}^{GGI}
-
U_{\min}^{Mean}
$$

and:

$$
\Delta U_{\min}^{Maximin}
=
U_{\min}^{Maximin}
-
U_{\min}^{Mean}.
$$

Report:

- all six raw differences;
- mean difference;
- median difference;
- number of positive differences;
- number of negative differences;
- 95% paired seed-level bootstrap CI.

Use:

$$
10{,}000
$$

bootstrap resamples.


# 33. Do not introduce a new p-value test

The primary replication inference uses:

- raw seed-level paired differences;
- effect estimates;
- bootstrap confidence intervals.

No new post-hoc significance test is introduced.


# 34. Competence robustness

For each welfare condition report how many new seeds satisfy:

$$
p_{\mathrm{completion}}
\ge0.90
$$

$$
p_{\mathrm{collision}}
\le0.05
$$

$$
p_{\mathrm{timeout}}
\le0.05.
$$

Report:

- Mean: $N_{\mathrm{pass}}/6$
- GGI: $N_{\mathrm{pass}}/6$
- Maximin: $N_{\mathrm{pass}}/6$

Failing this threshold does not remove a seed.


# 35. Severe-failure description

The new analysis should report the complete seed-level task-performance table.

If a new Mean seed shows very low completion or very high collision, report it directly.

Do not draw additional seeds until a similar failure appears.

Do not terminate the experiment because no failure appears.


# 36. Possible outcomes

## Outcome A — Mean failures reproduce

If one or more new Mean seeds show severe failure, the original 910102 outcome is less likely to be unique to one training lineage.

If matched GGI or Maximin branches also improve those lineages, this provides evidence that welfare-conditioned fine-tuning can sometimes recover a poor Mean training trajectory.

It does not establish a universal welfare advantage.


## Outcome B — Mean is robust across the new six seeds

If the six new Mean policies are consistently competent, the 910102 pattern should be interpreted as an uncommon seed-specific learned-policy outcome.

The original observation remains valid, but its generality is limited.


## Outcome C — GGI/Maximin consistently improve $U_{\min}$

If most new seeds show positive matched $U_{\min}$ differences, this strengthens evidence that the welfare intervention generalizes across training lineages.


## Outcome D — No consistent ordering

If Mean, GGI, and Maximin continue to change ordering across seeds, the correct conclusion is that welfare effects remain strongly training-seed dependent.


## Outcome E — Welfare improves but competence declines

If $U_{\min}$ improves while completion or collision performance deteriorates, the result must be reported as a welfare–task tradeoff rather than a simple fairness improvement.


# 37. Secondary pooled 12-seed analysis

Only after the new six-seed replication is reported separately may the original and new seeds be pooled.

The pooled sample is:

$$
6+6=12
$$

independent training seeds.

Report:

- Mean;
- median;
- seed-level 95% bootstrap CI;
- competence-pass frequency;
- matched $U_{\min}$ contrasts;
- utility Gini;
- burden metrics;
- $C_{95}$.

The pooled analysis is secondary to the independent replication result.

It must not hide whether the original pattern reproduced in the new six seeds.


# 38. Secondary RQ1 replication

Because Mean is evaluated on H0 and H1 for the new six seeds, report the H1-H0 matched differences for:

- $U_{\mathrm{mean}}$;
- $U_{\min}$;
- utility Gini;
- $C_{\mathrm{mean}}$;
- burden range.

This is a secondary replication of RQ1.

The original six-seed RQ1 analysis remains the original formal result.


# 39. Required result table

The main new-seed table should contain:

| Seed | Condition | Completion | Collision | Timeout | $U_{\mathrm{mean}}$ | $U_{\min}$ | Utility Gini | $C_{\mathrm{mean}}$ | $C_{\max}$ | $C_{95}$ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

A second table should contain:

| Seed | $\Delta U_{\min}^{GGI}$ | $\Delta U_{\min}^{Maximin}$ |
|---|---:|---:|


# 40. Main figure

The main replication figure should show:

**x-axis:** training seed

**y-axis:** $U_{\min}$

For each new seed, display:

- Mean;
- GGI;
- Maximin.

The figure should make seed dependence visible directly.


# 41. Tail figure

A secondary figure may show:

**x-axis:** welfare condition

**y-axis:** vehicle-level mobility burden

with emphasis on the upper tail or $C_{95}$.

The purpose is to show whether a condition improves the worst welfare outcome while shifting mobility burden elsewhere.


# 42. Behavioural figure

If the behavioural diagnostic is sufficiently populated, show:

**x-axis:** welfare condition

**y-axis:** probability that the high-burden vehicle subsequently receives earlier merge priority.

This figure is exploratory.


# 43. Governance rules

The following rules are frozen before new training begins.

1. The original six formal seeds remain unchanged.

2. No original seed is removed.

3. Six new seeds are added.

4. All six new seeds are reported.

5. New seeds are not selected based on outcome.

6. New seeds are not replaced because of poor performance.

7. A documented technical failure is rerun using the same seed.

8. No new seed is drawn to replace a technically failed seed.

9. Mean/GGI/Maximin for each seed start from the same task checkpoint.

10. $\lambda_W=0.5$ remains fixed.

11. Reward coefficients remain fixed.

12. Architecture remains fixed.

13. Observation structure remains fixed.

14. Evaluation banks remain fixed.

15. Checkpoint ensemble remains fixed.

16. No best-checkpoint selection is introduced.

17. No post-hoc hyperparameter tuning is conducted on the new six seeds.

18. The new experiment does not retroactively alter the original formal experiment.

19. Secondary tail and behavioural analyses must be labelled exploratory.

20. Results are reported whether they confirm or contradict the original pattern.


# 44. Directory structure

Recommended local structure:

`F:\正式训练`
- original formal package
- do not modify

`F:\正式训练_seed_replication_v1`
- copied scripts
- copied environment documentation
- copied evaluation banks
- copied README
- new seed manifest
- new outputs


Suggested internal structure:

`seed_replication_v1/`

`protocol/`
- `seed_replication_protocol_v1.md`
- `new_seed_manifest.csv`
- `replication_preflight_audit.md`

`curriculum/`
- `920101/`
- `920102/`
- `920103/`
- `920104/`
- `920105/`
- `920106/`

`welfare/`
- `920101/Mean/`
- `920101/GGI/`
- `920101/Maximin/`
- ...
- `920106/Maximin/`

`evaluation/`
- `H1/`
- `H0_mean/`
- `behavioral/`
- `success_only/`

`analysis/`
- `new6/`
- `pooled12/`


# 45. Required output files

At minimum save:

1. `replication_preflight_audit.md`
2. `new_seed_manifest.csv`
3. `new_seed_task_summary.csv`
4. `new_seed_formal_task_metrics.csv`
5. `new_seed_formal_welfare_metrics.csv`
6. `new_seed_umin_contrasts.csv`
7. `new_seed_tail_burden.csv`
8. `new_seed_success_only_burden.csv`
9. `new_seed_behavioral_diagnostic.csv`
10. `new_seed_competence_summary.csv`
11. `new6_replication_summary.md`
12. `pooled12_summary.csv`


# 46. Final diagnostic conclusion

The experiment must end by answering:

> **Does the strong training-seed dependence observed in the original formal experiment reproduce in six new independently trained policies?**

and:

> **Do GGI or Maximin show a consistent worst-off welfare advantage over Mean across independent training lineages?**

A third descriptive conclusion should answer:

> **When welfare changes, does it improve the welfare floor, redistribute mobility burden, alter the upper tail of burden, or change task competence?**

The conclusion must be reported regardless of direction.


# 47. Relationship to the thesis

The original six-seed experiment remains the original formal study.

The six-new-seed experiment is presented as:

**Independent replication and robustness analysis**

It strengthens the thesis by separating:

$$
\text{evaluation randomness}
$$

from:

$$
\text{training-lineage variability}.
$$

It also tests whether second-stage welfare adaptation is reproducible when agents remain decentralized and do not directly observe the welfare of other vehicles.