# Claude Code Autonomous Experiment Runbook — HighwayEnv Migration Edition
## Study B: Fair Decentralized Multi-Agent Highway Merging

**Status:** authoritative replacement for the previous autonomous runbook  
**Primary change:** migrate the simulator backend from the custom highway simulator to HighwayEnv before formal welfare experiments  
**Execution model:** autonomous bounded state machine  
**Formal policy:** parameter-shared local DQN  
**Formal fairness objectives:** Mean / Generalized Gini / Maximin  
**Do not ask the user for confirmation between ordinary gates. Stop only at explicit HARD STOP conditions or a genuine unrecoverable execution failure.**

---

# 0. Core scientific objective

The thesis studies:

> Whether successful decentralized coordination in a heterogeneous highway merge places unequal mobility burdens on different vehicles, and whether direct Mean, Generalized Gini, and Maximin welfare objectives change that distribution while preserving safety and task competence.

The scientific chain remains:

$$
\boxed{
\text{validate environment}
\rightarrow
\text{establish competence}
\rightarrow
\text{measure burden inequality}
\rightarrow
\text{compare welfare objectives}
\rightarrow
\text{explain behavioural changes}
}
$$

The HighwayEnv migration is an **engineering/backend amendment before formal fairness comparison**. It does not change the research question.

---

# 1. High-level autonomous state machine

Execute:

```text
M0  Code/provenance snapshot
 ↓
M1  Install and pin HighwayEnv
 ↓
M2  Decide subclass/wrapper vs fork
 ↓
M3  Build ThesisHighwayMergeEnv
 ↓
M4  Environment parity and validity gates
 ↓
M5  Oracle feasibility gate
 ↓
M6  RL single-scenario sanity gate
 ↓
C1 → C4 → C16 → C64 task-only curriculum
 ↓
Task-only competence gate
 ↓
Restore Mean welfare
 ↓
Mean qualification
 ↓
Freeze formal configuration
 ↓
Mean / GGI / Maximin formal training
 ↓
H0 / H1 held-out evaluation
 ↓
Seed-level statistics + figures
 ↓
Final thesis results package
```

At every gate:

1. calculate the predefined metrics;
2. classify `PASS`, `FAIL`, or `INCONCLUSIVE`;
3. append the result to the autonomous log;
4. update persistent state;
5. take the next allowed branch automatically;
6. never improvise a new algorithm outside this document.

---

# 2. Current historical evidence — do not rerun by default

The following is already established:

```text
MAPPO qualification
→ failed bounded reliability criterion

PBRS local DQN
→ 0/8 qualified

Direct Mean local DQN
→ qualification failed

Task-only local DQN
→ 31.2% at 800K before pipeline repair

Joint/global-information DQN
→ 25.0% at 800K
→ did not outperform local task-only DQN

Oracle on old custom simulator
→ 384/384 successful

DQN pipeline audit
→ target-network synchronization defect discovered and fixed

Corrected local DQN, old custom simulator, one fixed scenario
→ substantial learning (~72–81%)
→ did not cleanly cross old 0.95 overfit gate

Dense Q/TD/gradient audit
→ no additional numerical/mechanistic bug found
```

Therefore:

$$
\boxed{
\text{VDN gate remains CLOSED}
}
$$

and:

$$
\boxed{
\text{formal Mean/GGI/Maximin training has not started}
}
$$

The simulator migration is permitted because it occurs before the formal welfare-condition comparison.

---

# 3. Scientific invariants — do not change automatically

These define the thesis and must survive the backend migration.

## 3.1 Agents

- `N = 4`
- 2 ramp vehicles
- 2 mainline vehicles
- heterogeneous target speeds:
  - slow = 18 m/s
  - fast = 22 m/s
- physical ID must not be permanently tied to:
  - road role
  - target-speed class
  - front/rear social position

## 3.2 Timing

Target decision interval:

$$
\Delta t=0.2\ \mathrm{s}
$$

Target horizon:

$$
H=200
$$

therefore:

$$
T_{\max}=40\ \mathrm{s}
$$

The HighwayEnv backend must preserve this **decision frequency**, even if HighwayEnv performs smaller internal simulation substeps.

## 3.3 Matched TTC

Front-wave target:

$$
T_F^*=4.5\ \mathrm{s}
$$

Rear-wave target:

$$
T_R^*=6.5\ \mathrm{s}
$$

Shared jitter:

$$
\epsilon\sim\mathcal U(-0.25,0.25)
$$

Individual jitter:

$$
\zeta_i\sim\mathcal U(-0.05,0.05)
$$

Same-lane initial centre distance must remain at least 15 m.

## 3.4 Formal information restriction

The formal learned action remains:

$$
a_i=f_\theta(o_i)
$$

Other vehicles' hidden intrinsic target speeds must not enter ego observation.

Forbidden policy inputs:

```text
other vehicle target speed
other vehicle utility
other vehicle reward
other vehicle Q-value
other vehicle intended next action
full global simulator state
concatenated complete self-observation of all agents
```

## 3.5 Action semantics

The thesis action space remains conceptually:

```text
DECELERATE
MAINTAIN
ACCELERATE
```

Do not silently replace this with lane changes.

Do not silently change it into a different cruise-control problem.

The migration should preferentially use HighwayEnv's low-level/discretized longitudinal acceleration mechanism rather than default lane-changing meta-actions.

## 3.6 Utility

$$
u_{i,t}
=
\operatorname{clip}_{[0,1]}
\left(
\frac{v_{i,t}}{v_i^{\mathrm{target}}}
\right)
$$

$$
S_i
=
\begin{cases}
1,&\text{safe completion}\\
0,&\text{collision, timeout, or incomplete}
\end{cases}
$$

$$
U_i
=
S_i
\frac{
\sum_t I_{i,t}^{active}u_{i,t}
}{
\sum_t I_{i,t}^{active}
}
$$

## 3.7 Coordination-associated mobility burden

$$
C_i
=
\Delta t
\sum_t
I_{i,t}^{active}
(1-u_{i,t})
$$

Burden is an evaluation/mechanism metric, not a training reward.

## 3.8 Welfare objectives

Mean:

$$
W_{\mathrm{mean}}
=
\frac14\sum_{i=1}^{4}U_i
$$

GGI:

$$
W_{\mathrm{GGI}}
=
0.4U_{(1)}
+
0.3U_{(2)}
+
0.2U_{(3)}
+
0.1U_{(4)}
$$

Maximin:

$$
W_{\min}
=
\min_i U_i
$$

Condition names:

```text
mean
ggi
maximin
```

---

# 4. M0 — Snapshot current project before migration

Before modifying environment code:

1. run all currently passing Study B tests;
2. save current git commit/hash if repository uses git;
3. save `git status`;
4. save package/dependency environment;
5. snapshot old simulator configuration;
6. snapshot corrected DQN target-sync implementation;
7. record exact current DQN hyperparameters;
8. record replay, n-step, PER, optimizer, target-network settings;
9. record existing Q/M/H0/H1 scenario-bank generation seeds/IDs;
10. do not delete the old simulator.

Create:

```text
output/highwayenv_migration/
    PRE_MIGRATION_GIT.txt
    PRE_MIGRATION_DEPENDENCIES.txt
    PRE_MIGRATION_CONFIG.json
    PRE_MIGRATION_TEST_REPORT.txt
    CODE_PROVENANCE.md
```

The old simulator becomes:

```text
legacy_custom_backend
```

It remains available only for diagnostics/parity comparison.

---

# 5. M1 — Pin HighwayEnv

## 5.1 Required baseline

Use:

```text
HighwayEnv 1.12.0
```

unless installation proves that exact release cannot run with the project's Python environment.

Do not use a floating unpinned dependency such as:

```text
highway-env>=1.12
```

Prefer:

```text
highway-env==1.12.0
```

or the equivalent package-manager lock.

Record:

```text
HighwayEnv version
Python version
Gymnasium version
NumPy version
installation source
package hash/lock information when available
```

## 5.2 Why version 1.12 baseline is required

The migration must use connected-lane neighbour detection.

Do not base the new study on legacy same-segment neighbour behavior.

Set or inherit:

```text
neighbour_vehicles_connected_lanes = True
```

## 5.3 Installation smoke test

Verify:

```python
import highway_env
import gymnasium as gym
```

Then instantiate and reset:

```text
merge-v1
merge-generic-v1
```

No training yet.

### PASS

Both instantiate/reset successfully.

Proceed to M2.

### FAIL

Try only:

1. resolve dependency conflict;
2. recreate clean experiment virtual environment;
3. pin the compatible Gymnasium/NumPy versions required by HighwayEnv 1.12.0.

If HighwayEnv 1.12.0 is genuinely incompatible with the project's supported Python environment and resolving it would require a major unrelated downgrade, use the newest stable HighwayEnv version that contains connected-lane neighbour detection and record the amendment.

Do not fall back to old `merge-v0` merely because installation is easier.

---

# 6. M2 — Fork decision gate

## 6.1 Default decision

**Do not fork HighwayEnv initially.**

Preferred architecture:

```text
pinned HighwayEnv dependency
        ↓
project-local subclass / wrapper
        ↓
ThesisHighwayMergeEnv
```

The project should own only thesis-specific code.

## 6.2 Allowed extension mechanisms before fork

Claude Code must first attempt to implement the study using:

1. HighwayEnv configuration;
2. subclassing `ConnectedLaneMergeGenericEnv` or equivalent current connected-lane generic merge class;
3. overriding thesis-specific methods such as:
   - `default_config`
   - reset/vehicle creation
   - reward
   - termination/truncation
4. a project-local observation wrapper/builder;
5. HighwayEnv `MultiAgentObservation`;
6. HighwayEnv `MultiAgentAction`;
7. HighwayEnv `DiscreteAction` with:
   - longitudinal enabled
   - lateral disabled
   - three action bins
8. project-local Gymnasium registration.

## 6.3 Fork trigger

A fork is allowed only if a required scientific invariant cannot be implemented or tested through public extension points.

Examples:

```text
required multi-agent action behavior cannot be expressed correctly
required controlled-vehicle creation cannot be implemented safely
a confirmed HighwayEnv internal bug affects this exact experiment
connected-lane behaviour requires an internal patch not exposed to subclass/config
collision/road bug is reproduced and requires patching upstream code
```

The following are NOT fork triggers:

```text
"subclassing feels inconvenient"
"editing upstream is faster"
"the code is easier to copy"
"default reward is different"
"default vehicles are different"
"default observation is different"
```

Those should be solved in thesis-specific subclass/wrapper code.

## 6.4 If fork becomes necessary

Fork from the exact pinned upstream release/tag.

Required fork discipline:

```text
upstream base: HighwayEnv 1.12.0
upstream remote retained
MIT LICENSE retained
minimal patch commits only
no unrelated refactor
each upstream modification documented
```

Create:

```text
HIGHWAYENV_FORK_MANIFEST.md
```

containing:

```text
upstream repository
upstream tag/version
upstream base commit
local fork commit
files changed
reason for every change
tests proving the patch is necessary
```

After fork, never casually merge upstream changes during the formal study.

---

# 7. M3 — Implement ThesisHighwayMergeEnv

Recommended project structure:

```text
src/
  study_b/
    envs/
      highwayenv_merge.py
      local_observation.py
      scenario_adapter.py
      reward.py
      wrappers.py
```

The exact path should follow the existing repository layout.

Do not reorganize the entire project just to match this example.

---

# 8. HighwayEnv base environment

Prefer the current connected-lane generic merge implementation:

```text
ConnectedLaneMergeGenericEnv
```

or the registered equivalent:

```text
merge-generic-v1
```

Use the generic road as the physical backend unless parity tests show that the non-generic `merge-v1` better preserves the intended geometry with less custom code.

Do not use `merge-v0`.

The final chosen base must be written to:

```text
HIGHWAYENV_BACKEND_CONFIG.json
```

---

# 9. Decision/simulation frequency

The scientific decision timestep is:

$$
0.2s
$$

Therefore configure:

```text
policy_frequency = 5 Hz
```

Use a HighwayEnv simulation frequency that is an integer multiple of 5 Hz.

Preferred initial value:

```text
simulation_frequency = 15 Hz
policy_frequency = 5 Hz
```

This gives three internal physics updates per policy action.

Do not reduce the scientific policy frequency to HighwayEnv's default 1 Hz.

Gate this explicitly in M4.

---

# 10. Multi-agent control

Exactly four thesis vehicles must be controlled.

Set/construct:

```text
controlled_vehicles = 4
```

All four vehicles must receive actions at every policy step until they are considered completed/inactive by the thesis wrapper.

Use tuple multi-agent action dispatch.

The formal DQN still predicts independently:

$$
a_i
=
\arg\max_a Q_\theta(o_i,a)
$$

with shared parameters.

Do not concatenate observations for formal execution.

---

# 11. Action implementation

## 11.1 Preferred HighwayEnv action type

Prefer:

```text
MultiAgentAction
  └── DiscreteAction
      longitudinal = True
      lateral = False
      actions_per_axis = 3
```

This preserves a three-bin low-level longitudinal control interpretation.

## 11.2 Acceleration range

First read the acceleration magnitudes from the legacy custom simulator.

Configure HighwayEnv's `acceleration_range` to preserve those semantics as closely as possible.

Do not automatically adopt HighwayEnv's default acceleration magnitude if it materially differs from the old experiment.

## 11.3 Action semantic gate

Experimentally measure one step under each action from identical controlled initial states.

Classify action indices by observed acceleration:

```text
negative acceleration → DECELERATE
approximately zero    → MAINTAIN
positive acceleration → ACCELERATE
```

The DQN/logger/replay/environment mappings must use one common mapping.

Never assume action index order without testing it.

## 11.4 Fallback

If `DiscreteAction` cannot preserve the required semantics because of a demonstrated API limitation:

1. implement a minimal project-local adapter around `ContinuousAction`; or
2. implement a minimal project-local action type.

Only if HighwayEnv internals must be changed does the fork gate reopen.

---

# 12. Four controlled vehicles and matched-TTC initialization

Override thesis vehicle/scenario creation.

The environment must create exactly:

```text
Ramp-Fast
Ramp-Slow
Mainline-Fast
Mainline-Slow
```

The assignment of physical IDs and front/rear slots must remain counterbalanced.

For each vehicle:

$$
T_{i,0}^{nominal}
=
T_{slot(i)}^*
+
\epsilon
+
\zeta_i
$$

and:

$$
d_{i,0}
=
v_{i,0}
T_{i,0}^{nominal}
$$

with:

$$
v_{i,0}=v_i^{target}
$$

Translate that desired path distance into the HighwayEnv lane longitudinal coordinate using HighwayEnv lane/network geometry.

Do not approximate TTC using global x-coordinate if path distance along the lane is available.

Reject/resample invalid same-lane spawns.

---

# 13. Vehicle target speeds

Each thesis vehicle has an intrinsic thesis target:

```text
18 m/s
or
22 m/s
```

This target is used for:

- utility normalization;
- class identity;
- scenario construction.

Do not automatically expose the intrinsic target to other agents.

If HighwayEnv vehicle objects also contain their own control target-speed field, distinguish clearly between:

```text
thesis intrinsic target speed
HighwayEnv controller/internal target speed
```

Avoid accidental leakage.

Document this distinction in code and `OBSERVATION_SPEC.md`.

---

# 14. Local observation implementation

## 14.1 Principle

Use HighwayEnv for reliable physical state and neighbour search.

Use thesis code to construct the final policy observation.

Preferred flow:

```text
HighwayEnv physical state
      ↓
connected-lane local neighbour queries
      ↓
ThesisLocalObservationBuilder
      ↓
o_i
```

## 14.2 Allowed ego features

Retain the existing scientific observation specification where available:

- own road role;
- own speed;
- own target speed;
- own acceleration if previously present;
- own merge/path distance;
- previous action if previously present;
- active mask/state if previously present.

## 14.3 Allowed neighbour features

Use only observable motion/road relation features such as:

- relative path distance;
- relative speed;
- lane relation;
- approximate TTC relation;
- presence/active mask.

## 14.4 Forbidden leakage

Do not include:

- neighbour intrinsic target speed;
- neighbour utility;
- neighbour reward;
- neighbour intended action;
- global complete state.

Set HighwayEnv observation options so intentions are not exposed.

If raw `Kinematics` is used internally, the final DQN input must still pass through the thesis local observation builder.

---

# 15. Reward

Do not use HighwayEnv's default merge reward for the formal experiment.

Override it.

Task reward remains the Study B frozen task reward.

During backend validation and curriculum:

$$
\lambda_W=0
$$

Formal terminal welfare later:

$$
R_c^W
=
\lambda_W[W_c(\mathbf U)-1]
$$

The HighwayEnv migration must not simultaneously redesign the task reward.

---

# 16. Termination and truncation

Implement thesis semantics:

## Terminal success

All four controlled vehicles have safely completed the required merge/task.

## Terminal failure

Collision involving controlled thesis vehicles according to the frozen Study B collision definition.

## Truncation/task failure

Exactly 200 policy decisions / 40 seconds without safe completion.

For learning returns:

```text
completion → terminal, no bootstrap
collision  → terminal, no bootstrap
timeout    → learning terminal, no bootstrap
```

Do not inherit HighwayEnv default merge termination uncritically.

---

# 17. M4 — Environment parity and validity gates

No RL training is allowed before all mandatory M4 gates pass.

Create:

```text
output/highwayenv_migration/validation/
```

---

# 18. M4-A — Basic API gate

Verify:

```text
reset(seed)
step(tuple_of_four_actions)
four controlled vehicles
deterministic reset under identical seed
Gymnasium terminated/truncated semantics
```

### PASS
Proceed.

### FAIL
Fix thesis wrapper/subclass only.

If failure requires changing HighwayEnv internals, invoke M2 fork gate.

---

# 19. M4-B — Decision frequency gate

Run exactly 200 policy steps with no early terminal in a safe artificial setup.

Measured simulated time must equal approximately:

$$
40s
$$

and each action must persist for approximately:

$$
0.2s
$$

### PASS
Proceed.

### FAIL
Fix `policy_frequency`, simulation stepping, or wrapper step counting.

Do not train until exact.

---

# 20. M4-C — Action semantics gate

From identical states, test all three action indices.

Required:

```text
one action clearly lowers speed/has negative acceleration
one action approximately maintains speed
one action raises speed/has positive acceleration
```

Also verify:

```text
network index
epsilon-random index
replay action
environment action
logger label
```

all agree.

### PASS
Proceed.

### FAIL
Repair action adapter/config.

If no public extension route works, invoke fork gate.

---

# 21. M4-D — Matched TTC generator gate

Generate at least:

$$
10,000
$$

scenario initializations.

Check intended front cross-lane pair and rear cross-lane pair.

Required:

$$
|\Delta TTC|\le0.5s
$$

for at least:

$$
95\%
$$

of generated standard heterogeneous scenarios.

Also require:

$$
d_{\mathrm{same-lane,centre}}\ge15m
$$

for 100% accepted scenarios.

Save:

```text
matched_ttc_validation.csv
spawn_validity.csv
```

### PASS
Proceed.

### FAIL branch D1
Correct coordinate conversion/path-distance logic.

### FAIL branch D2
If geometry itself makes matched TTC infeasible, adjust only generic merge segment lengths before RL.

Do not change target speeds.

Do not change TTC targets unless a HARD STOP is reached and documented.

### FAIL branch D3
If the failure appears to come from HighwayEnv road/path APIs, inspect upstream source/tests.

Fork only if a confirmed internal defect requires a patch.

---

# 22. M4-E — Role/speed counterbalancing gate

Across generated scenarios verify balanced representation of:

```text
Ramp-Fast
Ramp-Slow
Mainline-Fast
Mainline-Slow
front/rear combinations
physical IDs
```

No physical ID may deterministically encode class or social position.

### PASS
Proceed.

### FAIL
Fix scenario assignment generator only.

---

# 23. M4-F — Local observation leakage gate

For each ego agent i:

1. hold every externally observable neighbour state fixed;
2. change neighbour j's hidden intrinsic target from 18 to 22;
3. recompute ego observation.

Required:

$$
o_i(s;v_j^{target}=18)
=
o_i(s;v_j^{target}=22)
$$

for:

$$
j\neq i
$$

within numerical tolerance.

Also inspect observation feature names and shapes.

### PASS
Proceed.

### FAIL
Fix observation builder.

Do not proceed to RL.

---

# 24. M4-G — Connected-lane neighbour gate

Construct deterministic cases with vehicles just before/after merge lane-segment boundaries.

Verify the relevant preceding/following vehicles remain detectable across connected segments.

### PASS
Proceed.

### FAIL
Confirm:

```text
neighbour_vehicles_connected_lanes = True
```

and connected-lane merge base class is actually in use.

If the official connected-lane behavior is demonstrably wrong for the chosen road graph, investigate upstream issue/source.

Fork only with a reproducible failing test.

---

# 25. M4-H — Collision gate

Construct:

1. obvious non-collision;
2. obvious controlled-vehicle overlap/collision;
3. near but non-overlapping same-lane state.

Verify collision flags and episode termination.

Also ensure no collision is triggered merely by road-segment transition.

### PASS
Proceed.

### FAIL
First check spawn geometry, lane indices, route, and vehicle dimensions.

If a minimal HighwayEnv reproduction reveals an upstream collision defect, reopen fork gate.

---

# 26. M4-I — Utility and burden gate

Test:

$$
u(18,18)=1
$$

$$
u(22,22)=1
$$

$$
u(15,18)<1
$$

$$
u(18,22)<1
$$

Test failure:

$$
U_i=0
$$

after collision/timeout/incomplete.

Check burden on a simple known-speed trajectory by hand.

### PASS
Proceed.

### FAIL
Fix thesis metric code only.

---

# 27. M4-J — Reward decomposition gate

For synthetic/controlled transitions verify every task-reward component independently.

Compare hand calculation with environment output.

Check:

```text
progress term
completion term
collision term
hard-braking term
timeout term
```

During migration:

```text
welfare term = 0
```

### PASS
Proceed.

### FAIL
Fix reward adapter.

Do not alter coefficients merely to make the gate pass.

---

# 28. M4-K — DQN terminal/return regression gate

Rerun all existing pipeline checks under the new backend interface:

```text
terminal bootstrap
timeout bootstrap
1-step target
3-step return
episode boundary
agent alignment
replay round-trip
target sync
PER index alignment
finite numerics
```

### PASS
Proceed to M5.

### FAIL
Repair only the interface regression.

Do not change algorithm.

---

# 29. M5 — Oracle feasibility gate on HighwayEnv

Adapt the existing rule-based oracle to the new physical backend.

The oracle may use diagnostic/full state because it is not the formal policy.

Evaluate on:

```text
Q = 64 fixed heterogeneous scenarios
M = 64 monitoring scenarios if currently available
H1 = 256 held-out heterogeneous scenarios if currently available
```

If those abstract scenario manifests exist, preserve their seeds/role/TTC assignments while regenerating physical coordinates through the HighwayEnv adapter.

## Strong PASS

$$
Completion\ge0.98
$$

with:

$$
Collision\le0.01
$$

$$
Timeout\le0.01
$$

Proceed to M6.

## Acceptable PASS

$$
Completion\ge0.95
$$

with combined failure:

$$
Collision+Timeout\le0.05
$$

Proceed to M6, but record the limitation.

## INCONCLUSIVE

$$
0.80\le Completion<0.95
$$

Execute Oracle Recovery OR1–OR3.

## FAIL

$$
Completion<0.80
$$

Execute Oracle Recovery OR1–OR4.

---

# 30. Oracle Recovery branches

## OR1 — oracle adaptation audit

Check whether the old oracle assumes old simulator-specific:

- coordinate system;
- lane IDs;
- merge-point coordinates;
- acceleration magnitude;
- neighbour ordering.

Repair oracle translation only.

Re-evaluate.

## OR2 — road-length/merge geometry adjustment

If oracle failures are caused by insufficient preparation/merge distance, adjust only HighwayEnv generic road-section lengths.

Freeze the first geometry that reaches the oracle acceptance gate.

Do not tune geometry for learned-policy success.

## OR3 — matched-TTC coordinate audit

Verify TTC is measured along route/lane path, not global x-coordinate.

Repair initialization if wrong.

## OR4 — upstream/backend failure audit

If the oracle still cannot solve a physically reasonable set of scenarios:

1. create minimal failing environment tests;
2. determine whether failure comes from:
   - thesis geometry/configuration;
   - action semantics;
   - HighwayEnv internal road/neighbour/collision behavior.

If thesis configuration is the cause, repair it.

If an upstream internal defect is proven and blocks the study, fork from the pinned release and patch minimally.

If no bounded repair makes oracle completion >= 0.80:

```text
HARD STOP ENV-A
```

Do not begin RL.

---

# 31. Freeze validated HighwayEnv environment

Once M5 passes, create:

```text
output/highwayenv_migration/HIGHWAYENV_ENV_FREEZE.json
```

Include:

```text
HighwayEnv version/fork commit
base environment class
road lengths
policy frequency
simulation frequency
vehicle dimensions/classes
acceleration range
action index mapping
target speeds
TTC parameters
scenario seeds
observation spec hash
reward config hash
termination rules
test suite result
oracle result
```

After this point, no environment geometry/observation/action changes are allowed without returning to M4/M5 and creating a documented amendment.

---

# 32. M6 — Single-scenario RL sanity gate

Purpose:

> Verify that the corrected parameter-shared local DQN can learn on the validated HighwayEnv backend before curriculum expansion.

Configuration:

```text
local information
parameter-shared DQN
task-only reward
lambda_W = 0
one fixed oracle-solvable scenario
corrected target sync
frozen absolute LR/epsilon schedule
```

Do not use joint/global information.

---

# 33. Freeze absolute training schedules

LR and epsilon must depend only on absolute environment/training step:

$$
LR=LR(t)
$$

$$
\epsilon=\epsilon(t)
$$

They must not be rescaled when a run is extended.

Create:

```text
ABSOLUTE_TRAINING_SCHEDULE.json
```

All later runs use it unless a bounded qualification amendment explicitly says otherwise.

---

# 34. M6 checkpoints

Evaluate deterministically at:

```text
25K
50K
100K
150K
200K
```

Log:

```text
completion
collision
timeout
episode length
task return
fixed-reference mean_Q
policy-visited mean_Q
Q action spread
TD target
TD error
loss
gradient norm
epsilon
LR
action distribution by class
```

---

# 35. M6 gate

## Strong PASS

At any checkpoint:

$$
Completion\ge0.95
$$

and:

$$
Collision+Timeout\le0.05
$$

Proceed to curriculum C4.

## Learnable PASS

At 200K:

$$
Completion\ge0.80
$$

with clear improvement over earlier checkpoints and healthy numerical diagnostics.

Continue same run to at most 400K.

If by 400K:

$$
Completion\ge0.90
$$

proceed to C4.

## INCONCLUSIVE

At 400K:

$$
0.60\le Completion<0.90
$$

with healthy diagnostics.

Run one second seed on the same fixed scenario using exactly the same schedule and maximum 300K steps.

If second seed reaches:

$$
Completion\ge0.80
$$

record:

```text
SINGLE_SCENARIO_LEARNABLE_WITH_VARIANCE
```

and proceed to C4.

## FAIL

If either:

```text
completion < 0.60 at 400K with no sustained improvement
```

or both single-scenario seeds remain poor, enter M6 recovery.

---

# 36. M6 recovery

## M6-R1 — backend interface audit

Recheck:

- action sign/index;
- observation normalization;
- terminal handling;
- reward decomposition;
- target sync.

If regression found, fix and restart M6.

## M6-R2 — minimal DQN

Diagnostic only:

```text
1-step TD
uniform replay
no PER
target network retained
same local observation
task-only reward
same fixed scenario
```

If minimal DQN reaches >=0.90, add:

```text
+ 3-step
then
+ PER
```

one at a time.

The first component causing major collapse is disabled/fixed before continuing.

Freeze the corrected learner.

## M6-R3 — action representation comparison

Only if both original and minimal DQN fail while oracle passes:

Compare two HighwayEnv-compatible three-action realizations on the fixed scenario:

```text
A: current DiscreteAction 3-bin low-level acceleration
B: longitudinal-only DiscreteMetaAction SLOWER/IDLE/FASTER
```

This is allowed because backend migration may reveal that the legacy fixed-acceleration action is poorly matched to HighwayEnv vehicle control.

Use identical local observations/reward/seed/budget.

Select the representation using **task competence only**, before welfare training.

If B is adopted, document this as a pre-formal environment/action amendment and rerun M4 action/utility/oracle gates.

Do not select based on fairness outcomes.

## M6-R4 — HARD STOP

If validated environment + oracle pass, all pipeline tests pass, and neither bounded action representation can produce meaningful fixed-scenario learning:

```text
HARD STOP SOLVER-A
```

Do not automatically switch to a broad MARL search.

---

# 37. Build nested curriculum sets

After M6 acceptance, create:

```text
C1
C4
C16
C64
```

with:

$$
C_1\subset C_4\subset C_{16}\subset C_{64}
$$

and:

$$
C_{64}=Q
$$

Use deterministic abstract scenario IDs/seeds.

Do not choose scenario membership based on learned-policy performance.

Write:

```text
SCENARIO_CURRICULUM_MANIFEST.json
```

---

# 38. Curriculum semantics

Use one sequential curriculum chain:

```text
C1
 ↓
C4
 ↓
C16
 ↓
C64
```

Continue network/optimizer state between accepted stages.

Do not restart LR/epsilon schedules.

Replay handling at stage transition:

- preserve replay by default;
- tag every transition with scenario ID/stage;
- if replay causes demonstrated catastrophic imbalance, use the bounded recovery branch below.

Reward remains task-only:

$$
\lambda_W=0
$$

---

# 39. C1 stage

M6 already functions as the C1 learning sanity stage.

Use the accepted M6 checkpoint as curriculum C1 start/end point.

Do not cherry-pick an older legacy-simulator checkpoint.

Proceed to C4.

---

# 40. C4 stage

Train on four nested scenarios.

Initial additional budget:

```text
200K
```

Evaluate every 50K on all C4 scenarios.

## PASS

$$
Completion\ge0.90
$$

$$
Collision\le0.05
$$

$$
Timeout\le0.05
$$

Proceed C16.

## SOFT PASS / improving

If:

$$
0.75\le Completion<0.90
$$

and final checkpoints improve, extend +100K.

If gate passes, proceed.

## FAIL

If completion <0.75 or severe collapse persists, run Diversity Recovery DR1–DR4.

---

# 41. C16 stage

Continue from accepted C4 checkpoint.

Initial additional budget:

```text
250K
```

Evaluate every 50K.

## PASS

Same preferred competence criteria:

$$
Completion\ge0.90
$$

$$
Collision\le0.05
$$

$$
Timeout\le0.05
$$

Proceed C64.

## SOFT PASS / improving

If 0.75–0.90 and improving, extend +100K.

## FAIL

Run Diversity Recovery DR1–DR4.

---

# 42. C64 task-only qualification stage

Continue from accepted C16 checkpoint.

Use complete frozen Q64 bank/training distribution.

Monitor every 50K.

## Task-only competence PASS

Require three consecutive final monitoring checkpoints with:

$$
Completion\ge0.90
$$

$$
Collision\le0.05
$$

$$
Timeout\le0.05
$$

and no adjacent completion collapse >0.10 after first reaching 0.90.

Then:

```text
TASK_ONLY_SOLVER_COMPETENCE = ESTABLISHED
```

Proceed to Mean qualification.

## FAIL

Run Diversity Recovery.

---

# 43. Diversity Recovery DR1–DR4

Triggered when a lower curriculum stage is learnable but a higher stage fails.

## DR1 — per-scenario failure map

Generate:

```text
per_scenario_metrics.csv
per_class_metrics.csv
action_distribution.csv
failure_mode_summary.md
```

Classify:

```text
collision dominated
timeout dominated
specific scenario subset
specific role/speed class
mixed
```

## DR2 — local observation aliasing audit

Using oracle/reference trajectories:

1. collect local observations;
2. pair with reference action;
3. find near-identical local observations associated with conflicting reference actions;
4. report clustering by role/scenario.

Do not expose hidden target speed automatically.

## DR3 — replay/curriculum retention audit

Measure replay composition by:

```text
scenario ID
terminal type
role-speed class
stage
```

If larger-stage training catastrophically forgets earlier scenarios, use a fixed mixed curriculum retaining previous-stage scenarios while adding new ones.

Freeze the mixture rule before rerunning.

Allow one bounded +150K rerun.

## DR4 — corrected joint-information diagnostic

Only if local training still fails after DR1–DR3:

Run one corrected joint/global-information DQN diagnostic on the SAME HighwayEnv stage with:

```text
task-only reward
same action space
same schedule
same budget
same scenarios
```

### If joint DQN reaches competence and local DQN remains far below

Set:

```text
VDN_GATE = OPEN
```

Proceed to conditional VDN section.

### If joint DQN does not materially outperform local DQN

Keep VDN CLOSED.

If no other bounded recovery applies:

```text
HARD STOP SOLVER-B
```

---

# 44. Conditional VDN branch

VDN is not part of the default path.

Only execute when DR4 provides strong evidence.

Requirements:

```text
oracle passes
basic curriculum stages learnable
higher-stage local DQN fails
corrected joint-information DQN clearly succeeds on same stage
```

Then follow the existing VDN conditional amendment protocol.

Use:

$$
Q_{\mathrm{tot}}
=
\sum_i Q_\theta(o_i,a_i)
$$

Execution remains local:

$$
a_i=\arg\max_a Q_\theta(o_i,a)
$$

First use task-only reward.

If VDN fails the bounded competence rescue:

```text
HARD STOP SOLVER-C
```

Do not proceed to QMIX/GNN/attention/recurrent architecture search.

---

# 45. Mean qualification

Enter only after task-only competence is established.

Restore Mean terminal welfare:

$$
R_{\mathrm{Mean}}^W
=
\lambda_W
[
W_{\mathrm{Mean}}(\mathbf U)-1
]
$$

Start with:

$$
\lambda_W=1
$$

Use the accepted solver/environment/training curriculum.

Do not change HighwayEnv backend during Mean qualification.

---

# 46. Mean qualification seeds and gate

Use two dedicated qualification seeds unless the final frozen thesis protocol already specifies a larger pre-formal number.

Do not reuse them as formal seeds.

Evaluate every 50K on frozen Q bank.

Initial maximum budget:

```text
800K
```

One extension to:

```text
1M
```

is allowed only if performance is clearly improving near 800K.

PASS requires the same competence standard:

$$
Completion\ge0.90
$$

$$
Collision\le0.05
$$

$$
Timeout\le0.05
$$

with stability across the final monitoring window.

If Mean passes, proceed to formal freeze.

If Mean fails while task-only passed, enter Mean Reward Recovery.

---

# 47. Mean Reward Recovery

The solver/environment is no longer the first suspect.

Do not modify HighwayEnv.

## MR1 — reward decomposition

Compare:

```text
task-only competent trajectories
Mean success
Mean collision
Mean timeout
```

Calculate discounted returns with actual gamma.

## MR2 — fixed welfare-scale ladder

Try Mean only:

```text
lambda_W = 0.5
lambda_W = 0.25
```

in that order.

Use the first value that passes competence.

Do not search more values.

Freeze the chosen lambda identically for:

```text
Mean
GGI
Maximin
```

## MR3 — timeout-dominated optional time cost

Only if all failed Mean runs are clearly timeout/stall dominated and scale ladder fails:

restore one small predeclared active-step time cost uniformly for all formal conditions.

Rerun Mean qualification.

If still fail:

```text
HARD STOP WELFARE-A
```

Do not invent a new reward after seeing results.

---

# 48. Formal freeze

Once Mean qualification passes, create:

```text
FORMAL_FREEZE_MANIFEST.json
```

Freeze:

```text
HighwayEnv version/fork commit
road geometry
simulation/policy frequency
action representation
acceleration range
observation builder
target speeds
TTC generator
scenario manifests
solver architecture
target sync
replay
n-step
PER
network
optimizer
LR/epsilon schedules
task reward
welfare lambda
time-cost setting
Mean/GGI/Maximin definitions
formal seeds
held-out banks
evaluation code
statistics plan
```

Set:

```text
formal_fairness_started = true
```

After this point:

> No outcome-dependent method modification is allowed.

---

# 49. Formal training

Train exactly:

```text
mean
ggi
maximin
```

using identical frozen configuration except welfare objective.

Target:

```text
6 independent seeds per condition
18 runs total
```

Emergency minimum:

```text
4 seeds per condition
```

only if compute makes six impossible and this reduction is decided before comparing welfare outcomes.

Do not drop poor seeds.

Repeat only genuine technical failures.

---

# 50. Held-out evaluation

Use common scenario IDs / common random numbers.

Required:

```text
H0 homogeneous held-out
H1 heterogeneous held-out
```

Target:

```text
256 scenarios each
```

Regenerate physical coordinates through the frozen HighwayEnv scenario adapter.

Do not change abstract role/TTC assignments between welfare conditions.

---

# 51. RQ1 evaluation

Use Mean policies only.

Compare H0 vs H1 on:

```text
U_mean
U_min
utility range
Gini
C_mean
C_max
burden range
burden by Ramp-Fast
burden by Ramp-Slow
burden by Mainline-Fast
burden by Mainline-Slow
worst-off identity
yielding
merge-first identity
below-target time
hard braking
```

Primary interpretation:

> Does target-speed heterogeneity increase inequality of welfare and coordination-associated mobility burden under a competent local-information policy?

---

# 52. RQ2 evaluation

Evaluate:

```text
Mean
GGI
Maximin
```

on H1.

Primary outcome:

$$
U_{\min}
$$

Safety/competence non-inferiority:

```text
collision difference < +0.03
completion difference > -0.05
```

Compare burden allocation and behavioral mechanisms.

---

# 53. Statistical analysis

Independent replication unit:

```text
training seed
```

Episodes are not independent algorithm replications.

Use:

```text
common scenario IDs
raw seed points
10,000 bootstrap resamples
95% confidence intervals
```

For primary RQ2 contrasts:

```text
GGI vs Mean
Maximin vs Mean
```

apply Holm correction.

---

# 54. Persistent autonomous state

Create:

```text
output/autonomous_highwayenv/
    AUTONOMOUS_EXPERIMENT_STATE.json
    AUTONOMOUS_EXPERIMENT_LOG.md
    GATE_RESULTS.json
    CODE_PROVENANCE.md
    HIGHWAYENV_BACKEND_CONFIG.json
    HIGHWAYENV_ENV_FREEZE.json
    ABSOLUTE_TRAINING_SCHEDULE.json
    SCENARIO_CURRICULUM_MANIFEST.json
    FORMAL_FREEZE_MANIFEST.json
```

State must include:

```json
{
  "backend": "highwayenv",
  "highwayenv_version": "1.12.0",
  "forked": false,
  "phase": "M0",
  "gate": null,
  "status": "not_started",
  "solver": "corrected_parameter_shared_local_dqn",
  "information": "local",
  "reward_mode": "task_only",
  "welfare_lambda": 0.0,
  "vdn_gate": "closed",
  "formal_fairness_started": false,
  "allowed_next_action": "snapshot_project"
}
```

Update atomically after every gate.

---

# 55. Automatic log format

Append only:

```markdown
## YYYY-MM-DD HH:MM — <phase/gate>

### Action
...

### Configuration
...

### Result
...

### Gate
PASS / FAIL / INCONCLUSIVE

### Evidence
...

### Automatic branch
...

### Files produced
...
```

Never erase failed history.

---

# 56. Code provenance requirements

Before formal freeze, classify every relevant component:

```text
Own implementation
HighwayEnv dependency
HighwayEnv subclass/wrapper
Adapted open-source code
Implementation reference only
Third-party dependency
```

If no fork:

Record:

```text
HighwayEnv 1.12.0
MIT license
project-local subclass/wrapper files
```

If fork:

Record exact upstream base and every patch.

Also document any code directly adapted from:

```text
MARL_CAVs
marlbenchmark/on-policy
PyMARL
DFRL
other repositories
```

Do not label conceptual/reference use as copied code.

---

# 57. HARD STOP conditions

## HARD STOP ENV-A

HighwayEnv migration cannot produce an oracle-solvable matched-TTC environment after bounded geometry/action/backend fixes.

Action:

```text
stop RL
write HARD_STOP_ENV_A.md
```

## HARD STOP SOLVER-A

Validated single scenario is oracle-solvable, pipeline checks pass, but bounded local-DQN/minimal-DQN/action-representation diagnostics cannot learn meaningful competence.

## HARD STOP SOLVER-B

Higher curriculum stage fails, diversity recovery fails, and corrected joint-information DQN does not materially outperform local DQN.

## HARD STOP SOLVER-C

Evidence-gated VDN is attempted and also fails bounded competence rescue.

## HARD STOP WELFARE-A

Task-only solver is competent, but Mean fails:

```text
lambda 1
lambda 0.5
lambda 0.25
and any allowed timeout-specific time-cost amendment
```

Do not redesign welfare after this.

## HARD STOP RESOURCE-A

Compute/time/storage cannot support the declared minimum formal design.

Preserve all completed runs and report the resource constraint.

---

# 58. Forbidden autonomous actions

Claude Code must not automatically:

```text
return to MAPPO
return to PBRS as formal mechanism
switch to QMIX
add GNN
add attention
add communication learning
add recurrence
change N
change 18/22 targets
remove local-information restriction
expose neighbour target speeds
change GGI weights
change H0/H1 after seeing outcomes
choose best seeds
discard bad seeds
change statistical tests after seeing significance
change HighwayEnv version during formal runs
merge upstream fork changes during formal runs
```

---

# 59. Final results package

If the full chain succeeds:

```text
output/final_study_b_highwayenv/
    CODE_PROVENANCE.md
    HIGHWAYENV_ENV_FREEZE.json
    FORMAL_FREEZE_MANIFEST.json
    TRAINING_RUN_MANIFEST.csv
    QUALIFICATION_RESULTS.csv
    RQ1_RESULTS.csv
    RQ2_RESULTS.csv
    SEED_LEVEL_SUMMARY.csv
    BOOTSTRAP_RESULTS.csv
    FIGURES/
    FIGURE_DATA/
    TABLES/
    FINAL_EXPERIMENT_SUMMARY.md
    REPRODUCIBILITY_NOTES.md
```

Every figure must have machine-readable source data.

---

# 60. Thesis reporting language

If the migration succeeds, Methods may describe the environment approximately as:

> The highway-merging simulator was implemented as a study-specific extension of HighwayEnv. HighwayEnv supplied the road-network representation, vehicle kinematics, connected-lane neighbour handling, collision mechanics and Gymnasium interface. The study-specific layer defined the four controlled vehicles, matched-TTC initialization, heterogeneous target speeds, local observation structure, three-action longitudinal control, task reward, welfare objectives and episode termination rules.

Do not say:

> "The experiment used the standard HighwayEnv merge environment."

That would be inaccurate because the study uses a customized environment.

If no fork was needed:

> The study extended a pinned HighwayEnv release through project-local subclasses and wrappers without modifying the upstream package.

If a fork was required:

> A pinned fork of HighwayEnv was used, with the upstream base version and all local modifications recorded in the reproducibility materials.

---

# 61. Final autonomous instruction

Execute this file as the authoritative remaining experiment state machine.

Priority:

$$
\boxed{
\text{trusted backend}
\rightarrow
\text{validated environment}
\rightarrow
\text{task competence}
\rightarrow
\text{Mean qualification}
\rightarrow
\text{formal welfare comparison}
}
$$

The migration is successful only when:

1. HighwayEnv backend gates pass;
2. oracle feasibility passes;
3. local-information solver competence passes;
4. Mean qualification passes;
5. formal Mean/GGI/Maximin held-out analysis is completed;

or an explicit HARD STOP is reached and documented.

Do not ask the user to choose between branches that are already ordered in this document.

---

# 62. AMENDMENT LOG

## Amendment 1 — 2026-08-16 — M6 continuation rule + action-adoption revalidation gate

Issued by the user BEFORE the M6-R3 (desired-speed/cruise-control)
400K continuation result was known. Full text:
`output/highwayenv_migration/PROTOCOL_AMENDMENT_M6_2026-08-16.md`.
Summary of what changed:

1. **Sec 35's M6 continuation rule is replaced** for this run: a
   single-scenario M6 run may continue 200K→400K even with completion
   below 0.80, provided numerics are healthy (no NaN/inf, no exploding
   gradients/TD-errors) AND the last 2-3 checkpoints show a clear
   positive trend AND no sustained collapse/plateau. Still a bounded
   extension -- never automatically beyond 400K. At 400K: PASS
   (>=0.90) → action-adoption revalidation then C4; INCONCLUSIVE
   (0.60-0.90) → one second seed at the SAME 400K budget (not the
   previously-specified shortened 300K), no cherry-picking; FAIL
   (<0.60, or two poor runs) → existing deeper recovery/HARD STOP path.
2. **New gate inserted between M6 acceptance and C4**: if the
   desired-speed representation is accepted, rerun M4-C/H/I/J/K and M5
   under it (not M4-D/geometry, unless a regression surfaces) before
   touching C4. On pass: update the freeze manifest, record the action
   space as a three-way desired-speed command (not direct
   acceleration), freeze it through C4/C16/C64/Mean/GGI/Maximin.
3. Methods-facing language for the action space must describe
   desired-speed commands translated to acceleration by HighwayEnv's
   own controller, not fixed -3/0/+2 m/s^2, if representation B is
   adopted.
4. Acceleration-DEPENDENT metrics (hard-braking, comfort penalty, any
   threshold-based accel metric) must be revalidated against the
   REALIZED physical acceleration distribution under the accepted
   controller, not the discrete action label -- any threshold amendment
   must happen before formal Mean/GGI/Maximin training and be applied
   identically to every formal condition, never chosen post hoc from
   fairness outcomes.
5. All other invariants (sec 3, sec 58's forbidden actions) explicitly
   reaffirmed, unchanged.

This amendment does not rewrite any prior gate outcome recorded in
`GATE_RESULTS.json`/`AUTONOMOUS_EXPERIMENT_LOG.md`.

## Amendment 2 — 2026-08-16 — expand M6-R3 INCONCLUSIVE resolution to a 4-seed design

Issued by the user after seed 900101's 400K result (0.818,
INCONCLUSIVE/LEARNABLE_WITH_VARIANCE per Amendment 1). Full text and
decision matrix: `output/highwayenv_migration/
M6_R3_ACTION_REPRESENTATION_COMPARISON.md`. Two more independent seeds
(900103, 900104) launched alongside the already-running 900102, all
identical (scenario, meta_speed representation, task-only reward, frozen
environment, network/replay/PER/n-step config, absolute schedule, 400K
budget each, no individual extensions). Superseded before completion by
Amendment 3 below (a material bug was found affecting all in-progress
runs).

## Amendment 3 — 2026-08-16 — completed-vehicle action-freeze bugfix (pre-formal implementation correction)

Full text: `output/highwayenv_migration/
BUGFIX_COMPLETED_VEHICLE_FREEZE_2026-08-16.md`. Found via a code audit
(user-requested, run in parallel with the Amendment-2 4-seed jobs):
completed vehicles kept receiving and applying policy actions
(including BRAKE) for the rest of the episode instead of freezing,
diverging from the legacy simulator's `a = 0.0` post-completion
semantics -- a plausible ADDITIONAL contributor (not claimed as the sole
cause) to the same-lane collision dominance already documented as M6's
primary failure mode. Fixed (`frozen` flag on both vehicle classes,
verified action-independent post-completion trajectories under both
representations). New gate **M4-L** (completed-vehicle inactivity
semantics) added and passed. M4-C/H/I/J/K + M5 oracle rerun and PASS
under the corrected code, including specifically under the accepted
`meta_speed` representation (satisfying Amendment 1 Change 2's
revalidation requirement in the same pass). Full suite: 235 passed, 0
failures.

Also renamed this backend's action terminology throughout its own
code/tests/docs: MAINTAIN->HOLD, DECELERATE->BRAKE (ACCELERATE
unchanged); "DECELERATE" is no longer used as a label anywhere in
`src/thesis/study_b/envs/` or its tests. The legacy, shared
`thesis.envs.stage10_symmetric_merge_env.HighLevelAction` module's own
names are explicitly NOT renamed (out of scope, shared code).

All Amendment-2 in-progress runs (seeds 900102 at step 350,000, 900103
and 900104 at step 325,000, all still running under the pre-fix code)
were terminated and marked `INVALID_FOR_POST_FIX_M6_DECISION` /
`PRE_FIX_M6_DIAGNOSTIC` (900101's completed 400K run). Checkpoints/logs
preserved, not deleted, not resumed. A clean 4-seed replication
(900101-900104, all fresh from step 0, `meta_speed`, corrected code,
identical 400K budget, no individual extensions) was launched under a
new four-tier decision rule (STRONG / LEARNABLE_WITH_VARIANCE / UNSTABLE
/ FAIL) that supersedes Amendment 1's simpler two-seed INCONCLUSIVE
branch for this specific comparison -- see `AUTONOMOUS_EXPERIMENT_LOG.md`
for the exact criteria and `GATE_RESULTS.json` for the outcome once
available. SUPERSEDED before completion by Amendment 4 below (a second
material issue -- unbounded control authority -- was found affecting
these same runs).

## Amendment 4 — 2026-08-16 — control-authority bound (CONTROL_AUTHORITY_MISMATCH)

Full text: `output/highwayenv_migration/
CONTROL_AUTHORITY_AMENDMENT_2026-08-16.md`. Found via a follow-up
acceleration audit on the Amendment-3 fix: the `meta_speed`
representation's realized physical acceleration was UNBOUNDED
([-18.09, +18.61] m/s^2), far outside the frozen legacy longitudinal
authority (BRAKE≈-3.0, HOLD≈0.0, ACCELERATE≈+2.0). Recorded as
`CONTROL_AUTHORITY_MISMATCH` (not an upstream HighwayEnv bug -- a
consequence of adopting desired-speed control without also bounding the
physical authority its controller may request).

All four Amendment-3 seeds (900101-900104, all still running,
step 25,000 reached) were terminated immediately, before any code
change, marked `M6_META_SPEED_UNBOUNDED_DIAGNOSTIC` +
`INVALID_FOR_FINAL_POST_FIX_M6_DECISION`; checkpoints/logs preserved,
not resumed
(`M6_META_SPEED_UNBOUNDED_DIAGNOSTIC_TERMINATION_RECORD.md`).

Fixed: `MetaSpeedControlledVehicle.act()` now clips HighwayEnv's own,
unmodified `speed_control()` output to `[-3.0, +2.0]` m/s^2 before it
reaches vehicle physics, applied on every physics substep (Python MRO
prevented intercepting via `super()` chaining, so the dispatch logic is
reproduced explicitly rather than the controller being reimplemented).
HOLD does not force acceleration to zero -- it only means target_speed
stops changing; realized acceleration under HOLD remains whatever
`speed_control()` requests, clipped to the same envelope. Completed
vehicles are unaffected (M4-L unchanged, still PASS). New gate **M4-M**
(physical control authority) added, PASS (Tests A-E).

Rerunning M5 oracle under the newly bounded controller initially
regressed to 0% completion (100% timeout) -- investigated per
instruction rather than loosening the clip, and found TWO oracle-side
issues (not clip defects): (1) the oracle's memoryless per-step
decisions caused unbounded `target_speed` windup (observed running to
-108 m/s during one sustained yield) -- fixed via a command debounce,
recovering to 96.9%; (2) the oracle's same-lane check grouped vehicles
by fixed scenario role rather than real physical lane, missing that an
already-merged ramp vehicle is in the same lane as a trailing mainline
vehicle -- fixed via an optional real-lateral-position parameter on
`oracle_actions()` (applied only for `meta_speed`; `direct_accel` and
the legacy backend are provably untouched, default `None` preserves
prior behavior exactly). Combined: **384/384 successful (0 collision, 0
timeout)** on Q/M/H1, matching the original target.

Full suite: 247 passed, 2 skipped, 0 failures. Freeze manifest updated
to record the clip and both oracle-side fixes. Four clean seeds
(900101-900104) relaunched from step 0 under the fully corrected
configuration (completed-vehicle freeze + control-authority clip +
`meta_speed`), same four-tier decision rule as Amendment 3.

## Amendment 5 — 2026-08-16 — final pre-formal end-to-end audit (user-supplied `audit.md`)

Full report: `output/pre_formal_audit/PRE_FORMAL_END_TO_END_AUDIT.md`
(gates A-T, per user-supplied `audit.md`). User instructed stopping
training FIRST (all 4 Amendment-4 seeds, step 25,000 each, preserved,
not resumed) before running this audit.

**Result: `AUDIT_PASS_WITH_LOW_RISK_NOTES`.**
`broad_engineering_audit_closed = true`.

One real issue found and repaired (AUDIT-1, severity 1):
`train_curriculum_stage_highwayenv.py`'s `--action-representation` CLI
flag defaulted to `direct_accel` (deprecated) instead of `meta_speed`
(the Amendment-4-accepted representation) -- no completed run was
affected (flag always passed explicitly), but this was a latent
default-value hazard for a future omitted-flag invocation. Fixed;
3 regression tests added. Three further severity-0, purely
informational notes documented (progress-reward denominator choice;
`MetaSpeedControlledVehicle.act()`'s idempotent double-call at frame 0;
explicit re-verification that the `target_speed` naming collision
between the thesis's intrinsic target and HighwayEnv's own control
setpoint does not leak into the observation).

New audit-only artifacts built: `evaluate_policy_highwayenv.py` (a
strictly-greedy, replay-write-back-free evaluation entrypoint that did
not previously exist for this backend), a combined action-pipeline +
single-trajectory trace tool (109-policy-step deterministic run on
Q_00000), and consolidated reward/metric hand-check, randomness, and
evaluation-pipeline reports.

Full suite: 252 passed, 2 skipped, 0 failures. Oracle re-confirmed
384/384 (Q=64, M=64, H1=256), 0 collision, 0 timeout, under the exact
configuration clean M6 uses. `FORMAL_SYSTEM_INVARIANTS.json` written
per audit.md sec 26, recording `allowed_next_action =
run_clean_m6_replication`.

Per the user's explicit instruction, clean M6 replication (4 seeds,
step 0, `meta_speed`, fully corrected+audited code) resumes after this
amendment.

## Amendment 6 — 2026-08-16 — M6_audited 4-seed replication result: LEARNABLE_WITH_VARIANCE

The 4 clean, fully-audited M6 seeds (900101-900104, `meta_speed`,
control-authority clip, completed-vehicle freeze, AUDIT-1-repaired
defaults) each completed the full 400K-step budget. Final completion/
collision/timeout: 900101=0.801/0.199/0.000, 900102=0.854/0.143/0.003,
900103=0.716/0.241/0.044, 900104=0.821/0.173/0.006. All 4 show clear,
sustained, non-collapsing learning curves; `mean_Q` stayed bounded
throughout every run.

Applying the four-tier decision rule: 3/4 seeds >=0.80, one at 0.716 --
substantial and sustained in every seed, but a meaningful final-value
spread. This matches `LEARNABLE_WITH_VARIANCE`, not `STRONG`.

**`SINGLE_SCENARIO_LEARNABILITY = LEARNABLE_WITH_VARIANCE`.** Per the
pre-specified rule this proceeds to C4, carrying the seed variance
forward as an explicit limitation rather than resolving it with further
M6 tuning (no outcome-dependent extension, no cherry-picking). The
action-adoption revalidation gate this branch requires is already
satisfied by Amendment 5's audit. Full evidence:
`output/autonomous_highwayenv/AUTONOMOUS_EXPERIMENT_LOG.md`'s
"2026-08-16 — M6_audited 4-seed replication complete" entry.

Before launching C4, one structural question not covered by the
original single-seed curriculum design (sec 38) was flagged to the
user rather than resolved unilaterally: with 4 independent accepted M6
seeds, does each continue independently into C4 (preserving seed as
the atomic replication unit through the whole curriculum), or some
other continuation rule?

## Amendment 7 — 2026-08-16 — C4 600K four-seed freeze (C4_STATUS = NOT_YET_QUALIFIED)

User answered Amendment 6's open question ("4 seeds continue
independently"); all 4 M6-accepted seeds ran the C4 200K extension
(400K->600K) identically. Final 600K completion/collision/timeout:
900101=0.738/0.262/0.000, 900102=0.783/0.204/0.013,
900103=0.814/0.186/0.000, 900104=0.781/0.210/0.008.

Sec 40's per-lineage gate (PASS>=0.90; SOFT PASS 0.75<=completion<0.90
+ improving -> +100K; FAIL <0.75 -> DR1-DR4) has no defined rule for
aggregating across 4 independent seeds when they land in different
bands, as happened here: 0/4 strict PASS, 3/4 SOFT PASS
(900102/900103/900104), 1/4 below the SOFT PASS floor at its final
checkpoint (900101 = 0.738, despite a 550K peak of 0.820 which is
explicitly NOT substituted as the gate value).

Rather than resolve this gap unilaterally, the gap was surfaced before
interpreting the result. The user supplied a full protocol
(2026-08-16): **freeze all 4 lineages at the common 600K checkpoint**
(`C4_STATUS = NOT_YET_QUALIFIED`, no C16, no seed-specific
retraining/dropping, all checkpoints/logs/configs preserved), run
identical DR1 (failure map) + DR2 (behavior/observation/control
diagnostics) on all 4 seeds, branch on whether a genuine
result-changing defect is found:

- **Defect found** -> classify severity, stop qualification, preserve
  evidence, add a minimal failing test, repair globally, full
  regression + oracle revalidation if environment/control/reward/
  termination changed, invalidate affected C4 evidence, restart
  affected qualification consistently across all four seeds (never
  fix only one seed).
- **No defect, healthy learning** -> write a
  `C4_FOUR_SEED_UNIFORM_EXTENSION_AMENDMENT` recording that the
  original single-lineage C4 rule did not define 4-seed aggregation,
  and that (pre-formal, no fairness outcome inspected, applied
  identically to all 4, no seed dropped) all four resume uniformly
  from their own 600K checkpoint for exactly +100K to 700K, preserving
  each seed's own network/optimizer/replay/absolute LR+epsilon
  schedule (no reset, no cross-seed exchange). Evaluate all 4 at the
  common 700K checkpoint using the ORIGINAL sec-40 numeric targets
  (no new threshold), report both per-seed and stage-level results,
  and do not advance only the best-performing seeds.

Explicit interpretation for the record (frozen at 600K): "C4 remained
learnable across all four independent qualification seeds, but the
predefined competence and safety target had not yet been reached.
Three seeds satisfied the original per-lineage soft-pass completion
range, while one fell slightly below it at the common final
checkpoint. Because the original C4 rule did not define aggregation
across four independent lineages, the qualification procedure was
paused at the common 600K checkpoint for uniform diagnostic analysis."
Not "C4 passed." Not "900101 failed while the others passed." The
stage as a whole is `NOT_YET_QUALIFIED`.

Full evidence:
`output/autonomous_highwayenv/AUTONOMOUS_EXPERIMENT_LOG.md`'s
"2026-08-16 — C4 600K results" entry; DR1/DR2 outputs under
`output/c4_diagnostics/`.

## Amendment 8 — 2026-08-16 — C4 four-seed uniform extension (NO-BUG branch, 600K -> 700K)

DR1 (600 diagnostic episodes/seed, N=2400 pooled, epsilon-greedy at
`epsilon_at_step_v12(600000, 640000)`=0.15625 -- the exact policy
behind the reported 600K window stats) and DR2 (full `tests/study_b/`
suite + 8 deterministically-traced representative episodes) ran
identically on all 4 frozen 600K checkpoints.

**No implementation defect found**: regression suite 252 passed/2
skipped/0 failures; 0/8 traced episodes violated any of invariants
1/2/3/4/7/8/9 (completed-vehicle inactivity, acceleration clip
[-3,+2] + clip-consistency, collision geometry, active-mask
consistency; no timeout episodes occurred so nothing to check there).

**A significant non-defect finding**: the 600K training-window
completion numbers are noisy small-N estimates. Under DR1's reliable
N=600-per-seed sample, the true ranking is 900102(0.985) >
900101(0.783) > 900103(0.688) > 900104(0.262) -- 900104, which read as
a comfortable SOFT PASS from its training-window number (0.781), is
in fact the weakest seed by a wide margin, while 900101, the one seed
that read as below the SOFT PASS floor (0.738), is actually solidly
mid-pack. Cross-checked and corroborated in direction/ranking by an
independent, already-audited greedy N=4 eval
(`evaluate_policy_highwayenv.py`), confirming this is a real property
of the checkpoints and not a bug in the new DR1 harness.

Per the user's protocol sec 6 (no-bug/healthy-learning branch), a
transparent amendment was written:
`output/highwayenv_migration/C4_FOUR_SEED_UNIFORM_EXTENSION_AMENDMENT_2026-08-16.md`
-- RUNBOOK sec 40's per-lineage gate has no rule for aggregating 4
independent seeds, so (pre-formal, applied identically to all 4, no
seed dropped) the procedure is amended so all four lineages receive
the same additional +100K exposure. All 4 seeds resumed from their own
600K checkpoint, unchanged config, absolute step 600000->700000.

Given the measurement-noise finding above, the 700K decision will
report BOTH the raw training-window number AND a dedicated N=600-
episode fixed-epsilon re-measurement (same DR1 methodology), against
the ORIGINAL sec-40 numeric targets -- no new threshold invented, a
measurement-reliability choice only.

## Amendment 9 — 2026-08-17 — authoritative C4 gate corrected to epsilon=0 greedy

User identified a further methodological issue: even DR1's N=600
"reliable" re-measurement (Amendment 8) was still exploration-on
(epsilon=0.15625) and must not be the authoritative C4/C16/C64 gate --
only the frozen GREEDY policy (epsilon=0) measures solver competence.
Both the 600K training-window numbers and DR1's diagnostic are
retroactively relabeled `TRAINING_WINDOW_ESTIMATE` /
`EXPLORATION_ON_DIAGNOSTIC` -- retained in full, no longer
authoritative.

The in-progress 600K->700K uniform extension (Amendment 8) had reached
step 650,000 on all 4 seeds when this instruction arrived. **Training
was stopped immediately**, all checkpoints (600K, 650K) preserved for
all 4 seeds, the partial run marked `SUPERSEDED_NOT_USED_FOR_GATE_
DECISION`.

New evaluation-correctness tests
(`tests/study_b/test_c4_greedy_gate_eval.py`, 4/4 passing) confirmed:
`epsilon=0.0, greedy=True` never touches the exploration RNG; the
checkpoint loader touches no optimiser/target state; different
checkpoints map to different seeds correctly; greedy evaluation
against a fixed scenario is bit-for-bit deterministic. That last
finding means the C4 evaluation bank's own frozen 4-scenario structure
makes N=4/seed an EXACT characterization of the frozen greedy policy,
not a statistical sample -- N=600 is not achievable (or meaningful)
under epsilon=0 against 4 fixed scenarios.

**AUTHORITATIVE_C4_GATE_EVALUATION at 600K** (new script
`c4_greedy_gate_eval.py`): 900101=0.500 (**FAIL**), 900102=1.000
(**PASS**), 900103=0.750 (**SOFT_PASS**, exact floor), 900104=0.000
(**FAIL**, total collapse -- 4/4 scenarios collide). Gate counts:
PASS=1/4, SOFT_PASS=1/4, FAIL=2/4. Pooled completion=0.5625.

Paired diagnostic vs. DR1's exploration-on estimate: greedy is
substantially WORSE for 900101 (delta=-0.283) and 900104
(delta=-0.262) -- exploration was incidentally masking these two
seeds' weaknesses, not causing them -- and marginally better for
900102/900103 (+0.015/+0.062), as naively expected.

**Branch B**: greedy performance is not substantially better than
exploration-on for any seed and is materially worse for two of four
-- the competence deficit is real, intrinsic to the learned policy,
not an exploration artifact. Stage-level PASS=1/SOFT_PASS=1/FAIL=2 is
neither "most/all seeds strong" nor "only one seed poor" -- **C4
remains `NOT_YET_QUALIFIED`.**

Per protocol sec 9 Branch B and the already-amended no-bug
qualification recovery logic (Amendment 8), the uniform +100K
extension is relaunched cleanly from each seed's own 600K checkpoint
(not resumed from the interrupted 650K point -- also technically
required, since the training script's exploration RNG is not part of
persisted checkpoint state, so only a fresh 600K restart reproduces
what an uninterrupted run would have produced). Unchanged config. The
700K decision will apply the authoritative greedy gate as primary,
with the training-window number and eps=0.15625 diagnostic retained as
secondary context, against the ORIGINAL sec-40 targets -- no new
threshold.

## Amendment 10 — 2026-08-17 — 700K authoritative greedy gate: severe checkpoint-to-checkpoint volatility, stopped for user input

The clean C4ext2 restart (600K->700K, all 4 seeds) completed. The
authoritative greedy gate at 700K: 900101=0.500 (FAIL, unchanged),
900102=0.000 (FAIL, **collapsed from PASS/1.000 at 600K**),
900103=0.500 (FAIL, declined from SOFT_PASS/0.750), 900104=1.000
(**PASS, recovered from total collapse/0.000 at 600K**). Gate counts
went from PASS=1/SOFT=1/FAIL=2 (600K) to PASS=1/SOFT=0/FAIL=3 (700K)
-- worse, despite the already-amended uniform +100K no-bug recovery
extension (Amendment 8).

Two of four seeds' PASS/FAIL classification fully inverted across a
single 100K-step window, decoupled from the training-window trend
(900104's window completion *declined* 0.781->0.541 while its greedy
score went from 0 to 1). The greedy-evaluated policy is evidently not
stable/converged at either 600K or 700K.

RUNBOOK sec 40's FAIL-branch text ("run Diversity Recovery DR1-DR4")
now literally applies, but that refers to this document's own
pre-existing DR1-DR4 sequence (secs ~1859-1908: per-scenario failure
map / local-observation-aliasing audit / replay-curriculum-retention
audit / corrected joint-information diagnostic) -- a different thing
from the "DR1"/"DR2" built 2026-08-16 for the bug-vs-nobug check, which
reused the same short names for a narrower purpose. DR4 specifically
borders joint-information/VDN territory the user separately instructed
elsewhere not to reopen without explicit authorization.

This checkpoint-volatility finding is not addressed by any branch in
the 2026-08-17 protocol message. Rather than unilaterally launch
RUNBOOK's own DR1-DR4 diversity-recovery sequence, this was flagged to
the user for a decision. `C4_STATUS` remains `NOT_YET_QUALIFIED`.

## Amendment 11 — 2026-08-17 — RUNBOOK Diversity Recovery DR1-DR3 executed (diagnostic only): primary mechanism identified, DR4 stays closed

User authorized this document's own pre-existing Diversity Recovery
sequence, DR1-DR3 only (DR4 explicitly withheld). No training
performed; all checkpoints (550K/600K/650K/700K, all 4 seeds) frozen
and preserved.

**DR1** (`output/c4_diagnostics/DR1_FAILURE_MATRIX/`): full scenario x
checkpoint x seed failure matrix, 64 greedy evaluations. Every (seed,
scenario) pair flips at least once across the 4 checkpoints. 900102/
900104 flip all 4 scenarios together uniformly (opposite directions);
900101/900103 show partial, mixed-direction trade-offs. No scenario/
role/speed-class concentration under greedy.

**DR2** (`output/c4_diagnostics/DR2_ALIASING/`): `NO_EVIDENCE_OF_
ALIASING`. An initial run produced a spurious STRONG-evidence result
traced to comparing the SAME fixed scenario across DIFFERENT seeds
(trivially observation-identical) -- caught, fixed, rerun; minimum
cross-scenario distance was ~48x the near-duplicate threshold.

**DR3** (`output/c4_diagnostics/DR3_REPLAY_RETENTION/`): `NO_
RETENTION_PROBLEM_FOUND` on reconstructable (DIRECTLY_OBSERVED,
per-scenario training-window) evidence -- balanced sampling, uniform
cross-scenario co-movement. Per-transition PER/TD-error data was never
logged and is marked `NOT_RECONSTRUCTABLE`, not invented.

**Additional action-flip + Q-margin analysis**
(`output/c4_diagnostics/ACTION_FLIP/`): 29/30 divergent greedy
decisions have a Q-margin under 0.005 on a ~0.36-0.65 scale
(near-degenerate ties), with a systematic 77% directional drift toward
more-conservative actions between 600K and 700K -- the single
mechanism explaining both 900104's recovery and 900102's collapse.

**Synthesis** (`output/c4_diagnostics/C4_DIVERSITY_RECOVERY_SYNTHESIS.md`):
primary classification **C -- SMALL_Q_MARGIN_POLICY_BOUNDARY_
INSTABILITY**, strongly supported. A and D ruled out; B not supported
on available evidence; E not supported (clean regression suite,
bounded Q-values). Recommendation: **`NO_SAFE_RECOVERY_SUPPORTED`** --
neither R1 (replay/curriculum) nor R2 (observation) nor DR4-
authorization is supported by the evidence; the mechanism doesn't map
cleanly onto the 5 listed categories. Returned to the user for a new
decision rather than force-fitting a category or unilaterally
proposing a stabilization change that would touch prohibited
invariants.

`C4 = UNSTABLE_NOT_QUALIFIED` / `C4_REASON =
CHECKPOINT_LEVEL_GREEDY_POLICY_INSTABILITY`. `C16`/Mean qualification/
formal welfare training remain `BLOCKED`. `DR4 =
CLOSED_PENDING_EXPLICIT_AUTHORIZATION`.

## Amendment 12 — 2026-08-17 — Bounded checkpoint-Q ensemble stabilization: CASE A SUCCESS

User authorized exactly one bounded pre-formal stabilization
amendment, motivated directly by Amendment 11's diagnosed
SMALL_Q_MARGIN_POLICY_BOUNDARY_INSTABILITY mechanism: a fixed
equal-weight arithmetic-mean Q-ensemble over each seed's own
{550K, 600K, clean-650K (C4ext2, NOT the superseded C4ext), 700K}
checkpoints. `Q_ensemble(o,a) = (1/4) sum_k Q_theta_k(o,a)`,
`a = argmax_a Q_ensemble(o,a)`. No training, no checkpoint selection
by outcome, no per-checkpoint weighting, no other invariant changed.

New additive module `src/thesis/study_b/q_ensemble.py` + script
`c4_q_ensemble_gate.py`; no legacy single-checkpoint evaluation code
modified. `load_ensemble_agents()` structurally enforces exactly the
4 expected steps, same-seed checkpoint directories, and each step's
saved `stage` field (`550K`->`"C4"`, `600/650/700K`->`"C4ext2"`) --
which structurally rejects the superseded `C4ext` 650K checkpoint. 10
new tests (`tests/study_b/test_q_ensemble.py`, A-J) all pass. Full
suite: 266 passed, 2 skipped, 0 failures.

**C4_Q_ENSEMBLE_GATE result** (N=4/seed, deterministic):
900101=1.000(PASS), 900102=1.000(PASS), 900103=1.000(PASS),
900104=0.750(SOFT_PASS). Strict PASS=3/4.

Diagnostic-only margin comparison (does not affect the decision):
ensemble median margin (0.00130) was NOT larger than the component
median margin (0.00255) -- stabilization is attributed to
cross-checkpoint variance reduction in which action wins the argmax,
not per-decision margin widening. Reported for honesty.

**Frozen decision rule applied: CASE A** (>=3/4 strict PASS) ->
`C4_Q_ENSEMBLE_STABILIZATION = SUCCESS`, **`C4 = QUALIFIED_FOR_NEXT_
CURRICULUM_STAGE`**. All 4 seeds preserved, no cherry-picking. The
checkpoint-Q-ensemble rule is now frozen as part of the solver
definition (`parameter-shared local DQN + fixed equal-weight recent-
checkpoint Q ensemble for deterministic execution/evaluation`) for all
future qualification and formal-condition evaluation. **C16 was NOT
started inside this task**, per explicit instruction -- a separate
continuation amendment must define the same fixed-window ensemble
rule for C16/C64 and later formal training BEFORE C16 begins.

`DR4 = CLOSED_PENDING_EXPLICIT_AUTHORIZATION`, unaffected by this
amendment. Full evidence:
`output/highwayenv_migration/C4_Q_ENSEMBLE_STABILIZATION_AMENDMENT_2026-08-17.md`.

## Amendment 13 — 2026-08-17 — Fixed checkpoint-Q ensemble continuation rule (C16/C64/Mean/GGI/Maximin, frozen prospectively) + C16 launch

**Interpretation correction to Amendment 12** (does not reopen the C4
decision): the ensemble did NOT increase Q-margins (component median
0.00255 vs. ensemble median 0.00130 -- ensemble smaller). Mechanism
relabeled `TEMPORAL_ARGMAX_VARIANCE_REDUCTION`: equal-weight
checkpoint averaging reduces checkpoint-to-checkpoint variation in
which action wins the greedy argmax, producing stable deterministic
performance despite a smaller median margin. Composes with, does not
replace, `SMALL_Q_MARGIN_POLICY_BOUNDARY_INSTABILITY` (Amendment 11).

Full continuation rule written and frozen BEFORE any C16 outcome
exists:
`output/highwayenv_migration/C16_C64_CHECKPOINT_Q_ENSEMBLE_CONTINUATION_AMENDMENT_2026-08-17.md`.
Solver definition: parameter-shared local DQN training (unchanged,
ordinary single online process) + fixed equal-weight checkpoint-Q
ensemble for authoritative greedy execution/evaluation ONLY. Future
window rule: for a stage ending at absolute step S,
`K(S)={S-150K,S-100K,S-50K,S}`, applied identically to C16, C64, Mean
qualification, and formal Mean/GGI/Maximin -- never a different
window/weighting per condition.

C16 budget verified unambiguous (sec 41: 250K from the accepted C4
checkpoint, now 700K -- no contradiction in the other 3 tracking
files, no HARD STOP). S16=950,000, `K(950000)={800K,850K,900K,950K}`,
already covered by the existing unmodified 50K checkpoint cadence.
C16 scenario bank verified (`C16.json`, 16 scenarios, contains C4's 4
scenarios as a subset). Full suite reconfirmed green (266/2/0) before
launch.

**C16 launched**: all 4 seeds resumed from their own 700K checkpoint,
`--start-step 700000 --max-additional-steps 250000`, C16.json bank,
`condition=mean welfare_lambda=0.0`, `meta_speed`, unchanged absolute
LR/epsilon schedules. C64's rule is frozen prospectively (same logic,
same `>=3/4` strict-PASS stage rule) but C64 itself was NOT started.

C4 final record: `C4_Q_ENSEMBLE_STABILIZATION=SUCCESS`;
`C4_STAGE=QUALIFIED`; `C4_STRICT_PASS_COUNT=3/4`;
`C4_WEAK_SEED_RETAINED=900104`; `ENSEMBLE_Q_MARGIN_INCREASED=false`;
`DR4=CLOSED_PENDING_EXPLICIT_AUTHORIZATION`.

**C16 result** (same session, 2026-08-17): all 4 seeds completed
training to `final_step=950000`. `q_ensemble.py` generalized to accept
an explicit window/stage-map (default preserves the original C4
behavior, all 10 existing tests reverified passing) + new generic
`stage_q_ensemble_gate.py`, so no future stage's evaluation can
accidentally reuse C4's hardcoded steps. Authoritative gate at
`K(950000)={800K,850K,900K,950K}`, N=16/seed: 900101=1.000(PASS),
900102=1.000(PASS), 900103=1.000(PASS), 900104=0.750(SOFT_PASS).
Strict PASS=3/4 -> **CASE A -> `C16 = QUALIFIED_FOR_NEXT_CURRICULUM_
STAGE`**, same margin-did-not-increase pattern as C4 (ensemble median
0.00187 vs. component median 0.00241). All 4 seeds retained. **C64 was
NOT started automatically**, per instruction -- returned for review.

**Full autonomous-controller authorization** (2026-08-17, same
session): user granted authority to proceed automatically through C64,
Mean qualification, formal freeze, and all 18 formal runs, stopping
only for a genuine protocol contradiction / unresolved defect / failed
gate with no safe recovery / missing state / formal completion.

Two RUNBOOK specification gaps resolved before C64 launch (not HARD
STOPs): (1) no `C64.json` existed -- `scenario_banks/Q.json` (already
present, 64 scenarios) verified to be an exact superset of C16's 16
scenarios, completing the nesting -- this IS the C64 bank, just not
renamed; (2) sec 42 has no explicit step budget (unlike sec 40/41) --
resolved by precedent, 250K matching C16's own flat budget under the
identical methodology, decided before any C64 outcome. C64 launched:
all 4 seeds resumed from `@950K`, `--max-additional-steps 250000`,
`Q.json` (64 scenarios), unchanged task-only config. S64=1,200,000,
`K(1200000)={1050K,1100K,1150K,1200K}`.

**C64 result**: all 4 seeds completed to 1,200,000. Authoritative gate:
900101=1.000(PASS,64/64), 900102=0.969(PASS,62/64),
900103=0.812(SOFT_PASS,52/64), 900104=0.938(PASS,60/64). Strict
PASS=3/4 -> **CASE A -> `C64 = QUALIFIED`, `SOLVER_TASK_QUALIFICATION
= PASS`**. Seed-role reversal vs. C4/C16 (900104 now PASS, 900103 now
SOFT_PASS, no outright FAIL this round) -- margin diagnostic again
shows no increase (ensemble median 0.00159 vs. component 0.00212),
`TEMPORAL_ARGMAX_VARIANCE_REDUCTION` confirmed a third independent
time. Per full-autonomy authorization, proceeding automatically to
Phase B — Mean welfare qualification.

**Phase B launched** (Mean welfare qualification, secs 45-47):
protocol resolved before launch -- seeds 900101/900102 (first two in
fixed ordinal order, not outcome-selected), continuation from each
seed's own `C64@1200K` checkpoint, `lambda_W=1`, `Q.json` bank,
`+800,000` additional steps (`Smean_initial=2,000,000`, one
pre-authorized extension to `+1,000,000` if clearly improving near
800K), stage gate frozen as 2/2 strict PASS for N=2 (the master
protocol's `>=3/4` rule is conditioned on 4 seeds). Pre-authorized
failure ladder if needed: sec 47 MR1->MR2(`lambda_W`=0.5 then 0.25,
stop at first pass)->MR3->`HARD STOP WELFARE-A`. Full evidence:
`AUTONOMOUS_EXPERIMENT_LOG.md` "2026-08-17 - Phase B launched" entry.

**Mean qualification result (lambda_W=1.0): FAILED.** 900101=1.000
(PASS), 900102=0.703 (FAIL, collision-dominated, timeout=0.000).
Strict PASS=1/2 < required 2/2. MR1 diagnosis: regression suite clean
(266/2/0), 900101 undegraded, 900102's failure has no NaN/collapse
signature -- classified MEAN-F2 (no bug). Proceeding to MR2's first
rung: `lambda_W=0.5` launched, both seeds fresh-restarted from
`C64@1,200,000` (not continuing the failed run), identical procedure,
same 2/2 gate at `Smean=2,000,000`.

**MR2 rung 1 (`lambda_W=0.5`) result: QUALIFIED.** 900101=0.984
(PASS), 900102=1.000 (PASS). Strict PASS=2/2, meets the required 2/2.
**`lambda_W=0.5` is now frozen** for Mean/GGI/Maximin per sec 47
("use the first value that passes... do not search more values") --
`lambda_W=0.25` was never needed. Phase C (formal freeze + 18 formal
runs) is deferred per the user's 2026-08-17 locality-sensing-amendment
instruction (separate, isolated investigation, no new training under
any observation config for now); will resume once that concludes.

**Locality-sensing-amendment investigation (2026-08-17, isolated,
`final_new_experiment/E34_locality_amendment_worktree/`)**: audited
the current nearest-3 neighbour selection (no distance cutoff; at
N=4 this always exposes all other vehicles), implemented an optional
`local_sensing_range_m` parameter (default `None` = old behaviour
exactly) reusing the existing `presence` mask, 13/13 new tests
passing, active tree confirmed untouched. **`RANGE_STATUS =
REQUIRES_PROSPECTIVE_FREEZE`** -- no numeric sensing range is frozen
anywhere in the project (`new_research_plan.md`'s `R_obs` entry is
explicitly provisional and is a different, normalization-scale
concept). No training launched under either configuration; Phase C
remains deferred pending either proceeding under the OLD config or a
prospective range freeze + fresh re-qualification under the NEW one.

---

## Amendment 14 — 2026-08-17 — R=50m locality range frozen and adopted; zero-training C64 bridge FAILED; HARD STOP

User froze **R=50m** as the prospective local-sensing range
(information-structure/geometry justification: Phase-8 spawn-time
audit shows R>=75m is approximately unlimited over the `Q.json` bank,
R=30m is a materially stronger restriction, R=50m gives meaningful
partial visibility -- explicitly NOT because it coincides numerically
with the unrelated `R_OBS_DEFAULT=50.0` normalization constant), and
directed a time-saving bridging route instead of a full curriculum
re-qualification from M6: adopt the amendment into the active tree,
then test whether the existing task-only-qualified C64 checkpoints
still pass the original C64 competence gate purely at inference time
under the new masking, before deciding whether any new training is
needed at all.

Amendment adopted into the ACTIVE tree (not just the isolated
worktree): `local_observation.py`'s `local_sensing_range_m` parameter,
`StudyBHighwayWrapperConfig.local_sensing_range_m`,
`stage_q_ensemble_gate.py --local-sensing-range-m` (all additive,
default `None` preserves every prior invocation's behaviour exactly).
Full `tests/study_b/` suite green after adoption (266 passed, 2
skipped, 0 failures, unchanged from before).

**Zero-training C64 compatibility bridge: FAILED.** Existing 4-seed
task-only C64 Q-ensemble (K(1,200,000), epsilon=0, no training/replay/
optimizer updates) evaluated under R=50m masking against the original
`>=3/4` strict-PASS gate: strict PASS = **1/4** (900101 SOFT_PASS
0.781/0.219, 900102 PASS 0.906/0.094, 900103 SOFT_PASS 0.844/0.156,
900104 FAIL 0.672/0.328). Collision rates rose for 3/4 seeds relative
to the OLD-observation C64 gate. The compatibility shortcut does not
hold.

**Consequence**: per the user's own explicit carve-out, bridge FAIL is
one of exactly 3 conditions requiring a stop-and-ask (the other two:
checkpoint fails to exact-load; an unfrozen locality semantic remains
-- neither applies). No Mean qualification or any new training has
been launched. Fresh task-only curriculum retraining under R=50m is
required; the runbook does not, by itself, resolve whether the correct
restart point is M6 (its R2 diagnostic was explicitly scoped to "same
local observation," so its finding does not automatically transfer) or
whether C4 is a defensible faster entry point -- this is left as an
explicit decision point, not self-simplified.

The OLD-observation `lambda_W=0.5` MR2 QUALIFIED result (2/2 strict
PASS) is preserved unchanged, explicitly labeled PRE-LOCALITY-AMENDMENT
evidence, not reused for the eventual R=50m formal freeze.

Evidence: `output/c64_diagnostics/LOCALITY_BRIDGE_R50/`,
`output/autonomous_highwayenv/AUTONOMOUS_EXPERIMENT_LOG.md` (2026-08-17
"R=50m locality range frozen and adopted" entry),
`output/autonomous_highwayenv/GATE_RESULTS.json`
(`LOCALITY_AMENDMENT_R50_FROZEN`, `C64_LOCALITY_BRIDGE_R50`).

---

## Amendment 15 — 2026-08-17 — R=50m curriculum rebuild: M6 restart LEARNABLE_WITH_VARIANCE, C4 launched

User chose "restart from M6" over "restart from C4". `M6_R50_audited`
(identical protocol to `M6_audited`, only `--local-sensing-range-m
50.0` added) completed all 4 seeds cleanly: 900101=0.777/0.212/0.012,
900102=0.730/0.238/0.032, 900103=0.737/0.245/0.018,
900104=0.768/0.214/0.019. Classified `LEARNABLE_WITH_VARIANCE` by the
same elimination logic used for the OLD-observation M6 result (no seed
reaches the ~0.80 STRONG bar; no seed collapses). Per the pre-specified
rule, proceeded automatically to C4 (all 4 seeds, `400K->600K`, R=50m,
otherwise identical to the OLD-observation C4 launch) -- within the
existing curriculum auto-branch table, not a new stop-and-ask point.

Full curriculum rebuild ahead: C4 -> C16 -> C64 -> fresh Mean/GGI/
Maximin qualification, all under R=50m from scratch (the zero-training
shortcut already failed once at C64 and will not be retried). Whether
Mean qualification restarts its lambda_W ladder at 0.5 (the
OLD-observation frozen value) or reopens from 1.0 is left open until a
fresh C64 gate under R=50m actually passes.

Evidence: `output/autonomous_highwayenv/AUTONOMOUS_EXPERIMENT_LOG.md`
(2026-08-17 "R=50m curriculum rebuild" entry), `GATE_RESULTS.json`
(`M6_R50_audited_4tier_classification`, `C4_R50_launch`).

C4 (R=50m) 600K result: all 4 seeds SOFT_PASS (0.813-0.868),
improving -- following the OLD-observation C4 precedent (Amendment 8),
extended uniformly +100K to 700K. Will apply the already-frozen
checkpoint-Q-ensemble gate (Amendment 13's mechanism, K(700000),
epsilon=0) directly at 700K rather than re-deriving DR1-DR4, since
that diagnostic sequence validated the ensemble mechanism itself, not
anything R-specific. Evidence: `AUTONOMOUS_EXPERIMENT_LOG.md` 2026-08-17
"C4 (R=50m) 600K result" entry.

C4 (R=50m) QUALIFIED at 700K: authoritative ensemble gate (K(700000),
mixed stage names C4_R50/C4_R50ext handled via a per-step
expected_stage_by_step dict, new one-off script
c4_r50_mixed_stage_ensemble_gate.py) gives 3/4 strict PASS
(900101/900102/900104=1.000 PASS, 900103=0.750 SOFT_PASS) -- same
TEMPORAL_ARGMAX_VARIANCE_REDUCTION pattern as every OLD-observation
stage. C16 under R=50m launched (700K->950K, all 4 seeds, otherwise
identical to the OLD-observation C16 launch). Evidence:
`AUTONOMOUS_EXPERIMENT_LOG.md` 2026-08-17 "C4 (R=50m) QUALIFIED at
700K" entry.

C16 (R=50m) QUALIFIED at 950K: 3/4 strict PASS (900101=1.000,
900102=0.938, 900104=1.000 PASS; 900103=0.625 outright FAIL, retained
not dropped). C64 under R=50m launched (950K->1200K, all 4 seeds,
otherwise identical to the OLD-observation C64 launch) -- the same
competence gate the zero-training bridge already failed once, now
trained fresh from scratch under R=50m. Evidence:
`AUTONOMOUS_EXPERIMENT_LOG.md` 2026-08-17 "C16 (R=50m) QUALIFIED at
950K" entry.

C64 (R=50m) QUALIFIED at 1,200,000: 3/4 strict PASS (900101=0.984,
900102=1.000, 900104=1.000 PASS; 900103=0.453 outright FAIL, a
notable regression from the OLD-observation C64's 0.812 SOFT_PASS for
the same seed). Curriculum rebuild (M6->C4->C16->C64) complete,
confirming the zero-training bridge's failure was genuine and the
rebuild was necessary. **STOPPED before launching Mean qualification**:
whether to start the lambda_W ladder at 0.5 (OLD-observation frozen
value) or reopen from 1.0 is an open question not covered by the
pre-specified auto-branch table, and 900103's regression is a concrete
reason the OLD choice may not transfer -- returned to the user rather
than decided unilaterally, per pre-registration discipline (deciding
after seeing any Mean-qualification result would be outcome-dependent
tuning). Evidence: `AUTONOMOUS_EXPERIMENT_LOG.md` 2026-08-18 "C64
(R=50m) QUALIFIED... STOP before Mean qualification" entry.

User decided: lambda_W=0.5 FIXED for Mean qualification under R=50m
(no re-test of 1.0/0.25). Mean qualification launched (seeds
900101/900102, 1200K->2000K, otherwise identical to the OLD-observation
Mean qualification launch), now also with `OMP_NUM_THREADS=1` per
process (new additive performance change -- reduces thread
oversubscription across parallel seed-processes for this tiny network,
does not affect determinism/results, not applied retroactively).
PASS -> continue to frozen formal conditions; FAIL -> STOP, no
auto-tuning of lambda_W or R. Evidence: `AUTONOMOUS_EXPERIMENT_LOG.md`
2026-08-18 "User decision: lambda_W=0.5 FIXED... launched" entry.

Mean qualification (R=50m, lambda_W=0.5) QUALIFIED with a PERFECT 2/2
(both seeds 1.000 completion, 0 collision). Closes the locality
amendment end-to-end -- R=50m+lambda_W=0.5 is now the frozen protocol.
**STOPPED before Phase C**: sec 46's "do not reuse qualification seeds
as formal seeds" was interpreted earlier this session as excluding all
four curriculum seeds (900101-900104), meaning all 6 formal seeds need
their own fresh full curriculum build under R=50m (~1.2M steps each)
before any formal training starts -- a large compute commitment not
pre-sized/approved, returned to the user rather than committed to
silently. H0.json/H1.json held-out banks already exist. Evidence:
`AUTONOMOUS_EXPERIMENT_LOG.md` 2026-08-18 "Mean qualification (R=50m,
lambda_W=0.5) QUALIFIED... STOP before Phase C" entry.

**Amendment 16 (2026-08-18, pre-outcome)**: user resolved the formal-seed
scope question, superseding the earlier all-four-excluded reading of
sec 46: formal 6-seed matched block = {900101,900102,900103,900104,
910101,910102}, same 6 across Mean/GGI/Maximin, lambda_W=0.5 fixed for
all 3. 900101-900104 (task-only curriculum only, never welfare-trained
during qualification) are reusable; 900103 stays despite its C64 FAIL.
910101/910102 (verified unused) need a fresh R=50m task-only curriculum
build, now in progress -- prerequisite work, not formal welfare
training. Formal training itself stays gated until the manifest is
written and held-out banks/hashes confirmed. Evidence:
`AUTONOMOUS_EXPERIMENT_LOG.md` 2026-08-18 "Formal-seed scope resolved"
entry.
