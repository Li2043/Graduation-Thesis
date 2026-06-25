# V1 System Specification

> Status: **Formal V1 design specification.** This document defines the V1
> research system as an independent, clean redesign. It specifies *what* the
> system is and *how* it must be evaluated; it contains no implementation
> details and makes no empirical claims. Design decisions recorded here are
> subject to the governance process in `V1_DECISION_LOG.md`.

---

## 1. System Overview

The V1 system is a **fairness-aware multi-agent reinforcement learning (MARL)
system** for cooperative highway merging. It studies whether a fairness
objective grounded in Rawlsian maximin reasoning changes learned driving
behaviour relative to a purely self-interested objective.

The system defines exactly **two policies**, and no others:

1. **Egoistic DQN** — each controlled agent optimises its own individual return.
2. **Rawlsian DQN** — the objective is to maximise the expected experience of
   the worst-off controlled agent.

V1 is a **clean redesign**, not an extension or continuation of any earlier
work. It does not inherit prior code, parameters, reward formulations, or
results. All constructs used in V1 (the experience function, the metrics, the
seed protocol, and the evaluation procedure) are defined within this document
and its companion V1 documents, independently of any pre-existing material.

---

## 2. Environment Definition

### 2.1 Scenario

The environment is a **highway merging scenario** in which a main road and a
merging lane converge into a shared downstream segment. The scenario is chosen
because merging creates a natural conflict of interest between vehicles, making
it suitable for studying fairness between agents.

### 2.2 Controlled agents

The system contains **two controlled agents**:

- **Agent M (main road):** a vehicle travelling on the main carriageway.
- **Agent R (merging lane):** a vehicle attempting to merge from the on-ramp.

Both agents are learning agents driven by the policy under study (Egoistic or
Rawlsian). The two agents share a single policy specification within a given
condition; the fairness objective is defined over the set of controlled agents
\(\{M, R\}\).

### 2.3 Background traffic

All non-controlled vehicles follow a **fixed background policy** that is held
constant across both conditions and across training and evaluation. Background
traffic is part of the environment, not part of the agent set over which
fairness is computed, and its behaviour is never adapted to either policy.

### 2.4 Episode termination

An episode terminates when any of the following conditions occurs:

1. **Collision:** a controlled agent is involved in a collision.
2. **Completion:** all controlled agents have cleared the merge region and
   reached the downstream segment.
3. **Horizon:** a fixed maximum number of environment steps is reached
   (time-out / truncation).

Termination conditions are identical for both policies and for both the
training and evaluation phases.

---

## 3. Experience Function (Core Contribution)

The central construct of V1 is a per-agent **experience** \(E_i\) defined over
the controlled agents \(i \in \{M, R\}\). The experience function expresses, on
a single comparable scale, how well-off an agent is in terms of progress,
safety, and delay. It is defined here independently and is not derived from any
external reward formulation.

### 3.1 Components

For controlled agent \(i\) at the episode level (or as an episodic aggregate of
per-step quantities, to be fixed during calibration):

- **Mobility** \(m_i \in [0, 1]\): a normalised measure of forward progress
  toward the agent's intended downstream goal (higher is better).
- **Safety** \(s_i \in [0, 1]\): a normalised measure of safety margin based on
  **time-to-collision (TTC)** with respect to the nearest relevant vehicle,
  with a clearly specified proxy permitted only where TTC is undefined (e.g.
  diverging trajectories). Higher \(s_i\) denotes a safer situation.
- **Waiting time** \(t_i \ge 0\): a normalised measure of accumulated delay
  attributable to the agent (e.g. time spent unable to progress or yielding at
  the merge). Higher \(t_i\) denotes worse outcomes and therefore enters the
  experience with a negative sign.

Each component is normalised to a common scale so that the weighted combination
below is dimensionally consistent. Exact normalisation constants are fixed
during the calibration phase and recorded in `V1_DECISION_LOG.md`.

### 3.2 Final formula

The per-agent experience is the weighted combination

\[
E_i = w_1 \cdot \text{mobility}_i + w_2 \cdot \text{safety}_i - w_3 \cdot \text{waiting\_time}_i,
\]

with non-negative weights \(w_1, w_2, w_3 \ge 0\) that are fixed before the
frozen evaluation phase. Higher \(E_i\) denotes a better-off agent. The sign
convention guarantees that improved mobility and safety raise experience while
increased waiting lowers it.

### 3.3 Least-advantaged agent

The **least-advantaged agent** at a given decision point is the controlled
agent with the lowest experience:

\[
i^{\*} = \arg\min_{i \in \{M, R\}} E_i .
\]

The quantity \(\min_i E_i = E_{i^{\*}}\) is the worst-off experience and is the
basis of the Rawlsian objective (Section 4). Ties are resolved by a fixed,
documented rule.

---

## 4. Policy Definitions

Both policies operate over the same environment, observation space, action
space, and learning algorithm class. The **only** intended difference between
conditions is the objective each policy optimises.

### 4.1 Egoistic policy

Each controlled agent maximises its own individual expected return:

\[
\max \; \mathbb{E}\!\left[\sum_{t} \gamma^{t}\, r_{i,t}\right],
\]

where \(r_{i,t}\) is agent \(i\)'s individual per-step return signal. The
egoistic policy contains no term that references other agents' experiences.

### 4.2 Rawlsian policy

The Rawlsian policy maximises the **expected minimum experience** across the
controlled agents:

\[
\max \; \mathbb{E}\!\left[\min_{i \in \{M, R\}} E_i\right].
\]

This operationalises the Rawlsian maximin principle: improvements are valued
according to their effect on the worst-off agent rather than on aggregate or
individual returns.

---

## 5. Evaluation Protocol

### 5.1 Train/evaluation separation

Evaluation is **strictly separated** from training. The set of evaluation
scenarios is held out and never used to train, select, calibrate, or tune any
policy or parameter. No information from the evaluation set may influence any
design choice.

### 5.2 Fixed evaluation seeds

A fixed set of **evaluation seeds** defines the held-out evaluation scenarios.
This set is:

- defined and frozen before the frozen evaluation phase;
- **disjoint** from all training seeds and from any calibration seeds;
- **identical** across the Egoistic and Rawlsian conditions, so that both
  policies are assessed on exactly the same scenarios.

### 5.3 No leakage

There must be no leakage between training and evaluation. In particular,
evaluation scenarios are not reused as training initial conditions, evaluation
results are not consulted when choosing weights or hyperparameters, and no
quantity optimised during training is used as an evaluation outcome
(see Section 9).

### 5.4 Metric groups

Evaluation metrics are reported in three explicit groups:

- **Fairness metrics**
  - Minimum experience \(\min_i E_i\) (worst-off agent).
  - Gini coefficient of the experience distribution across controlled agents.
- **Safety metrics**
  - Collision rate (fraction of episodes containing a collision involving a
    controlled agent).
  - Near-collision rate (fraction of episodes/steps in which a controlled agent
    falls below a defined safety / TTC threshold without colliding).
- **Efficiency metrics**
  - Mean experience across controlled agents.
  - Time-to-merge (steps required for the merging agent to clear the merge
    region; reported only for completed episodes).

The fairness group is primary for RQ1; safety and efficiency jointly
characterise the trade-off in RQ2. Diagnostic quantities used to explain
behaviour (RQ3) are reported separately and are not used to declare success.

---

## 6. Research Questions

### RQ1 — Fairness improvement
Does the Rawlsian policy improve the experience of the least-advantaged
controlled agent relative to the egoistic policy, as measured by the fairness
metric group on the held-out evaluation set?

### RQ2 — Trade-off structure
What is the structure of the trade-off between safety and efficiency induced by
the Rawlsian objective relative to the egoistic objective, characterised jointly
by the safety and efficiency metric groups?

### RQ3 — Failure modes
Under what conditions does the Rawlsian objective fail to improve, or degrade,
the worst-off agent's experience, and what behavioural mechanisms explain such
failures?

---

## 7. Experimental Design

The experiment comprises exactly **two conditions**:

1. **Egoistic** — controlled agents trained under the egoistic objective.
2. **Rawlsian** — controlled agents trained under the Rawlsian maximin objective.

All other factors — environment, background-traffic policy, observation and
action spaces, learning algorithm and capacity, training budget, evaluation
seeds, and metric definitions — are held identical across the two conditions.
The presence or absence of the Rawlsian maximin objective is the single
manipulated factor. **No additional conditions, extensions, sweeps, or variants
are included in V1.**

---

## 8. Out of Scope

The following are explicitly excluded from the V1 system and must not be
introduced without a formal scope change:

- **Human-in-the-loop** feedback, preference elicitation, or participant studies.
- **RIPPER / symbolic rule induction** or any interpretable-norm extraction.
- **Automatic hyperparameter tuning** or automated search over design parameters.
- **Multi-agent communication learning** or any learned inter-agent messaging
  channel.

---

## 9. Scientific Constraints

V1 is bound by the following methodological constraints to protect the validity
of its conclusions:

- **No seed-selection bias.** Training and evaluation seeds are fixed in advance.
  Results are reported over the full predefined seed set; seeds are never chosen,
  dropped, or reordered on the basis of observed outcomes.
- **No metric hacking.** Metrics, their directions, and their groupings (fairness,
  safety, efficiency, diagnostic) are defined before the frozen evaluation phase
  and are not redefined to favour a particular conclusion.
- **No reward-based evaluation leakage.** The quantity an agent optimises during
  training is not used as an evaluation outcome, and a shaped or
  objective-specific training signal is never compared across conditions as if it
  were a neutral metric. Cross-condition comparisons use only condition-neutral
  evaluation metrics.
- **No best-only reporting.** Results are reported as full distributions or
  appropriate central tendency with dispersion across all seeds and evaluation
  scenarios. Single best runs, best seeds, or cherry-picked episodes are not
  presented as representative.
