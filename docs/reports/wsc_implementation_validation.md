# WSC Implementation and Validation Report

Implementation + validation only. **No formal WSC training runs were launched.** All new code is additive; the Original 18D path was verified unchanged after the change (TEST 13). All validation artifacts live under `F:\正式训练_seed_replication_v1\validation_artifacts\wsc\`, never under a formal checkpoint/output/Chapter-5 directory.

---

## 1. Source-of-truth formal pipeline

Confirmed (per the prior preflight audit, re-verified here): the real formal training entry point is

```
experiments/pilots/study_b_fairness_mappo/scripts/train_curriculum_stage_highwayenv.py
```

using `thesis.study_b.envs.highwayenv_wrapper.StudyBHeterogeneousHighwayEnv`. This file was **not modified** (hash `55cdde81...f0ad`, matches its pre-change state — never touched by any edit this session). `train_dqn_direct_welfare.py` / `thesis.study_b.heterogeneous_env` were **not** used as the implementation basis for anything in this task, per instructions.

Real checkpoint schema confirmed unchanged and used as the baseline for all expansion logic: `step, stage, scenario_ids, online, target, optimiser, update_count, replay_size`.

---

## 2. Files changed

| File | Change | Scientific role |
|---|---|---|
| `src/thesis/study_b/local_observation.py` | Additive: `SELF_OBS_DIM_WSC`/`NEIGHBOUR_OBS_DIM_WSC`/`LOCAL_OBS_DIM_WSC`/`WSC_NEUTRAL_M` constants; `VehicleSnapshot.welfare_state: float = 1.0` (new field, default preserves old callers); `include_welfare_state: bool = False` parameter threaded through `build_self_observation`/`build_neighbour_observations`/`build_local_observation`. Original 6/4/18 constants and behaviour untouched. | A. WSC observation support |
| `src/thesis/study_b/utility.py` | Additive: new function `running_active_attainment(trace) -> float` (M_i(t)), added to `__all__`. No existing function modified. | A. WSC observation support |
| `src/thesis/study_b/envs/highwayenv_wrapper.py` | Additive: `StudyBHighwayWrapperConfig.include_welfare_state: bool = False`; `_snapshot()` computes `welfare_state` from `self._traces[vid]` via `running_active_attainment` only when the flag is set; `_build_observations()` passes `include_welfare_state` through. Original code paths (flag=False) unaffected — verified by TEST 13. | A. WSC observation support |
| `src/thesis/study_b/shared_local_dqn.py` | Additive: `build_study_b_dqn_config(..., obs_dim: int = LOCAL_OBS_DIM)` — new parameter, default preserves all existing callers exactly. | B. 22D configuration support |
| `src/thesis/study_b/q_ensemble.py` | Additive: `load_ensemble_agents(..., obs_dim: int = LOCAL_OBS_DIM)`, threaded to `build_study_b_dqn_config`. Default preserves existing callers exactly (TEST 13). | B/D. 22D config + evaluator support |
| `src/thesis/study_b/wsc_checkpoint_expansion.py` | **New file.** Zero-column expansion of online/target state dicts; Adam optimizer moment expansion for `net.0.weight`'s slot; provenance metadata. Never writes to the source checkpoint. | C. Checkpoint/optimizer expansion |
| `experiments/pilots/study_b_fairness_mappo/scripts/train_curriculum_stage_highwayenv_wsc.py` | **New file**, sibling of the real formal script (which remains byte-unchanged). Mirrors its continuation protocol exactly except: WSC observation mode on, `obs_dim=22`, checkpoint loaded via `wsc_checkpoint_expansion`, additive provenance metadata in saved checkpoints/manifest. | A+B+C combined (new entry point) |
| `validation_artifacts/wsc/*.py` (7 scripts) | **New files**, validation/test code only. | E. Validation/test code |

**No file outside these categories was changed.** Explicitly re-verified by hash (see §16): `welfare_reward.py`, `stage11_welfare.py` (welfare formulas), `stage11_dyad_merge_pilot_config.py` (all schedules/hyperparameters), `highwayenv_merge.py` (target speeds, scenario/termination geometry), `scenario_generator.py`, `independent_dqn_v2.py` (QNetwork/DQNConfig), `stage10_shared_dqn.py`, `replay_buffer_v2.py`, and `train_curriculum_stage_highwayenv.py` itself are all **untouched**.

**Scope note**: this task modified only the `F:\正式训练_seed_replication_v1` source tree. `F:\正式训练`'s identical copies of the 5 modified files were confirmed byte-identical to their pre-change originals (untouched) — since every evaluation/training script in this project (including the new WSC script) resolves its Python imports through `F:\正式训练_seed_replication_v1\project\src` regardless of which bundle's checkpoints it loads, this is not a functional gap for this validation pass, but is noted for completeness before any future formal WSC run.

---

## 3. Final WSC definition

```
M_i(t) = (1/n) * sum_{k=1..n} u_i,k          if n >= 1 already-realised active samples
M_i(t) = 1.0                                  if no active samples yet (neutral init)
```

where `u_i,k = clip(v_i,k / v_i_target, 0, 1)` — the exact same `target_speed_attainment` construct the terminal utility pipeline already uses. Implemented as `thesis.study_b.utility.running_active_attainment(trace)`, reusing `trace.active_flags`/`trace.attainments()` unchanged (the same selection rule `episode_utilities` uses, just evaluated on a prefix of the trace instead of only at termination). Not a second/independent accumulator. Never zeroed by collision (collision simply stops the trace from growing further, same as normal completion). This is a **pre-decision, accumulated mobility-history summary** and may lag the newest physical observation by one policy-decision interval, by construction of the existing trace semantics (documented explicitly in the function's own docstring; not silently redefined away).

---

## 4. Observation layout

**Original 18D (unchanged):**

| Index | Feature |
|---|---|
| 0 | role |
| 1 | own speed |
| 2 | own target speed |
| 3 | own acceleration |
| 4 | distance to merge |
| 5 | previous action |
| 6-9 | neighbour-0: presence, Δdistance, Δspeed, lane relation |
| 10-13 | neighbour-1: same 4 fields |
| 14-17 | neighbour-2: same 4 fields |

**New WSC 22D:**

| Index | Feature |
|---|---|
| 0-5 | same as Original |
| 6 | **M_i(t)** |
| 7-11 | neighbour-0: presence, Δdistance, Δspeed, lane relation, **M_j(t)** |
| 12-16 | neighbour-1: same 5 fields |
| 17-21 | neighbour-2: same 5 fields |

`7 + 3*5 = 22`, confirmed programmatically (TEST 14). Absolute `M_i`/`M_j` values used, not `M_j - M_i` (per spec). Absent neighbour slot: `presence=0`, `M_j=0.0` (matches existing all-zero-row convention, verified TEST 6/9).

---

## 5. Checkpoint expansion

For **both** `online` and `target` independently (never `target_new = online_new`):

```
net.0.weight  (64,18) -> (64,22):  new[:, 0:18] = old;  new[:, 18:22] = 0
net.0.bias, net.2.weight, net.2.bias, net.4.weight, net.4.bias:  copied exactly, unchanged
```

Verified: max |ΔQ| = **3.58e-7** (online, TEST 2), **2.98e-7** (target, TEST 3) — both at float32 machine-precision noise level, not a modeling approximation. Greedy-action agreement: **100.00%** (12,200/12,200 comparisons, TEST 4). Source checkpoint file is never written to (`torch.load` only; `expand_checkpoint` returns a new in-memory dict).

---

## 6. Optimizer expansion

Adam state is keyed by **integer parameter index**, not name — verified empirically (not assumed): `online_state_dict.keys()` order is `['net.0.weight', 'net.0.bias', 'net.2.weight', 'net.2.bias', 'net.4.weight', 'net.4.bias']`, and `optimiser['param_groups'][0]['params'] == [0,1,2,3,4,5]`, so index `0` is confirmed to correspond to `net.0.weight`. Only that index's `exp_avg`/`exp_avg_sq` are expanded (same zero-padding rule as the network weights); `step` preserved exactly; every other parameter's optimizer state copied byte-for-byte; `param_groups` (lr, betas, eps, weight_decay, amsgrad) preserved exactly. No AMSGrad/extra tensors were present in the real checkpoint (generic handling code exists and would warn if any appeared, but none did). **TEST 5: PASS**, all sub-checks true (first-18-columns-equal, new-columns-zero, step-preserved, other-params-byte-identical, param-groups-preserved).

---

## 7. Replay-buffer handling

WSC continuation constructs a **fresh, empty 22D `ReplayBuffer`** (`obs_dim=22`), exactly the way `SharedDQNLearner.__init__` already constructs a fresh buffer for every Original branch at the C64→continuation boundary — confirmed in the preflight audit that the real formal protocol **never** persists or restores replay contents across this boundary for any of the 48 existing runs. This is therefore not a deviation but a continuation of the existing protocol; no padding/mixing of 18D and 22D transitions was implemented or needed.

---

## 8. Reward identity

`welfare_reward.py` is **byte-unchanged** (hash `d012d2ab...a6bbc`, never edited). TEST 15 additionally verifies numerically: for a test utility vector `{0.9, 0.4, 0.95, 0.6}` and λ=0.5, `terminal_welfare_bonus` returns exactly the closed-form Mean/GGI/Maximin values computed independently in the test (`Mean=-0.14375`, `GGI=-0.1925` using ascending sort + weights `(0.4,0.3,0.2,0.1)`, `Maximin=-0.3`), and λ=0 gives exactly 0.0 (Baseline case). **TEST 15: PASS.** WSC has zero code path that touches reward computation — it only adds observation features.

---

## 9. Decentralization properties

Confirmed unaffected by the implementation as built:
- **No centralized selector**: `SharedLocalDQNAgent.select_actions` still calls `select_action` independently per vehicle on that vehicle's own (now 22D) observation.
- **No centralized critic**: this is a value-only DQN architecture; `build_global_state`/`GLOBAL_STATE_DIM` (MAPPO-only, never used by the DQN path) were not touched.
- **No global observation vector**: each vehicle's 22D observation is still built from its own self-block + up to 3 neighbours' restricted relative fields — never a concatenation of all 4 vehicles' observations.
- **No Q-value sharing, no intended-action sharing**: `M_j(t)` is a mobility-satisfaction scalar, not a value estimate or action signal.
- **Exactly one M scalar per visible neighbour slot**, attached to the identical slot/vehicle as the other 4 neighbour fields for that slot (TEST 9, including an observed rank-swap case).

---

## 10. Test results

| Test | Result | Evidence | Pass/Fail |
|---|---|---|---|
| TEST 1 — real checkpoint structure (12 seeds) | keys/shapes correct for all 12 real C64 checkpoints | `test1_checkpoint_structure.json` | PASS |
| TEST 2 — online zero-expansion | max\|ΔQ\| = 3.58e-7, n=12,200 comparisons | `wsc_validation_results.json` | PASS |
| TEST 3 — target zero-expansion | max\|ΔQ\| = 2.98e-7 | same | PASS |
| TEST 4 — greedy action identity | 12,200/12,200 = 100.00% | same | PASS |
| TEST 5 — optimizer state expansion | all sub-checks true (see §6) | same | PASS |
| TEST 6 — WSC range | min=0.2385, max=1.0 (present); absent slots all exactly 0 | same | PASS |
| TEST 7 — trace consistency | 0/288 mismatches; empty-history=1.0 confirmed | same | PASS |
| TEST 8 — no future leakage | 108 step-checks, 0 leaks; step-order evidence saved | `test8_step_order_evidence.json` | PASS |
| TEST 9 — neighbour slot alignment | 240 checks, 0 mismatches; **rank swap observed and correctly tracked** | `wsc_validation_results_part2.json` | PASS |
| TEST 10 — completion/collision/timeout | success + collision cases captured and verified (no NaN, values bounded); **timeout case not reproduced within 150 random-action attempts** | `test10_lifecycle_cases.json` | **PARTIAL** (see §16) |
| TEST 11 — WSC smoke continuation | Baseline+WSC (3,000 steps) and GGI+WSC (3,000 steps) both completed, checkpoints saved, no shape/runtime errors | `smoke_baseline/`, `smoke_ggi/` run logs | PASS |
| TEST 12 — WSC ensemble evaluation | 4 throwaway 22D checkpoints loaded, averaged, argmax, 5 H1 scenarios run, all Q/utility values finite | `test12_result.json` | PASS |
| TEST 13 — Original 18D regression | 5/5 fixed H1 scenarios byte-for-byte identical (actions, U, C, Gini, range) vs. Phase-0 reference | `wsc_prechange_reference/`, part2 log | PASS |
| TEST 14 — parameter count | 5,571 → 5,827, diff=256, +4.5952% | same | PASS |
| TEST 15 — reward identity | exact closed-form match for Mean/GGI/Maximin/Baseline | same | PASS |
| TEST 16 — observation-only diff audit | all changes classified A-F; no unexpected file touched (hash-verified) | §2, §16 | PASS |

---

## 11. Q-equivalence results

- Online: max\|ΔQ\| = **3.5762786865234375e-07** (12,200 (observation × M-vector) comparisons across 5 scenarios × up to 4 vehicles × ~50 steps × 10 M-vectors)
- Target: max\|ΔQ\| = **2.980232238769531e-07**
- Greedy-action agreement: **12,200/12,200 = 100.000%**

Both differences are at float32 rounding-noise level (≈1e-7), consistent with exact zero-padding rather than any approximation.

---

## 12. Optimizer-equivalence results

- `net.0.weight` optimizer slot resolved to **index 0** (verified against real `param_groups[0]['params'] = [0,1,2,3,4,5]` and `online_state_dict` key order).
- `exp_avg`/`exp_avg_sq`: first 18 columns bit-for-bit equal to source; new 4 columns exactly `0.0`; shapes `(64,22)` confirmed.
- `step` scalar: preserved exactly (`torch.equal` true).
- Every other parameter's optimizer state (indices 1-5): max abs difference = **0.0** (byte-identical).
- `param_groups` (lr, betas, eps, weight_decay, amsgrad): preserved exactly (`==` true).

---

## 13. WSC feature validation

- Min M observed (present slots, random-action rollout): **0.2385**; max: **1.0000**. All values in [0,1] as required.
- Empty-history behavior: confirmed **M=1.0 exactly** at episode reset, before any active sample exists (TEST 7).
- Slot consistency: 240/240 checks correct including a naturally-occurring neighbour rank swap (TEST 9) — M_j always followed the reassigned slot, never a stale physical identity.
- Completion behavior: a completed vehicle's trace stops growing (existing `_completed`/`active` gate, untouched); M freezes automatically as a consequence, verified via TEST 8/10's evidence (no post-completion mutation observed).
- Leakage check (TEST 8): 108 step-level checks; the newest trace sample appended each step was, in every case, equal to the vehicle's PRE-physics speed (the state already visible before that step's action was chosen) — zero leaks found. Full step-order evidence sample saved to `test8_step_order_evidence.json`.

---

## 14. Original-path regression

Phase-0 reference captured **before** any code change (seed 900101, Baseline/taskonly ensemble, 5 fixed H1 scenarios: `H1_00000`-`H1_00004`), then re-run **after** all WSC code changes, through the identical (default `include_welfare_state=False`) 18D path. Compared: term_reason, action trace (first 5 actions/episode), per-vehicle U and C, U_mean, U_min, Utility Gini, utility range. **Result: all 5 scenarios identical in every field.** No unexplained Original-path behavioral difference was found.

---

## 15. Smoke-run results (technical only — no fairness interpretation)

- **Baseline+WSC**: 1,200,000 → 1,203,000 steps (3,000 steps), checkpoints saved every 1,000 steps, no shape/runtime errors, optimizer updates and target syncs executed, replay buffer accepted 22D transitions throughout. Elapsed 145.6s.
- **GGI+WSC**: same budget, λ=0.5, terminal GGI welfare bonus path exercised at every episode boundary, no errors. Elapsed 137.3s.
- Both are explicitly **not** interpreted for fairness/performance content, per instructions — completion/collision numbers during these 3,000-step smoke windows are not reported as a scientific finding, only as evidence the technical pipeline (shapes, optimizer, replay, target sync, terminal welfare path, checkpoint save/load) runs end-to-end.

---

## 16. Remaining risks

- **+256 parameters / +4.60%** (5,571 → 5,827) — small, quantified, confined to first-layer input columns; cannot add nonlinear capacity beyond a linear function of 4 new scalars feeding 64 units. Disclosed, not eliminated.
- **Hand-defined communication message**: `M_i(t)`/`M_j(t)` are not learned end-to-end; only their *use* is learned by the network.
- **Historical one-step timing convention**: `M_i(t)` reflects the trace as of the start of the current policy step (pre-physics), so it may lag the very latest physical state by one decision interval — documented explicitly in `running_active_attainment`'s docstring, not hidden.
- **TEST 10 partial**: success and collision lifecycle cases were captured and verified clean (no NaN, values bounded, trace-freeze consistent with TEST 8's leak-free evidence); a timeout case was **not** reproduced within 150 random/BRAKE-biased-action attempts in this short validation run. Per instructions, this is reported explicitly rather than fabricated. This is not one of the items the pass/fail rule (§40 of the task) lists as required for GO, but it is an honest, unresolved test gap that should be closed (e.g. via an existing unit fixture, if one exists, or a longer/more targeted validation run) before formal launch.
- **Single-bundle scope**: only `F:\正式训练_seed_replication_v1`'s source tree was modified; `F:\正式训练`'s identical copy remains untouched (verified by hash) and would need the same additive patch mirrored before formally training WSC branches that load C64 checkpoints for seeds 900101-104/910101-102 through that bundle's own path, if any script ever does so (current scripts, including the new WSC one, resolve imports through the `_seed_replication_v1` tree regardless of checkpoint source, so this is currently inert).
- **No other code-path differences** were found between the Original and WSC training scripts beyond the four explicitly intended ones (WSC mode, obs_dim, checkpoint loading via expansion, optimizer-state expansion) — confirmed by writing the WSC script as a direct structural mirror of the real formal script and diffing the two by inspection during authoring.

---

## 17. Formal-run readiness

### GO

Justification: every critical test required by the task's pass/fail rule (§40) passed — Original 18D path unchanged (TEST 13), online/target zero-expansion equivalent (TEST 2/3), greedy actions 100% identical at initialization (TEST 4), Adam state correctly expanded (TEST 5), reward identity confirmed (TEST 15), no future leakage (TEST 8), WSC values bounded (TEST 6), neighbour slots correct (TEST 9), evaluator supports 22D (TEST 12), smoke continuation succeeded for both Baseline+WSC and GGI+WSC (TEST 11), and no formal output/checkpoint was modified (verified by hash throughout). The one open item (TEST 10's missing timeout case) is a reporting gap, not a failure of any implemented mechanism, and does not implicate any of the critical GO criteria.

**No formal WSC training runs were launched.** The 48-row future 4×2 matrix configuration (`wsc_future_4x2_matrix_dryrun.csv`) was generated as a planning table only — no training process was started for any row in it.
