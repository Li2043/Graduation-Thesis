# WSC Corrected-Mapping Validation and Re-Freeze Report

**This report covers the post-incident repair, validation, scientific re-freeze, and relaunch preparation for the formal WSC (Welfare-State Communication) campaign. The new 48-run campaign has been prepared but NOT launched. See Section M.**

---

## A. Executive summary

**GO** — the bug is understood, fixed, independently re-verified by two methods, covered by a new permanent regression test, and does not require any change to a scientific constant (reward, lambda, GGI weights, seeds, checkpoints, schedules). A new scientific freeze (v2) has been issued and a new, non-overlapping 48-run campaign matrix has been generated and dry-run tested. **The campaign has not been launched and awaits explicit user approval.**

Summary of what happened: the first formal WSC campaign (48 runs) collapsed to near-random, collision-dominated behavior across all 4 reward conditions, starting within the first ~200,000 of 800,000 continuation steps. Two hypotheses (Adam optimizer step-count inheritance; small Q-value decision-margin fragility, with an associated gradient-warmup fix) were investigated and refuted in turn. The actual, confirmed root cause is a data-alignment bug: `wsc_checkpoint_expansion.py` assumed the WSC 22D observation appends its 4 new features as a contiguous suffix, when the real layout interleaves them, so every original network weight from column 6 onward was fed the wrong input feature from the first forward pass. This has been fixed, and the fix has been verified two independent ways (exact Q-equivalence on real matched observations; a healthy 50,000-step short continuation with no gradient tricks).

---

## B. Final root cause, with evidence, and refuted/superseded earlier hypotheses

**FINAL ROOT CAUSE (confirmed): incorrect semantic column mapping in the 18D→22D checkpoint expansion.**

`local_observation.build_local_observation`'s WSC layout interleaves the 4 new welfare-state features (M_i after the self block, each M_j at the end of its own neighbour block) rather than appending them as a suffix. The pre-incident `wsc_checkpoint_expansion.py` assumed a simple prefix/suffix split (`new[:, :18] = old; new[:, 18:] = 0`), so every original weight column from index 6 onward was multiplying the WRONG physical quantity from the very first WSC forward pass — a pure data-corruption bug, unrelated to gradient magnitude, optimizer bookkeeping, or Q-value margins.

**Refuted / superseded hypotheses (preserved as audit trail, not deleted — see `wsc_formal_campaign_incident_diagnosis.md` for full detail):**

1. **Adam optimizer step-count inheritance** — tested via A/B (inherited step ~1.2M vs. reset to 1); both variants collapsed nearly identically within 5,000 steps. **REFUTED.**
2. **Small Q-value decision-margin fragility** — the C64 checkpoint's median decision margin is 0.0042 (91.7% of sampled observations below 0.01); hypothesized that any new trainable capacity would flip decisions. This was the working diagnosis that led to selecting and implementing a gradient-warmup fix. **REFUTED** by the failure of that fix (next item) — a pure gradient-magnitude mechanism predicts that scaling down the new columns' gradient should reduce/delay collapse, and it did not.
3. **Gradient-warmup fix** (`NewColumnGradientRamp`, ramping the 4 new columns' gradient 0→1 over N steps) — tested at N=20,000 and N=50,000; **both collapsed nearly identically to no ramp at all**, even at 10% gradient strength for most of the window. This result is what prompted re-examining the WSC layout directly and finding the actual bug.

**Confirmed root cause, with two independent pieces of evidence:**

- **Q-equivalence on real, matched observations** (`reverify_q_equivalence_fixed.py`): runs an Original-18D env and a WSC-22D env in lockstep on identical scenarios/actions (both driven by the same Original-ensemble policy) so both visit the same underlying states, then compares `Q_old(obs_18d)` vs. `Q_new(obs_22d)` for the SAME state. `n_compared=2208`, `max_abs_diff=4.77e-07`, greedy-action agreement `2208/2208 = 100.000%`.
- **Corrected short continuation, no gradient tricks** (`diagnostic_fixed_expansion_test.py`): 50,000-step Baseline+WSC (λ=0) continuation from the real seed-900101 C64 checkpoint, corrected mapping, no ramp installed at all (full strength from step 0) — healthy trajectory (Section G), closely matching the Original-18D control's shape.

This also means the ORIGINAL WSC implementation validation (`wsc_implementation_validation.md`, TEST 2–4) was itself invalid: it used synthetic `np.concatenate([old_obs, new_features])` vectors, which encode the SAME wrong prefix/suffix assumption as the buggy production code — the test and the bug agreed with each other by construction, not because the mapping was correct.

---

## C. Exact old-vs-corrected mapping table

| Old (18D) index | Feature | New (22D) index (CORRECTED) | New (22D) index (WRONG, pre-fix assumption) |
|---:|---|---:|---:|
| 0 | role | 0 | 0 |
| 1 | speed | 1 | 1 |
| 2 | target_speed | 2 | 2 |
| 3 | acceleration | 3 | 3 |
| 4 | dist_to_merge | 4 | 4 |
| 5 | prev_action | 5 | 5 |
| — | **M_i (new)** | **6** | 18 (wrong) |
| 6 | n0.presence | 7 | 6 (wrong) |
| 7 | n0.delta_d | 8 | 7 (wrong) |
| 8 | n0.delta_v | 9 | 8 (wrong) |
| 9 | n0.lane_relation | 10 | 9 (wrong) |
| — | **n0.M_j (new)** | **11** | 19 (wrong) |
| 10 | n1.presence | 12 | 10 (wrong) |
| 11 | n1.delta_d | 13 | 11 (wrong) |
| 12 | n1.delta_v | 14 | 12 (wrong) |
| 13 | n1.lane_relation | 15 | 13 (wrong) |
| — | **n1.M_j (new)** | **16** | 20 (wrong) |
| 14 | n2.presence | 17 | 14 (wrong) |
| 15 | n2.delta_d | 18 | 15 (wrong) |
| 16 | n2.delta_v | 19 | 16 (wrong) |
| 17 | n2.lane_relation | 20 | 17 (wrong) |
| — | **n2.M_j (new)** | **21** | 21 (coincidentally correct) |

```
OLD_TO_NEW_COLUMN_MAP = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20]
NEW_ONLY_COLUMNS      = [6, 11, 16, 21]
```

Derived programmatically in `wsc_checkpoint_expansion._build_column_map()` from `local_observation.py`'s own `SELF_OBS_DIM`/`NEIGHBOUR_OBS_DIM`/`NEIGHBOUR_SLOTS` constants (never hand-written a second time), with module-level `assert` self-tests pinning these exact values at import time.

---

## D. Checkpoint expansion validation

- **Online network**: `expand_state_dict(online_old)` — `net.0.weight`'s 18 old columns copied to `OLD_TO_NEW_COLUMN_MAP`, the 4 `NEW_ONLY_COLUMNS` left at `torch.zeros()` default (explicitly re-set to 0.0 for clarity). Every other tensor (`net.0.bias`, `net.2.*`, `net.4.*`) deep-copied unchanged. Verified by `test_expand_state_dict_copies_each_old_column_to_its_semantic_target` (sentinel values 1000+i per old column, checked at the correct target column; zero-columns checked exactly).
- **Target network**: `expand_state_dict(target_old)` called **independently** — `expand_checkpoint()` never derives `target_new` from `online_new`. Confirmed by direct code inspection (`wsc_checkpoint_expansion.py` line: `target_new = expand_state_dict(target_old)  # independent expansion, NOT target_new = online_new`).
- **Optimizer state**: `expand_optimizer_state()` locates `net.0.weight`'s integer parameter index by walking `online_state_dict`'s own key order (`nn.Module.state_dict()` ordering matches `model.parameters()` ordering for a plain `nn.Sequential` — confirmed index 0). Only that parameter's `exp_avg`/`exp_avg_sq` are expanded, using the SAME `OLD_TO_NEW_COLUMN_MAP`/`NEW_ONLY_COLUMNS` mapping. `step` and every other parameter's optimizer state are copied unchanged via `copy.deepcopy` — **no speculative reset of any optimizer state.** Any other same-shape tensor (e.g. AMSGrad's `max_exp_avg_sq`) is handled generically with an explicit warning, not silently dropped.
- **Q-equivalence / greedy-action agreement**: see Section B — `max_abs_diff=4.77e-07`, `100.000%` agreement over 2,208 real, matched-state comparisons (`reverify_q_equivalence_fixed.py`), independently reproduced in-repo by `test_end_to_end_zero_init_q_equivalence_on_real_observations` (random-weight network, `torch.testing.assert_close(atol=1e-5, rtol=1e-5)`, argmax equality).
- **Negative control**: `test_end_to_end_detects_a_reintroduced_prefix_suffix_bug` deliberately reintroduces the OLD, WRONG prefix/suffix expansion and confirms it does NOT reproduce `Q_old` on a real WSC observation — i.e. the test methodology is proven capable of catching this exact class of bug, not just vacuously passing.

---

## E. WSC observation validation

- **Self-state (M_i) correctness**: `build_self_observation(..., include_welfare_state=True)` appends `ego.welfare_state` as the 7th field (new index 6). Verified: `test_M_i_lands_exactly_at_index_6_not_appended_at_end`.
- **Neighbour-state (M_j) correctness and slot alignment**: `build_neighbour_observations` appends the SAME selected neighbour's `welfare_state` as the 5th field of that neighbour's row, using the identical `scored[:k]` selection/sort as the other 4 fields (never a separate ranking). Verified: `test_each_M_j_lands_at_end_of_its_own_neighbour_block_not_grouped_at_end` (checks M_j at indices 11/16/21 against the correct nearest-first-sorted neighbour).
- **Absent-neighbour masking**: a padded (presence=0) slot's M_j is `0.0` (matching the existing all-zero-row convention), not the `WSC_NEUTRAL_M=1.0` value used only for "no history yet" at the per-vehicle level. Verified: `test_absent_neighbour_slot_M_j_is_zero_not_arbitrary` (confirms no leaked value from a since-deactivated vehicle).
- **Temporal timing (pre-decision, causally safe)**: `highwayenv_wrapper._append_pre_step_trace_sample()` runs BEFORE `self._env.step(action_tuple)` (the physics step) inside `step()`. `_snapshot()`'s `welfare_state` reads `running_active_attainment(self._traces[vid])` — the trace as it stands after that pre-step append, i.e. only already-realized samples, never the consequence of an action not yet chosen. Confirmed by direct code inspection of `highwayenv_wrapper.py`.
- **No target-speed leakage**: `running_active_attainment` is built from `target_speed_attainment(speed, target_speed) = clip_01(speed/target_speed)` — a bounded [0,1] ratio, never the raw `target_speed` value itself. Confirmed by direct code inspection of `stage11_welfare.target_speed_attainment` and `utility.running_active_attainment`.
- **No terminal-utility / reward leakage**: `running_active_attainment` reuses the exact same `active_flags`/`attainments()` per-step construct as `episode_burdens` (a running, per-step quantity), NOT `episode_utilities`'s terminal `episode_mobility_outcome` (which has collision-zeroing and a different empty-history convention, 0.0 vs. 1.0). The terminal welfare bonus is computed separately, only at episode end, from `episode_utilities` — M_i/M_j at decision time cannot encode information about a reward not yet computed.
- **No Q-value/intended-action leakage**: unrelated code paths; not touched by this feature.
- **Observation range/normalization**: M_i/M_j ∈ [0, 1] (bounded ratio), consistent in scale with the project's other normalized observation fields (Δd, Δv are clipped to [-1,1]; role/lane_relation are small discrete values).
- **Baseline+WSC reward = task-reward-only equivalence**: Baseline uses `welfare_lambda=0.0`, so `terminal_welfare_bonus` contributes exactly 0 regardless of WSC; reward construction code is byte-identical to the Original script's call to the same `welfare_reward`/`utility` functions. Confirmed by direct comparison of `train_curriculum_stage_highwayenv_wsc.py`'s reward-construction block against `train_curriculum_stage_highwayenv.py` (both call `condition_by_name`/`terminal_welfare_bonus`/`episode_utilities` identically).
- **Mean/GGI/Maximin reward/lambda equivalence to Original**: same `condition_by_name`/`terminal_welfare_bonus` functions, same `lambda=0.5`, same GGI weights `[0.4,0.3,0.2,0.1]` — no WSC-specific reward-path branching exists anywhere in `welfare_reward.py` or `utility.py` (neither file was modified by the WSC feature or this bug fix).

**Fairness outcomes (U_min, Utility Gini, Fast-vs-Slow disadvantage, WSC interaction effects) were NOT computed, inspected, or used to tune or approve the mapping at any point in this validation, per explicit instruction.**

---

## F. Original 18D regression result

Full `tests/study_b/` suite (which exercises `local_observation.py`'s default `include_welfare_state=False` path, among everything else in Study B): **278 passed, 12 skipped, 0 failed.** The new WSC test file (9 tests) is included in this count. No pre-existing test was modified to make it pass. This confirms the Original 18D path is byte-identical in behavior to before the fix (as it must be, since the corrected `wsc_checkpoint_expansion.py` and the training script's warmup-default change do not touch `local_observation.py`, `utility.py`, `highwayenv_wrapper.py`, `welfare_reward.py`, or the Original training script at all — all confirmed unchanged by SHA256 in Section J).

---

## G. Corrected Baseline+WSC short continuation: trajectory and interpretation

50,000-step Baseline+WSC (λ=0) continuation, seed 900101, real C64 checkpoint, corrected `expand_checkpoint`, **no gradient ramp/hook installed at all** (full strength from step 0):

| Step (+N) | Completion | Collision |
|---:|---:|---:|
| 5,000  | 0.458 | 0.542 |
| 10,000 | 0.868 | 0.132 |
| 15,000 | 0.915 | 0.085 |
| 20,000 | 0.942 | 0.058 |
| 25,000 | 0.861 | 0.139 |
| 30,000 | 0.763 | 0.237 |
| 35,000 | 0.684 | 0.316 |
| 40,000 | 0.662 | 0.338 |
| 45,000 | 0.863 | 0.137 |
| 50,000 | 0.903 | 0.097 |

**Interpretation**: after an initial adjustment window (first 5,000 steps, expected — the network has just been expanded and is seeing 4 new input columns for the first time), completion recovers to 0.87–0.94 and stays there with the kind of window-to-window fluctuation (dipping to 0.66–0.76 around steps 30–40K before recovering to 0.86–0.90) that is normal for this training procedure — this is the same qualitative shape as the Original-18D control run (Section 5 of the incident diagnosis: 0.78→1.00→0.81 over the same budget). This is **not** the old failure pattern, which was an immediate, near-total, collision-dominated collapse (completion ~0.10–0.30) that never recovered through the full 800,000-step budget. No new stabilization mechanism was introduced merely because of the mid-run dip — per instruction, normal training fluctuation is not grounds for additional intervention.

---

## H. Warmup status: DISABLED

`--wsc-weight-warmup-steps` now defaults to `0` in `train_curriculum_stage_highwayenv_wsc.py` (previously a required argument with no default). `0` means the `NewColumnGradientRamp` hook is never installed — `net.0.weight` trains at full strength from step 0, for every column, identical to how the Original script trains every column. The new campaign's generated launch commands (`wsc_formal_launch_commands_v2.txt`) do not pass this flag at all, so every one of the 48 prepared runs uses the disabled default. The `wsc_gradient_warmup.py` module and its CLI flag remain in the codebase for possible future diagnostic use only, explicitly marked as not for formal use in both the module docstring and the v2 freeze manifest.

---

## I. Invalid-run registry: exclusion and preservation confirmed

All 48 rows of `F:\正式训练_seed_replication_v1\validation_artifacts\wsc_formal_launch\wsc_formal_run_registry.csv` have been updated:

| Original status | Count | New status | Disposition |
|---|---:|---|---|
| COMPLETED | 20 (seeds 900101–900104, 910101 × 4 conditions) | `INVALID_PRE_FIX_COLUMN_MAPPING` | Checkpoints/logs preserved on disk, unmodified. Must never be used in formal WSC evaluation, fairness comparison, or thesis tables/figures. |
| BLOCKED (partial) | 4 (seed 910102 × 4 conditions) | `INVALID_PRE_FIX_COLUMN_MAPPING` | Last valid (but invalid-mapping) checkpoints preserved (steps 1,750,000/1,750,000/1,700,000/1,600,000). **Must NOT be resumed.** |
| PENDING (never started) | 24 (seeds 920101–920106 × 4 conditions) | `INVALID_PRE_FIX_COLUMN_MAPPING` | No run was ever launched under these rows; superseded by the new v2 matrix. |

No row was deleted. Each row's `notes` field records the specific reason and preserves any prior note. The new campaign (Section K) writes exclusively to a new output root (`checkpoints\wsc_formal_runs_v2\`) and a new registry (`wsc_formal_run_registry_v2.csv`), so there is no path collision with any invalid pre-fix output.

---

## J. New freeze details

- **Version/tag**: v2 (supersedes v1; v1 files not overwritten).
- **Manifest**: `F:\正式训练_seed_replication_v1\validation_artifacts\wsc_formal_launch_v2\wsc_scientific_freeze_manifest_v2.json` (machine-readable) and `wsc_scientific_freeze_manifest_v2.md` (human-readable).
- **Changed files** (SHA256, v1 → v2): `wsc_checkpoint_expansion.py` (`8bb761dd...` → `f85494cf...`), `train_curriculum_stage_highwayenv_wsc.py` (`cd9bb51c...` → `1daaa707...`). New file: `tests/study_b/test_wsc_semantic_column_alignment.py` (`9a16bec5...`).
- **All 15 other listed scientific source files**: SHA256 unchanged from v1 (confirmed by direct re-hash this session).
- **Source checkpoints**: unchanged (spot-checked seeds 900101 and 920106 against `source_checkpoint_hashes.json`; both match exactly).
- **Reason string** (verbatim, as recorded in the manifest): *"Correction of WSC observation-to-network semantic column mapping bug discovered during the first formal campaign."*

---

## K. Prepared 48-run matrix

`F:\正式训练_seed_replication_v1\validation_artifacts\wsc_formal_launch_v2\wsc_formal_matrix_v2.csv` — 12 seeds × 4 conditions = 48 rows. Every row's `source_checkpoint` is that seed's **ORIGINAL** 1.2M C64 checkpoint (the same file used by the v1/invalid campaign — never modified); every row's `output_dir`/`checkpoint_dir` is under the **NEW** root `F:\正式训练_seed_replication_v1\checkpoints\wsc_formal_runs_v2\<run_id>\`.

| Seed group | Source checkpoint root |
|---|---|
| 900101–900104 | `F:\正式训练\checkpoints\formal_init\<seed>\C64_R50\seed_<seed>_C64_R50\ckpt_step_1200000.pt` |
| 910101, 910102 | `F:\正式训练\checkpoints\curriculum_910101_910102\<seed>\C64_R50\seed_<seed>_C64_R50\ckpt_step_1200000.pt` |
| 920101–920106 | `F:\正式训练_seed_replication_v1\checkpoints\seed_replication_v1\curriculum\<seed>\C64_R50\seed_<seed>_C64_R50\ckpt_step_1200000.pt` |

Each row's target: `checkpoints\wsc_formal_runs_v2\{condition}_wsc_{seed}\` (e.g. `baseline_wsc_900101`, `mean_wsc_900101`, ..., `maximin_wsc_920106`). `stage_name` uses a `_v2` suffix (e.g. `Formal_baseline_WSC_v2`) so v2 checkpoint metadata is unambiguous vs. any invalid v1 artifact if ever inspected side by side. `welfare_lambda`/`obs_dim`/`start_step`/`additional_steps`/`checkpoint_every` are identical to v1 (0.0/0.5 by condition, 22, 1,200,000, 800,000, 50,000).

---

## L. Launch readiness

- **Orchestrator**: `experiments/pilots/study_b_fairness_mappo/scripts/launch_wsc_formal_batch_v2.py` — same process-management logic as the original `launch_wsc_formal_batch.py` (preserved unmodified for audit), pointed at the new matrix/registry/commands-archive paths, with an added runtime assertion that every row's `output_dir` contains `wsc_formal_runs_v2` (refuses to run against a mismatched matrix).
- **Command construction**: identical non-scientific hyperparameters to every other formal run in this project (`--device cpu --replay-warmup 512 --eps-decay-steps-absolute 640000 --lr-decay-steps-absolute 800000 --action-representation meta_speed --local-sensing-range-m 50.0`), with `--wsc-weight-warmup-steps` **omitted** (Section H).
- **Dry-run tested this freeze**: `python launch_wsc_formal_batch_v2.py --dry-run --max-concurrent 4` → 48 unique run_ids, `wsc_formal_launch_commands_v2.txt` and `wsc_formal_run_registry_v2.csv` (all `PENDING`) written correctly. **No process was launched.**
- **Concurrency**: 4 (same as the v1 campaign, consistent with this project's own established `launch_formal.py` concurrency formula for this machine).
- **Persistence**: if/when launched, the same detached, session-independent process discipline used for the v1 campaign applies (PowerShell `Start-Process -WindowStyle Hidden -PassThru`, so the campaign survives beyond any single Claude Code session).
- **Logging/monitoring**: per-run stdout/stderr logs under `checkpoints\wsc_formal_runs_v2\_logs\`, registry CSV updated with PID/status/latest-checkpoint on the same polling discipline as v1.

---

## M. Final statement

**Formal WSC relaunch has NOT been started and requires explicit user approval.** Everything above (semantic mapping audit, permanent regression test, full `tests/study_b/` re-validation, invalid-run registry update, incident-report update, new v2 freeze, new 48-run matrix, new launcher — dry-run only) is prepared and ready. No training process for the new campaign has been launched. Please review this report and `wsc_formal_campaign_incident_diagnosis.md` (updated with the final root cause) before authorizing the launch of `launch_wsc_formal_batch_v2.py`.
