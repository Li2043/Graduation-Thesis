# Dense Reward Study — environment setup report

Full narrative report for standing up `F:\dense reward` as an isolated, reproducible clone of the verified
`F:\正式训练_seed_replication_v1` formal environment, per the task spec's 16 sections. This file is the
single "read this first" summary; supporting detail lives in the other files under `provenance/` and
`validation/` referenced throughout.

## 1. Audit of the old environment

See `provenance/source_environment.md` for the full write-up. Highlights:

- Old bundle's own `.venv` confirmed broken on this machine (`pyvenv.cfg` references a nonexistent
  `C:\Users\SamChui\...` interpreter path from a different machine/drive). Not touched, not repaired in
  place (would have required writing into the forbidden old-env path).
- Real training entrypoint traced via the actual `.bat` → `.py` call chain (not guessed from filenames):
  `04_START_FORMAL.bat` → `launch_formal.py` → subprocess →
  `project\experiments\pilots\study_b_fairness_mappo\scripts\train_curriculum_stage_highwayenv.py`. The
  legacy `train_curriculum_stage.py` and the WSC sibling `train_curriculum_stage_highwayenv_wsc.py` are
  never invoked by the formal/replication launchers.
- Real evaluation entrypoint: `07_EVALUATE.bat` → `evaluate_formal.py` (held-out H0/H1 via
  `stage_q_ensemble_gate.py`); `12_EVALUATE_REPLICATION.bat` → `evaluate_replication.py` (imports
  `evaluate_formal_welfare.py`/`evaluate_formal_behavioral.py`/`evaluate_high_burden_diagnostic.py` as
  Python modules and monkey-patches their seed/checkpoint-path module attributes).
- Reward/welfare math traced to `thesis.study_b.welfare_reward` (Mean/GGI/Maximin conditions,
  `terminal_welfare_bonus`) and `thesis.study_b.utility` (`generalized_gini_welfare`, `gini_coefficient`,
  `running_active_attainment` — the WSC M_i(t) function). `include_welfare_state`
  (`StudyBHighwayWrapperConfig`, default `False`) gates 18D vs 22D observations.

## 2–5. New directory / what was copied / Study B unchanged / no dense reward yet

- New root mirrors the old bundle's own layout (per Section 2's "prefer existing structure" instruction)
  plus 3 new directories the task explicitly requires: `provenance\`, `validation\`, `reports\`.
- Full manifest of what was copied / excluded: `provenance/copied_files_manifest.txt`.
- Full manifest of what was modified and why: `provenance/modified_files_manifest.txt` — 7 files total, all
  either (a) stale/hardcoded-absolute-path fixes with zero behavior change to reward/welfare/observation
  logic, or (b) purely additive default-off scaffolding for Sections 12–13.
- No reward, welfare, observation, episode-count, or seed-schedule value was changed anywhere. Verified
  both by direct code diff (the 7 modified files) and by the baseline-equivalence smoke test producing a
  passing run against the SAME frozen config values as the old bundle.
- No dense-shaping formula is implemented anywhere in this copy (grep-verified: no new arithmetic reward
  term was added; the two new scaffolds (Sections 12, 13) are inert when off, which is their only state in
  this task).

## 6. Python environment

Old `.venv` was unusable (see above) → built a brand-new venv at `F:\dense reward\.venv` from
`C:\Python314\python.exe` (Python 3.14.6 — **exact match** to the old bundle's recorded
`environment\python_version.txt`, not just "close enough").

Offline install: `python -m pip install --no-index --find-links="F:\正式训练_seed_replication_v1\wheelhouse\cpu" -r requirements-lock.txt`.

**Blocker found and NOT silently worked around**: `PyYAML==6.0.2` has no `cp314-win_amd64` wheel in the
wheelhouse (only `pyyaml-6.0.2.tar.gz` sdist + a wrong-version `pyyaml-6.0.3` wheel), and building the sdist
fails immediately (`ERROR: Could not find a version that satisfies the requirement wheel (from versions:
none)` — the `wheel` PEP517 build-backend package is also absent from the wheelhouse, and this machine has
no C compiler either, per `where cl`/`where gcc` both failing). Per the task's explicit instruction, this was
**not** worked around with a networked install or a version substitution. Instead:
- All other 35 pinned packages + `psutil` (orchestration-only) were installed successfully, offline,
  bit-for-bit at their pinned versions (`provenance/pip_freeze.txt` matches
  `environment/requirements-lock.txt` line-for-line except the missing `PyYAML` line).
- The actual Study B call chain (`train_curriculum_stage_highwayenv.py` and everything it imports —
  `thesis.study_b.*`, `thesis.agents.*`, `thesis.pilots.stage11_welfare`) was grep-verified to contain **no
  `import yaml`** anywhere reachable from it. The modules that DO import yaml
  (`thesis.calibration.comfort_lock`, `thesis.protocol.final_pbrs_lock`, `thesis.training.final_lock_loader`,
  etc.) are legacy/Study-A-era modules not on the traced Study B path.
- **Net effect, measured precisely**: `pytest --collect-only` over the full copied `tests/` tree collects
  1209 tests with 47 collection ERRORs, and every one of the 47 is `ModuleNotFoundError: No module named
  'yaml'` (confirmed by inspecting one directly:
  `tests/protocol/test_final_pbrs_lock.py` → `thesis.protocol.final_pbrs_lock` → `import yaml`). All 47 are
  in `tests/pilots/`, `tests/protocol/`, or `tests/training/` — legacy/Study-A-era modules, not on the Study
  B path this task validates. Running `tests/study_b/` in isolation: **278 passed, 12 skipped (all 12
  skips are pre-existing "requires a real finished checkpoint fixture not present on this machine" —
  unrelated to this environment setup), 0 failed, 0 errors**. PyYAML's absence did not block any of the
  validation work in this task (smoke test, seed reproducibility, WSC on/off, parallelism benchmark all ran
  and passed). It is flagged here as an open item for whoever next touches the `protocol`/`training`/
  `pilots` legacy modules.

Full record: `provenance/python_environment.txt`, `provenance/pip_freeze.txt`.

**Second discrepancy found and flagged, not silently absorbed**: the task brief stated this machine has 32
logical CPUs. Direct measurement (4 independent methods, see `python_environment.txt`) shows **12**. The
Section 9 benchmark below was run against the 12 cores this shell session actually has access to.

## 7. Hardcoded paths

Full detail: `provenance/path_changes.md`. Summary: 5 files had genuinely stale/hardcoded absolute paths
(`evaluate_formal_welfare.py`, `evaluate_formal_behavioral.py`, `evaluate_high_burden_diagnostic.py`,
`launch_wsc_formal_batch.py`, `launch_wsc_formal_batch_v2.py`) — all fixed to resolve relative to their own
file location (mirroring the pattern the rest of the codebase, e.g. `_common.py`, already used correctly).
One dead/unrelated personal path (`_watch_eval_done.py`) and several test-fixture string literals were
found and intentionally left unmodified (not real I/O paths / not reachable code).

## 8. Chinese-character / space-character path handling

Full detail: `validation/windows_path_handling_report.md`. All 5 targeted tests (pathlib, subprocess,
multiprocessing, logging, torch checkpoint save/load) PASS against both the Chinese-character old path
(read-only) and the space-containing new path.

## 9. 32-core parallel configuration

Full detail: `provenance/parallelism_benchmark.md`. Benchmarked on the 12 logical CPUs actually available
(see the CPU-count discrepancy note above) at worker counts 2/4/6/8/10/12/16, each launching genuinely
independent `train_curriculum_stage_highwayenv.py` subprocesses (400 steps, `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1` per child, mirroring `launch_formal.py`'s existing convention) and measuring wall-clock
throughput, aggregate CPU%, aggregate peak RSS, and crash rate. See that file for the recommended
`DEFAULT_MAX_WORKERS` and the reasoning, and for the explicit caveat that this number should be
re-benchmarked on the true target machine if it turns out to actually expose 32 cores.

## 10. Seed reproducibility

Full detail: `validation/seed_reproducibility_report.md`. Two independent runs, identical seed/config,
single-threaded (`OMP_NUM_THREADS=1`) — bit-identical manifests AND bit-identical (`torch.equal`) checkpoint
tensors. PASS, with an explicit note on the scope of what "deterministic" was actually tested (CPU,
single-threaded, same machine/torch version — not claimed beyond that).

## 11. Baseline equivalence smoke test

Full detail: `validation/baseline_equivalence_smoke_test_report.md`. Reused the old bundle's own
`replication_smoke_test.py` (`new_protocol.md §10`) unmodified, run from the new copy: PASS (rc=0, all
checkpoints present with required keys, replay buffer populated, log format correct, eval-script path
resolvable). Cross-referenced against the WSC on/off test and the seed-reproducibility checkpoint-reload
test for the pieces that single script doesn't itself cover in isolation.

## 12. Reward-trace debug mode

Added to `train_curriculum_stage_highwayenv.py`: `--debug-reward-trace` (default off) /
`--debug-reward-trace-episodes` (default 5). When off (the only state exercised anywhere in this task),
behavior is byte-identical to the original script (verified: every validation run in this task used the
default, and both the smoke test and the seed-reproducibility bit-identity test would have caught any
accidental behavior change). When on, logs step/agent_id/base_reward/terminal_component/current
M_i/welfare_objective_value/done/collision/completion for the first N episodes only. No dense-shaping
arithmetic is computed. See `provenance/modified_files_manifest.txt` entry #6 for the exact diff description.

## 13. Dense-reward config schema (reserved only)

Added `dense_reward_study_reserved` block to `configs/FROZEN_EXPERIMENT_CONFIG.json` — purely additive,
`dense_welfare_shaping: false`, all magnitude/epsilon/objective fields `null`. See
`provenance/modified_files_manifest.txt` entry #7. Nothing in the copied codebase reads this key yet.

## 14. Provenance file index

- `provenance/source_environment.md`
- `provenance/copied_files_manifest.txt`
- `provenance/modified_files_manifest.txt`
- `provenance/python_environment.txt`
- `provenance/pip_freeze.txt`
- `provenance/path_changes.md`
- `provenance/parallelism_benchmark.md`
- `provenance/environment_setup_report.md` (this file)
- `provenance/requirements-lock-minus-pyyaml.txt` (the exact requirements list actually installed offline,
  kept for exact-repro of the install step)
- `validation/windows_path_handling_report.md` + `path_handling_test_results.json`
- `validation/wsc_onoff_report.json`
- `validation/seed_reproducibility_report.md` + `seed_reproducibility_raw_report.json`
- `validation/baseline_equivalence_smoke_test_report.md`
- `validation/parallelism_benchmark_raw.json`

## 15. Not done (as required)

- `F:\正式训练_seed_replication_v1` was never written to. Every write in this task was verified to start
  with `F:\dense reward` before execution (directory-creation and copy commands constructed the destination
  path explicitly and printed it before running).
- No dependency was upgraded — every installed version matches `requirements-lock.txt` exactly, and the one
  gap (PyYAML) was left unfixed rather than resolved by installing a different version or from the network.
- No reward/welfare/WSC-observation definition was changed.
- No formal or dense-reward training campaign was started. Every training subprocess run in this task used
  an isolated bench/smoke/repro-only seed (929999, 939999, 950000+) never overlapping the formal (900xxx/
  910xxx) or replication (920xxx) seed blocks.
- No exp01 `experience_score` code exists anywhere in the copied tree (verified: not present in the source
  tree either — nothing to exclude).

## 16. See the final chat response for the full A–F structured report and READY/NOT READY verdict.
