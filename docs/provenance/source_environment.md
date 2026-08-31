# Source environment audit — F:\正式训练_seed_replication_v1

Audited read-only on 2026-08-26. Nothing in this path was modified.

## Python / environment

- Recorded in `environment/python_version.txt`: Python 3.14.6, Windows-11-10.0.26200-SP0, CPython 3.14.6.
- `environment/pip_freeze.txt` and `environment/requirements-lock.txt` list the same 36 pinned packages
  (cloudpickle, colorama, contourpy, cycler, Farama-Notifications, filelock, fonttools, fsspec, gymnasium,
  highway-env, iniconfig, Jinja2, kiwisolver, MarkupSafe, matplotlib, mpmath, networkx, numpy, packaging,
  pandas, pettingzoo, pillow, pluggy, pygame-ce, Pygments, pyparsing, pytest, python-dateutil, pytz, PyYAML,
  setuptools, six, sympy, torch, typing_extensions, tzdata).
- `environment/requirements-orchestration.txt`: `psutil` only (explicitly documented as NOT part of the
  frozen scientific environment — orchestration/hardware-detection tooling only).
- The old bundle's own `.venv` is **confirmed broken on this machine**: `pyvenv.cfg` records
  `home = C:\Users\SamChui\AppData\Local\Programs\Python\Python314` (a different user/machine, drive `D:\`
  originally). `F:\正式训练_seed_replication_v1\.venv\Scripts\python.exe` was not executed/repaired (would
  have required writing into the forbidden path) — its brokenness was accepted at face value per the
  coordinator's prior confirmation.
- `torch==2.13.0` (CPU wheel: `wheelhouse\cpu\torch-2.13.0-cp314-cp314-win_amd64.whl`; a CUDA build
  `torch-2.13.0+cu126-cp314-cp314-win_amd64.whl` also exists under `wheelhouse\gpu\` but is not used —
  `scripts\verify_cuda.py` in the old bundle is a CPU-vs-GPU smoke test whose own recommendation text
  states the tiny (18→64→64→3) network is CPU-bound and recommends staying on CPU + process-level
  parallelism; the machine has no CUDA device in any case).

## Wheelhouse

`wheelhouse\cpu\` holds 35 of the 36 pinned packages as `cp314-win_amd64` wheels (or `py3-none-any` for
pure-Python packages) plus `psutil-7.2.2-cp37-abi3-win_amd64.whl`. **`PyYAML==6.0.2` has no matching wheel**
for this platform/interpreter in the wheelhouse — only `pyyaml-6.0.2.tar.gz` (sdist) and an
`pyyaml-6.0.3-cp314-cp314-win_amd64.whl` (wrong pinned version) are present. See
`provenance/environment_setup_report.md` for how this was handled (not silently substituted or
network-installed).

## Real call chain (traced by reading code / imports, not filenames)

- `04_START_FORMAL.bat` → `python scripts\launch_formal.py` → for each of the 18 formal Mean/GGI/Maximin
  runs, subprocess-launches
  `project\experiments\pilots\study_b_fairness_mappo\scripts\train_curriculum_stage_highwayenv.py`
  (never `train_curriculum_stage.py`, the legacy-backend sibling kept only for diagnostic parity, and never
  `train_curriculum_stage_highwayenv_wsc.py`, the WSC sibling used only by the separate WSC campaign).
- `11_START_WELFARE.bat` → `python scripts\launch_replication_welfare.py` → same
  `train_curriculum_stage_highwayenv.py` entrypoint, different seed/step budget (the 6×3 welfare fine-tune
  matrix, 1.2M→2.0M steps, λ_W=0.5).
- `08_SMOKE_REPLICATION.bat` → `python scripts\replication_smoke_test.py` → same entrypoint again, seed
  929999 only, 2000 steps, `--welfare-lambda 0.0`. This is the only genuinely tiny/technical (non-scientific)
  smoke definition already in the old bundle; the dense-reward baseline-equivalence smoke test (Section 11)
  reuses this script unmodified rather than inventing a new definition.
- `07_EVALUATE.bat` → `scripts\evaluate_formal.py` → per completed run, subprocess to
  `study_b_fairness_mappo\scripts\stage_q_ensemble_gate.py` for held-out H0/H1 evaluation.
- `12_EVALUATE_REPLICATION.bat` → `scripts\evaluate_replication.py` (--kind all) → `analyze_replication.py`.
  `evaluate_replication.py` imports (as Python modules, not subprocesses)
  `evaluate_formal_welfare.py` / `evaluate_formal_behavioral.py` / `evaluate_high_burden_diagnostic.py` from
  `study_b_fairness_mappo\scripts\` and monkey-patches their `SEEDS`/`BANK_ROOT`/`CKPT_ROOT`/
  `checkpoint_paths_for` module attributes for the replication seed set. It does NOT patch their `OUT_ROOT`
  module attribute (unused by the patched code path) or their `sys.path.insert` side effects at import time.

## Reward / welfare implementation (traced, not guessed)

- `thesis.study_b.welfare_reward` (`project\src\thesis\study_b\welfare_reward.py`): `MEAN`/`GGI`/`MAXIMIN`
  `WelfareCondition`s wrapping `thesis.study_b.utility.mean_welfare` / `generalized_gini_welfare` /
  `min_welfare`. `terminal_welfare_bonus(condition, episode_utilities, lam)` computes
  `R_c^W = lam * (W_c(U) - 1)`, called exactly once per finished episode and added identically to every
  active agent's terminal task reward (not per-agent-recomputed — see that file's own docstring and
  `tests/study_b/test_welfare_reward.py::test_bonus_added_once_per_agent_not_multiplied`).
- `thesis.study_b.utility` (`project\src\thesis\study_b\utility.py`): `GGI_WEIGHTS_N4 = (0.4, 0.3, 0.2, 0.1)`
  (frozen, worst-off-first ascending), `gini_coefficient`, `generalized_gini_welfare`, `coordination_burden`,
  `episode_utilities`/`episode_burdens` (per-vehicle terminal U_i/C_i from `EpisodeVehicleTrace`), and
  `running_active_attainment(trace)` — the WSC pre-decision mobility-history summary M_i(t): mean of already
  realised ACTIVE attainment samples so far, returning 1.0 (neutral) if none yet recorded. Reuses
  `thesis.pilots.stage11_welfare` for `mean_welfare`/`min_welfare`/`target_speed_attainment`/
  `stakeholder_experience`/`episode_mobility_outcome` rather than reimplementing them.
- `include_welfare_state` (`thesis.study_b.envs.highwayenv_wrapper.StudyBHighwayWrapperConfig`, default
  `False`): when `True`, `_snapshot()` computes `running_active_attainment(self._traces[vid])` instead of the
  constant `1.0`, and `_build_observations()` passes `include_welfare_state=True` through to
  `build_local_observation`, producing 22D observations (`LOCAL_OBS_DIM_WSC`) instead of the Original 18D
  (`LOCAL_OBS_DIM`). The formal (non-WSC) training/eval scripts never set this to `True`; only
  `train_curriculum_stage_highwayenv_wsc.py` does, and only that sibling script's checkpoints are loaded
  through `wsc_checkpoint_expansion.expand_checkpoint`'s zero-column expansion.
- Task/terminal/collision reward and completion handling live inside
  `thesis.study_b.envs.highwayenv_merge.ThesisHighwayMergeEnv` / `highwayenv_wrapper.py`'s `step()`, which
  returns `per_vehicle_reward` (the base task reward per agent, unmodified by this copy), `collision_event`,
  `completed_this_step` (`exit_event`), and `term_reason` ∈ {collision, success, truncation, ongoing}.

## Seeds

- Formal seed block (frozen, `configs/FROZEN_EXPERIMENT_CONFIG.json`): 900101, 900102, 900103, 900104,
  910101, 910102 — same 6 seeds for all 3 conditions (18 runs total), decided before outcome; 900103 kept
  despite a C64_R50 qualification FAIL, per the "no seed exclusion based on outcome" rule.
- Replication seeds (`scripts/replication_common.py`): 920101–920106; smoke seed 929999 (isolated, never a
  formal/replication seed).
- Seed wiring: `--master-seed` → `SharedLocalDQNAgent(dqn_config, seed=master_seed)` →
  `SharedDQNLearner.__init__` sets `self._rng = np.random.default_rng(seed)` (action epsilon-greedy sampling)
  and `torch.manual_seed(seed)` (network init / any torch-level randomness). Independently,
  `scenario_rng = np.random.default_rng(master_seed * 7919 + 1)` controls which scenario is drawn each
  episode. `env.reset(seed=0, scenario=...)` always passes a constant `seed=0` when an explicit `scenario`
  object is supplied (the scenario content itself, not the wrapper's internal RNG, is what varies with
  `master_seed` via `scenario_rng`).

## Parallelism (existing, validated prior)

- `scripts/launch_formal.py::compute_safe_concurrency`: `by_cpu = max(1, cpu_count() - 2)`, capped by an
  estimated-RAM budget (400MB/process placeholder) and by 18 (total run count), overridable via
  `--max-concurrent`. Every subprocess launched via `run_subprocess(..., env_overrides={"OMP_NUM_THREADS": "1"})`.
  `launch_replication_welfare.py` does the same (`env["OMP_NUM_THREADS"] = "1"`), default `--max-concurrent 18`.
  No code in the traced training path calls `torch.set_num_threads()` explicitly — thread-limiting is done
  entirely via the `OMP_NUM_THREADS` environment variable set by the launcher, one level up from the
  training script itself.
