# E34 — Study B HighwayEnv backend migration

Isolated per user instruction (2026-08-15): this is a **separate physical
copy**, not a junction, of the Study B code that was living under
`E33_stage11_n4_extended_convergence/`. It exists so the HighwayEnv
migration (per `Claude_Code_Autonomous_Experiment_Runbook_HighwayEnv.md`,
copied into this folder's `RUNBOOK.md`) can proceed without touching or
risking E33's existing custom-simulator diagnostic work (Diagnostics 1-6,
the target-sync fix, the curriculum infrastructure, all of which stay
frozen and citable as `legacy_custom_backend` history).

## What's here vs. what's excluded

Copied from E33 (2026-08-15 snapshot): `src/`, `tests/`,
`experiments/pilots/study_b_fairness_mappo/{scripts,configs,scenario_banks,
README.md,STATUS_SUMMARY_2026-08-13.md,unit_test_report.txt}`,
`pytest.ini`, `requirements-stage*.txt`.

**Not copied** (too large / not needed for the new backend): old training
checkpoints (`checkpoints/`, `*.pt`), per-step trajectory logs (`*.jsonl`),
`logs/`, `.pytest_cache/`, `__pycache__/`. If you need to diff against a
specific old checkpoint or trajectory, go back to
`E33_stage11_n4_extended_convergence/`.

`.venv/` is a **Windows junction** back to
`E33_stage11_n4_extended_convergence/.venv` (not a copy — saves several GB;
verified working from this path: `highway_env==1.12.0`, `torch==2.13.0+cpu`,
Python 3.14.6). Both folders share the same interpreter/site-packages;
installing a new package from either location affects both. If E34's
migration work ever needs a package version E33 doesn't have (should not
happen — HighwayEnv 1.12.0 was already pinned in E33 before this folder was
created), break the junction and give E34 its own venv rather than
upgrading shared packages in place.

## Scope

Everything under `output/highwayenv_migration/` and
`output/autonomous_highwayenv/` in this folder is **new work**, starting
from the M0 snapshot. The `legacy_custom_backend`
(`src/thesis/envs/stage10_symmetric_merge_env.py`,
`src/thesis/study_b/heterogeneous_env.py`) is carried over **unmodified**,
kept only for diagnostic/parity comparison per the runbook's explicit
instruction not to delete it.

Verified immediately after setup: `pytest tests/study_b/ -q` → 193 passed,
2 skipped (both skips require an old checkpoint file that was
intentionally excluded from this copy — not a regression; E33 shows
195 passed with those checkpoints present).
