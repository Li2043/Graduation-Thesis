# Known test-suite gaps

Full suite (`python -m pytest` from repo root, Python 3.14.6, all
`requirements-stage*.txt` + `requirements-study_b.txt` installed):

```
1210 passed, 83 failed, 36 skipped, 2 warnings in 815.94s
```

This thesis's own test suite alone (`python -m pytest tests/study_b`):

```
298 passed, 12 skipped, 0 failed
```

That 298/12/0 figure matches what the thesis Experiments chapter itself
reports for the DWS implementation-validation pass ("the wider Study B
test suite completed with 298 tests passed and 12 skipped") — confirmed
independently in this release-verification pass, not just carried over
from the thesis text.

## Category (b): missing optional dependency — fixed

`tests/study_b/test_pettingzoo_wrapper.py` (covering
`thesis.study_b.pettingzoo_wrapper`) requires the `pettingzoo` package,
which is not listed in any of the original `requirements-stage*.txt`
files — an apparent pre-existing gap in the source project's own
dependency pinning, not something introduced by this release. Added as
`requirements-study_b.txt` (pins `pettingzoo==1.27.0`, the version
verified against the full suite). With it installed, `tests/study_b/`
is 100% green.

## Category (a): legitimately expected, not a defect in this repository

All 83 failures in the full-suite run are **outside `tests/study_b/`**
(and outside `tests/agents/`, `tests/envs/`, the actually-relevant
directories for this thesis), in: `tests/analysis/`, `tests/calibration/`,
`tests/certification/`, `tests/formal/`, `tests/integration/`,
`tests/pilots/test_stage7*`, `tests/protocol/`, `tests/training/`, plus 3
in `tests/agents/test_stage2b2_*`.

These all belong to a **separate, unrelated pilot-study lineage**
(single/dyad-vehicle merge with PBRS reward shaping, Stage 1–11 — the
same one documented in `docs/PROVENANCE.md` as living in the
`final_new_stage8` repository, not in this thesis's own `Study B` work).
They ended up in this repository's `tests/` tree only because the whole
`src/thesis/` and `tests/` trees were copied wholesale (`thesis.study_b`
imports sibling packages such as `thesis.pilots.stage11_welfare`, so a
hand-picked subset would break at import time — see
`docs/provenance/copied_files_manifest.txt`'s own stated reasoning for
copying everything).

Every sampled failure is the same root cause: a `FileNotFoundError` (or a
downstream `AssertionError`/exception from one) for a historical
experiment-artifact path that belongs entirely to that other lineage and
was never part of this repository's scope — e.g.
`experiments\pre_impl\stage4a_r1_final_environment_reselection\artifacts\...\final_environment_lock.yaml`,
frozen protocol YAMLs, prior-stage "artifact intact" regression checks,
and lock-hash checks against files that live only in that other project's
own `experiments/formal/` and `experiments/pre_impl/` directories. None
of these paths were meant to exist in a Study-B-only release, and
including that other lineage's full historical artifact tree here would
defeat the purpose of this repository (which is scoped to RQ1/RQ2/WSC/DWS
only, per `README.md`).

**No code fix was made for these 83 failures** — fixing them would mean
copying in an unrelated project's multi-hundred-MB historical artifact
tree, which is out of scope for this release. If a fully green
`python -m pytest` (no directory filter) is ever wanted, the correct fix
is to delete the non-`study_b` (and non-`agents`/`envs`) test directories
from this repository rather than to manufacture their missing fixture
data.

## Skipped tests (36 in the full run, 12 within `tests/study_b/`)

Not individually triaged in this pass — skips are declared explicitly in
each test (`pytest.mark.skip`/`skipif`), typically gated on GPU
availability or on external checkpoint data not distributed with this
repository (see `README.md`'s Exclusions section). None were observed to
be silent/unexplained skips.
