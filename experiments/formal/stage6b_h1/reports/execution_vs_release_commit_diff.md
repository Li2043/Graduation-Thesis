# Execution commit vs release commit diff (H1 → H1.1)

Compared:

- execution-recorded HEAD at first H1 run: `c54905ece91ffb5c8f5ec4634b65c457a102e0d5`
- candidate release commit: `ce6a27ba98dd60bb89a324491fc3f0702ecf9d71`

Command basis: `git diff --stat` and `git diff --name-only` between these SHAs.

## Classification

### A. Evaluation-affecting code
- `src/thesis/analysis/reconstruct_eval.py` (trajectory utility sampling wired into evaluator)

### B. Utility calculation code
- `src/thesis/analysis/episode_utility_accumulator.py` (**new**)

### C. Statistical analysis code
- `experiments/formal/stage6b_h1/scripts/run_stage6b_h1.py` (seed aggregation, contrasts, controller-swap, acceptance)

### D. Seed/protocol/config code
- None beyond using existing locked Stage 6A seeds via reconstruct_eval (no protocol file content change in this diff).

### E. Manifest/release-only code
- Partial: same runner also writes manifests/logs; not separable from C in the single H1 commit.

### F. Tests only
- `tests/analysis/test_episode_utility_accumulator.py`
- `tests/analysis/test_stage6b_h1_*.py`

### G. Reports/documentation only
- `experiments/formal/stage6b_h1/reports/*`
- `PAPER_CHANGES_REQUIRED_LATER.md`
- audit / status / logs

### H. .gitignore or packaging only
- None in this commit range.

## Decision

`REQUIRES_EVALUATION_RERUN = true`

Reason:

1. Diff contains evaluation-affecting and utility-calculation code (A, B, C).
2. First H1 runner log recorded `git_commit=c54905e...` while the working tree already contained uncommitted A/B changes later committed as `ce6a27ba`.
3. Release provenance therefore requires a fresh 480-episode execution under a committed SHA that includes the evaluator/utility code, with H1.1 release metadata written afterward.
