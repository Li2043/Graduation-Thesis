# Stage 7A-0 — Baseline Competence Diagnostic Pilot

## A. Status

```text
Diagnostic pipeline status: PARTIAL
Baseline competence status: NOT PASSED
Continuation probe status: BLOCKED
```

PASS here means the diagnostic workflow completed for available artifacts.
It does **not** mean Baseline competence is adequate.

## B. Input integrity

- Published Baseline final weights: 10
- Missing full resumable checkpoints: 50
- Formal checkpoint hash changes: 0
- Paper file changes: 0
- Stage 6A / Stage 6B-H1: not modified by this pilot

## C. Checkpoint learning trajectory

| step | status |
|-----:|--------|
| 0K | unavailable |
| 10K | unavailable (local-only ckpt missing) |
| 25K | unavailable |
| 50K | unavailable |
| 75K | unavailable |
| 100K | available (final_online_target_weights.pt) |
| 125K–200K | BLOCKED (continuation) |

## D. 100K outcomes (reconstructed diagnostic; n=160)

- Success: 0.3500
- Collision: 0.0437
- Truncation: 0.6062
- Non-diagnostic mismatches vs H1: 0

## E. Failure taxonomy (truncated episodes)

{
  "unilateral_stall": 74,
  "mutual_yielding": 17,
  "downstream_completion_failure": 5,
  "other_unresolved": 1
}

## F. Seed bifurcation

- High (>=0.75): [61004, 61007]
- Intermediate: [61003, 61005, 61006]
- Low (<0.25): [61001, 61002, 61008, 61009, 61010]

## G–I. Policy / reward / continuation

- Replay/TD diagnostics: unavailable (no replay in published weights)
- Reward weak separation: False
- Continuation: BLOCKED — Published Stage 6A retains only final_online_target_weights.pt. Full ckpt_step_*.pt files were local_only_intermediate_or_replay_checkpoint and are absent from this results worktree; resume/continuation is impossible.

## J. Root-cause matrix

See `output/endpoint_tables/baseline_root_cause_matrix.csv`.

## K. Recommendation

Do **not** implement interventions in this stage.

Highest-priority next experiment candidates (choose one primary axis):

1. If full checkpoints can be recovered: unchanged continuation / longer Baseline-only budget pilot on **new** seeds.
2. If truncation taxonomy is dominated by mutual yielding / post-exit stall and reward separation is weak: base-task deadlock-resolution reward pilot (new experiment version; re-audit PBRS boundary; retrain all conditions).
3. If seed bifurcation dominates with some high competence seeds: increase independent Baseline seeds and stabilise before treatment comparison.

This pilot reused formal seeds and is exploratory only.
