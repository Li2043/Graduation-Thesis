# Dissertation Results Narrative Draft (Stage 6B)

## Scope

This draft summarises performance at the preregistered 100,000-step formal
endpoint under the H1-R1 equal-coefficient PBRS protocol
(`λ_mean = λ_min = 0.2`). It does not claim that training has converged solely
because the budget ended at 100,000 steps.

## Descriptive results

Across 30 completed formal runs (failed=0), seed-level primary
endpoints were computed from 16 validation evaluation episodes per
condition×seed at step 100000. Condition-level means and medians are reported
in the descriptives table.

## Uncertainty

Paired percentile bootstrap confidence intervals (10,000 replicates; seed 91001)
quantify uncertainty in mean paired differences. These intervals are descriptive
of sampling variability across the ten shared master seeds.

## Statistical tests

Two-sided paired Wilcoxon signed-rank tests were applied where defined, with
Holm adjustment within each primary endpoint across the three preregistered
contrasts. Undefined Wilcoxon cases (for example all-zero differences) are
reported explicitly rather than imputed.

## Substantive interpretation

Interpretations must remain within the randomised paired-seed design comparing
baseline, mean-PBRS, and min-PBRS under equal shaping coefficients. The analysis
does not claim magnitude-matched or RMS-matched shaping, and therefore does not
control realised shaping magnitudes across conditions.

## Limitations

1. Episode-level evaluation records at intermediate checkpoints were not
   published by Stage 6A; learning-curve trajectories for primary endpoints
   before step 100000 are unavailable and were not interpolated.
2. Step-100000 episode fields were reconstructed from published final weights
   using the locked evaluation protocol.
3. Missing convention-consistency values are retained as missing (never
   zero-filled).
4. Ending training at 100,000 steps does not by itself establish convergence.
