# Dense Reward Study — frozen protocol v1 (c recalibrated)

Full machine-readable version: `configs/dense_reward_protocol_v1.json`. This file is the human-readable
record of the same freeze.

## Previous provisional c revoked

```
Previous provisional c: 0.1
```

**Never used in any formal or Priority 1 training run** — it was frozen during the initial 10-minute
pre-Priority-1 check but flagged there as a disclosed tension, not committed to before this recalibration.

**Why revoked**: Actual Study B reward traces showed that c=0.1 would materially dominate the typical
non-terminal base reward. `median(abs(base_reward))` across all (step, agent) entries in the real
WSC+Dense trace probe was only `0.004244` — so `c=0.1` alone is ~23.6x that typical per-step magnitude,
far outside any defensible "small nudge" role for a shaping term.

## Recalibration method

Used the existing trace (`validation/wsc_dense_trace_probe_output.jsonl`, 73 real steps from a WSC+Dense
rollout resumed from the real seed-900101 C64 checkpoint) — no new training, no C64 retrain, no formal
12-seed run.

1. **Dense-active frequency** `p`: fraction of steps where `abs(DeltaPhi) > epsilon(1e-6)`.
2. **Base reward scale**: `median(abs(base_reward))` over all (step, agent) entries (absolute, not signed
   — signed values partially cancel and would understate true magnitude).
3. **Target interval**: chose `c` so that `p * c / median(abs(base_reward))` (the average per-timestep
   absolute dense contribution as a fraction of typical base reward) falls in `[0.10, 0.25]`.
4. **Candidate selection**: from `{0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05}`, picked the value(s)
   inside the target interval, then applied "prefer the smaller, more conservative value when more than
   one qualifies."
5. **Episode-level sanity check**: compared total absolute dense contribution over the one available
   episode against that episode's absolute base return, confirmed it is far below the 0.5 danger
   threshold.

## Results

```
Frozen epsilon:                        1e-6   (unchanged from the original freeze)
Dense-active frequency (p):            0.9863  (72/73 steps: 70 positive, 2 negative, 1 dead-zone)
Median |base_reward| (step, agent):    0.004244
Mean |base_reward| (step, agent):      0.012011
P75 / P90 / P95 |base_reward|:         0.004301 / 0.004361 / 0.004800
Target c interval (10%-25% of scale):  [0.00043, 0.00108]
Candidates inside target interval:     0.0005, 0.001
Final frozen c:                        0.0005   (the smaller of the two, per the conservative tie-break rule)
Estimated timestep dense/base ratio:   0.1146   (within target)
Estimated episode dense/base ratio:    0.0431 (absolute) / 0.0408 (net)   (well below 0.5 danger threshold)
```

## Explicit non-use statement

`c` was selected **before** any formal Dense training. Selection used **only** reward-scale and
signal-frequency information from the existing trace. It did **not** use: fairness outcomes, completion
outcomes, collision outcomes, formal Dense training results, or any parameter-sweep training.

## Final smoke test after recalibration

See `validation/dense_shaping_smoke_test_report.md` addendum (recalibration section) for the short
confirmation run with the new `c=0.0005` — dense component uses the new value, all values finite, shared
shaping unchanged, no NaN/Inf. No exit test, full regression suite, or 32-core benchmark was re-run (not
needed — nothing about those areas changed).
