# NEEDS_USER_DECISION

Written: 2026-08-19 13:10:00
Resolved: 2026-08-20 (training machine is now the 32-core / RTX 4090 host)

## Issue

`protocol/new_protocol.md` (the independent-seed replication protocol,
FINAL VERSION 1) requires execution "on the same 32-core machine" used
for the original 18-run formal experiment (§16), sized for up to 18-way
process parallelism (§17) and a total of 21.6M new training steps (§18).
Stage 0 (audit + reproduction) has been completed and **passes** — see
`replication_preflight_audit.md`. The machine this bundle is currently
being operated from does **not** match the machine the protocol assumes,
and training should not be launched here without a decision.

## Evidence

Directly queried on the machine this bundle is currently on
(`replication_preflight_audit.md` §1, §5):

- CPU: Intel Core i7-13355U, 10 cores / **12 logical processors**
  (original machine: 32 logical processors).
- GPU: `torch.cuda.is_available() == False`, no usable CUDA device
  (original machine: RTX 4090 Laptop GPU, 15.99 GB VRAM — though also
  not the bottleneck-relieving factor for this network size per
  `verification/cuda_report.json`'s own finding, `gpu_speedup_factor_vs_cpu=0.418`,
  i.e. GPU was slower than CPU for this workload even on the original
  machine).
- RAM: ~31.7 GB total, ~13 GB free (original machine: 63.6 GB).
- Measured throughput on this machine: 131.4s for 512 evaluation
  episodes (2 conditions × 256 episodes), i.e. ~65s per single-condition
  256-episode evaluation pass. Training throughput will be worse per
  step than evaluation throughput, and the protocol's planned
  concurrency (up to 18 simultaneous processes) is sized for 32 cores,
  not 12.

## Options

1. **Migrate this directory (`F:\正式训练_seed_replication_v1`, already
   created — Stage 1 is done, see `README.md` §14) to the original
   32-core/GPU machine**, the same way the original bundle reached
   wherever it is now, before running `00_SETUP.bat` and launching any
   real training there. Preserves the protocol's implicit throughput and
   timeline assumptions.
2. **Run on this machine with deliberately reduced parallelism** (e.g.
   4–6 concurrent processes instead of up to 18) and an explicitly
   re-estimated, much longer timeline (plausibly days to weeks rather
   than the original 18-run experiment's timeframe) — feasible, not
   recommended without the user accepting that timeline up front.
3. **Do neither yet** — leave Stage 0 as the last completed step, keep
   this bundle as-is, and revisit once the target machine/logistics are
   clarified.

## Scientific consequences

None of these options change any frozen scientific parameter (seed IDs,
$\lambda_W$, network architecture, evaluation protocol) — this is purely
an infrastructure/throughput decision, not a methodology one. Whichever
option is chosen, `protocol/new_protocol.md`'s governance rules (§43)
still apply unchanged: no seed selection by outcome, no coefficient
tuning, all six new seeds reported regardless of what quality they turn
out to have.

## Recommended conservative action

Option 1 (migrate to the original machine) — matches what the protocol
document already assumes throughout (§16's explicit instruction), avoids
committing to a multi-week timeline on underprovisioned hardware, and is
the same operational pattern already used successfully once for this
project (the original bundle's own migration). This was the option
selected in conversation on 2026-08-19; **the blocker is not "which
option" but "the physical transfer to that machine has not happened
yet"** — see `README.md` §14 for exactly what is prepared and waiting.

## Resolution (2026-08-20)

Option 1 is now satisfied. This bundle is at `D:\正式训练_seed_replication_v1`
on the original training machine, confirmed by live hardware probe:

- CPU: 13th Gen Intel Core i9-13980HX, 24 cores / **32 logical processors**
- RAM: ~63.6 GB
- GPU: NVIDIA GeForce RTX 4090 Laptop GPU

Training is therefore allowed to start here (`00_SETUP` → preflight →
§10 smoke test seed 929999 → §11 six-seed curriculum). CPU vs GPU remains
a measured execution-target choice only (`scripts/verify_cuda.py`); no
scientific parameter changes.
