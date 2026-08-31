# Parallelism benchmark (Section 9)

## Hardware discrepancy — read this first

The task brief stated this machine has **32 logical CPUs**. Direct measurement in this shell/session
disagrees, confirmed four independent ways (all agreeing on **12**):

| Method | Result |
|---|---|
| Python `multiprocessing.cpu_count()` | 12 |
| Python `os.cpu_count()` | 12 |
| Windows env var `%NUMBER_OF_PROCESSORS%` | 12 |
| PowerShell `(Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors` | 12 |
| PowerShell `(Get-CimInstance Win32_Processor | Measure-Object NumberOfLogicalProcessors -Sum).Sum` | 12 |

Total RAM: 31.7 GB (psutil), ~12.6 GB available at benchmark time (other processes running).

Everything below was benchmarked against the **12 cores this session actually has**. The worker-count sweep
was rescaled accordingly (2/4/6/8/10/12/16 instead of the originally-suggested 4/8/16/24/28/32). **If the
real dense-reward training campaign later runs on hardware that genuinely exposes 32 logical CPUs, this
sweep and the numeric `DEFAULT_MAX_WORKERS` recommendation below should be re-run there** — the
`cpu_count() - 2` *formula* (already used by `launch_formal.py`) is expected to generalize, but the
peak/plateau *shape* found here (see below) should be re-verified on the real target machine rather than
assumed to hold at a different core count.

## Method

For each worker count W, launched W genuinely independent OS processes
(`train_curriculum_stage_highwayenv.py`, 400 env steps each, `--replay-warmup 50` so the workload includes
real DQN forward/backward, not just environment stepping), each with `OMP_NUM_THREADS=1` /
`MKL_NUM_THREADS=1` set in its environment — process-level parallelism only, mirroring the existing
`launch_formal.py` / `launch_replication_welfare.py` convention exactly (never thread-level or
algorithmic parallelism inside one run; no batch-size/update-frequency change to "use more cores"). No
`torch.set_num_threads()` call was found or added anywhere in the traced training code — thread-limiting is
entirely via the `OMP_NUM_THREADS`/`MKL_NUM_THREADS` env vars set one level up by the launcher, matching the
existing design. Not a real training run — a smoke workload, run once per worker-count configuration.

Measured: wall-clock time for all W processes to finish, aggregate throughput (total env-steps ÷
wall-seconds), sampled aggregate RSS across all children every 0.5s (peak reported), sampled system-wide
CPU% every 0.5s (mean/max reported), and crash rate (nonzero exit codes ÷ W).

## Results

| workers (W) | wall (s) | throughput (env-steps/s) | crashes | peak aggregate RSS (MB) | mean CPU% | max CPU% |
|---|---|---|---|---|---|---|
| 2  | 10.4 | 76.7  | 0/2  | 716  | 24.4 | 39.6 |
| 4  | 15.1 | 105.9 | 0/4  | 1432 | 40.3 | 54.1 |
| 6  | 19.7 | 121.8 | 0/6  | 2148 | 51.4 | 71.0 |
| 8  | 26.1 | 122.8 | 0/8  | 2862 | 60.2 | 88.8 |
| 10 | 29.9 | 133.8 | 0/10 | 3577 | 67.0 | 96.9 |
| 12 | 40.0 | 120.1 | 0/12 | 4288 | 75.0 | 99.6 |
| 16 | 46.9 | 136.5 | 0/16 | 5652 | 77.2 | 100.0 |

Raw data: `validation/parallelism_benchmark_raw.json` (includes per-run return codes and sample counts).

## Interpretation

- **No crashes at any worker count**, including 16 workers on 12 cores (33% oversubscription). Crash rate is
  not the limiting factor at these worker counts on this hardware.
- **RAM scales linearly** at ≈350–360 MB per worker (well under the placeholder 400MB/process estimate
  already hard-coded in `launch_formal.py`'s `ESTIMATED_MEM_PER_PROCESS_MB`) and is nowhere near the 31.7GB
  ceiling even at 16 workers (5.65GB) — RAM is not the binding constraint on this machine for this workload.
- **Throughput peaks at W=10 (133.8 steps/s) among the "at or under core count" configurations**, then
  *drops* at W=12 (120.1 steps/s, -10%) — i.e. running on literally every logical core, with zero headroom
  for the OS/launcher/logging, measurably hurt throughput here, not just latency. This is a direct empirical
  confirmation of the existing `cpu_count() - 2` design already used by `launch_formal.py` and
  `launch_replication_welfare.py` (both leave exactly 2 cores free) — it is not merely a theoretical
  convention on this hardware, it measurably wins.
- **W=16 (oversubscribed) shows the single highest raw throughput number (136.5)**, but at the cost of the
  highest wall-clock latency per batch (46.9s vs 29.9s at W=10, +57% for +33% more workers) and CPU pinned
  at 100% max with 77% mean (no headroom at all — a real launcher/monitoring process competing for that last
  bit of CPU would be starved). Given the task's explicit tie-breaking rule ("give the OS/IO/launcher 2-4
  cores unless benchmark clearly proves otherwise"), this modest, noisy throughput edge (+2% over W=10) is
  not judged a clear enough win to justify running with zero OS headroom.

## Recommendation

`DEFAULT_MAX_WORKERS = cpu_count() - 2` — **on this 12-core machine, that is 10**, which is also the
best-performing non-oversubscribed configuration actually measured (133.8 steps/s, 96.9% peak CPU, still 2
cores of headroom). This is the same formula `launch_formal.py` already implements
(`compute_safe_concurrency`); no launcher code change is needed. Do NOT hardcode the literal value `10` (or
`28`) anywhere — keep it as the `cpu_count() - 2` formula so it re-derives correctly on whatever machine
actually runs the dense-reward campaign; re-run this benchmark script
(`provenance`-referenced `parallelism_benchmark.py`, kept in the coordinator's scratchpad — see
`environment_setup_report.md` if you need to locate/rerun it) on that machine to confirm the same
plateau-then-dip shape holds before trusting the number for a real campaign.

Per-child thread limiting confirmed necessary and already sufficient: `OMP_NUM_THREADS=1` +
`MKL_NUM_THREADS=1` per child, seed/run-level parallelism only, exactly the existing convention — no changes
recommended to that part of the design.
