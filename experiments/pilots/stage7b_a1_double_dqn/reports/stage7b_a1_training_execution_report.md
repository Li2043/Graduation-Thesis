# Stage 7B-A1 Training Execution Report

## Status

**PASS** — 40/40 formal runs completed to 300K; post-hoc evaluation produced 6,400 episodes with no duplicate keys and no evaluation isolation violations.

## Frozen protocol

| Field | Value |
| --- | --- |
| Tag | `stage7b-a1-protocol-v1` |
| Commit | `3a190d6763120e7f4b60a1f9e2412c0c3c31954c` |
| Protocol hash | `32f5707e2e9f1ccefcdc48f712e94ff4bd96ae12ea1d1558b68b2c0d3b3afea4` |
| Environment lock | `d2d82ac02feb5bb2f5217f8e399972b91bd56cce343f60861954f66d7f70bf12` |
| Comfort lock | `1d9439c211955f9a8a177e455b6b5ff34aa98f85f3ff0677bbd62abd6d29b061` |

Training code was not modified on the experiment machine. Jobs executed from the frozen tag.

## Machine environment

- OS: Windows 11 (10.0.26200)
- CPU: 13th Gen Intel Core i9-13980HX (24 physical / 32 logical)
- RAM: ~63.6–68 GB
- Device: CPU (`torch 2.13.0+cpu`)
- Python: 3.12.10
- NumPy 2.3.2 / Gymnasium 1.2.0

## Parallel execution

Frozen in `manifests/parallel_execution_config.json`:

- `max_workers = 12`
- `threads_per_worker = 1`
- `checkpoint_write_slots = 12` (raised to match workers because the frozen launcher holds the write-slot semaphore for the full job lifetime; protocol default slots=2 would have capped concurrency at 2)
- BLAS / OpenMP / Torch threads = 1 per worker

Resource probe (2 Vanilla + 2 Double to 10K): peak RAM ~20.5%, wall ~39 s. Selected 12 workers (not 16/32).

Launcher summary: 40 jobs, `failed=[]`, elapsed ~6985 s (~1.94 h).

## Resume equivalence

Isolated test (`double_dqn`, seed logic 63001): Path A uninterrupted 0→50K vs Path B 0→25K save/terminate/resume→50K.

- Passed: **true**
- Network / optimizer / replay / RNG / schedule / evaluation mismatch: **all 0**

Formal seed 63001 was restarted from step 0 after the test artifacts were isolated.

## Run completion

| Metric | Value |
| --- | --- |
| Planned | 40 |
| Completed | 40 |
| Failed | 0 |
| Resumed jobs | 0 (clean formal start) |
| Final step | 300000 for all |

## Checkpoints

- External root: `C:\thesis_checkpoints\stage7b_a1_double_dqn`
- Outside git repository: **true**
- Full milestone checkpoint sets: **400** (40 × 10)
- Weights extracted post-hoc: **400**
- Metadata JSON: **400**
- Hash failures: **0**
- GitHub checkpoint uploads: **0**
- Checkpoint backup redundancy: **not available** (only C: filesystem present; no second disk/backup location verified)

## Evaluation (post-hoc)

In-loop `evaluation_steps` were empty in the frozen runner; greedy evaluation was run from full checkpoints after training.

| Metric | Value |
| --- | --- |
| Expected episodes | 6400 |
| Actual episodes | 6400 |
| Duplicate keys | 0 |
| Isolation violations | 0 |

## Preliminary descriptive outcomes (300K)

| Condition | Success | Collision | Truncation | Late collapses (seeds) |
| --- | --- | --- | --- | --- |
| Vanilla DQN | 0.6125 | 0.06875 | 0.31875 | 2 |
| Double DQN | 0.75625 | 0.08125 | 0.1625 | 4 |

These are descriptive only. Final paired statistical inference is reserved for the local analysis machine.

## Integrity

- Stage 6 / formal / thesis tracked files: **unchanged** relative to frozen tag
- No `.pt` / `.pth` / `.ckpt` under the published results paths

## Results branch

Branch to publish: `results/stage7b-a1-double-dqn` (created from `stage7b-a1-protocol-v1`).
