# Stage 7C-Q1 Experiment Machine Report

## Protocol
- protocol tag: `stage7c-q1-protocol-v1`
- training commit: `c8c75207c06c6a0511cac5fb24b644a61def8d14`
- algorithm: Double DQN
- condition: Baseline only
- reward: Base Reward V2 (active_time_cost_per_step=0.0005)
- max steps: 400000

## Machine
- Python / PyTorch: see ENVIRONMENT_MANIFEST.json
- workers: 8
- threads per worker: 1
- checkpoint write slots: 8

## Completeness
- validator status: `COMPLETE`
- completed seeds: 20 / 20
- max step per seed: 400000
- logical checkpoints: 340 / 340
- formal evaluation episodes: 14080 / 14080
- duplicate episode keys: 0
- conflict rows: 0
- cross-seed eval overlap ok: True
- incomplete seeds: none
- training failures: none
- resumed jobs: 0 (clean start)

## Storage
- checkpoint root: `C:\Users\SamChui\graduation_thesis_runs\stage7c_q1_v1_checkpoints`
- output root: `C:\Users\SamChui\graduation_thesis_runs\stage7c_q1_v1`
- checkpoint inventory: present (SHA-256)
- checkpoint backup redundancy: not available
- GitHub checkpoint uploads: 0

## Notes
- Full `pytest -q` reported 14 failures due to missing historical `experiments/pre_impl` artifacts on this machine; Stage 7C-relevant suites (`tests/pilots`, `tests/training`, `tests/envs`, `tests/rewards`) = 220 passed.
- Scripted Base Reward V2 audit: PASS.
- Formal competence gate PASS/FAIL is deferred to local Prompt 3 analysis.
