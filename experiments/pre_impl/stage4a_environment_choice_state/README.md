# Stage 4A — Final Environment Candidate Selection and Choice-State Certification

Freezes road geometry, physics/decision timing, IDM background traffic, initial-condition
blocks, and genuine strategic choice-state semantics **before** any policy training.

## Scope

- Core reward only: `0.4 Δρ + 0.6 exit − 1.0 collision` (no hard-braking / comfort term).
- No DQN updates, replay sampling, PBRS λ tuning, or comfort-parameter freeze.
- Comfort remains unresolved after Stage 3B failure.

## Run

```bash
# from repository root, with PYTHONPATH=src
python experiments/pre_impl/stage4a_environment_choice_state/scripts/run_stage4a_certification.py \
  --config experiments/pre_impl/stage4a_environment_choice_state/configs/environment_candidates.yaml
```

## Outputs

Per `run_id` under `data/raw`, `data/processed`, `reports`, `logs`, `artifacts`.
On PASS only: `artifacts/<run_id>/final_environment_lock.yaml` (+ `.sha256`).
