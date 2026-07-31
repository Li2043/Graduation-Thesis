# Stage 7B-A1 Final Analysis Report

## 1. Protocol and integrity

| Field | Value |
| --- | --- |
| Frozen tag | `stage7b-a1-protocol-v1` |
| Frozen protocol commit | `3a190d6763120e7f4b60a1f9e2412c0c3c31954c` |
| Results commit | `b47d6fd4c7c8c6832fdc9ae25a4daa5d27ce59b7` |
| Protocol hash | `32f5707e2e9f1ccefcdc48f712e94ff4bd96ae12ea1d1558b68b2c0d3b3afea4` |
| Environment lock | `d2d82ac02feb5bb2f5217f8e399972b91bd56cce343f60861954f66d7f70bf12` |
| Comfort lock | `1d9439c211955f9a8a177e455b6b5ff34aa98f85f3ff0677bbd62abd6d29b061` |
| Local validation | `OK` |
| Stage 7B git checkpoints | `0` |
| Checkpoints downloaded locally | `False` |

Critical failures: `none`.

## 2. Run completion

- Planned / completed runs: 40 / 40
- Paired seeds: 20 (`63001`–`63020`)
- Conditions: ['double_dqn', 'vanilla_dqn']
- Checkpoints: 10
- Evaluation episodes: 6400 (expected 6400)
- Duplicate keys: 0
- Missing condition–seed–checkpoint: 0
- Evaluation isolation violations: 0
- Checkpoint inventory SHA-256 complete: True
- Storage location: experiment_machine_local for all inventory rows
- Thesis / Stage-6 formal tracked file changes (training summary): 0 / 0

## 3. Descriptive trajectories

Seed-level endpoints are in `output/analysis/stage7b_a1_seed_checkpoint_endpoints.csv`. Condition×checkpoint descriptives (mean/SD/median/IQR/min/max/bootstrap CI over seeds) are in `output/analysis/stage7b_a1_descriptives.csv`.

300K means (seed-level):

| Condition | Success | Collision | Truncation | Unilateral stall | Seeds ≥0.75 | Seeds <0.50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla | 0.6125 | 0.0688 | 0.3187 | 0.2531 | 10 | 6 |
| Double | 0.7562 | 0.0813 | 0.1625 | 0.1562 | 13 | 3 |

## 4. 300K primary paired contrasts

Statistical unit = paired training seed. Differences = `double_dqn - vanilla_dqn`. Bootstrap resamples the 20 paired differences. Holm adjustment applied only to the pre-defined 300K primary family.

| Endpoint | Mean Δ | 95% CI | Wilcoxon p | Holm p | Cohen dz | Double higher / lower / tied |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| success | 0.1437 | [-0.0656, 0.3375] | 0.1526 | 0.6105 | 0.302 | 12 / 7 / 1 |
| truncation | -0.1562 | [-0.3406, 0.0437] | 0.12 | 0.5999 | -0.344 | 7 / 12 / 1 |
| collision | 0.0125 | [-0.0250, 0.0469] | 0.4323 | 0.8284 | 0.151 | 10 / 5 / 5 |
| unilateral_stall | -0.0969 | [-0.2844, 0.1000] | 0.2655 | 0.7964 | -0.214 | 7 / 11 / 2 |

Full table: `output/analysis/stage7b_a1_paired_contrasts.csv`.

## 5. Late collapse

Frozen rule: success at 200K or 250K ≥ 0.75 and success at 300K < 0.50.

- Vanilla collapse count: **2**
- Double collapse count: **4**
- Discordant (Vanilla only / Double only): 2 / 4
- Exact McNemar p (exploratory): 0.6875

Per-seed table: `output/analysis/stage7b_a1_late_collapse_by_seed.csv`.

## 6. Failure taxonomy

Primary failure labels at 300K (episode counts by condition) are summarised from `output/diagnostics/baseline_algorithm_failure_taxonomy.csv`:

```json
{
  "mutual_yielding": {
    "double_dqn": 2,
    "vanilla_dqn": 20
  },
  "other_unresolved": {
    "double_dqn": 0,
    "vanilla_dqn": 1
  },
  "post_exit_survivor_stall": {
    "double_dqn": 50,
    "vanilla_dqn": 81
  }
}
```

Unilateral stall remains present under both algorithms at 300K (see descriptives and paired contrasts).

## 7. Q/TD diagnostics

| Metric | Vanilla | Double |
| --- | ---: | ---: |
| mean Q margin @300K | 0.0150 | 0.0235 |
| mean abs best Q @300K | 0.8536 | 0.7034 |
| TD abs p95 @300K | 0.0420 | 0.0435 |
| mean replay terminal frac | 0.0092 | 0.0145 |

Judgments (not claims of eliminated overestimation without direct target-gap evidence):

```json
{
  "Vanilla_overestimation_unstable_bootstrap": "NOT IDENTIFIABLE",
  "Double_reduces_late_collapse": "NOT SUPPORTED",
  "Double_improves_action_separation": "PARTIALLY SUPPORTED",
  "Double_merely_increases_aggressiveness": "PARTIALLY SUPPORTED",
  "Double_converts_truncation_into_collision": "PARTIALLY SUPPORTED",
  "seed_bifurcation_remains": "SUPPORTED",
  "reward_related_stall_remains": "SUPPORTED"
}
```

## 8. Competence gate

Provisional gate (unchanged; not lowered for Double): ≥16/20 seeds success≥0.75; mean success≥0.75; collision≤0.05; truncation≤0.15; swap eligibility≥0.75.

| Condition | 300K gate | Failed components | Consecutive primary pass | Stable budget |
| --- | --- | --- | --- | --- |
| Vanilla | False | `seeds_ge_0_75,mean_success,collision,truncation,swap_eligibility` | False | `` |
| Double | False | `seeds_ge_0_75,collision,truncation,swap_eligibility` | False | `` |

Table: `output/analysis/stage7b_a1_competence_gate.csv`.

## 9. Scientific interpretation

Decision class **B**: algorithm stabilisation was beneficial but insufficient.

- Double raises mean success and lowers truncation at 300K in the paired seed analysis, but does **not** pass the frozen competence gate.
- Late collapses: Vanilla 2 vs Double 4 — Double does **not** reduce late collapse in this pilot.
- Collision direction: mean paired Δ = 0.0125 (positive ⇒ Double higher collision). No formal non-inferiority margin; do not claim safety non-inferiority.
- Safety note: competence improvement accompanied by safety degradation.
- Seed bifurcation remains under both algorithms.
- Unilateral stall is not eliminated; reward/credit structure remains a live failure mode.

## 10. Next experiment decision

- Use Double DQN for next pilot: **True**
- Modify reward next: **True**
- Extend budget alone: **False** (gate not passed; bifurcation/stall persist)
- Required new seeds: **True** (do not reuse 610xx/620xx/630xx blocks without a new frozen plan)

Recommended path: treat Double as optional algorithmic default only if retained for engineering reasons; prioritise a **single-factor active-time-cost / stall-resolution reward pilot** because algorithm change alone was insufficient and late collapse did not improve.

## 11. Limitations

- Exploratory algorithm pilot; competence gate is provisional and not a confirmatory preregistered test.
- In-loop evaluation was empty on the training machine; episodes are post-hoc greedy reconstructions from full checkpoints (hashes inventoried; weights not downloaded here).
- No local checkpoint download; Q/TD diagnostics rely on experiment-machine summaries.
- Holm correction covers only the 300K primary family; other checkpoints/endpoints are exploratory.
- Cannot claim Double DQN eliminated overestimation without direct online/target overestimation gap series.
- Do not pool with formal 610xx or Stage 7A-1 620xx seeds.

## Figures

- `output/figures/fig_success_by_algorithm_checkpoint.png`
- `output/figures/fig_success_paired_300k.png`
- `output/figures/fig_truncation_by_algorithm_checkpoint.png`
- `output/figures/fig_collision_by_algorithm_checkpoint.png`
- `output/figures/fig_unilateral_stall_comparison.png`
- `output/figures/fig_q_margin_comparison.png`
- `output/figures/fig_seed_trajectory_vanilla.png`
- `output/figures/fig_seed_trajectory_double.png`
- `output/figures/fig_late_collapse_comparison.png`
- `output/figures/fig_seed_success_distribution_300k.png`
- `output/figures/fig_competence_gate_components.png`
