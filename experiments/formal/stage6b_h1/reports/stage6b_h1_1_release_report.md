# Stage 6B-H1.1 Release Metadata and Reproducibility Report

## 19.1 Status

```text
PASS
```

## 19.2 Evaluation rerun decision

```text
Evaluation rerun required: yes
```

Evidence: `git diff c54905ece91ffb5c8f5ec4634b65c457a102e0d5 ce6a27ba98dd60bb89a324491fc3f0702ecf9d71`
includes evaluation-affecting paths `src/thesis/analysis/reconstruct_eval.py` and
`src/thesis/analysis/episode_utility_accumulator.py`. The first H1 run recorded
`HEAD=c54905e` while the working tree already contained those evaluator/utility
changes later committed as `ce6a27ba`. See
`reports/execution_vs_release_commit_diff.md` and
`output/manifests/h1_1_release_decision.json`.

Therefore H1.1 re-ran all 480 evaluation episodes under the committed H1.1 code
SHA before packaging.

## 19.3 Commit provenance

```text
Original H1 execution commit (first logged HEAD):
  c54905ece91ffb5c8f5ec4634b65c457a102e0d5

H1.1 evaluation execution commit:
  1a6e361a4f31f30fd74edc4b0bf8b7ebfae556d3

H1.1 release commit:
  <filled after packaging commit>
```

## 19.4 Manifest fixes

- `nonutility_mismatches.csv` is header-only with a single LF newline; zero data rows;
  SHA-256 = `058f70d94399aabc7a91305fb4429a13b9a24aa21ced6af74447946a7d2ad0fc`
- `paper_file_integrity_before.csv` and `paper_file_integrity_after.csv` both registered
  in `output_hashes`; `changed_file_count = 0`
- `figure_paths` use paths relative to `experiments/formal/stage6b_h1/`
- `reference_tolerance = 1e-6`; `maximum_absolute_reference_error ≈ 4.35e-7`
- `verify_manifest_hashes()` passes for all registered relative paths

## 19.5 Archive

```text
archive path: <filled after packaging>
archive SHA-256: <filled after packaging>
archive size: <filled after packaging>
extracted validation: <filled after packaging>
```

## 19.6 Experiment result preservation

```text
480 episodes
30 checkpoints
0 nonutility mismatches
mean utilities unchanged vs H1 corrected references
swap estimability unchanged: Baseline 4 / Mean-PBRS 0 / Min-PBRS 4
```

Corrected mean stakeholder utility:

| Condition | Mean utility |
|-----------|-------------:|
| Baseline  | 0.605213 |
| Mean-PBRS | 0.527772 |
| Min-PBRS  | 0.586206 |

## 19.7 Thesis integrity

```text
No thesis or dissertation files were modified.
```

Paper integrity before/after hashes match (`verified_unchanged: true`).
