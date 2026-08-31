# Provenance

This repository has **no inherited git history**. It was consolidated,
in a single pass, from three timestamped, checksum-verified snapshot
bundles that were never git repositories themselves. This is stated
plainly here rather than left implicit, since a fresh `git init` could
otherwise be mistaken for a project with no prior history at all.

## Why there is no commit history

Each source bundle independently confirms it was never a git working
tree. Quoted verbatim:

- `F:\正式训练\MIGRATION_MANIFEST.json`, field `git_note`:
  > "confirmed NOT a git repository on the source machine (git status/rev-parse
  > both fail with 'not a git repository'); provenance is tracked via
  > CHECKSUMS.sha256 + the experiment_records/ tracking files instead of a
  > commit hash"

- `C:\dense reward\project\LOCALITY_AMENDMENT_ISOLATION_RECORD.md` (line 52):
  > "`git status` and `git rev-parse HEAD` both fail with 'not a git
  > repository (or any of the parent directories)'"

So provenance for the code and results in this repository rests on file
checksums and hand-written manifests, not commit hashes. The original
manifest files are copied into `docs/provenance/` alongside this document
so the chain of custody remains inspectable.

## The three source bundles and their role

The formal training/evaluation for this thesis ran from three snapshot
bundles, each a superset of the previous one, forming a chronological
chain:

1. **`F:\正式训练\`** — the original 6-seed formal campaign
   (seeds `900101–900104`, `910101`, `910102`), Baseline / Mean / GGI /
   Maximin, Original (18D) observation only. Backs RQ1 and the Original
   half of RQ2.
2. **`F:\正式训练_seed_replication_v1\`** — a second-generation working
   copy (built on different hardware — see
   `docs/protocol/START_HERE_seed_replication_v1.md` for the documented
   CPU-count discrepancy) that adds the 6 replication seeds
   (`920101–920106`), the Welfare-State Communication (WSC, 22D)
   continuation runs, and all of the formal RQ1/RQ2/WSC statistical
   analysis scripts (`analysis/wsc_v2_formal/`, `analysis/ch5_baseline/`,
   `analysis/pooled12/`, `analysis/wsc_v2_behavioural/` in this
   repository).
3. **`C:\dense reward\`** — the most recent bundle, a superset again,
   adding `dense_shaping.py` (Dense Welfare Shaping / DWS) and the full
   DWS training/evaluation/analysis/whole-thesis-synthesis toolchain
   (most of `scripts/` in this repository).

All three copy the same underlying `src/thesis/` package tree (not just
`study_b/`, since `study_b` imports sibling packages such as
`thesis.pilots.stage11_welfare`) — confirmed by `C:\dense reward`'s own
`copied_files_manifest.txt`, which documents exactly this reasoning for
copying the whole tree rather than a hand-picked subset. This repository
uses the `C:\dense reward\project\` copy as the source of truth for
`src/` and `tests/`, since it is the chronologically latest and only
fully-complete one.

## What's in `docs/provenance/`

The original manifest and environment-capture files copied from the
source bundles: `copied_files_manifest.txt`, `MIGRATION_MANIFEST.json`,
`CHECKSUMS_正式训练.sha256`, `LOCALITY_AMENDMENT_ISOLATION_RECORD.md`,
`modified_files_manifest.txt`, `path_changes.md`,
`environment_setup_report.md`, `source_environment.md`,
`python_environment.txt`, `pip_freeze.txt`,
`requirements-lock-minus-pyyaml.txt`, `parallelism_benchmark.md`.

See `docs/reports/` for the full statistical write-ups (WSC and DWS
formal evaluation, behavioural mechanism analysis, whole-thesis evidence
synthesis) and `docs/protocol/` for the frozen experiment configs and
manifests these bundles were built and verified against.
