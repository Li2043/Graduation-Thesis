# Fairness in Multi-Agent Highway Merging — Thesis Code Release

Code and formal-analysis pipeline for a thesis studying distributional
fairness in a 4-vehicle highway-merge RL environment (role×speed classes:
Fast/Slow, Ramp/Mainline). Four experiments are covered:

- **RQ1 (Baseline)** — does task-only training produce systematic
  differences in individual welfare?
- **RQ2 (Mean / GGI / Rawlsian Maximin)** — do social-welfare reward
  objectives improve worst-off welfare and reduce inequality, relative to
  Baseline?
- **WSC (Welfare-State Communication)** — does expanding the observation
  from 18D to 22D (adding running local welfare) change the effect of
  Mean/GGI/Maximin?
- **DWS (Dense Welfare Shaping)** — does adding step-wise shared Maximin
  potential-based reward shaping, on top of the terminal welfare reward,
  change realised fairness?

There is **no git history before this repository's first commit** — see
[`docs/PROVENANCE.md`](docs/PROVENANCE.md) for why, and for how the code
here was consolidated from three non-git snapshot bundles.

## Environment setup

This project's Python stack has one documented gotcha: **the system
Python 3.12 on the original development machine has a broken stdlib
`json` module — do not use it.** Development used Python 3.14.6.

Install in order (each file `-r`s the previous):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    POSIX: source .venv/bin/activate
pip install -r requirements-stage1.txt
pip install -r requirements-stage2b1.txt
pip install -r requirements-stage2b2.txt   # pulls in torch
pip install -r requirements-study_b.txt    # adds pettingzoo, needed by tests/study_b/test_pettingzoo_wrapper.py
```

Run the test suite from the repo root (`pytest.ini` sets `pythonpath=src`,
`testpaths=tests`):

```bash
python -m pytest             # full suite (includes an unrelated pilot lineage's tests -- see docs/KNOWN_TEST_GAPS.md)
python -m pytest tests/study_b   # just this thesis's own test suite -- 298 passed, 12 skipped, 0 failed
```

See [`docs/KNOWN_TEST_GAPS.md`](docs/KNOWN_TEST_GAPS.md) before treating any
test failure as a repository defect: this codebase's `tests/` directory was
copied whole (see `docs/PROVENANCE.md`) and includes ~83 tests for a
completely separate, unrelated pilot-study lineage that share the same `src/`
tree only because `thesis.study_b` imports sibling packages.

## Pipeline overview

1. **Curriculum task training** (task reward only): `C1 → C4 → C16 → C64`
   scenario stages, 1.2M environment steps total, common starting point
   for all conditions.
2. **Matched continuation**: the 1.2M-step checkpoint is copied into four
   branches — Baseline / Mean / GGI / Maximin — each continued for another
   800K steps (2.0M total), Original 18D observation.
3. **WSC continuation**: the same four conditions, but the first network
   layer is expanded 18D→22D (new welfare-state columns zero-initialised)
   before an equivalent 800K-step continuation.
4. **DWS continuation**: restricted to Maximin; two more 800K-step
   branches (Original+DWS, WSC+DWS) add the step-wise dense welfare
   shaping term on top of the terminal reward.

The actual training entry points invoked as subprocesses are under
`experiments/pilots/study_b_fairness_mappo/scripts/`:
`train_curriculum_stage_highwayenv.py` (task-only curriculum),
`train_curriculum_stage_highwayenv_wsc.py` (WSC continuation),
`train_dqn_direct_welfare.py` (Mean/GGI/Maximin continuation). These are
orchestrated by the launchers in `scripts/`: `scripts/launch_formal.py`
(original 6-seed campaign), `scripts/launch_replication_curriculum.py` +
`launch_replication_welfare.py` (the 6 replication seeds + WSC),
`scripts/launch_dense_priority.py` (DWS). Evaluation:
`scripts/evaluate_formal.py`, `scripts/evaluate_replication.py`,
`scripts/evaluate_dense_interim.py` (all use a deterministic 4-checkpoint
Q-ensemble on the held-out `H1` scenario bank, `scenario_banks/H1.json`,
which is also mirrored at
`experiments/pilots/study_b_fairness_mappo/scenario_banks/H1.json`).

**Model checkpoints are intentionally excluded from this repository** (see
Exclusions below). Regenerate them by rerunning the launch scripts above
against the configs in `configs/` — see `docs/protocol/RUNBOOK.md` and
`docs/protocol/FORMAL_EXPERIMENT_FREEZE_MANIFEST.md` for the exact frozen
seed lists and step budgets.

## Repository layout

```
src/thesis/            core package (envs, agents, welfare objectives, study_b/)
tests/                 pytest suite mirroring src/
scripts/               training launchers, evaluators, DWS + whole-thesis analysis
analysis/              RQ1/RQ2/WSC formal statistical analysis (see below)
configs/, protocol/    frozen experiment configs
scenario_banks/        held-out and curriculum scenario banks (JSON)
docs/provenance/       source-bundle manifests and checksums (no git history — see PROVENANCE.md)
docs/protocol/         RUNBOOK, freeze manifests, setup notes from the source bundles
docs/reports/          full statistical write-ups (WSC, DWS, whole-thesis synthesis)
```

`analysis/` contains four sub-pipelines, each copied whole from
`F:\正式训练_seed_replication_v1\analysis_scripts\`:
`wsc_v2_formal/` (WSC outcome-level bootstrap/Holm analysis),
`ch5_baseline/` (RQ1/RQ2 baseline and welfare-objective analysis, Tables
5.2–5.13 in that pipeline's own numbering), `pooled12/` (merges the
original 6 seeds + 6 replication seeds into one 12-seed dataset),
`wsc_v2_behavioural/` (WSC behavioural-mechanism analysis). `analysis/data/`
holds the three merged Original-observation evaluation CSVs these scripts
read as input.

## Thesis table/figure → source file mapping

Table numbers below are as labelled by the analysis pipeline's own output
filenames, not necessarily identical to the final thesis chapter's
renumbered Table/Figure sequence — match by content/column, not by
assuming the numbers line up exactly, since the thesis manuscript went
through a later renumbering pass independent of this repository.

| Thesis content | Source script | Source output file (in this repo) |
|---|---|---|
| RQ1 baseline welfare / worst-off identity | `analysis/ch5_baseline/phase1_analysis*.py` | `analysis/ch5_baseline/outputs/table5_2.csv`, `table5_3.csv`, `table5_4.csv` |
| RQ1 role-conditional burden | `analysis/ch5_baseline/phase1_analysis_part2.py`/`part3.py` | `analysis/ch5_baseline/outputs/table5_5_panelA.csv`, `table5_5_panelB.csv`, `table5_6_panelA.csv`, `table5_6_panelB.csv` |
| RQ2 condition-minus-baseline contrasts | `analysis/ch5_baseline/phase2_analysis.py` | `analysis/ch5_baseline/outputs/table_condition_minus_baseline_12seed.csv`, `table5_7.csv`, `table5_8.csv` |
| Outcome decomposition of mobility burden | `analysis/ch5_baseline/phase2_analysis_part2.py` | `analysis/ch5_baseline/outputs/table5_9.csv`, `table5_9_gini_contrasts.csv`, `table5_9_outcome_decomposition_seedlevel.csv` |
| Task performance / safety (Baseline+Mean+GGI+Maximin) | `analysis/ch5_baseline/phase1_analysis.py` | `analysis/ch5_baseline/outputs/table5_10.csv`–`table5_13_success.csv` |
| Pooled 12-seed RQ1/RQ2 summary | `analysis/pooled12/merge_and_audit.py` | `analysis/pooled12/outputs/pooled12_rq1_seed_level_metrics.csv`, `pooled12_rq1_h0_h1_summary.csv`, `pooled12_umin_holm.csv`, `pooled12_noninferiority.csv`, `pooled12_worst_off_tie_corrected.csv` |
| WSC outcome-level fairness interactions | `analysis/wsc_v2_formal/wsc_v2_formal_analysis.py` | `analysis/wsc_v2_formal/outputs/wsc_v2_formal_fairness_summary.csv`, `wsc_v2_formal_bootstrap_results.json`, `wsc_v2_leave_one_seed_out.csv`, `wsc_v2_formal_sensitivity_910102.csv` |
| WSC task performance/safety | `analysis/wsc_v2_formal/wsc_v2_formal_analysis.py` | `analysis/wsc_v2_formal/outputs/wsc_v2_formal_safety_summary.csv` |
| WSC behavioural mechanisms (yielding, priority, burden transfer, recovery) | `analysis/wsc_v2_behavioural/wsc_v2_behavioural_run.py` + `_aggregate.py` | `analysis/wsc_v2_behavioural/outputs/wsc_behavioural_primary_effects.csv`, `wsc_behavioural_interactions.csv`, `wsc_behavioural_leave_one_seed_out.csv` |
| DWS primary fairness effects, four-cell outcomes, behavioural/signal diagnostics, task-fairness coupling, seed heterogeneity | `scripts/analyze_dws_formal.py`, `scripts/dws_analyze_primary.py`, `scripts/dws_analyze_mechanisms.py`, `scripts/dws_analyze_signal_and_actions.py` | `C:\dense reward\outputs\dws_final_reevaluation_v1\*.csv` — **not copied into this repository** (outside the approved packaging scope); the scripts that produced them are included under `scripts/`, but their output CSVs are not. Flag for the user: copy that output directory in a follow-up pass if a fully self-contained figure rebuild is wanted. |
| The 7 new Chapter-5 figures added in the visual-revision pass (WSC behavioural forest, DWS primary fairness forest, four-cell seed outcomes, DWS behavioural forest, reward-state diagnostics, task-fairness coupling, seed-effect heatmap) | `ch5_stage1_generate_figures.py`, `ch5_stage1_figG_heatmap.py` | Both scripts live at `C:\dense reward\outputs\whole_thesis_evidence_synthesis_v1\` — **source not identified inside this repository**; not copied per the approved packaging plan. Flag for the user. |

## See also

- [`docs/PROVENANCE.md`](docs/PROVENANCE.md) — why there is no git history,
  and the three-bundle chain this repository was consolidated from.
- [`docs/reports/`](docs/reports/) — full statistical write-ups (DWS
  statistical protocol, final DWS re-evaluation, WSC formal evaluation and
  safety, WSC behavioural mechanism report, whole-thesis evidence
  synthesis and hierarchy).
- [`docs/protocol/`](docs/protocol/) — frozen experiment manifests, RUNBOOK,
  and setup notes carried over from the source bundles.

## Exclusions

Model checkpoints (`*.pt`/`*.pth`/`*.ckpt`), Python virtual environments
and wheel caches, `__pycache__`/`.pytest_cache`, and raw per-step
trajectory/replay data are intentionally not included — they are large and
regenerable by rerunning the scripts in `scripts/` against the frozen
configs in `configs/`.
