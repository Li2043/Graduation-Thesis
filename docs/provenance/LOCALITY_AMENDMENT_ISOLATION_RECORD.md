# Locality Amendment Isolation Record

## Status of the lambda_W=0.5 (MR2 rung 1) experiment at the time this task began

**Correction to this task's stated premise**: by the time this
provenance check was performed, the `lambda_W=0.5` Mean-qualification
run (`MeanQualMR2`, seeds 900101/900102) had **already completed**
(`final_step=2,000,000` written to both logs, both processes exited
naturally) and had already been evaluated and recorded as
**QUALIFIED** (2/2 strict PASS; `lambda_W=0.5` now frozen for Mean/
GGI/Maximin -- see `output/autonomous_highwayenv/GATE_RESULTS.json`
`MEAN_QUALIFICATION_MR2_lambda0.5_AUTHORITATIVE_ENSEMBLE_GATE` and
`MEAN_LAMBDA_FROZEN`, and `AUTONOMOUS_EXPERIMENT_LOG.md`'s "MR2 rung 1
(lambda_W=0.5) QUALIFIED" entry). This was recorded via a read-only
evaluation of the already-finished checkpoints (the standard
`stage_q_ensemble_gate.py` ensemble-gate procedure used at every prior
stage this session) -- **no training process was started, stopped, or
modified to produce that result.**

This record therefore documents the (now-historical) run's provenance
for the purpose the task requested -- confirming the locality
amendment work below did not, and structurally could not, touch it --
rather than describing a process still in flight.

## Provenance (read-only, single check, not repeatedly polled)

- **Experiment identifier**: `MeanQualMR2` (lambda_W=0.5, MR2 rung 1
  of the Mean-qualification reward-recovery ladder, RUNBOOK sec 47).
- **Seeds**: 900101, 900102.
- **Training script**: `experiments/pilots/study_b_fairness_mappo/
  scripts/train_curriculum_stage_highwayenv.py`, invoked with
  `--start-step 1200000 --max-additional-steps 800000
  --welfare-lambda 0.5 --condition mean --resume-from
  checkpoints/autonomous_highwayenv/C64_{seed}/seed_{seed}_C64/
  ckpt_step_1200000.pt --scenario-bank
  scenario_banks/Q.json --action-representation meta_speed`.
- **Working directory used by that run**:
  `C:\Users\HP\Desktop\毕业项目\thesis\final_new_experiment\E34_study_b_highwayenv_migration`.
- **Output/state directories**: `experiments/pilots/study_b_fairness_mappo/
  output/autonomous_highwayenv/MeanQualMR2_{seed}/`,
  `experiments/pilots/study_b_fairness_mappo/checkpoints/
  autonomous_highwayenv/MeanQualMR2_{seed}/`, logs at
  `logs/autonomous_highwayenv_MeanQualMR2_{seed}.log`.
- **Python executable**: `.venv/Scripts/python.exe`, Python 3.14.6.
  This `.venv` is itself a Windows junction into a SEPARATE physical
  location (`E33_stage11_n4_extended_convergence/.venv`) -- it is a
  **shared** environment across at least E33 and E34, not private to
  this run. It is treated below as read-only shared infrastructure
  (running code against it does not modify it; no package
  install/upgrade is performed by this task).
- **Git commit/hash**: **UNKNOWN / N/A** -- verified directly:
  `git status` and `git rev-parse HEAD` both fail with "not a git
  repository (or any of the parent directories)" from
  `E34_study_b_highwayenv_migration`. This directory is not, and is
  not inside, a git working tree. This is recorded as a fact, not
  assumed.
- **Config hash**: not applicable -- configuration is passed entirely
  via CLI flags to the training script (recorded above), not a
  separate config file with its own hash.

## What will remain unchanged

Everything under `experiments/pilots/study_b_fairness_mappo/{output,
checkpoints}/autonomous_highwayenv/MeanQualMR2_{seed}/`,
`logs/autonomous_highwayenv_MeanQualMR2_{seed}.log`, and the tracking
files' `MeanQualMR2`-related entries are historical evidence for the
**old** (unmodified, nearest-3-with-no-range-cutoff) local observation
configuration. The locality amendment work below:

- does not modify `src/thesis/study_b/local_observation.py` or any
  other file in the active `E34_study_b_highwayenv_migration` source
  tree;
- does not modify any config, checkpoint, replay, or output path
  listed above;
- does not touch git state (there is none to touch);
- does not launch any GPU-heavy or CPU-competing work concurrently
  with any active training (none is active at the time this record
  was written).
