# DWS Final Training Audit — All Four Maximin Cells

Audit for the Dense Welfare Shaping (DWS) final re-evaluation. Verifies, per
cell x seed (48 combinations = 4 cells x 12 formal seeds), that training is
complete and technically valid before any re-evaluation or inferential
analysis is run.

**Verdict: PASS for all 48 cell x seed combinations. No missing checkpoints,
no technical-failure signatures, no re-evaluation failures.**

## Method

- **Checkpoint existence**: verified directly on disk that both the final
  checkpoint (`ckpt_step_2000000.pt`) and the full frozen "final-four"
  ensemble window ({1,850,000; 1,900,000; 1,950,000; 2,000,000}) exist for
  every cell x seed, using the exact directory-naming rules each cell's own
  launcher/evaluation script uses (Cell 1/Cell 3 rules copied verbatim from
  `F:\正式训练_seed_replication_v1\analysis_scripts\wsc_v2_behavioural\wsc_v2_behavioural_run.py`'s
  `original_checkpoint_paths`/`wsc_checkpoint_paths`; Cell 2/Cell 4 rules
  from `C:\dense reward\scripts\launch_dense_priority.py`).
- **No technical failure**: `grep -r "Traceback\|Exception"` across every
  manifest and log file under Cell 1's (`checkpoints/formal_runs/`,
  `checkpoints/seed_replication_v1/welfare/`) and Cell 3's
  (`checkpoints/wsc_formal_runs_v2/`) checkpoint trees returned **zero**
  matches. Cell 2/Cell 4's clean completion (12/12 seeds, exact
  `{"final_step": 2000000, ...}` line, no traceback) was already confirmed
  earlier in this session by direct log inspection.
- **Observation type / DWS on-off / condition**: determined from the
  training-script identity and checkpoint directory naming — NOT from the
  launcher's run-manifest `condition` field, which is **known unreliable**
  (`launch_dense_priority.py` hardcodes `"condition": "maximin"` in every
  manifest it writes regardless of the actual `--priority`/condition used;
  confirmed independently in the GGI branch's own results report and in
  `DENSE_PROTOCOL_VERIFICATION_FOR_THESIS.md`). Cell 1/Cell 2 checkpoints
  come from `train_curriculum_stage_highwayenv.py` (18D, no WSC observation
  argument); Cell 3/Cell 4 come from `train_curriculum_stage_highwayenv_wsc.py`
  (22D, WSC observation). This is authoritative because
  `dws_eval_worker.py`'s `load_ensemble_agents(..., obs_dim=...)` call would
  itself raise `EnsembleValidationError` on a dimension mismatch, and none
  of the 48 re-evaluation jobs raised.
- **DWS frozen shaping parameters** (Cells 2 and 4 only): a single global
  file, `C:\dense reward\configs\dense_reward_protocol_v1.json`, applies
  uniformly to every seed — `dense_shaping_mode="discrete"`,
  `dense_shaping_magnitude=0.0005`, `dense_shaping_epsilon=1e-6`,
  `primary_objective="maximin"`, `shaping_scope="shared_global"`,
  `cohort="fixed_four_vehicle"`, `welfare_source="running_active_attainment"`.
  These are not per-seed values, so there is nothing to verify per-row beyond
  confirming the training script that consumed them (`train_curriculum_stage_highwayenv[_wsc].py`)
  is the one actually invoked, which the checkpoint provenance above confirms.
- **Eval-rollout success**: every one of the 48 `dws_eval_worker.py`
  subprocess invocations (via `dws_final_eval_launcher.py`) returned exit
  code 0 and produced its 256-episode shard; the launcher's own
  "MISSING checkpoint directories, aborting before any launch" precondition
  check (which would have stopped everything before any rollout) never
  triggered.

## Per-cell x seed table (48 rows)

Cell 2 (Maximin + DWS) is the branch that completed training only earlier
in this same session (all 12 seeds reached step 2,000,000 cleanly, resumed
once after an interrupting machine reboot — see
`reports/priority1_maximin_wsc_dense_results.md`'s sibling context and this
session's own resume log for that history). Cells 1, 3, and 4 were already
complete before this audit.

| Cell | Seed | Obs dim | Condition | DWS | Ckpt dir | Final ckpt (2.0M) | Ensemble window complete | Eval rollout succeeded |
|---|---:|---:|---|---|---|---|---|---|
| Cell 1 | 900101 | 18 | maximin | No | `F:\正式训练_seed_replication_v1\checkpoints\formal_runs\maximin_900101\seed_900101_Formal_maximin` | YES | YES | YES |
| Cell 1 | 900102 | 18 | maximin | No | `F:\正式训练_seed_replication_v1\checkpoints\formal_runs\maximin_900102\seed_900102_Formal_maximin` | YES | YES | YES |
| Cell 1 | 900103 | 18 | maximin | No | `F:\正式训练_seed_replication_v1\checkpoints\formal_runs\maximin_900103\seed_900103_Formal_maximin` | YES | YES | YES |
| Cell 1 | 900104 | 18 | maximin | No | `F:\正式训练_seed_replication_v1\checkpoints\formal_runs\maximin_900104\seed_900104_Formal_maximin` | YES | YES | YES |
| Cell 1 | 910101 | 18 | maximin | No | `F:\正式训练_seed_replication_v1\checkpoints\formal_runs\maximin_910101\seed_910101_Formal_maximin` | YES | YES | YES |
| Cell 1 | 910102 | 18 | maximin | No | `F:\正式训练_seed_replication_v1\checkpoints\formal_runs\maximin_910102\seed_910102_Formal_maximin` | YES | YES | YES |
| Cell 1 | 920101 | 18 | maximin | No | `F:\正式训练_seed_replication_v1\checkpoints\seed_replication_v1\welfare\920101\Maximin\seed_920101_Formal_maximin` | YES | YES | YES |
| Cell 1 | 920102 | 18 | maximin | No | `F:\正式训练_seed_replication_v1\checkpoints\seed_replication_v1\welfare\920102\Maximin\seed_920102_Formal_maximin` | YES | YES | YES |
| Cell 1 | 920103 | 18 | maximin | No | `F:\正式训练_seed_replication_v1\checkpoints\seed_replication_v1\welfare\920103\Maximin\seed_920103_Formal_maximin` | YES | YES | YES |
| Cell 1 | 920104 | 18 | maximin | No | `F:\正式训练_seed_replication_v1\checkpoints\seed_replication_v1\welfare\920104\Maximin\seed_920104_Formal_maximin` | YES | YES | YES |
| Cell 1 | 920105 | 18 | maximin | No | `F:\正式训练_seed_replication_v1\checkpoints\seed_replication_v1\welfare\920105\Maximin\seed_920105_Formal_maximin` | YES | YES | YES |
| Cell 1 | 920106 | 18 | maximin | No | `F:\正式训练_seed_replication_v1\checkpoints\seed_replication_v1\welfare\920106\Maximin\seed_920106_Formal_maximin` | YES | YES | YES |
| Cell 2 | 900101 | 18 | maximin | Yes | `C:\dense reward\checkpoints\maximin_dense\maximin_dense_900101\seed_900101_Dense_maximin_dense` | YES | YES | YES |
| Cell 2 | 900102 | 18 | maximin | Yes | `C:\dense reward\checkpoints\maximin_dense\maximin_dense_900102\seed_900102_Dense_maximin_dense` | YES | YES | YES |
| Cell 2 | 900103 | 18 | maximin | Yes | `C:\dense reward\checkpoints\maximin_dense\maximin_dense_900103\seed_900103_Dense_maximin_dense` | YES | YES | YES |
| Cell 2 | 900104 | 18 | maximin | Yes | `C:\dense reward\checkpoints\maximin_dense\maximin_dense_900104\seed_900104_Dense_maximin_dense` | YES | YES | YES |
| Cell 2 | 910101 | 18 | maximin | Yes | `C:\dense reward\checkpoints\maximin_dense\maximin_dense_910101\seed_910101_Dense_maximin_dense` | YES | YES | YES |
| Cell 2 | 910102 | 18 | maximin | Yes | `C:\dense reward\checkpoints\maximin_dense\maximin_dense_910102\seed_910102_Dense_maximin_dense` | YES | YES | YES |
| Cell 2 | 920101 | 18 | maximin | Yes | `C:\dense reward\checkpoints\maximin_dense\maximin_dense_920101\seed_920101_Dense_maximin_dense` | YES | YES | YES |
| Cell 2 | 920102 | 18 | maximin | Yes | `C:\dense reward\checkpoints\maximin_dense\maximin_dense_920102\seed_920102_Dense_maximin_dense` | YES | YES | YES |
| Cell 2 | 920103 | 18 | maximin | Yes | `C:\dense reward\checkpoints\maximin_dense\maximin_dense_920103\seed_920103_Dense_maximin_dense` | YES | YES | YES |
| Cell 2 | 920104 | 18 | maximin | Yes | `C:\dense reward\checkpoints\maximin_dense\maximin_dense_920104\seed_920104_Dense_maximin_dense` | YES | YES | YES |
| Cell 2 | 920105 | 18 | maximin | Yes | `C:\dense reward\checkpoints\maximin_dense\maximin_dense_920105\seed_920105_Dense_maximin_dense` | YES | YES | YES |
| Cell 2 | 920106 | 18 | maximin | Yes | `C:\dense reward\checkpoints\maximin_dense\maximin_dense_920106\seed_920106_Dense_maximin_dense` | YES | YES | YES |
| Cell 3 | 900101 | 22 | maximin | No | `F:\正式训练_seed_replication_v1\checkpoints\wsc_formal_runs_v2\maximin_wsc_900101\seed_900101_Formal_maximin_WSC_v2` | YES | YES | YES |
| Cell 3 | 900102 | 22 | maximin | No | `F:\正式训练_seed_replication_v1\checkpoints\wsc_formal_runs_v2\maximin_wsc_900102\seed_900102_Formal_maximin_WSC_v2` | YES | YES | YES |
| Cell 3 | 900103 | 22 | maximin | No | `F:\正式训练_seed_replication_v1\checkpoints\wsc_formal_runs_v2\maximin_wsc_900103\seed_900103_Formal_maximin_WSC_v2` | YES | YES | YES |
| Cell 3 | 900104 | 22 | maximin | No | `F:\正式训练_seed_replication_v1\checkpoints\wsc_formal_runs_v2\maximin_wsc_900104\seed_900104_Formal_maximin_WSC_v2` | YES | YES | YES |
| Cell 3 | 910101 | 22 | maximin | No | `F:\正式训练_seed_replication_v1\checkpoints\wsc_formal_runs_v2\maximin_wsc_910101\seed_910101_Formal_maximin_WSC_v2` | YES | YES | YES |
| Cell 3 | 910102 | 22 | maximin | No | `F:\正式训练_seed_replication_v1\checkpoints\wsc_formal_runs_v2\maximin_wsc_910102\seed_910102_Formal_maximin_WSC_v2` | YES | YES | YES |
| Cell 3 | 920101 | 22 | maximin | No | `F:\正式训练_seed_replication_v1\checkpoints\wsc_formal_runs_v2\maximin_wsc_920101\seed_920101_Formal_maximin_WSC_v2` | YES | YES | YES |
| Cell 3 | 920102 | 22 | maximin | No | `F:\正式训练_seed_replication_v1\checkpoints\wsc_formal_runs_v2\maximin_wsc_920102\seed_920102_Formal_maximin_WSC_v2` | YES | YES | YES |
| Cell 3 | 920103 | 22 | maximin | No | `F:\正式训练_seed_replication_v1\checkpoints\wsc_formal_runs_v2\maximin_wsc_920103\seed_920103_Formal_maximin_WSC_v2` | YES | YES | YES |
| Cell 3 | 920104 | 22 | maximin | No | `F:\正式训练_seed_replication_v1\checkpoints\wsc_formal_runs_v2\maximin_wsc_920104\seed_920104_Formal_maximin_WSC_v2` | YES | YES | YES |
| Cell 3 | 920105 | 22 | maximin | No | `F:\正式训练_seed_replication_v1\checkpoints\wsc_formal_runs_v2\maximin_wsc_920105\seed_920105_Formal_maximin_WSC_v2` | YES | YES | YES |
| Cell 3 | 920106 | 22 | maximin | No | `F:\正式训练_seed_replication_v1\checkpoints\wsc_formal_runs_v2\maximin_wsc_920106\seed_920106_Formal_maximin_WSC_v2` | YES | YES | YES |
| Cell 4 | 900101 | 22 | maximin | Yes | `C:\dense reward\checkpoints\maximin_wsc_dense\maximin_wsc_dense_900101\seed_900101_Dense_maximin_wsc_dense` | YES | YES | YES |
| Cell 4 | 900102 | 22 | maximin | Yes | `C:\dense reward\checkpoints\maximin_wsc_dense\maximin_wsc_dense_900102\seed_900102_Dense_maximin_wsc_dense` | YES | YES | YES |
| Cell 4 | 900103 | 22 | maximin | Yes | `C:\dense reward\checkpoints\maximin_wsc_dense\maximin_wsc_dense_900103\seed_900103_Dense_maximin_wsc_dense` | YES | YES | YES |
| Cell 4 | 900104 | 22 | maximin | Yes | `C:\dense reward\checkpoints\maximin_wsc_dense\maximin_wsc_dense_900104\seed_900104_Dense_maximin_wsc_dense` | YES | YES | YES |
| Cell 4 | 910101 | 22 | maximin | Yes | `C:\dense reward\checkpoints\maximin_wsc_dense\maximin_wsc_dense_910101\seed_910101_Dense_maximin_wsc_dense` | YES | YES | YES |
| Cell 4 | 910102 | 22 | maximin | Yes | `C:\dense reward\checkpoints\maximin_wsc_dense\maximin_wsc_dense_910102\seed_910102_Dense_maximin_wsc_dense` | YES | YES | YES |
| Cell 4 | 920101 | 22 | maximin | Yes | `C:\dense reward\checkpoints\maximin_wsc_dense\maximin_wsc_dense_920101\seed_920101_Dense_maximin_wsc_dense` | YES | YES | YES |
| Cell 4 | 920102 | 22 | maximin | Yes | `C:\dense reward\checkpoints\maximin_wsc_dense\maximin_wsc_dense_920102\seed_920102_Dense_maximin_wsc_dense` | YES | YES | YES |
| Cell 4 | 920103 | 22 | maximin | Yes | `C:\dense reward\checkpoints\maximin_wsc_dense\maximin_wsc_dense_920103\seed_920103_Dense_maximin_wsc_dense` | YES | YES | YES |
| Cell 4 | 920104 | 22 | maximin | Yes | `C:\dense reward\checkpoints\maximin_wsc_dense\maximin_wsc_dense_920104\seed_920104_Dense_maximin_wsc_dense` | YES | YES | YES |
| Cell 4 | 920105 | 22 | maximin | Yes | `C:\dense reward\checkpoints\maximin_wsc_dense\maximin_wsc_dense_920105\seed_920105_Dense_maximin_wsc_dense` | YES | YES | YES |
| Cell 4 | 920106 | 22 | maximin | Yes | `C:\dense reward\checkpoints\maximin_wsc_dense\maximin_wsc_dense_920106\seed_920106_Dense_maximin_wsc_dense` | YES | YES | YES |

## Stop-condition check (Section 1/3 of the re-evaluation prompt)

No cell x seed combination failed any precondition. Re-evaluation was **not**
stopped/blocked — proceeding to build and run the unified held-out evaluation
was correct per the prompt's own rule ("If any required cell-seed combination
is missing or technically invalid, stop before inferential analysis").
