# FORMAL_EXPERIMENT_FREEZE_MANIFEST

Frozen: 2026-08-18T21:51:02

6-seed matched block (same seeds across all 3 conditions): `[900101, 900102, 900103, 900104, 910101, 910102]`
Conditions: `['mean', 'ggi', 'maximin']` -- lambda_W = **0.5** (frozen, do not re-search)
GGI weights (ascending): `[0.4, 0.3, 0.2, 0.1]`
Total formal runs: **18**

## Run matrix

| run_id | seed | condition | init checkpoint |
|---|---|---|---|
| mean_900101 | 900101 | mean | `D:\正式训练\checkpoints\formal_init\900101\C64_R50\seed_900101_C64_R50\ckpt_step_1200000.pt` |
| ggi_900101 | 900101 | ggi | `D:\正式训练\checkpoints\formal_init\900101\C64_R50\seed_900101_C64_R50\ckpt_step_1200000.pt` |
| maximin_900101 | 900101 | maximin | `D:\正式训练\checkpoints\formal_init\900101\C64_R50\seed_900101_C64_R50\ckpt_step_1200000.pt` |
| mean_900102 | 900102 | mean | `D:\正式训练\checkpoints\formal_init\900102\C64_R50\seed_900102_C64_R50\ckpt_step_1200000.pt` |
| ggi_900102 | 900102 | ggi | `D:\正式训练\checkpoints\formal_init\900102\C64_R50\seed_900102_C64_R50\ckpt_step_1200000.pt` |
| maximin_900102 | 900102 | maximin | `D:\正式训练\checkpoints\formal_init\900102\C64_R50\seed_900102_C64_R50\ckpt_step_1200000.pt` |
| mean_900103 | 900103 | mean | `D:\正式训练\checkpoints\formal_init\900103\C64_R50\seed_900103_C64_R50\ckpt_step_1200000.pt` |
| ggi_900103 | 900103 | ggi | `D:\正式训练\checkpoints\formal_init\900103\C64_R50\seed_900103_C64_R50\ckpt_step_1200000.pt` |
| maximin_900103 | 900103 | maximin | `D:\正式训练\checkpoints\formal_init\900103\C64_R50\seed_900103_C64_R50\ckpt_step_1200000.pt` |
| mean_900104 | 900104 | mean | `D:\正式训练\checkpoints\formal_init\900104\C64_R50\seed_900104_C64_R50\ckpt_step_1200000.pt` |
| ggi_900104 | 900104 | ggi | `D:\正式训练\checkpoints\formal_init\900104\C64_R50\seed_900104_C64_R50\ckpt_step_1200000.pt` |
| maximin_900104 | 900104 | maximin | `D:\正式训练\checkpoints\formal_init\900104\C64_R50\seed_900104_C64_R50\ckpt_step_1200000.pt` |
| mean_910101 | 910101 | mean | `D:\正式训练\checkpoints\curriculum_910101_910102\910101\C64_R50\seed_910101_C64_R50\ckpt_step_1200000.pt` |
| ggi_910101 | 910101 | ggi | `D:\正式训练\checkpoints\curriculum_910101_910102\910101\C64_R50\seed_910101_C64_R50\ckpt_step_1200000.pt` |
| maximin_910101 | 910101 | maximin | `D:\正式训练\checkpoints\curriculum_910101_910102\910101\C64_R50\seed_910101_C64_R50\ckpt_step_1200000.pt` |
| mean_910102 | 910102 | mean | `D:\正式训练\checkpoints\curriculum_910101_910102\910102\C64_R50\seed_910102_C64_R50\ckpt_step_1200000.pt` |
| ggi_910102 | 910102 | ggi | `D:\正式训练\checkpoints\curriculum_910101_910102\910102\C64_R50\seed_910102_C64_R50\ckpt_step_1200000.pt` |
| maximin_910102 | 910102 | maximin | `D:\正式训练\checkpoints\curriculum_910101_910102\910102\C64_R50\seed_910102_C64_R50\ckpt_step_1200000.pt` |

## Keep-all-seeds rule

900103 retained despite C64_R50 FAIL (0.453 completion, 0.547 collision); no seed selection/exclusion based on any outcome, ever

After this manifest is frozen: **no outcome-dependent method modification is allowed** (RUNBOOK sec 48).