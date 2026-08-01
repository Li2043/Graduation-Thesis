# Stage 7C-Q1 Integrity Report

Status: **VALID**

Failed checks: 0 / 27

| check | ok | detail |
|-------|----|--------|
| `protocol_tag_git` | True | tag_object=2b10297094984564e3936af124393994b21ffe01 tagged_commit=c8c75207c06c6a0511cac5fb24b644a61def8d14 |
| `protocol_tag_commit` | True | expected=c8c75207c06c6a0511cac5fb24b644a61def8d14 got=c8c75207c06c6a0511cac5fb24b644a61def8d14 |
| `protocol_tag_in_episodes` | True | unique=['stage7c-q1-protocol-v1'] |
| `single_code_commit` | True | commits=['c8c75207c06c6a0511cac5fb24b644a61def8d14'] |
| `config_sha256` | True | {"C:\\Users\\HP\\Desktop\\\u6bd5\u4e1a\u9879\u76ee\\thesis\\final_new_stage7c_q1\\configs\\stage7c_q1.yaml": "df64cc71c3c221e22b1abdb714ff6a45850ae32162e1b8ac8672cf23dc20e248", "C: |
| `inventory_config_sha256` | True | unique=['df64cc71c3c221e22b1abdb714ff6a45850ae32162e1b8ac8672cf23dc20e248'] |
| `seeds_exact_64001_64020` | True | seeds=[64001, 64002, 64003, 64004, 64005, 64006, 64007, 64008, 64009, 64010, 64011, 64012, 64013, 64014, 64015, 64016, 64017, 64018, 64019, 64020] |
| `no_foreign_formal_seeds` | True | foreign=[] |
| `algorithm_double_dqn` | True | double_dqn |
| `condition_baseline` | True | baseline |
| `base_reward_v2` | True | v2_active_time |
| `active_time_cost_0_0005` | True | 0.0005 |
| `max_steps_400000` | True | 400000 |
| `checkpoint_schedule_17` | True | ckpts=[0, 25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000, 225000, 250000, 275000, 300000, 325000, 350000, 375000, 400000] |
| `training_complete_20_seeds` | True | {"planned_seeds": 20, "completed_seeds": 20, "failed_or_incomplete_seeds": [], "max_steps_required": 400000, "logical_checkpoints_per_seed": 17, "logical_seed_checkpoints_expected" |
| `logical_seed_checkpoints_340` | True | n_pairs=340 |
| `episode_count_14080` | True | n=14080 |
| `no_duplicate_episode_keys` | True | duplicate_rows=0 |
| `no_same_key_conflicts` | True | conflict_groups=0 |
| `no_cross_seed_eval_overlap` | True | ok |
| `role_swap_pairs_complete` | True | ok |
| `reward_decomposition_sum` | True | bad_A=0 bad_B=0 |
| `checkpoint_inventory_hash_format` | True | sha_lines=680 full_ckpt_rows=340 |
| `inventory_340_full_checkpoints` | True | n=340 |
| `git_results_no_weight_or_replay` | True | bad=[] |
| `episodes_per_seed_checkpoint` | True | ok |
| `machine_integrity_complete` | True | {"status": "COMPLETE", "actual_episodes": 14080, "duplicate_episode_keys": 0, "cross_seed_eval_overlap_ok": true} |
